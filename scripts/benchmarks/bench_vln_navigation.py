#!/usr/bin/env python3
"""
scripts/benchmarks/bench_vln_navigation.py — P4-3 VLN 导航评测脚本

读题库（gen_vln_testset.py 产出）→ 对每个配置（B0/B1/B2/B3）逐题调用
app.run_vln_episode_headless（真实感知/规划/复核/记忆链路）→ 由 report + GT
计算 NE / SR / SPL / Steps / ΔU / 判定准确率 → 落 runs/benchmarks/<run_id>/。

配置开关（B0~B3）通过在 import 后**直接改 app 模块级全局**切换（这些开关是
import 时读 env 的常量，运行期改 app.VLN_* 即生效）。

用法（务必先 source .env，让本地 VLM/planner 配置生效）：
    cd backend && set -a && source ../.env && set +a && \
      python ../scripts/benchmarks/bench_vln_navigation.py \
        --configs B0,B1,B2,B3 --limit 8 --repeat 1 --grounder vlm

结果一键复现：固定随机种子，结果自描述（带 env_snapshot），每条 episode 即时落盘。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))  # 必须在 import app 之前，app 内有 `from geo import ...`

DEFAULT_TESTSET = BACKEND / "data" / "benchmarks" / "vln_testset.json"
RUNS_DIR = REPO_ROOT / "runs" / "benchmarks"

# 配置 → (planner, recheck, memory)
CONFIGS = {
    "B0": {"planner": "legacy", "recheck": False, "memory": False, "desc": "baseline 关键词+贪心"},
    "B1": {"planner": "hspm", "recheck": False, "memory": False, "desc": "+HSPM 三层规划"},
    "B2": {"planner": "hspm", "recheck": True, "memory": False, "desc": "+不确定性复核"},
    "B3": {"planner": "hspm", "recheck": True, "memory": True, "desc": "+记忆拓扑图"},
}


def _geodesic_m(lat1, lon1, lat2, lon2) -> float:
    from geo import latlon_to_meters
    n, e = latlon_to_meters(lat1, lon1, lat2, lon2)
    return math.hypot(n, e)


# ── 语义判定：同类最近建筑 ───────────────────────────────────────────────
# class 级指令（如"寻找完全损毁建筑"）灾区遍地同类目标，GT 取了特定一栋并不公平。
# 故额外按"到达任一同类受损建筑"判定：semNE = 终点到瓦片内最近同类建筑的距离。
_DATASET_ROOT = None
_tile_cache: dict = {}


def goal_class_buildings(tile_id: str, goal_class: str) -> list[dict]:
    global _DATASET_ROOT
    import xbd_map
    import xbd_store
    from gen_vln_testset import tile_buildings
    if _DATASET_ROOT is None:
        _DATASET_ROOT = xbd_map.resolve_dataset_root()
    if tile_id not in _tile_cache:
        entry = xbd_store.get_entry(tile_id)
        _tile_cache[tile_id] = tile_buildings(_DATASET_ROOT, entry) if entry else []
    return [b for b in _tile_cache[tile_id] if b.get("class") == goal_class]


def semantic_ne_m(tile_id: str, goal_class: str, lat: float, lon: float) -> float | None:
    cands = goal_class_buildings(tile_id, goal_class)
    if not cands:
        return None
    return min(_geodesic_m(lat, lon, b["lat"], b["lon"]) for b in cands)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def apply_config(app, cfg: dict, grounder: str, mem_path: str) -> None:
    """运行期切换 app 的模块级开关。"""
    app.VLN_PLANNER = cfg["planner"]
    app.VLN_RECHECK = bool(cfg["recheck"])
    app.VLN_MEMORY = bool(cfg["memory"])
    app.VLN_GROUNDER = grounder
    app.VLN_MEMORY_PATH = mem_path
    # 重置记忆图单例，避免跨配置/跨 run 串味
    app._memory_graph = None


def _predicted_class(report: dict) -> str | None:
    """从轨迹末点的类别计数里取占比最高的受损类别（粗略判定准确率用）。"""
    traj = report.get("trajectory") or []
    if not traj:
        return None
    labels = traj[-1].get("labels") or {}
    damaged = {k: v for k, v in labels.items() if "建筑" in k and "无损伤" not in k}
    pool = damaged or labels
    if not pool:
        return None
    return max(pool.items(), key=lambda kv: kv[1])[0]


def eval_episode(report: dict, item: dict) -> dict:
    """由 report + GT 算单题指标。"""
    goals = item.get("goals") or []
    sr_radius = float(item.get("success_radius_m", 25))
    shortest = float(item.get("shortest_path_m", 0.0)) or 0.0
    out: dict = {
        "id": item.get("id"),
        "disaster": item.get("disaster"),
        "difficulty": item.get("difficulty"),
        "multi": bool(item.get("multi")),
        "with_direction": bool(item.get("with_direction")),
        "shortest_path_m": round(shortest, 1),
        "error": report.get("error"),
    }
    if report.get("error") or not report.get("final_pos") or not goals:
        out.update({"success": False, "ne_m": None, "spl": 0.0,
                    "steps": report.get("steps_executed"), "path_len_m": report.get("path_len_m"),
                    "delta_u": None, "pred_class": None, "judge_ok": None})
        return out

    fp = report["final_pos"]
    last = goals[-1]
    ne = _geodesic_m(fp["lat"], fp["lon"], last["lat"], last["lon"])
    success = ne <= sr_radius

    # 语义判定：终点到瓦片内最近"同类"建筑的距离（class 级指令更公平）。
    sem_ne = None
    sem_success = None
    try:
        sem_ne = semantic_ne_m(item.get("tile_id"), last.get("class"), fp["lat"], fp["lon"])
        if sem_ne is not None:
            sem_success = sem_ne <= sr_radius
    except Exception:
        pass
    path_len = float(report.get("path_len_m") or 0.0)
    if success and path_len > 0 and shortest > 0:
        spl = shortest / max(path_len, shortest)
    else:
        spl = 1.0 if (success and shortest <= 0) else 0.0

    rc = report.get("recheck") or {}
    delta_u = rc.get("avg_uncertainty_reduction") if isinstance(rc, dict) else None

    pred = _predicted_class(report)
    judge_ok = (pred == last.get("class")) if (pred is not None) else None

    out.update({
        "success": bool(success),
        "ne_m": round(ne, 2),
        "sem_ne_m": round(sem_ne, 2) if sem_ne is not None else None,
        "sem_success": sem_success,
        "spl": round(spl, 4),
        "steps": report.get("steps_executed"),
        "path_len_m": round(path_len, 2),
        "delta_u": delta_u,
        "pred_class": pred,
        "goal_class": last.get("class"),
        "judge_ok": judge_ok,
        "arrived": bool(report.get("arrived")),
        "planner": report.get("planner"),
        "grounder": report.get("grounder"),
    })
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    succ = [r for r in rows if r.get("success")]
    sem_rows = [r for r in rows if r.get("sem_success") is not None]
    sem_succ = [r for r in sem_rows if r.get("sem_success")]
    agg = {
        "n": n,
        "SR": round(len(succ) / n, 4),
        "sem_SR": round(len(sem_succ) / len(sem_rows), 4) if sem_rows else None,
        "NE_mean_m": _mean([r.get("ne_m") for r in rows]),
        "NE_mean_success_m": _mean([r.get("ne_m") for r in succ]),
        "sem_NE_mean_m": _mean([r.get("sem_ne_m") for r in rows]),
        "SPL_mean": _mean([r.get("spl") for r in rows]),
        "steps_mean": _mean([r.get("steps") for r in rows]),
        "path_len_mean_m": _mean([r.get("path_len_m") for r in rows]),
        "delta_u_mean": _mean([r.get("delta_u") for r in rows]),
        "judge_acc": _mean([1.0 if r.get("judge_ok") else 0.0
                            for r in rows if r.get("judge_ok") is not None]),
    }
    # 难度分桶（E8）
    by_diff = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r.get("difficulty") == d]
        if sub:
            by_diff[d] = {"n": len(sub),
                          "SR": round(sum(1 for r in sub if r.get("success")) / len(sub), 4),
                          "NE_mean_m": _mean([r.get("ne_m") for r in sub])}
    agg["by_difficulty"] = by_diff
    return agg


def md_table(per_config: dict) -> str:
    lines = [
        "| 配置 | 说明 | n | SR | semSR | NE(m) | semNE(m) | SPL | Steps | ΔU |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg_name, blk in per_config.items():
        a = blk["agg"]
        lines.append(
            f"| {cfg_name} | {CONFIGS[cfg_name]['desc']} | {a.get('n')} | "
            f"{a.get('SR')} | {a.get('sem_SR')} | {a.get('NE_mean_m')} | {a.get('sem_NE_mean_m')} | "
            f"{a.get('SPL_mean')} | {a.get('steps_mean')} | {a.get('delta_u_mean')} |"
        )
    return "\n".join(lines)


def env_snapshot() -> dict:
    info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": os.environ.get("DATASET_MODE", "xbd"),
        "perception_device": os.environ.get("PERCEPTION_DEVICE", "cuda"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="VLN 导航评测（消融成绩单）")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--configs", default="B0,B1,B2,B3")
    ap.add_argument("--grounder", default="vlm", choices=["yolo", "vlm", "hybrid"])
    ap.add_argument("--limit", type=int, default=0, help="每个配置最多跑前 N 题（0=全部）")
    ap.add_argument("--repeat", type=int, default=1, help="每题重复次数取平均")
    ap.add_argument("--difficulty", default="", help="只跑某难度 easy/medium/hard（空=全部）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="", help="run_id 附加标签")
    args = ap.parse_args()

    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            print(f"[ERROR] 未知配置 {c}，可选 {list(CONFIGS)}", file=sys.stderr)
            return 2

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    items = testset.get("items", [])
    if args.difficulty:
        items = [it for it in items if it.get("difficulty") == args.difficulty]
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        print("[ERROR] 题库为空。", file=sys.stderr)
        return 2

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] run_id={run_id}  out_dir={out_dir}")
    print(f"[bench] configs={configs} grounder={args.grounder} items={len(items)} repeat={args.repeat}")

    print("[bench] 正在导入 app（首次会加载/预热感知 + 本地 VLM，可能耗时数分钟）...")
    t_import = time.time()
    import app  # noqa: E402  —— 触发 AppState() warmup，加载真实模型
    print(f"[bench] app ready in {time.time() - t_import:.1f}s")

    raw_rows: list[dict] = []
    per_config: dict = {}
    raw_path = out_dir / "episodes.jsonl"
    raw_fp = open(raw_path, "w", encoding="utf-8")

    for cfg_name in configs:
        cfg = CONFIGS[cfg_name]
        mem_path = str(out_dir / f"memory_{cfg_name}.json")
        apply_config(app, cfg, args.grounder, mem_path)
        print(f"\n[bench] ===== {cfg_name} ({cfg['desc']}) "
              f"planner={cfg['planner']} recheck={cfg['recheck']} memory={cfg['memory']} =====")
        cfg_rows: list[dict] = []
        for idx, item in enumerate(items):
            for rep in range(args.repeat):
                set_seed(args.seed + rep)
                t0 = time.time()
                try:
                    report = app.run_vln_episode_headless(item["instruction"], item["start"], source="bench")
                except Exception as exc:
                    report = {"ok": False, "error": f"crash: {exc}", "arrived": False}
                    traceback.print_exc()
                dt = time.time() - t0
                row = eval_episode(report, item)
                row.update({"config": cfg_name, "repeat": rep, "wall_s": round(dt, 1),
                            "instruction": item["instruction"]})
                cfg_rows.append(row)
                raw_rows.append(row)
                raw_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_fp.flush()
                ok = "OK " if row.get("success") else "miss"
                ne = row.get("ne_m")
                print(f"  [{cfg_name}] {idx + 1}/{len(items)} r{rep} {ok} "
                      f"NE={ne}m steps={row.get('steps')} {dt:.1f}s :: {item['instruction'][:28]}")
        per_config[cfg_name] = {"agg": aggregate(cfg_rows)}

    raw_fp.close()

    results = {
        "run_id": run_id,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "env": env_snapshot(),
        "testset": str(args.testset),
        "n_items": len(items),
        "configs": {c: {"switches": CONFIGS[c], **per_config[c]} for c in configs},
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    table = md_table(per_config)
    summary_md = (
        f"# VLN 评测成绩单 — {run_id}\n\n"
        f"- 题库：`{args.testset}`（{len(items)} 题，grounder={args.grounder}，repeat={args.repeat}）\n"
        f"- 设备：{results['env'].get('gpu', 'cpu')}\n\n"
        f"## 主消融表（E1）\n\n{table}\n\n"
        f"> SR=到达指定 GT 那一栋（严格）；semSR=到达瓦片内任一**同类**受损建筑（class 级指令更公平）。\n"
        f"> NE/semNE 同理（米，越小越好）；SPL 越高越好；Steps 越少越好；ΔU 越大说明复核越值。\n"
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print("\n" + table)
    print(f"\n[bench] 完成。结果目录：{out_dir}")
    print(f"[bench]   - results.json  （机器可读全量指标）")
    print(f"[bench]   - summary.md    （成绩单表格）")
    print(f"[bench]   - episodes.jsonl（每条 episode 明细）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
