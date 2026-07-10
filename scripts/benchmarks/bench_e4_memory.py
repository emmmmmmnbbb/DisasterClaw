#!/usr/bin/env python3
"""
scripts/benchmarks/bench_e4_memory.py — E4：记忆"越用越熟"实验（对应 H3）

问题：同一片区域走过一次后，第二趟相似指令是否更省力（步数/路径↓）且成功率不降？

机制（见 memory_graph.py / app._vln_memory_prefly）：episode **成功到达**才把轨迹沉淀进
记忆图，终点节点打上地标标签；下次相似指令在 episode 开始时按 landmarks[-1] 匹配节点 +
地理门控，命中则沿"熟路"预飞到目标附近再精定位。

题材：题库里 14 个瓦片各有 2 条指令，且两条的**最终目标常是同一栋楼**（如都终于"完全损毁建筑"）。
取每瓦片的多目标指令作"第二趟" pass2，单目标指令作"第一趟" pass1。

三种条件，在**同一组 pass2** 上对比（隔离变量=记忆内容）：
  - cold：空记忆（VLN_MEMORY=1 但图为空）——对照。
  - warm：先跑 pass1（真实积累，仅到达才沉淀）再跑 pass2——端到端真实。
  - seed：用 pass2 的 GT 目标**播种**记忆（模拟"上一趟成功"），再跑 pass2——oracle，
          隔离当前 grounding/到达率瓶颈，验证 prefly 机制本身能否省步降 NE。

用法：
    cd backend && set -a && source ../.env && set +a && \
      python ../scripts/benchmarks/bench_e4_memory.py --modes cold,warm,seed
输出：runs/benchmarks/<run_id>_e4/{episodes.jsonl, summary.md}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BENCH_DIR))

DEFAULT_TESTSET = BACKEND / "data" / "benchmarks" / "vln_testset.json"
RUNS_DIR = REPO_ROOT / "runs" / "benchmarks"


def pick_pairs(items: list[dict]) -> list[dict]:
    """取每个含 >=2 指令的瓦片：pass1=单目标(短)、pass2=多目标(长，最终目标作记忆目标)。"""
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_tile[it["tile_id"]].append(it)
    pairs = []
    for tile, group in by_tile.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: len(x.get("goals") or []))
        pass1, pass2 = group[0], group[-1]
        pairs.append({"tile": tile, "pass1": pass1, "pass2": pass2})
    return pairs


def seed_memory(app, mem_path: Path, pass2: dict, merge_m: float) -> bool:
    """用 pass2 最终 GT 目标播种一张记忆图（模拟上一趟成功到达并沉淀）。"""
    from memory_graph import MemoryGraph
    goals = pass2.get("goals") or []
    if not goals:
        return False
    g = MemoryGraph(merge_radius_m=merge_m)
    for goal in goals:  # 把途经 + 最终目标都种上，终点为 landmarks[-1] 匹配目标
        g.add_trajectory(
            [{"lat": float(goal["lat"]), "lon": float(goal["lon"]), "alt": 30.0,
              "labels": {goal["class"]: 1}, "risk": "high", "summary": "seeded"}],
            instruction=pass2["instruction"],
            landmarks=[goal["class"]],
            success=True,
        )
    g.save(mem_path)
    return True


def run_one(app, eval_episode, item: dict) -> dict:
    report = app.run_vln_episode_headless(item["instruction"], item["start"], source="e4")
    m = eval_episode(report or {}, item)
    # 记忆诊断：prefly 是否真触发（report.memory 节点数 > 0 即记忆图非空）
    mem = (report or {}).get("memory") or {}
    m["mem_nodes"] = mem.get("nodes") if isinstance(mem, dict) else None
    return m


def agg(rows: list[dict]) -> dict:
    def mean(key, cond=lambda r: True):
        xs = [r[key] for r in rows if cond(r) and r.get(key) is not None]
        return round(st.mean(xs), 2) if xs else None
    n = len(rows)
    return {
        "n": n,
        "SR": round(sum(1 for r in rows if r.get("success")) / n, 3) if n else None,
        "semSR": round(sum(1 for r in rows if r.get("sem_success")) / n, 3) if n else None,
        "NE": mean("ne_m"),
        "semNE": mean("sem_ne_m"),
        "Steps": mean("steps"),
        "path_m": mean("path_len_m"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--modes", default="cold,warm,seed")
    ap.add_argument("--grounder", default="hybrid", choices=["yolo", "vlm", "hybrid"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个瓦片对（0=全部）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="e4")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    items = json.loads(Path(args.testset).read_text(encoding="utf-8"))["items"]
    pairs = pick_pairs(items)
    if args.limit:
        pairs = pairs[:args.limit]

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.tag}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_f = (out_dir / "episodes.jsonl").open("w", encoding="utf-8")
    print(f"[e4] run_id={run_id}  pairs={len(pairs)}  modes={modes}  grounder={args.grounder}")

    print("[e4] 导入 app（加载模型，数分钟）...")
    import app
    from bench_vln_navigation import set_seed, apply_config, eval_episode

    mem_path = out_dir / "_e4_mem.json"
    B3 = {"planner": "hspm", "recheck": True, "memory": True}
    results: dict[str, list[dict]] = {m: [] for m in modes}

    for pi, pair in enumerate(pairs):
        for mode in modes:
            set_seed(args.seed)
            # 每个 (pair,mode) 用独立记忆文件，互不串味
            mp = mem_path.with_name(f"_e4_mem_{pi}_{mode}.json")
            if mp.exists():
                mp.unlink()
            apply_config(app, B3, args.grounder, str(mp))

            if mode == "warm":
                # 先跑 pass1 真实积累（仅到达才沉淀）
                app.run_vln_episode_headless(pair["pass1"]["instruction"],
                                             pair["pass1"]["start"], source="e4-warm1")
                app._memory_graph = None  # 重载，确保读到 pass1 写盘后的图
            elif mode == "seed":
                seed_memory(app, mp, pair["pass2"], app.VLN_MEMORY_MERGE_M)
                app._memory_graph = None

            m = run_one(app, eval_episode, pair["pass2"])
            m.update({"mode": mode, "tile": pair["tile"], "pair_idx": pi})
            results[mode].append(m)
            ep_f.write(json.dumps(m, ensure_ascii=False) + "\n")
            ep_f.flush()
            print(f"[e4] pair {pi+1}/{len(pairs)} [{mode}] tile={pair['tile'][:30]} "
                  f"NE={m.get('ne_m')} steps={m.get('steps')} mem_nodes={m.get('mem_nodes')}")

    ep_f.close()

    # 成绩单
    lines = [f"# E4 记忆越用越熟成绩单 — {run_id}", "",
             f"- pairs={len(pairs)}（同瓦片重访），grounder={args.grounder}",
             "- pass2（第二趟，多目标指令）在三种记忆状态下对比：",
             "  cold=空记忆 / warm=先跑pass1真实积累 / seed=GT播种(oracle机制上界)", "",
             "| 模式 | n | SR | semSR | NE(m) | semNE(m) | Steps | 路径(m) |",
             "|---|---|---|---|---|---|---|---|"]
    desc = {"cold": "空记忆(对照)", "warm": "真实积累", "seed": "GT播种(oracle)"}
    for mode in modes:
        a = agg(results[mode])
        lines.append(f"| {mode}（{desc.get(mode,'')}） | {a['n']} | {a['SR']} | {a['semSR']} | "
                     f"{a['NE']} | {a['semNE']} | {a['Steps']} | {a['path_m']} |")
    lines += ["", "> 期望：seed 相对 cold 步数/路径/NE 下降 → 记忆机制有效；warm≈cold 说明端到端受"
              "到达率限制（pass1 少有成功→记忆图空）。SR 不应下降。"]
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"[e4] → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
