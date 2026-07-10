#!/usr/bin/env python3
"""
scripts/benchmarks/bench_report.py — P4 成绩单聚合（多 run → 表）

把一个或多个 bench 运行目录里的 episodes.jsonl 读进来，按配置/难度/灾种聚合，产出：
    1) 主消融表（E1）：每个配置 SR/semSR/NE/semNE/SPL/Steps/ΔU/judge_acc
    2) E8 难度分桶：配置 × {easy,medium,hard} → SR/NE
    3) E3 复核价值：B1(无复核) vs B2(复核) → ΔU/judge_acc/步数代价
    4) E6 跨灾种：配置 × 灾种 → SR/NE
    5) （若多 run 含不同 grounder）E2 grounder 对比：按 grounder 聚合

用法：
    # 单个 E1 run（一个目录里就含 B0~B3 全部 episode）
    python scripts/benchmarks/bench_report.py runs/benchmarks/<run_id>
    # 多个 run 合并（如 E1 + 各 grounder 的 E2）
    python scripts/benchmarks/bench_report.py runs/benchmarks/<r1> runs/benchmarks/<r2> ...
    # 不传参 = 自动取最新一个 run
输出：在第一个 run 目录写 report.md（也打印到 stdout）。
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs" / "benchmarks"
CONFIG_DESC = {
    "B0": "baseline 关键词+贪心",
    "B1": "+HSPM 三层规划",
    "B2": "+不确定性复核",
    "B3": "+记忆拓扑图",
}
CONFIG_ORDER = ["B0", "B1", "B2", "B3"]
DIFF_ORDER = ["easy", "medium", "hard"]


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _rate(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(1 for x in xs if x) / len(xs), 3) if xs else None


def load_episodes(run_dirs: list[Path]) -> list[dict]:
    eps: list[dict] = []
    for d in run_dirs:
        f = d / "episodes.jsonl"
        if not f.exists():
            print(f"[warn] 无 episodes.jsonl: {d}", file=sys.stderr)
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                eps.append(json.loads(ln))
    return eps


def agg(rows: list[dict]) -> dict:
    """一组 episode → 指标聚合。"""
    return {
        "n": len(rows),
        "SR": _rate([r.get("success") for r in rows]),
        "semSR": _rate([r.get("sem_success") for r in rows]),
        "NE": _mean([r.get("ne_m") for r in rows]),
        "semNE": _mean([r.get("sem_ne_m") for r in rows]),
        "SPL": _mean([r.get("spl") for r in rows]),
        "Steps": _mean([r.get("steps") for r in rows]),
        "dU": _mean([r.get("delta_u") for r in rows]),
        "judge": _rate([r.get("judge_ok") for r in rows]),
    }


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r.get(key)].append(r)
    return out


def table_main(eps: list[dict]) -> list[str]:
    by_cfg = group_by(eps, "config")
    L = ["## 主消融表（E1）", "",
         "| 配置 | 说明 | n | SR | semSR | NE(m) | semNE(m) | SPL | Steps | ΔU | judge_acc |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONFIG_ORDER:
        if c not in by_cfg:
            continue
        a = agg(by_cfg[c])
        L.append(f"| {c} | {CONFIG_DESC.get(c,'')} | {a['n']} | {_fmt(a['SR'])} | {_fmt(a['semSR'])} | "
                 f"{_fmt(a['NE'])} | {_fmt(a['semNE'])} | {_fmt(a['SPL'])} | {_fmt(a['Steps'])} | "
                 f"{_fmt(a['dU'])} | {_fmt(a['judge'])} |")
    L += ["", "> SR=到达指定 GT；semSR=到达瓦片内任一同类受损建筑；ΔU/judge_acc 仅复核配置(B2/B3)有值。", ""]
    return L


def table_difficulty(eps: list[dict]) -> list[str]:
    by_cfg = group_by(eps, "config")
    L = ["## E8 难度分桶（SR / NE）", "",
         "| 配置 | " + " | ".join(f"{d} SR" for d in DIFF_ORDER) + " | "
         + " | ".join(f"{d} NE" for d in DIFF_ORDER) + " |",
         "|---|" + "---|" * (2 * len(DIFF_ORDER))]
    for c in CONFIG_ORDER:
        if c not in by_cfg:
            continue
        bd = group_by(by_cfg[c], "difficulty")
        srs = [_rate([r.get("success") for r in bd.get(d, [])]) for d in DIFF_ORDER]
        nes = [_mean([r.get("ne_m") for r in bd.get(d, [])]) for d in DIFF_ORDER]
        L.append(f"| {c} | " + " | ".join(_fmt(x) for x in srs) + " | "
                 + " | ".join(_fmt(x) for x in nes) + " |")
    L.append("")
    return L


def table_recheck(eps: list[dict]) -> list[str]:
    """E3：B1(无复核) vs B2(复核)。"""
    by_cfg = group_by(eps, "config")
    if "B1" not in by_cfg or "B2" not in by_cfg:
        return []
    a1, a2 = agg(by_cfg["B1"]), agg(by_cfg["B2"])
    d_steps = None
    if a1["Steps"] is not None and a2["Steps"] is not None:
        d_steps = round(a2["Steps"] - a1["Steps"], 3)
    L = ["## E3 复核到底值不值（B1 无复核 vs B2 复核）", "",
         "| 指标 | B1 | B2 | 变化 |",
         "|---|---|---|---|",
         f"| ΔU（不确定性下降） | {_fmt(a1['dU'])} | {_fmt(a2['dU'])} | 复核独有 |",
         f"| judge_acc（判定准确率） | {_fmt(a1['judge'])} | {_fmt(a2['judge'])} | — |",
         f"| semNE(m) | {_fmt(a1['semNE'])} | {_fmt(a2['semNE'])} | — |",
         f"| Steps（代价） | {_fmt(a1['Steps'])} | {_fmt(a2['Steps'])} | {_fmt(d_steps)} |",
         "",
         "> 论证：B2 用「多花的步数」换来 ΔU>0 与更高 judge_acc，即「没把握就再看一眼」是否划算。", ""]
    return L


def table_disaster(eps: list[dict]) -> list[str]:
    by_cfg = group_by(eps, "config")
    disasters = sorted({r.get("disaster") for r in eps if r.get("disaster")})
    if len(disasters) < 2:
        return []
    L = ["## E6 跨灾种（SR / NE，按配置）", ""]
    for c in CONFIG_ORDER:
        if c not in by_cfg:
            continue
        bd = group_by(by_cfg[c], "disaster")
        L.append(f"### {c}")
        L.append("| 灾种 | n | SR | NE(m) |")
        L.append("|---|---|---|---|")
        for ds in disasters:
            rows = bd.get(ds, [])
            if not rows:
                continue
            L.append(f"| {ds} | {len(rows)} | {_fmt(_rate([r.get('success') for r in rows]))} | "
                     f"{_fmt(_mean([r.get('ne_m') for r in rows]))} |")
        L.append("")
    return L


def table_grounder(eps: list[dict]) -> list[str]:
    """E2：若数据里含多种 grounder，按 grounder 聚合（固定看 B1）。"""
    grounders = sorted({r.get("grounder") for r in eps if r.get("grounder")})
    if len(grounders) < 2:
        return []
    b1 = [r for r in eps if r.get("config") == "B1"]
    if not b1:
        return []
    L = ["## E2 grounder 三选一（固定 B1）", "",
         "| grounder | n | SR | semSR | NE(m) | semNE(m) | Steps | 平均耗时(s) |",
         "|---|---|---|---|---|---|---|---|"]
    for g in grounders:
        rows = [r for r in b1 if r.get("grounder") == g]
        if not rows:
            continue
        a = agg(rows)
        wall = _mean([r.get("wall_s") for r in rows])
        L.append(f"| {g} | {a['n']} | {_fmt(a['SR'])} | {_fmt(a['semSR'])} | {_fmt(a['NE'])} | "
                 f"{_fmt(a['semNE'])} | {_fmt(a['Steps'])} | {_fmt(wall)} |")
    L.append("")
    return L


def main() -> int:
    args = sys.argv[1:]
    if args:
        run_dirs = [Path(a) if Path(a).is_absolute() else (REPO_ROOT / a) for a in args]
    else:
        runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)
        if not runs:
            print("无 run 可聚合", file=sys.stderr)
            return 1
        run_dirs = [runs[-1]]
        print(f"[report] 未指定，取最新 run: {run_dirs[0].name}", file=sys.stderr)

    eps = load_episodes(run_dirs)
    if not eps:
        print("无 episode 数据", file=sys.stderr)
        return 1

    out = [f"# VLN P4 成绩单聚合（{len(eps)} episodes，{len(run_dirs)} run）", "",
           "- 来源：" + ", ".join(f"`{d.name}`" for d in run_dirs), ""]
    out += table_main(eps)
    out += table_recheck(eps)
    out += table_difficulty(eps)
    out += table_grounder(eps)
    out += table_disaster(eps)

    text = "\n".join(out)
    report_path = run_dirs[0] / "report.md"
    report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[report] → {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
