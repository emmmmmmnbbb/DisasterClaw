#!/usr/bin/env python3
"""
scripts/benchmarks/bench_report.py — P4 成绩单聚合（多 run → 表）

把一个或多个 bench 运行目录里的 episodes.jsonl 读进来，按配置/难度/灾种聚合，产出：
    1) 主消融表（E1）：每个配置 SR/semSR/NE/semNE/SPL/Steps/ΔU/judge_acc
    2) E8 难度分桶：配置 × {easy,medium,hard} → SR/NE
    3) E3 复核价值：B1(无复核) vs B2(复核) → ΔU/judge_acc/步数代价
    4) E6 跨灾种：配置 × 灾种 → SR/NE
    5) （若多 run 含不同 grounder）E2 grounder 对比：按 grounder 聚合
    6) P6：配置两两 SR 差异的 bootstrap 95% CI + 配对检验（paired permutation test），
       给消融结论一个"差异是不是噪声"的显著性判断，不再只看点估计。

用法：
    # 单个 E1 run（一个目录里就含 B0~B3 全部 episode）
    python scripts/benchmarks/bench_report.py runs/benchmarks/<run_id>
    # 多个 run 合并（如 E1 + 各 grounder 的 E2）
    python scripts/benchmarks/bench_report.py runs/benchmarks/<r1> runs/benchmarks/<r2> ...
    # 不传参 = 自动取最新一个 run
    # 关闭 P6 显著性表（题量很小时 CI 区间没有意义，可跳过）： --no-significance
输出：在第一个 run 目录写 report.md（也打印到 stdout）。
"""

from __future__ import annotations

import argparse
import json
import random
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
    "E11_NONE": "E11: 不复核（=B1）",
    "E11_RANDOM": "E11: 随机复核 p=0.5",
    "E11_FIXED": "E11: 固定降高复核",
    "E11_HEURISTIC": "E11: 现有启发式（=B2）",
    "E11_ENTROPY": "E11: 校准熵驱动",
    "E11_INFOGAIN": "E11: 校准熵+信息增益触发",
    "E12_OFF": "E12: OROI 自由选择（=B1）",
    "E12_ON": "E12: OROI 打分融合",
}
CONFIG_ORDER = ["B0", "B1", "B2", "B3"]
DIFF_ORDER = ["easy", "medium", "hard"]


def config_order(by_cfg: dict) -> list[str]:
    """B0~B3 在前（保持原有顺序，若存在），其余任意配置名（如 E11_*/E12_*）
    按首次出现顺序追加——支持 bench_vln_navigation.py 里新增的 E11/E12/E13 配置，
    不再要求 report.py 提前认识每一个新配置名。"""
    known = [c for c in CONFIG_ORDER if c in by_cfg]
    others = [c for c in by_cfg if c not in CONFIG_ORDER]
    return known + others


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _rate(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(1 for x in xs if x) / len(xs), 3) if xs else None


def damage_macro_f1(rows: list[dict]) -> float | None:
    classes = sorted({r.get("goal_class") for r in rows if r.get("goal_class")})
    if not classes:
        return None
    scores = []
    for cls in classes:
        tp = sum(1 for r in rows if r.get("goal_class") == cls and r.get("pred_class") == cls)
        fp = sum(1 for r in rows if r.get("goal_class") != cls and r.get("pred_class") == cls)
        fn = sum(1 for r in rows if r.get("goal_class") == cls and r.get("pred_class") != cls)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return round(sum(scores) / len(scores), 3)


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
        "recheck_triggered": _mean([r.get("recheck_triggered") for r in rows]),
        "recheck_completed": _mean([r.get("recheck_completed") for r in rows]),
        "recheck_pending": _mean([r.get("recheck_pending") for r in rows]),
        "recheck_extra_actions": _mean([r.get("recheck_extra_actions") for r in rows]),
        "recheck_extra_horizontal_m": _mean(
            [r.get("recheck_extra_horizontal_m") for r in rows]
        ),
        "judge": _rate([r.get("judge_ok") for r in rows]),
        "macro_f1": damage_macro_f1(rows),
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
         "| 配置 | 说明 | n | SR | semSR | NE(m) | semNE(m) | SPL | Steps | ΔU | judge_acc | macro-F1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in config_order(by_cfg):
        a = agg(by_cfg[c])
        L.append(f"| {c} | {CONFIG_DESC.get(c,'')} | {a['n']} | {_fmt(a['SR'])} | {_fmt(a['semSR'])} | "
                 f"{_fmt(a['NE'])} | {_fmt(a['semNE'])} | {_fmt(a['SPL'])} | {_fmt(a['Steps'])} | "
                 f"{_fmt(a['dU'])} | {_fmt(a['judge'])} | {_fmt(a['macro_f1'])} |")
    L += ["", "> SR=到达指定 GT；semSR=到达瓦片内任一同类受损建筑；ΔU/judge_acc 仅复核配置(B2/B3)有值。", ""]
    return L


def table_difficulty(eps: list[dict]) -> list[str]:
    by_cfg = group_by(eps, "config")
    L = ["## E8 难度分桶（SR / NE）", "",
         "| 配置 | " + " | ".join(f"{d} SR" for d in DIFF_ORDER) + " | "
         + " | ".join(f"{d} NE" for d in DIFF_ORDER) + " |",
         "|---|" + "---|" * (2 * len(DIFF_ORDER))]
    for c in config_order(by_cfg):
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


def table_recheck_strata(eps: list[dict]) -> list[str]:
    """复核配置按 episode 是否实际触发证据分层，避免零触发样本稀释 ΔU。"""
    rows = [r for r in eps if r.get("evidence_stratum") in {"evidence", "no_evidence"}]
    if not rows:
        return []
    by_cfg = group_by(rows, "config")
    L = [
        "## 复核证据链分层（逐 episode）",
        "",
        "| 配置 | 分层 | n | 触发数 | completed | pending | ΔU | macro-F1 | 额外动作 | 额外水平距离(m) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in config_order(by_cfg):
        by_stratum = group_by(by_cfg[cfg], "evidence_stratum")
        for stratum in ("evidence", "no_evidence"):
            subset = by_stratum.get(stratum, [])
            if not subset:
                continue
            a = agg(subset)
            L.append(
                f"| {cfg} | {stratum} | {a['n']} | {_fmt(a['recheck_triggered'])} | "
                f"{_fmt(a['recheck_completed'])} | {_fmt(a['recheck_pending'])} | "
                f"{_fmt(a['dU'])} | {_fmt(a['macro_f1'])} | {_fmt(a['recheck_extra_actions'])} | "
                f"{_fmt(a['recheck_extra_horizontal_m'])} |"
            )
    L += [
        "",
        "> completed 是正式确认/排除/未定论；pending 是 episode 结束时截断但已纳入 ΔU 的闭环。",
        "",
    ]
    return L


def table_disaster(eps: list[dict]) -> list[str]:
    by_cfg = group_by(eps, "config")
    disasters = sorted({r.get("disaster") for r in eps if r.get("disaster")})
    if len(disasters) < 2:
        return []
    L = ["## E6 跨灾种（SR / NE，按配置）", ""]
    for c in config_order(by_cfg):
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


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float]:
    """P6：均值的 bootstrap 95% CI（默认）。返回 (point_estimate, lo, hi)。

    values 为空/长度<2 时退化返回 (point, point, point)（区间宽度 0，标明"无法估计"）。
    """
    xs = [x for x in values if x is not None]
    if len(xs) < 2:
        point = xs[0] if xs else 0.0
        return point, point, point
    rng = random.Random(seed)
    n = len(xs)
    point = sum(xs) / n
    means = []
    for _ in range(n_resamples):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    lo = means[max(0, lo_idx)]
    hi = means[min(n_resamples - 1, hi_idx)]
    return point, lo, hi


def paired_permutation_test(
    a: list[float], b: list[float], n_perm: int = 2000, seed: int = 42,
) -> float | None:
    """P6：配对置换检验（paired permutation test），返回双尾 p 值。

    针对同一批题目在两个配置下的差值 d_i = a_i - b_i，随机对每个 d_i 翻符号做
    n_perm 次重排，统计 |mean(d_perm)| >= |mean(d_obs)| 的比例——不需要假设正态性，
    比 t 检验更适合 SR(0/1) 这类指标，也适合样本量不大的场景。
    a/b 必须等长且按题目一一配对（同一 id 顺序）；长度<2 或全同则返回 None。
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [ai - bi for ai, bi in zip(a, b) if ai is not None and bi is not None]
    if len(diffs) < 2:
        return None
    obs = abs(sum(diffs) / len(diffs))
    if obs == 0:
        return 1.0
    rng = random.Random(seed)
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signed = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(signed / n) >= obs:
            count += 1
    return count / n_perm


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm family-wise error correction, preserving the input order."""
    valid = sorted(
        ((p, idx) for idx, p in enumerate(p_values) if p is not None),
        key=lambda pair: pair[0],
    )
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    m = len(valid)
    for rank, (p_value, idx) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[idx] = running
    return adjusted


def paired_wilcoxon(a: list[float], b: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p value when SciPy and variation exist."""
    diffs = [left - right for left, right in zip(a, b)]
    if len(diffs) < 2 or all(abs(diff) < 1e-12 for diff in diffs):
        return None
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(a, b, alternative="two-sided", zero_method="zsplit").pvalue)
    except (ImportError, ValueError):
        return None


def table_significance(eps: list[dict]) -> list[str]:
    """P6：主配置两两 SR 差异的 bootstrap CI + 配对检验。"""
    by_cfg = group_by(eps, "config")
    present = config_order(by_cfg)
    if len(present) < 2:
        return []

    L = ["## P6 显著性：配置两两 SR 差异（bootstrap 95% CI + 配对检验）", ""]
    L.append("| 配置 | SR (95% CI) | n |")
    L.append("|---|---|---|")
    for c in present:
        srs = [1.0 if r.get("success") else 0.0 for r in by_cfg[c]]
        point, lo, hi = bootstrap_ci(srs)
        L.append(f"| {c} | {point:.3f} ({lo:.3f}, {hi:.3f}) | {len(srs)} |")
    L.append("")

    L.append("| 配置对 | ΔSR | p (配对置换) | p (Wilcoxon) | Holm p | 说明 |")
    L.append("|---|---|---|---|---|---|")
    # 同一题多次 repeat 先在 item 内平均，避免把相关重复当成独立样本。
    by_id_cfg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in eps:
        rid = r.get("id")
        cfg = r.get("config")
        if rid is None or cfg is None:
            continue
        by_id_cfg[rid][cfg].append(1.0 if r.get("success") else 0.0)
    comparisons = []
    for i, c1 in enumerate(present):
        for c2 in present[i + 1:]:
            a, b = [], []
            for rid, vals in by_id_cfg.items():
                if c1 in vals and c2 in vals:
                    a.append(sum(vals[c1]) / len(vals[c1]))
                    b.append(sum(vals[c2]) / len(vals[c2]))
            if len(a) < 2:
                continue
            p = paired_permutation_test(a, b)
            p_wilcoxon = paired_wilcoxon(a, b)
            delta = round(sum(a) / len(a) - sum(b) / len(b), 3)
            comparisons.append((c1, c2, delta, p, p_wilcoxon, len(a)))
    adjusted = holm_adjust([comparison[3] for comparison in comparisons])
    for (c1, c2, delta, p, p_wilcoxon, n_pairs), p_holm in zip(comparisons, adjusted):
        sig = "显著 (Holm p<0.05)" if (p_holm is not None and p_holm < 0.05) else "不显著"
        L.append(
            f"| {c1} vs {c2} | {delta:+.3f} | "
            f"{_fmt(round(p, 4) if p is not None else None)} | "
            f"{_fmt(round(p_wilcoxon, 4) if p_wilcoxon is not None else None)} | "
            f"{_fmt(round(p_holm, 4) if p_holm is not None else None)} | "
            f"{sig} (n={n_pairs}) |"
        )
    L.append("")
    L.append("> 题量较小时 CI 很宽、p 值不稳定，属正常现象；扩题库规模（E13）后应重跑本表。")
    L.append("")
    return L


def table_recheck_significance(eps: list[dict]) -> list[str]:
    """Evidence-positive policy comparisons for ΔU and damage correctness."""
    rows = [
        row for row in eps
        if str(row.get("config", "")).startswith("E11_")
        and row.get("evidence_stratum") == "evidence"
    ]
    configs = config_order(group_by(rows, "config"))
    if len(configs) < 2:
        return []
    lines = [
        "## E11 证据阳性配对检验（item 内先聚合 repeats）",
        "",
    ]
    for metric, label in (("delta_u", "ΔU"), ("judge_ok", "damage correctness")):
        by_item: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            by_item[row["id"]][row["config"]].append(float(value))
        comparisons = []
        for index, left in enumerate(configs):
            for right in configs[index + 1:]:
                a, b = [], []
                for values in by_item.values():
                    if left in values and right in values:
                        a.append(sum(values[left]) / len(values[left]))
                        b.append(sum(values[right]) / len(values[right]))
                if len(a) < 2:
                    continue
                p_perm = paired_permutation_test(a, b)
                p_wilcoxon = paired_wilcoxon(a, b)
                effect = sum(a_i - b_i for a_i, b_i in zip(a, b)) / len(a)
                comparisons.append((left, right, effect, p_perm, p_wilcoxon, len(a)))
        if not comparisons:
            continue
        adjusted = holm_adjust([comparison[3] for comparison in comparisons])
        lines += [
            f"### {label}",
            "",
            "| 配置对 | 平均配对差 | p (置换) | p (Wilcoxon) | Holm p | n |",
            "|---|---|---|---|---|---|",
        ]
        for comparison, p_holm in zip(comparisons, adjusted):
            left, right, effect, p_perm, p_wilcoxon, n_pairs = comparison
            lines.append(
                f"| {left} vs {right} | {effect:+.4f} | "
                f"{_fmt(round(p_perm, 4) if p_perm is not None else None)} | "
                f"{_fmt(round(p_wilcoxon, 4) if p_wilcoxon is not None else None)} | "
                f"{_fmt(round(p_holm, 4) if p_holm is not None else None)} | "
                f"{n_pairs} |"
            )
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--no-significance", action="store_true", help="跳过 P6 bootstrap CI/配对检验表")
    args = ap.parse_args()

    if args.run_dirs:
        run_dirs = [Path(a) if Path(a).is_absolute() else (REPO_ROOT / a) for a in args.run_dirs]
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
    out += table_recheck_strata(eps)
    out += table_difficulty(eps)
    out += table_grounder(eps)
    out += table_disaster(eps)
    if not args.no_significance:
        out += table_significance(eps)
        out += table_recheck_significance(eps)

    text = "\n".join(out)
    report_path = run_dirs[0] / "report.md"
    report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[report] → {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
