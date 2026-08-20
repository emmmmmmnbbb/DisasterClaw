#!/usr/bin/env python3
"""scripts/benchmarks/export_agent_vqa_assets.py — Agent-VQA 论文资产生成 (计划 9.3)

从冻结的正式结果 (report_agent_vqa.py 产出的 aggregate.json + paired_tests.json)
生成 LaTeX 表格与 CSV。禁止硬编码论文结果; 所有数字可追溯到结果 JSON。

产物:
  - tables/main_ablation.tex: 主消融表 (E1-E5), accuracy/abstain/flip/steps
  - tables/by_question_type.tex: 分题型表
  - tables/by_event.tex: 分事件表 (跨事件泛化, 计划 P0)
  - tables/paired_tests.tex: 配对显著性表
  - figures/budget_utility.csv: 预算效用曲线数据 (供 pgfplots)

用法:
    python scripts/benchmarks/export_agent_vqa_assets.py \\
        --report-dir runs/benchmarks/cja_agent_vqa/<run_id>/reports \\
        --out-dir paper_cja/assets/agent_vqa
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def tex_main_ablation(agg: dict) -> str:
    cfgs = sorted(agg)
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Config & n & Accuracy & Abstain & Flip & Steps \\\\",
        "\\midrule",
    ]
    for c in cfgs:
        a = agg[c]
        lines.append(
            f"{_esc(c)} & {a.get('n','')} & {a.get('accuracy','')} & "
            f"{a.get('abstain_rate','')} & {a.get('flip_rate','')} & "
            f"{a.get('n_steps_mean','')} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def tex_by_question_type(agg: dict) -> str:
    qts = ("presence", "damage", "count", "spatial")
    cfgs = sorted(agg)
    lines = [
        "\\begin{tabular}{l" + "ccc" * len(qts) + "}",
        "\\toprule",
        "Config & " + " & ".join(
            f"{qt} acc & {qt} abst & {qt} n" for qt in qts) + " \\\\",
        "\\midrule",
    ]
    for c in cfgs:
        bt = agg[c].get("by_question_type", {})
        cells = []
        for qt in qts:
            rec = bt.get(qt, {})
            cells.extend([rec.get("accuracy", "--"), rec.get("abstain_rate", "--"),
                          rec.get("n", "--")])
        lines.append(f"{_esc(c)} & " + " & ".join(str(x) for x in cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def tex_by_event(agg: dict) -> str:
    events = sorted({e for a in agg.values() for e in a.get("by_event", {})})
    cfgs = sorted(agg)
    lines = [
        "\\begin{tabular}{l" + "cc" * len(events) + "}",
        "\\toprule",
        "Config & " + " & ".join(f"{_esc(e)} acc & {_esc(e)} n" for e in events) + " \\\\",
        "\\midrule",
    ]
    for c in cfgs:
        be = agg[c].get("by_event", {})
        cells = []
        for e in events:
            rec = be.get(e, {})
            cells.extend([rec.get("accuracy", "--"), rec.get("n", "--")])
        lines.append(f"{_esc(c)} & " + " & ".join(str(x) for x in cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def tex_paired_tests(paired: dict) -> str:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & n & Mean diff & 95\\% CI & Sig. \\\\",
        "\\midrule",
    ]
    for k, v in paired.items():
        if not v.get("n_paired"):
            continue
        ci = v.get("ci95", ["--", "--"])
        sig = "$\\bullet$" if v.get("excludes_zero") else ""
        lines.append(
            f"{_esc(k)} & {v['n_paired']} & {v['mean_difference']} & "
            f"[{ci[0]}, {ci[1]}] & {sig} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent-VQA 论文资产生成 (LaTeX/CSV)")
    ap.add_argument("--report-dir", required=True,
                    help="report_agent_vqa.py 输出目录 (含 aggregate.json/paired_tests.json/curves.json)")
    ap.add_argument("--out-dir", default="", help="资产输出目录")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"[ERROR] 报告目录不存在: {report_dir}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else (report_dir.parent / "assets" / "agent_vqa")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    agg = json.loads((report_dir / "aggregate.json").read_text(encoding="utf-8"))
    (out_dir / "tables" / "main_ablation.tex").write_text(
        tex_main_ablation(agg), encoding="utf-8")
    (out_dir / "tables" / "by_question_type.tex").write_text(
        tex_by_question_type(agg), encoding="utf-8")
    (out_dir / "tables" / "by_event.tex").write_text(tex_by_event(agg), encoding="utf-8")

    paired_fp = report_dir / "paired_tests.json"
    if paired_fp.is_file():
        paired = json.loads(paired_fp.read_text(encoding="utf-8"))
        (out_dir / "tables" / "paired_tests.tex").write_text(
            tex_paired_tests(paired), encoding="utf-8")

    curves_fp = report_dir / "curves.json"
    if curves_fp.is_file():
        curves = json.loads(curves_fp.read_text(encoding="utf-8"))
        # 预算效用曲线 CSV (供 pgfplots): config, n_steps, accuracy
        with open(out_dir / "figures" / "budget_utility.csv", "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["config", "n_steps", "n", "accuracy"])
            for c, rec in curves.items():
                for pt in rec.get("budget_utility", []):
                    w.writerow([c, pt["n_steps"], pt["n"], pt["accuracy"]])
        # 风险覆盖曲线 CSV
        with open(out_dir / "figures" / "risk_coverage.csv", "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["config", "threshold", "coverage", "accuracy", "n"])
            for c, rec in curves.items():
                for pt in rec.get("risk_coverage", []):
                    w.writerow([c, pt["threshold"], pt["coverage"], pt["accuracy"], pt["n"]])

    print(f"[export] 完成。资产目录: {out_dir}")
    print(f"[export]   - tables/main_ablation.tex, by_question_type.tex, by_event.tex, paired_tests.tex")
    print(f"[export]   - figures/budget_utility.csv, risk_coverage.csv")
    print(f"[export] 所有数字源自 {report_dir} 的冻结结果, 未硬编码任何论文数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
