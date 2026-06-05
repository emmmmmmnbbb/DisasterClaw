#!/usr/bin/env python
"""
plot_benchmarks.py — 把 Bench-A/B/C 的 JSON 渲染为论文级 PNG + Markdown 表格。

读取:
    runs/benchmarks/<run-id>/task_latency.json
    runs/benchmarks/<run-id>/resolution.json
    runs/benchmarks/<run-id>/exception.json

输出:
    /home/lc/ppt_figures/fig5_5_bench_task_latency.png    (堆叠柱 + 30s/10s/3s 预算线)
    /home/lc/ppt_figures/fig5_5_bench_resolution.png      (双轴: YOLO/SegFormer 延迟 + 显存峰值)
    /home/lc/ppt_figures/fig5_5_bench_exception.png       (异常占比柱 + 累计成功率 CDF)
    runs/benchmarks/<run-id>/tables.md                    (3 张论文用表)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RUNS_DIR, SLA  # noqa: E402

PPT_FIG_DIR = Path("/home/lc/ppt_figures")

PHASE_COLORS = {
    "good": "#4ade80",   # green
    "warn": "#fbbf24",   # amber
    "bad": "#ef4444",    # red
    "neutral": "#94a3b8",
}


def _setup_chinese_font() -> None:
    """Try to enable a CJK font so Chinese labels render correctly."""
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "Source Han Sans CN", "WenQuanYi Zen Hei", "Microsoft YaHei",
        "SimHei", "Arial Unicode MS",
        # Debian/Ubuntu often ship "Droid Sans Fallback" as the only CJK font.
        "Droid Sans Fallback",
    ]
    from matplotlib.font_manager import fontManager

    available = {f.name for f in fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        # matplotlib only renders with a single font per text element; pick the
        # CJK-capable family as the primary so Chinese labels render. Latin
        # letters render fine because every CJK font ships with ASCII glyphs.
        matplotlib.rcParams["font.sans-serif"] = [chosen]
        matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


def _bucket_color(value_ms: float | None, budget_ms: float) -> str:
    if value_ms is None:
        return PHASE_COLORS["neutral"]
    if value_ms <= budget_ms:
        return PHASE_COLORS["good"]
    if value_ms <= budget_ms * 1.5:
        return PHASE_COLORS["warn"]
    return PHASE_COLORS["bad"]


# ===================== Bench-A =====================

def plot_task_latency(data: dict, out_png: Path, out_md_lines: list[str]) -> None:
    agg = data.get("aggregate") or {}
    tasks = list(agg.keys())
    if not tasks:
        print("[plot] task_latency: no aggregate data; skipping")
        return
    means = {phase: [] for phase in ("plan_ms", "fly_to_perception_ms", "perception_compute_ms", "report_residual_ms")}
    p50s_total = []
    p95s_total = []
    sla_total = SLA["total_ms"]
    sla_plan = SLA["plan_ms"]
    sla_first_perc = SLA["first_perception_ms"]
    sla_report = SLA["report_ms"]
    for tid in tasks:
        ph = agg[tid]["phases"]
        plan = (ph.get("plan_ms") or {}).get("mean", 0)
        fly = (ph.get("fly_to_perception_ms") or {}).get("mean", 0)
        perc = (ph.get("perception_compute_ms") or {}).get("mean", 0)
        total = (ph.get("total_ms") or {}).get("mean", 0)
        residual = max(0, total - (plan + fly + perc))
        means["plan_ms"].append(plan)
        means["fly_to_perception_ms"].append(fly)
        means["perception_compute_ms"].append(perc)
        means["report_residual_ms"].append(residual)
        p50s_total.append((ph.get("total_ms") or {}).get("p50", 0))
        p95s_total.append((ph.get("total_ms") or {}).get("p95", 0))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(tasks))
    bottom = np.zeros(len(tasks))
    phase_labels = [
        ("plan_ms", "规划 (plan)"),
        ("fly_to_perception_ms", "飞行+对位"),
        ("perception_compute_ms", "感知推理"),
        ("report_residual_ms", "报告/其他"),
    ]
    palette = ["#60a5fa", "#fbbf24", "#34d399", "#a78bfa"]
    for (key, label), color in zip(phase_labels, palette):
        vals = np.array(means[key], dtype=float)
        ax.bar(x, vals / 1000.0, bottom=bottom / 1000.0, label=label, color=color, edgecolor="white")
        bottom = bottom + vals
    # SLA lines
    ax.axhline(sla_total / 1000, color="#ef4444", linestyle="--", linewidth=1.2, label=f"30s 总预算")
    ax.axhline(sla_first_perc / 1000, color="#fbbf24", linestyle=":", linewidth=1.2, label="10s 首帧")
    ax.axhline(sla_plan / 1000, color="#60a5fa", linestyle=":", linewidth=1.0, label="3s 规划")
    # p95 markers
    ax.scatter(x, np.array(p95s_total) / 1000.0, marker="x", color="#dc2626", zorder=5, label="p95 总耗时")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{tid}\n{agg[tid]['task_name']}" for tid in tasks], fontsize=10)
    ax.set_ylabel("耗时 (s)")
    ax.set_title("Bench-A: 五类巡查任务全链路毫秒级耗时分解 (mean ± p95)")
    ax.legend(loc="upper left", fontsize=9, ncols=2)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[plot] task_latency -> {out_png}")

    # ----- Markdown table -----
    out_md_lines.append("\n### 表 5-5  五类巡查任务全链路耗时统计 (单位: ms)\n")
    out_md_lines.append(
        "| 任务 | 名称 | 规划 mean / p95 | 首帧感知 mean / p95 | 感知推理 mean | 总耗时 mean / p95 | < 30s 命中数 |"
    )
    out_md_lines.append("|---|---|---|---|---|---|---|")
    for tid in tasks:
        ph = agg[tid]["phases"]
        c = agg[tid]["sla_compliance"]

        def m(key):
            return (ph.get(key) or {})

        plan = m("plan_ms")
        first_perc = m("submit_to_first_perception_ms")
        perc = m("perception_compute_ms")
        total = m("total_ms")
        out_md_lines.append(
            f"| {tid} | {agg[tid]['task_name']} | "
            f"{int(plan.get('mean',0))} / {int(plan.get('p95',0))} | "
            f"{int(first_perc.get('mean',0))} / {int(first_perc.get('p95',0))} | "
            f"{int(perc.get('mean',0))} | "
            f"{int(total.get('mean',0))} / {int(total.get('p95',0))} | "
            f"{c.get('total_under_30s', 0)} / {agg[tid]['n_valid']} |"
        )


# ===================== Bench-B =====================

def plot_resolution(data: dict, out_png: Path, out_md_lines: list[str]) -> None:
    results = data.get("results") or {}
    res_keys = sorted(results.keys(), key=lambda r: int(r))
    if not res_keys:
        print("[plot] resolution: no results; skipping")
        return
    yolo_mean = [results[r]["yolo_ms"]["mean"] for r in res_keys]
    seg_mean = [results[r]["segformer_ms"]["mean"] for r in res_keys]
    yolo_peak = [results[r]["yolo_peak_mb"]["mean"] for r in res_keys]
    seg_peak = [results[r]["segformer_peak_mb"]["mean"] for r in res_keys]

    x = np.arange(len(res_keys))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    b1 = ax1.bar(x - width / 2, yolo_mean, width, label="YOLO 推理 (ms)", color="#60a5fa")
    b2 = ax1.bar(x + width / 2, seg_mean, width, label="SegFormer 推理 (ms)", color="#34d399")
    ax1.set_xlabel("Patch 分辨率 (px)")
    ax1.set_ylabel("推理延迟 (ms, mean of 20 runs)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r}×{r}" for r in res_keys])
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    for bars in (b1, b2):
        for rect in bars:
            ax1.annotate(
                f"{rect.get_height():.0f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax2 = ax1.twinx()
    if any(p is not None and p > 0 for p in yolo_peak + seg_peak):
        l1, = ax2.plot(x, yolo_peak, marker="o", color="#1d4ed8", label="YOLO 显存峰值 (MB)")
        l2, = ax2.plot(x, seg_peak, marker="s", color="#047857", label="SegFormer 显存峰值 (MB)")
        ax2.set_ylabel("GPU 显存峰值 (MB)")
        ax2.legend(handles=[l1, l2], loc="upper right", fontsize=9)
    else:
        ax2.set_visible(False)

    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Bench-B: YOLO / SegFormer 推理延迟 & 显存峰值随 patch 分辨率变化")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[plot] resolution -> {out_png}")

    # ----- Markdown table -----
    out_md_lines.append("\n### 表 5-6  视觉感知模型在 768→2048 patch 下的推理延迟与显存峰值\n")
    out_md_lines.append(
        "| 分辨率 | YOLO mean / p95 (ms) | SegFormer mean / p95 (ms) | 总流水线 mean (ms) | YOLO peak (MB) | SegFormer peak (MB) |"
    )
    out_md_lines.append("|---|---|---|---|---|---|")
    for r in res_keys:
        row = results[r]
        out_md_lines.append(
            f"| {r}×{r} | "
            f"{row['yolo_ms']['mean']:.1f} / {row['yolo_ms']['p95']:.1f} | "
            f"{row['segformer_ms']['mean']:.1f} / {row['segformer_ms']['p95']:.1f} | "
            f"{row['total_pipeline_ms']['mean']:.1f} | "
            f"{(row['yolo_peak_mb']['mean'] or 0):.0f} | "
            f"{(row['segformer_peak_mb']['mean'] or 0):.0f} |"
        )


# ===================== Bench-C =====================

def plot_exception(data: dict, out_png: Path, out_md_lines: list[str]) -> None:
    counts = data.get("counts") or {}
    rates = data.get("rates") or {}
    cdf = data.get("cdf") or []
    if not counts:
        print("[plot] exception: no counts; skipping")
        return
    cats = ["success", "flight_timeout", "perception_failure", "report_format_error"]
    cat_labels = ["成功", "飞行超时", "感知失败", "报告格式错误"]
    cat_colors = ["#22c55e", "#f97316", "#dc2626", "#a855f7"]

    counts_arr = [counts.get(c, 0) for c in cats]
    rates_arr = [rates.get(c, 0) * 100 for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    bars = ax1.bar(cat_labels, counts_arr, color=cat_colors, edgecolor="white")
    for bar, pct in zip(bars, rates_arr):
        ax1.annotate(
            f"{int(bar.get_height())}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    total_runs = sum(counts_arr)
    ax1.set_ylabel(f"次数 (共 {total_runs} 次任务)")
    ax1.set_title(f"Bench-C(a): {total_runs} 次连续任务异常事件分布")
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    if cdf:
        idx = [c["index"] for c in cdf]
        rate = [c["cum_success_rate"] * 100 for c in cdf]
        ax2.plot(idx, rate, color="#0ea5e9", linewidth=2)
        ax2.axhline(95, color="#22c55e", linestyle="--", linewidth=1.0, label="95% 目标线")
        ax2.fill_between(idx, rate, 0, alpha=0.1, color="#0ea5e9")
        ax2.set_xlim(1, max(idx))
        ax2.set_ylim(0, 105)
        ax2.set_xlabel("任务序号")
        ax2.set_ylabel("累计成功率 (%)")
        ax2.set_title("Bench-C(b): 累计成功率随序号变化")
        ax2.legend(loc="lower right")
        ax2.grid(linestyle=":", alpha=0.4)
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[plot] exception -> {out_png}")

    # ----- Markdown table -----
    total = sum(counts_arr)
    out_md_lines.append(f"\n### 表 5-7  {total} 次连续任务异常事件统计\n")
    out_md_lines.append("| 类别 | 次数 | 占比 |")
    out_md_lines.append("|---|---|---|")
    for c, lab in zip(cats, cat_labels):
        cnt = counts.get(c, 0)
        pct = cnt / max(total, 1) * 100
        out_md_lines.append(f"| {lab} | {cnt} | {pct:.1f}% |")
    out_md_lines.append(f"| **合计** | **{total}** | **100%** |")


# ===================== main =====================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ppt-dir", default=str(PPT_FIG_DIR))
    args = ap.parse_args()

    _setup_chinese_font()

    run_dir = RUNS_DIR / args.run_id
    if not run_dir.is_dir():
        print(f"[plot] run_dir {run_dir} not found", file=sys.stderr)
        return 1
    print(f"[plot] run_dir = {run_dir}")

    ppt_dir = Path(args.ppt_dir)
    ppt_dir.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = [f"# DisasterClaw 仿真平台性能基准测试 (run_id={args.run_id})\n"]

    # Bench-A
    a_path = run_dir / "task_latency.json"
    if a_path.is_file():
        plot_task_latency(json.load(open(a_path)), ppt_dir / "fig5_5_bench_task_latency.png", md_lines)
    else:
        print(f"[plot] {a_path} missing, skip Bench-A figure")

    # Bench-B
    b_path = run_dir / "resolution.json"
    if b_path.is_file():
        plot_resolution(json.load(open(b_path)), ppt_dir / "fig5_5_bench_resolution.png", md_lines)
    else:
        print(f"[plot] {b_path} missing, skip Bench-B figure")

    # Bench-C
    c_path = run_dir / "exception.json"
    if c_path.is_file():
        plot_exception(json.load(open(c_path)), ppt_dir / "fig5_5_bench_exception.png", md_lines)
    else:
        print(f"[plot] {c_path} missing, skip Bench-C figure")

    # SLA reference appendix
    md_lines.append("\n### 实时性预算参考\n")
    md_lines.append("| 阶段 | 预算 |")
    md_lines.append("|---|---|")
    md_lines.append(f"| 全链路总耗时 | < {SLA['total_ms']/1000:.0f} s |")
    md_lines.append(f"| 规划 (planning) | < {SLA['plan_ms']/1000:.0f} s |")
    md_lines.append(f"| 首帧感知 (submit→first perception) | < {SLA['first_perception_ms']/1000:.0f} s |")
    md_lines.append(f"| 报告生成 (submit→ai_execution_report) | < {SLA['report_ms']/1000:.0f} s |")

    md_path = run_dir / "tables.md"
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_lines) + "\n")
    print(f"[plot] tables -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
