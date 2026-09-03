#!/usr/bin/env python3
"""Export CJA paper tables/figures from X1/X2/X3/X5/RescueNet JSON products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


ZH_CLASS = {
    "no-damage": "完好",
    "minor-damage": "轻微损伤",
    "major-damage": "严重损伤",
    "destroyed": "损毁",
}


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write_gsd_table(curve: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"尺度 / 有效 GSD & 精度 & macro-F1 & ECE & 平均熵 \\",
        r"\hline",
    ]
    for row in curve:
        lines.append(
            f"{row['scale']:.2f}$\\times$ / {row['gsd_m']:.2f} m "
            f"& {row['accuracy']:.3f} & {row['macro_f1']:.3f} "
            f"& {row['ece']:.3f} & {row['mean_entropy']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_budget_table(curves: dict, path: Path, budget: float = 0.25) -> None:
    names = [
        ("none", "不下降"),
        ("random", "随机"),
        ("entropy_uncal", "未校准熵"),
        ("entropy_cal", "校准熵"),
        ("expected_gain", "条件期望熵"),
        ("conformal", "Conformal"),
        ("oracle", "Oracle 上界"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"策略 & macro-F1 & 精度 & ECE & Brier & NLL & 净纠正 \\",
        r"\hline",
    ]
    for key, zh in names:
        rows = curves.get(key) or []
        row = next((r for r in rows if abs(float(r["budget"]) - budget) < 1e-9), None)
        if not row:
            continue
        lines.append(
            f"{zh} & {row['macro_f1']:.3f} & {row['accuracy']:.3f} & {row['ece']:.3f} "
            f"& {row['brier']:.3f} & {row['nll']:.3f} "
            f"& {int(row.get('net_corrected', 0))} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_fov_table(curve: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"视图档 & 高度 (m) & GSD (m/px) & 精度 & macro-F1 & Brier & NLL \\",
        r"\hline",
    ]
    labels = {"cruise": "巡航", "mid": "中间", "floor": "下限"}
    for row in curve:
        lines.append(
            f"{labels.get(row['view'], row['view'])} & {row['alt_m']:.1f} & {row['gsd_m']:.2f} "
            f"& {row['accuracy']:.3f} & {row['macro_f1']:.3f} "
            f"& {row['brier']:.3f} & {row['nll']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_agent_strategy_tables(aggregate: dict, out: Path) -> None:
    strategy_order = [
        "A0_HOLD", "A1_RANDOM_MATCHED", "A2_FIXED_MATCHED", "A3U_RAW_ENTROPY",
        "A3_ENTROPY", "A4_CONFORMAL", "A5_EXPECTED", "O_REF",
    ]
    action_order = ["AB_HOLD", "AB_CENTER", "AB_DESCEND", "AB_FULL"]

    def render(names: list[str], path: Path) -> None:
        lines = [
            r"\begin{tabular}{lrrrrr}",
            r"\hline",
            r"配置 & $n$ & 精度 & 复观测 & 净纠正 & 航时 (s) \\",
            r"\hline",
        ]
        for name in names:
            row = aggregate.get(name)
            if not row:
                continue
            flip = row.get("flip_matrix") or {}
            lines.append(
                f"{_tex_escape(name)} & {int(row.get('n', 0))} & {_fmt(row.get('accuracy'))} "
                f"& {int(row.get('n_reobservations', 0))} & {int(flip.get('net_corrected', 0))} "
                f"& {_fmt(row.get('reobserve_flight_time_s'), 1)} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}", ""]
        path.write_text("\n".join(lines), encoding="utf-8")

    render(strategy_order, out / "recheck_strategy_table.tex")
    render(action_order, out / "recheck_motion_ablation_table.tex")


def write_class_table(gsd: dict, path: Path) -> None:
    names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    gt = gsd.get("gt_counts") or {}
    pred = gsd.get("pred_counts_native") or {}
    native = next((r for r in (gsd.get("curve") or []) if abs(float(r["scale"]) - 1.0) < 1e-9), {})
    per = native.get("per_class") or {}
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"类别 & 标注数 & 原生档预测数 & 原生档召回 \\",
        r"\hline",
    ]
    for name in names:
        rec = (per.get(name) or {}).get("recall")
        rec_s = f"{rec:.3f}" if rec is not None else "--"
        lines.append(
            f"{ZH_CLASS.get(name, _tex_escape(name))} & {int(gt.get(name, 0))} "
            f"& {int(pred.get(name, (per.get(name) or {}).get('pred_n', 0)))} "
            f"& {rec_s} \\\\"
        )
    n_flip = gsd.get("n_pred_flip_native_vs_cruise")
    if n_flip is not None:
        lines.append(rf"\hline")
        lines.append(rf"巡航$\leftrightarrow$原生预测翻转 & \multicolumn{{3}}{{r}}{{{n_flip}}} \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rescuenet_table(data: dict, path: Path) -> None:
    names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    per = data.get("per_class") or {}
    acc = data.get("accuracy")
    f1 = data.get("macro_f1")
    ece = data.get("ece")
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"项 & 支持数 & 指标 \\",
        r"\hline",
        f"图像数 & {data.get('n_images')} & -- \\\\",
        f"建筑连通域 & {data.get('n_boxes')} & -- \\\\",
        f"精度 & -- & {'--' if acc is None else f'{acc:.3f}'} \\\\",
        f"macro-F1 & -- & {'--' if f1 is None else f'{f1:.3f}'} \\\\",
        f"ECE & -- & {'--' if ece is None else f'{ece:.3f}'} \\\\",
        r"\hline",
    ]
    for name in names:
        blk = per.get(name) or {}
        rec = blk.get("recall")
        rec_s = f"{rec:.3f}" if rec is not None else "--"
        lines.append(f"{ZH_CLASS.get(name, _tex_escape(name))}召回 & {int(blk.get('n', 0))} & {rec_s} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_identifiability_table(data: dict, path: Path) -> None:
    bl = data.get("building_level") or {}
    ql = data.get("question_level") or {}
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"量 & 建筑级 & 问题级 \\",
        r"\hline",
        f"$n$ & {bl.get('n_buildings', '--')} & {ql.get('n_questions', '--')} \\\\",
        f"巡航精度 & {_fmt(bl.get('acc_cruise'))} & -- \\\\",
        f"下限精度 & {_fmt(bl.get('acc_floor'))} & -- \\\\",
        f"$n_{{\\mathrm{{flip}}}}$ & {bl.get('n_flip', '--')} & {ql.get('n_flip', '--')} \\\\",
        f"可纠正 & {bl.get('n_correctable', '--')} & {ql.get('n_correctable', '--')} \\\\",
        f"有害 & {bl.get('n_harmful', '--')} & {ql.get('n_harmful', '--')} \\\\",
        f"净收益 & {bl.get('net_gain', '--')} & -- \\\\",
        f"判定 & \\multicolumn{{2}}{{r}}{{{_tex_escape(str(data.get('verdict') or '--'))}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(v, nd=3):
    if v is None:
        return "--"
    return f"{float(v):.{nd}f}"


def write_oracle_table(results: dict, path: Path) -> None:
    """X3 oracle 阶梯 L0–L3 四行表。"""
    desc = {
        "L0": "无 oracle（现状）",
        "L1": "oracle 指认",
        "L2": "oracle 初始定位（指认仍生效，M7）",
        "L3": "oracle 初始定位+指认",
    }
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"层级 & 设定 & $n$ & SR (95\% CI) & semSR & NE (m) & SPL \\",
        r"\hline",
    ]
    for name in ("L0", "L1", "L2", "L3"):
        blk = (results.get("configs") or {}).get(name)
        if not blk:
            continue
        a = blk.get("agg") or {}
        n = int(a.get("n") or 0)
        sr = a.get("SR")
        sr_s = "--"
        if sr is not None and n:
            k = round(float(sr) * n)
            lo, hi = _wilson_ci(k, n)
            sr_s = f"{sr:.3f} [{lo:.3f},{hi:.3f}]"
        lines.append(
            f"{name} & {desc[name]} & {n} & {sr_s} "
            f"& {_fmt(a.get('sem_SR'))} & {_fmt(a.get('NE_mean_m'), 1)} & {_fmt(a.get('SPL_mean'))} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_e2e_table(runs: list[tuple[str, Path]], path: Path) -> dict:
    """X6 端到端 + 扰动套件表（M1：附 Wilson 95% CI）。

    runs = [(row_label, run_dir), ...]，run_dir 含 results.json/episodes.jsonl。
    返回 {(label, cfg_name): (k, n)} 供调用方做组间显著性检验（如强制退化几何 vs 主套件 B2）。
    """
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"套件 & 配置 & $n$ & SR (95\% CI) & semSR & NE (m) & 复检触发 \\",
        r"\hline",
    ]
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for label, run_dir in runs:
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        trig_by_cfg: dict[str, int] = {}
        ep_path = run_dir / "episodes.jsonl"
        if ep_path.is_file():
            for line in ep_path.open(encoding="utf-8"):
                r = json.loads(line)
                if r.get("recheck_triggered"):
                    trig_by_cfg[r.get("config", "?")] = trig_by_cfg.get(r.get("config", "?"), 0) + 1
        for cfg_name, blk in (results.get("configs") or {}).items():
            a = blk.get("agg") or {}
            n = int(a.get("n") or 0)
            sr = a.get("SR")
            sr_s = "--"
            if sr is not None and n:
                k = round(float(sr) * n)
                counts[(label, cfg_name)] = (k, n)
                lo, hi = _wilson_ci(k, n)
                sr_s = f"{sr:.3f} [{lo:.3f},{hi:.3f}]"
            trig_s = str(trig_by_cfg.get(cfg_name, 0)) if ep_path.is_file() else "--"
            lines.append(
                f"{_tex_escape(label)} & {_tex_escape(cfg_name)} & {n} & {sr_s} "
                f"& {_fmt(a.get('sem_SR'))} & {_fmt(a.get('NE_mean_m'), 1)} & {trig_s} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return counts


def write_significance_macros(counts: dict, path: Path) -> None:
    """M1：强制退化几何 vs 主套件 B2 的 Fisher 精确检验，写成 \\newcommand 供正文引用。"""
    base = counts.get(("主套件", "B2"))
    degraded = counts.get(("强制退化几何", "B2"))
    lines = []
    if base and degraded:
        k1, n1 = base
        k2, n2 = degraded
        p = _fisher_exact_p(k1, n1 - k1, k2, n2 - k2)
        p_s = f"{p:.3f}" if p is not None else "\\text{n/a}"
        lines.append(f"\\renewcommand{{\\DegradedFisherP}}{{{p_s}}}")
        lines.append(f"\\renewcommand{{\\DegradedFisherBaseK}}{{{k1}}}")
        lines.append(f"\\renewcommand{{\\DegradedFisherBaseN}}{{{n1}}}")
        lines.append(f"\\renewcommand{{\\DegradedFisherDegK}}{{{k2}}}")
        lines.append(f"\\renewcommand{{\\DegradedFisherDegN}}{{{n2}}}")
    else:
        lines.append(r"\renewcommand{\DegradedFisherP}{\text{n/a}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval，用于小 n 下的 SR 95% 置信区间（M1）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((center - half) / denom, (center + half) / denom)


def _fisher_exact_p(a: int, b: int, c: int, d: int) -> float | None:
    """2x2 Fisher 精确检验双侧 p 值（M1 强制退化几何 vs 基线对比），无 scipy 依赖。"""
    try:
        from math import comb
    except ImportError:
        return None
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1 = a + c
    if row1 == 0 or row2 == 0 or col1 == 0 or col1 == n:
        return None

    def hyper_p(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)

    obs_p = hyper_p(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    total = 0.0
    for x in range(lo, hi + 1):
        p = hyper_p(x)
        if p <= obs_p * (1 + 1e-9):
            total += p
    return min(1.0, total)


def write_training_curve_table(curves: dict, path: Path) -> None:
    """C1 证据：train/val macro-F1 逐 epoch 曲线（类别加权训练）。"""
    labels = {
        "v2_weighted_diff_attention": "差分注意力（事件不相交，加权）",
        "v2_weighted_concat": "拼接（事件不相交，加权）",
        "standard_split_weighted_concat": "拼接（官方标准切分，加权）",
    }
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"配置 & epoch & train macro-F1 & val macro-F1 & train acc & val acc \\",
        r"\hline",
    ]
    for key, zh in labels.items():
        rows = curves.get(key) or []
        for row in rows:
            lines.append(
                f"{_tex_escape(zh) if row['epoch'] == 1 else ''} & {row['epoch']} "
                f"& {row['train_macro_f1']:.3f} & {row['val_macro_f1']:.3f} "
                f"& {row['train_acc']:.3f} & {row['val_acc']:.3f} \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_change_perception_v2_table(report: dict, path: Path) -> None:
    """C1/C2/H2 汇总表：新旧 checkpoint × train/val/test 三切分 macro-F1 + destroyed 召回。"""
    labels = [
        ("old_unweighted_diff_attention", "旧（未加权，论文原用）"),
        ("v2_weighted_diff_attention", "v2 差分注意力（加权）"),
        ("v2_weighted_concat", "v2 拼接（加权，H2 对照）"),
        ("standard_split_weighted_concat", "标准切分拼接（加权，协议对照）"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"checkpoint & train F1 & val F1 & test F1 & test 完好召回 & test 损毁召回 & 切分协议 \\",
        r"\hline",
    ]
    ckpts = report.get("checkpoints") or {}
    for key, zh in labels:
        entry = ckpts.get(key)
        if not entry:
            continue
        sp = entry.get("splits") or {}
        train_f1 = (sp.get("train") or {}).get("macro_f1")
        val_f1 = (sp.get("val") or {}).get("macro_f1")
        test = sp.get("test") or {}
        test_f1 = test.get("macro_f1")
        nodmg_rec = ((test.get("per_class") or {}).get("no-damage") or {}).get("recall")
        destroyed_rec = ((test.get("per_class") or {}).get("destroyed") or {}).get("recall")
        protocol = "标准（非事件不相交）" if "standard" in key else "事件不相交"
        lines.append(
            f"{_tex_escape(zh)} & {_fmt(train_f1)} & {_fmt(val_f1)} & {_fmt(test_f1)} "
            f"& {_fmt(nodmg_rec)} & {_fmt(destroyed_rec)} & {protocol} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_power_table(data: dict, path: Path) -> None:
    cur = data.get("current_design") or {}
    lines = [
        r"\begin{tabular}{lr}",
        r"\hline",
        r"量 & 取值 \\",
        r"\hline",
        f"基线 SR & {data.get('baseline_sr')} \\\\",
        f"目标绝对增益 & {data.get('target_delta')} \\\\",
        f"Holm $\\alpha$ & {data.get('holm_alpha')} \\\\",
        f"功效 0.8 所需每策略题数 & {data.get('n_items_per_policy_needed')} \\\\",
        f"旧设计最小可检测 SR & {cur.get('min_detectable_sr')} \\\\",
        r"\hline",
        r"\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(
    gsd: dict | None,
    budget: dict | None,
    fig_dir: Path,
    fov: dict | None = None,
    aggregate: dict | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable: {exc}")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    if gsd and gsd.get("curve"):
        xs = [r["gsd_m"] for r in gsd["curve"]]
        f1 = [r["macro_f1"] for r in gsd["curve"]]
        ece = [r["ece"] for r in gsd["curve"]]
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.plot(xs, f1, "o-", label="macro-F1")
        ax.plot(xs, ece, "s--", label="ECE")
        ax.set_xlabel("Effective GSD (m)")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "gsd_ladder.pdf")
        plt.close(fig)
    if budget and budget.get("curves"):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        labels = {
            "none": "None", "random": "Random", "entropy_uncal": "Uncal. entropy",
            "entropy_cal": "Cal. entropy", "expected_gain": "Cond. E[U]",
            "conformal": "Conformal", "oracle": "Oracle",
        }
        for key, rows in budget["curves"].items():
            xs = [r["budget"] for r in rows]
            ys = [r["macro_f1"] for r in rows]
            ax.plot(xs, ys, marker="o", label=labels.get(key, key))
        ax.set_xlabel("Mean extra observations per sample")
        ax.set_ylabel("Damage macro-F1")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "budget_allocation.pdf")
        plt.close(fig)
    if fov and fov.get("curve"):
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        xs = [r["gsd_m"] for r in fov["curve"]]
        f1 = [r["macro_f1"] for r in fov["curve"]]
        nll = [r.get("nll") for r in fov["curve"]]
        ax.plot(xs, f1, "o-", label="macro-F1")
        if all(v is not None for v in nll):
            ax.plot(xs, nll, "s--", label="NLL")
        ax.set_xlabel("Effective GSD (m/px)")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fov_ladder.pdf")
        plt.close(fig)
    if aggregate:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        plotted = 0
        for name, row in aggregate.items():
            acc = row.get("accuracy")
            t = row.get("reobserve_flight_time_s")
            if acc is None or t is None:
                continue
            ax.scatter([t], [acc], label=name)
            plotted += 1
        if plotted:
            ax.set_xlabel("Extra reobservation flight time (s)")
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / "accuracy_flight_time.pdf")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(REPO / "runs/benchmarks/paper_cja_mech_v1"))
    ap.add_argument("--out-dir", default=str(REPO / "paper_cja/generated"))
    args = ap.parse_args()
    root, out = Path(args.run_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fov_path = root / "fov_ladder_test.json"
    if not fov_path.is_file():
        fov_path = root / "fov_ladder_val.json"
    fov = json.loads(fov_path.read_text(encoding="utf-8")) if fov_path.is_file() else None
    if fov:
        write_fov_table(fov.get("curve") or [], out / "fov_ladder_table.tex")
    gsd = json.loads((root / "gsd_ladder_test.json").read_text()) if (root / "gsd_ladder_test.json").is_file() else None
    items_path = root / "gsd_ladder_test_items.jsonl"
    if gsd and items_path.is_file() and not gsd.get("gt_counts"):
        from collections import Counter
        names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
        ys, p1, p4 = [], [], []
        for line in items_path.open(encoding="utf-8"):
            it = json.loads(line)
            ys.append(int(it["y"]))
            p1.append(it["views"]["1.0"]["pred"])
            p4.append(it["views"]["4.0"]["pred"])
        gsd["gt_counts"] = {names[c]: ys.count(c) for c in range(4)}
        gsd["pred_counts_native"] = dict(Counter(p1))
        gsd["n_pred_flip_native_vs_cruise"] = int(sum(a != b for a, b in zip(p1, p4)))
        native = next((r for r in (gsd.get("curve") or []) if abs(float(r["scale"]) - 1.0) < 1e-9), None)
        if native is not None:
            native["per_class"] = {}
            for c, name in enumerate(names):
                support = ys.count(c)
                pred_n = p1.count(name)
                tp = sum(1 for y, p in zip(ys, p1) if y == c and p == name)
                native["per_class"][name] = {
                    "support": support,
                    "pred_n": pred_n,
                    "recall": (tp / support) if support else 0.0,
                }
    if gsd:
        write_gsd_table(gsd.get("curve") or [], out / "gsd_ladder_table.tex")
        write_class_table(gsd, out / "gsd_class_table.tex")
    budget_path = root / "budget_allocation.json"
    if not budget_path.is_file():
        budget_path = root / "selection" / "budget_allocation.json"
    budget = json.loads(budget_path.read_text()) if budget_path.is_file() else None
    if budget:
        write_budget_table(budget.get("curves") or {}, out / "budget_table.tex")
    aggregate_candidates = [
        root / "final_reports/aggregate.json",
        root / "agent_vqa_reports/aggregate.json",
        root / "aggregate.json",
    ]
    aggregate_path = next((p for p in aggregate_candidates if p.is_file()), None)
    aggregate = None
    if aggregate_path:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        write_agent_strategy_tables(aggregate, out)
    power = json.loads((root / "power_analysis.json").read_text()) if (root / "power_analysis.json").is_file() else None
    if power:
        write_power_table(power, out / "power_table.tex")
    rescue = json.loads((root / "rescuenet_shift.json").read_text()) if (root / "rescuenet_shift.json").is_file() else None
    if rescue:
        write_rescuenet_table(rescue, out / "rescuenet_table.tex")
    x3 = root / "x3_oracle" / "results.json"
    if x3.is_file():
        write_oracle_table(json.loads(x3.read_text(encoding="utf-8")), out / "oracle_table.tex")
    e2e_runs: list[tuple[str, Path]] = []
    for label, sub in [
        ("主套件", "x6_main"),
        ("复检策略族", "x6_e11"),
        ("GPS $\\sigma{=}2$ m", "x6_gps2"),
        ("GPS $\\sigma{=}5$ m", "x6_gps5"),
        ("GPS $\\sigma{=}10$ m", "x6_gps10"),
        ("强制退化几何", "x6_degraded"),
    ]:
        if (root / sub / "results.json").is_file():
            e2e_runs.append((label, root / sub))
    if e2e_runs:
        counts = write_e2e_table(e2e_runs, out / "e2e_table.tex")
        write_significance_macros(counts, out / "significance_macros.tex")
    cpv2_path = root / "change_perception_v2_report.json"
    if cpv2_path.is_file():
        report = json.loads(cpv2_path.read_text(encoding="utf-8"))
        write_change_perception_v2_table(report, out / "change_perception_v2_table.tex")
        write_training_curve_table(report.get("training_curves") or {}, out / "training_curve_table.tex")
    ident_path = root / "boundary_identifiability_eventdisjoint.json"
    if ident_path.is_file():
        write_identifiability_table(
            json.loads(ident_path.read_text(encoding="utf-8")),
            out / "identifiability_eventdisjoint_table.tex",
        )
    plot_curves(gsd, budget, out, fov=fov, aggregate=aggregate)
    print(f"[ok] wrote CJA assets → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
