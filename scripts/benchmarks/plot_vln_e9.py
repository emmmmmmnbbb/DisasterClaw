#!/usr/bin/env python3
"""
scripts/benchmarks/plot_vln_e9.py — E9 汇报用图批量生成

两类输出（默认全做）：
  A) 汇总图（读已有 bench 结果，秒级）：E1 消融 / E2 grounder / E8 难度 / E4 记忆 / E6 灾种 / E3 复核
  B) 定性案例图（重跑精选题拿 trajectory，叠在 xBD 瓦片底图上）：成功/失败/记忆 cold vs seed 等

用法：
    cd backend && set -a && source ../.env && set +a && \
      python ../scripts/benchmarks/plot_vln_e9.py \
        --e1-run ../runs/benchmarks/20260623_202108_e1full \
        --e4-run ../runs/benchmarks/20260624_222318_e4 \
        --out ../runs/benchmarks/e9_figures

    # 只要汇总图、不重跑案例：
    python ../scripts/benchmarks/plot_vln_e9.py --summary-only --out ../runs/benchmarks/e9_figures
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BENCH_DIR))

DEFAULT_TESTSET = BACKEND / "data" / "benchmarks" / "vln_testset.json"
CONFIG_ORDER = ["B0", "B1", "B2", "B3"]
CONFIG_LABEL = {
    "B0": "Baseline",
    "B1": "+HSPM",
    "B2": "+复核",
    "B3": "+记忆",
}
DIFF_ORDER = ["easy", "medium", "hard"]


def _geodesic_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from geo import latlon_to_meters
    n, e = latlon_to_meters(lat1, lon1, lat2, lon2)
    return math.hypot(n, e)


def _setup_chinese_font() -> None:
    candidates = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Droid Sans Fallback",
    ]
    from matplotlib.font_manager import fontManager
    available = {f.name for f in fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_episodes(run_dir: Path) -> list[dict]:
    f = run_dir / "episodes.jsonl"
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def agg_rows(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    return {
        "n": n,
        "SR": sum(1 for r in rows if r.get("success")) / n,
        "semSR": sum(1 for r in rows if r.get("sem_success")) / n,
        "NE": st.mean([r["ne_m"] for r in rows if r.get("ne_m") is not None]) if rows else 0,
        "semNE": st.mean([r["sem_ne_m"] for r in rows if r.get("sem_ne_m") is not None]) if rows else 0,
        "Steps": st.mean([r["steps"] for r in rows if r.get("steps") is not None]) if rows else 0,
    }


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[e9] → {path}")


# ── A) 汇总图 ─────────────────────────────────────────────────────

def plot_e1_ablation(eps: list[dict], out: Path) -> None:
    by_cfg = defaultdict(list)
    for r in eps:
        by_cfg[r.get("config")].append(r)
    cfgs = [c for c in CONFIG_ORDER if c in by_cfg]
    if not cfgs:
        return
    metrics = [agg_rows(by_cfg[c]) for c in cfgs]
    x = np.arange(len(cfgs))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1, ax2 = axes
    ax1.bar(x - w / 2, [m["SR"] * 100 for m in metrics], w, label="SR(%)", color="#3b82f6")
    ax1.bar(x + w / 2, [m["semSR"] * 100 for m in metrics], w, label="semSR(%)", color="#22c55e")
    ax1.set_xticks(x)
    ax1.set_xticklabels([CONFIG_LABEL.get(c, c) for c in cfgs])
    ax1.set_ylabel("成功率 (%)")
    ax1.set_title("E1 主消融：成功率")
    ax1.legend()
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    ax2.bar(x - w / 2, [m["NE"] for m in metrics], w, label="NE(m)", color="#f97316")
    ax2.bar(x + w / 2, [m["semNE"] for m in metrics], w, label="semNE(m)", color="#a855f7")
    ax2.set_xticks(x)
    ax2.set_xticklabels([CONFIG_LABEL.get(c, c) for c in cfgs])
    ax2.set_ylabel("导航误差 (m)")
    ax2.set_title("E1 主消融：导航误差")
    ax2.legend()
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("E1 主消融（40题，grounder=hybrid）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, out / "01_e1_ablation.png")


def plot_e2_grounder(e1: list[dict], e2_yolo: list[dict], e2_vlm: list[dict], out: Path) -> None:
    groups = {
        "hybrid": [r for r in e1 if r.get("config") == "B1"],
        "yolo": e2_yolo,
        "vlm": e2_vlm,
    }
    names, semne, sr, steps = [], [], [], []
    for g in ["hybrid", "yolo", "vlm"]:
        rows = groups[g]
        if not rows:
            continue
        a = agg_rows(rows)
        names.append(g)
        semne.append(a["semNE"])
        sr.append(a["semSR"] * 100)
        steps.append(a["Steps"])
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    ax.bar(x, semne, color=["#22c55e", "#60a5fa", "#f472b6"][: len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("semNE (m) ↓")
    ax.set_title("E2 Grounder 三选一（B1×40）")
    for i, (s, stp) in enumerate(zip(sr, steps)):
        ax.text(i, semne[i] + 8, f"semSR={s:.0f}%\nSteps={stp:.1f}", ha="center", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "02_e2_grounder.png")


def plot_e8_difficulty(eps: list[dict], out: Path) -> None:
    by_cfg = defaultdict(lambda: defaultdict(list))
    for r in eps:
        by_cfg[r.get("config")][r.get("difficulty")].append(r)
    cfgs = [c for c in CONFIG_ORDER if c in by_cfg]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DIFF_ORDER))
    w = 0.18
    colors = ["#94a3b8", "#3b82f6", "#f97316", "#22c55e"]
    for i, c in enumerate(cfgs):
        srs = []
        for d in DIFF_ORDER:
            rows = by_cfg[c].get(d, [])
            srs.append(sum(1 for r in rows if r.get("sem_success")) / len(rows) * 100 if rows else 0)
        ax.bar(x + (i - 1.5) * w, srs, w, label=CONFIG_LABEL.get(c, c), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_ORDER)
    ax.set_ylabel("semSR (%)")
    ax.set_title("E8 难度分桶（semSR）")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "03_e8_difficulty.png")


def plot_e4_memory(e4: list[dict], out: Path) -> None:
    by_mode = defaultdict(list)
    for r in e4:
        by_mode[r.get("mode")].append(r)
    modes = [m for m in ("cold", "warm", "seed") if m in by_mode]
    if not modes:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = {"cold": "空记忆", "warm": "真实积累", "seed": "GT播种"}
    metrics = ["SR", "semNE", "Steps"]
    for ax, met, ylab in zip(
        axes,
        ["SR", "semNE", "Steps"],
        ["SR", "semNE(m)", "Steps"],
    ):
        vals = []
        for m in modes:
            a = agg_rows(by_mode[m])
            v = a["SR"] if met == "SR" else a.get("semNE" if met == "semNE" else "Steps", 0)
            if met == "SR":
                v *= 100
            vals.append(v)
        ax.bar(range(len(modes)), vals, color=["#94a3b8", "#fbbf24", "#22c55e"][: len(modes)])
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([titles.get(m, m) for m in modes], fontsize=9)
        ax.set_title(ylab)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("E4 记忆二趟（pass2×14）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, out / "04_e4_memory.png")


def plot_e6_disaster(eps: list[dict], out: Path) -> None:
    rows = [r for r in eps if r.get("config") == "B1"]
    by_d = defaultdict(list)
    for r in rows:
        by_d[r.get("disaster")].append(r)
    disasters = sorted(by_d.keys())
    if not disasters:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nes = [st.mean([r["ne_m"] for r in by_d[d]]) for d in disasters]
    srs = [sum(1 for r in by_d[d] if r.get("sem_success")) / len(by_d[d]) * 100 for d in disasters]
    x = np.arange(len(disasters))
    ax.bar(x, nes, color="#60a5fa")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("-", "\n") for d in disasters], fontsize=8)
    ax.set_ylabel("NE (m)")
    ax.set_title("E6 跨灾种（B1 hybrid，平均 NE）")
    for i, s in enumerate(srs):
        ax.text(i, nes[i] + 10, f"semSR={s:.0f}%", ha="center", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "05_e6_disaster.png")


def plot_e3_recheck(eps: list[dict], out: Path) -> None:
    b1 = [r for r in eps if r.get("config") == "B1"]
    b2 = [r for r in eps if r.get("config") == "B2"]
    if not b1 or not b2:
        return
    a1, a2 = agg_rows(b1), agg_rows(b2)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["semNE(m)", "Steps"]
    v1 = [a1["semNE"], a1["Steps"]]
    v2 = [a2["semNE"], a2["Steps"]]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, v1, w, label="B1 无复核", color="#94a3b8")
    ax.bar(x + w / 2, v2, w, label="B2 复核", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("E3 复核代价 vs 收益（B1 vs B2）")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "06_e3_recheck.png")


def plot_grounding_ablation(out: Path) -> None:
    """静态示意：RescueNet YOLO vs xBD 微调 vs hybrid（来自 E2 结论）。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["vlm-only", "yolo-only\n(xBD)", "hybrid"]
    semne = [242, 242, 193]
    colors = ["#f472b6", "#60a5fa", "#22c55e"]
    bars = ax.barh(labels, semne, color=colors)
    ax.set_xlabel("semNE (m) ↓ 越小越好")
    ax.set_title("Grounding 对比（B1×40，E2）")
    ax.axvline(193, color="#22c55e", linestyle="--", alpha=0.5)
    for b, v in zip(bars, semne):
        ax.text(v + 3, b.get_y() + b.get_height() / 2, f"{v:.0f}m", va="center")
    fig.tight_layout()
    save_fig(fig, out / "07_grounding_story.png")


def plot_pipeline_diagram(out: Path) -> None:
    """VLN 管线示意（纯 matplotlib 框图，汇报用）。"""
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, "语言指令", "#dbeafe"),
        (1.6, "HSPM\n地标拆解", "#e0e7ff"),
        (3.0, "记忆图\n预飞", "#fef3c7"),
        (4.4, "Hybrid\nGrounding", "#dcfce7"),
        (5.8, "飞行\n机动", "#ffedd5"),
        (7.2, "感知\nYOLO+Seg", "#fce7f3"),
        (8.6, "不确定性\n复核", "#f3e8ff"),
        (10.0, "到达\n判定", "#d1fae5"),
    ]
    for x, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x, 0.5), 1.1, 1.0, fc=col, ec="#64748b", lw=1.2))
        ax.text(x + 0.55, 1.0, txt, ha="center", va="center", fontsize=9)
    for x in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[x + 1][0], 1.0), xytext=(boxes[x][0] + 1.15, 1.0),
                    arrowprops=dict(arrowstyle="->", color="#475569"))
    ax.set_title("DisasterClaw VLN 闭环管线（P0~P3）", fontsize=12, pad=12)
    fig.tight_layout()
    save_fig(fig, out / "08_pipeline.png")


def plot_e1_heatmap_semne(eps: list[dict], out: Path) -> None:
    by_cfg = defaultdict(lambda: defaultdict(list))
    for r in eps:
        by_cfg[r.get("config")][r.get("difficulty")].append(r)
    cfgs = [c for c in CONFIG_ORDER if c in by_cfg]
    if not cfgs:
        return
    mat = np.zeros((len(cfgs), len(DIFF_ORDER)))
    for i, c in enumerate(cfgs):
        for j, d in enumerate(DIFF_ORDER):
            rows = by_cfg[c].get(d, [])
            mat[i, j] = st.mean([r["sem_ne_m"] for r in rows if r.get("sem_ne_m") is not None]) if rows else np.nan
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DIFF_ORDER)))
    ax.set_xticklabels(DIFF_ORDER)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([CONFIG_LABEL.get(c, c) for c in cfgs])
    for i in range(len(cfgs)):
        for j in range(len(DIFF_ORDER)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if mat[i, j] > np.nanmean(mat) else "black")
    ax.set_title("E1 semNE 热力图（m，越低越好）")
    fig.colorbar(im, ax=ax, label="semNE (m)")
    fig.tight_layout()
    save_fig(fig, out / "09_e1_heatmap_semne.png")


def plot_e4_paired_cold_seed(e4: list[dict], out: Path) -> None:
    cold = {r["id"]: r for r in e4 if r.get("mode") == "cold"}
    seed = {r["id"]: r for r in e4 if r.get("mode") == "seed"}
    ids = sorted(set(cold) & set(seed))[:10]
    if not ids:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ids))
    w = 0.35
    cold_ne = [cold[i].get("ne_m") or 0 for i in ids]
    seed_ne = [seed[i].get("ne_m") or 0 for i in ids]
    ax.bar(x - w / 2, cold_ne, w, label="cold（空记忆）", color="#94a3b8")
    ax.bar(x + w / 2, seed_ne, w, label="seed（GT播种）", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p{r.get('pair_idx', '?')}" for r in [cold[i] for i in ids]], fontsize=8)
    ax.set_xlabel("pair 序号")
    ax.set_ylabel("NE (m) ↓")
    ax.set_title("E4 同题对照：cold vs seed（pass2×10）")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "10_e4_paired_cold_seed.png")


def plot_e2_full_metrics(e1: list[dict], e2_yolo: list[dict], e2_vlm: list[dict], out: Path) -> None:
    groups = {
        "hybrid": [r for r in e1 if r.get("config") == "B1"],
        "yolo": e2_yolo,
        "vlm": e2_vlm,
    }
    names = [g for g in ("hybrid", "yolo", "vlm") if groups[g]]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, ylab, pct in zip(
        axes,
        ["SR", "semSR", "Steps"],
        ["SR (%)", "semSR (%)", "Steps"],
        [True, True, False],
    ):
        vals = []
        for g in names:
            a = agg_rows(groups[g])
            v = a[key]
            if pct:
                v *= 100
            vals.append(v)
        ax.bar(range(len(names)), vals, color=["#22c55e", "#60a5fa", "#f472b6"][: len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(ylab)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("E2 Grounder 多维对比（B1×40）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, out / "11_e2_full_metrics.png")


def plot_e8_semne(eps: list[dict], out: Path) -> None:
    by_cfg = defaultdict(lambda: defaultdict(list))
    for r in eps:
        by_cfg[r.get("config")][r.get("difficulty")].append(r)
    cfgs = [c for c in CONFIG_ORDER if c in by_cfg]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DIFF_ORDER))
    w = 0.18
    colors = ["#94a3b8", "#3b82f6", "#f97316", "#22c55e"]
    for i, c in enumerate(cfgs):
        nes = []
        for d in DIFF_ORDER:
            rows = by_cfg[c].get(d, [])
            nes.append(st.mean([r["sem_ne_m"] for r in rows if r.get("sem_ne_m") is not None]) if rows else 0)
        ax.bar(x + (i - 1.5) * w, nes, w, label=CONFIG_LABEL.get(c, c), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_ORDER)
    ax.set_ylabel("semNE (m) ↓")
    ax.set_title("E8 难度分桶（semNE）")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "12_e8_semne.png")


def plot_e4_semSR(e4: list[dict], out: Path) -> None:
    by_mode = defaultdict(list)
    for r in e4:
        by_mode[r.get("mode")].append(r)
    modes = [m for m in ("cold", "warm", "seed") if m in by_mode]
    if not modes:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    titles = {"cold": "空记忆", "warm": "真实积累", "seed": "GT播种"}
    srs = [sum(1 for r in by_mode[m] if r.get("sem_success")) / len(by_mode[m]) * 100 for m in modes]
    ax.bar(range(len(modes)), srs, color=["#94a3b8", "#fbbf24", "#22c55e"])
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([titles.get(m, m) for m in modes])
    ax.set_ylabel("semSR (%)")
    ax.set_title("E4 记忆二趟：语义成功率")
    for i, v in enumerate(srs):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_ylim(0, max(srs) * 1.25 + 5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "13_e4_semSR.png")


def plot_b1_scatter(eps: list[dict], out: Path) -> None:
    rows = [r for r in eps if r.get("config") == "B1" and r.get("sem_ne_m") is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"easy": "#22c55e", "medium": "#fbbf24", "hard": "#ef4444"}
    for d in DIFF_ORDER:
        pts = [r for r in rows if r.get("difficulty") == d]
        if not pts:
            continue
        ax.scatter(
            [r["steps"] for r in pts],
            [r["sem_ne_m"] for r in pts],
            c=colors.get(d, "#64748b"),
            label=d,
            alpha=0.75,
            s=60,
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel("semNE (m)")
    ax.set_title("B1 hybrid：步数 vs 语义误差（按难度着色）")
    ax.legend()
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out / "14_b1_scatter_steps_semne.png")


# ── B) 案例轨迹图 ─────────────────────────────────────────────────

def pick_cases(e1: list[dict], e4: list[dict], test_items: dict, max_cases: int) -> list[dict]:
    """挑选多样化案例用于重跑+绘图。"""
    picks: list[dict] = []
    seen: set[str] = set()

    def add(item_id: str, tag: str, run_cfg: dict):
        if item_id in seen or item_id not in test_items:
            return
        seen.add(item_id)
        picks.append({"id": item_id, "tag": tag, "run_cfg": run_cfg, "item": test_items[item_id]})

    # E4 seed 成功
    seed_ok = sorted(
        [r for r in e4 if r.get("mode") == "seed" and r.get("success")],
        key=lambda r: r.get("ne_m") or 999,
    )
    for r in seed_ok[:3]:
        add(r["id"], f"E4-seed成功 NE={r.get('ne_m')}m", {"config": "B3", "grounder": "hybrid", "e4_mode": "seed", "pair_idx": r.get("pair_idx")})

    # E4 cold 失败但 seed 成功（同 id）
    for r in seed_ok[:3]:
        cold = next((x for x in e4 if x.get("mode") == "cold" and x["id"] == r["id"]), None)
        if cold and not cold.get("success"):
            add(r["id"], f"E4-cold对照 NE={cold.get('ne_m')}m", {"config": "B3", "grounder": "hybrid", "e4_mode": "cold", "pair_idx": r.get("pair_idx"), "compare": "cold"})

    # E1 sem 成功
    sem_ok = sorted(
        [r for r in e1 if r.get("config") == "B1" and r.get("sem_success")],
        key=lambda r: r.get("sem_ne_m") or 999,
    )
    for r in sem_ok[:2]:
        add(r["id"], f"E1-sem成功 semNE={r.get('sem_ne_m')}m", {"config": "B1", "grounder": "hybrid"})

    # 典型失败（NE 大）
    fails = sorted(
        [r for r in e1 if r.get("config") == "B1" and not r.get("success")],
        key=lambda r: r.get("ne_m") or 0,
        reverse=True,
    )
    for r in fails[:3]:
        add(r["id"], f"E1-失败 NE={r.get('ne_m')}m {r.get('difficulty')}", {"config": "B1", "grounder": "hybrid"})

    # easy 近失
    easy = [r for r in e1 if r.get("config") == "B1" and r.get("difficulty") == "easy"]
    easy.sort(key=lambda r: r.get("ne_m") or 999)
    for r in easy[:2]:
        add(r["id"], f"easy NE={r.get('ne_m')}m", {"config": "B1", "grounder": "hybrid"})

    return picks[:max_cases]


def _geo_to_px(entry: dict, lat: float, lon: float) -> tuple[float, float]:
    from xbd_map import geo_to_pixel
    t = {"pixel_to_geo": entry["pixel_to_geo"], "geo_to_pixel": entry["geo_to_pixel"]}
    return geo_to_pixel(t, lon, lat)


def plot_case_map(
    item: dict,
    report: dict,
    out_path: Path,
    title: str,
) -> None:
    import xbd_store
    from xbd_map import resolve_dataset_root

    tile_id = item.get("tile_id") or item.get("id", "").rsplit("__", 1)[0]
    entry = xbd_store.get_entry(tile_id)
    if not entry or not entry.get("has_georef"):
        print(f"[e9] skip map (no georef): {tile_id}")
        return
    manifest, _ = xbd_store.load_cached()
    root = Path((manifest or {}).get("dataset_root") or resolve_dataset_root())
    img_path = root / entry["image_relpath"]
    if not img_path.exists():
        print(f"[e9] skip map (no image): {img_path}")
        return

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    def xy(lat, lon):
        x, y = _geo_to_px(entry, lat, lon)
        return x, y

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)

    # 轨迹
    traj = report.get("trajectory") or []
    if traj:
        xs, ys = zip(*[xy(p["lat"], p["lon"]) for p in traj])
        ax.plot(xs, ys, color="#facc15", linewidth=2.5, label="轨迹", zorder=3)
        ax.scatter(xs[0], ys[0], c="#facc15", s=40, zorder=4)

    # 起点
    st = item["start"]
    sx, sy = xy(st["lat"], st["lon"])
    ax.scatter([sx], [sy], c="#3b82f6", s=120, marker="^", label="起点", zorder=5, edgecolors="white")

    # 目标
    for i, g in enumerate(item.get("goals") or []):
        gx, gy = xy(g["lat"], g["lon"])
        ax.scatter([gx], [gy], c="#22c55e", s=150, marker="*", label="目标" if i == 0 else "", zorder=5, edgecolors="white")

    # 终点
    fp = report.get("final_pos") or {}
    if fp.get("lat"):
        fx, fy = xy(fp["lat"], fp["lon"])
        ax.scatter([fx], [fy], c="#ef4444", s=120, marker="X", label="终点", zorder=5, edgecolors="white")

    ne = report.get("final_pos") and item.get("goals")
    ne_txt = ""
    if fp.get("lat") and item.get("goals"):
        g = item["goals"][-1]
        ne = _geodesic_m(fp["lat"], fp["lon"], g["lat"], g["lon"])
        ne_txt = f"NE={ne:.0f}m  SR={'✓' if ne <= item.get('success_radius_m', 25) else '✗'}"

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    instr = item.get("instruction", "")[:36]
    ax.set_title(f"{title}\n{instr}\n{ne_txt}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    save_fig(fig, out_path)


def run_cases_and_plot(picks: list[dict], out: Path, cache_dir: Path) -> None:
    print(f"[e9] 重跑 {len(picks)} 个案例拿 trajectory（有缓存则跳过）...")
    import app
    from bench_vln_navigation import set_seed, apply_config, eval_episode
    from bench_e4_memory import seed_memory, pick_pairs

    testset = json.loads(DEFAULT_TESTSET.read_text(encoding="utf-8"))
    pairs_by_tile = {p["tile"]: p for p in pick_pairs(testset["items"])}

    case_dir = out / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, pick in enumerate(picks):
        item = pick["item"]
        cfg = pick["run_cfg"]
        tag = pick["tag"].replace("/", "-").replace(" ", "_")[:40]
        cache = cache_dir / f"{pick['id']}_{cfg.get('config')}_{cfg.get('e4_mode','')}.json"
        report = None
        if cache.exists():
            report = json.loads(cache.read_text(encoding="utf-8"))
        else:
            set_seed(42)
            mem_path = cache_dir / f"_mem_{i}.json"
            if mem_path.exists():
                mem_path.unlink()
            B = {
                "B0": {"planner": "legacy", "recheck": False, "memory": False},
                "B1": {"planner": "hspm", "recheck": False, "memory": False},
                "B3": {"planner": "hspm", "recheck": True, "memory": True},
            }[cfg.get("config", "B1")]
            apply_config(app, B, cfg.get("grounder", "hybrid"), str(mem_path))
            if cfg.get("e4_mode") == "seed":
                pi = cfg.get("pair_idx")
                if pi is not None:
                    pair_list = pick_pairs(testset["items"])
                    if pi < len(pair_list):
                        seed_memory(app, mem_path, pair_list[pi]["pass2"], app.VLN_MEMORY_MERGE_M)
                        app._memory_graph = None
            report = app.run_vln_episode_headless(item["instruction"], item["start"], source="e9")
            cache.write_text(json.dumps(report or {}, ensure_ascii=False, indent=2), encoding="utf-8")

        plot_case_map(item, report or {}, case_dir / f"case_{i+1:02d}_{tag}.png", pick["tag"])


def write_index(out: Path, fig_paths: list[Path]) -> None:
    lines = ["# E9 汇报用图索引", "", f"生成目录：`{out}`", "", "## 汇总图", ""]
    for p in sorted(p for p in out.glob("*.png") if p.parent == out and p.name[:1].isdigit()):
        lines.append(f"- `{p.name}`")
    lines += ["", "## 定性案例（轨迹叠瓦片）", ""]
    for p in sorted((out / "cases").glob("*.png")):
        lines.append(f"- `cases/{p.name}`")
    lines += [
        "",
        "## 使用建议",
        "- 汇报开场：`08_pipeline.png` + `01_e1_ablation.png`",
        "- Grounding 故事：`07_grounding_story.png` + `02_e2_grounder.png`",
        "- 记忆 H3：`04_e4_memory.png` + cases 里 E4-seed / E4-cold 对照",
        "- 瓶颈分析：`03_e8_difficulty.png` + `05_e6_disaster.png` + `12_e8_semne.png`",
        "- 记忆对照：`10_e4_paired_cold_seed.png` + `13_e4_semSR.png`",
        "- 散点分析：`14_b1_scatter_steps_semne.png` + `09_e1_heatmap_semne.png`",
        "- 失败案例：cases 里 E1-失败 / easy 近失",
    ]
    (out / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[e9] → {out / 'INDEX.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-run", default=str(REPO_ROOT / "runs/benchmarks/20260623_202108_e1full"))
    ap.add_argument("--e4-run", default=str(REPO_ROOT / "runs/benchmarks/20260624_222318_e4"))
    ap.add_argument("--e2-yolo", default=str(REPO_ROOT / "runs/benchmarks/20260624_180229_e2yolo"))
    ap.add_argument("--e2-vlm", default=str(REPO_ROOT / "runs/benchmarks/20260624_185755_e2vlm"))
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs/benchmarks/e9_figures"))
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--cases-only", action="store_true")
    args = ap.parse_args()

    _setup_chinese_font()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    e1 = load_episodes(Path(args.e1_run))
    e4 = load_episodes(Path(args.e4_run))
    e2y = load_episodes(Path(args.e2_yolo))
    e2v = load_episodes(Path(args.e2_vlm))

    if not args.cases_only:
        print("[e9] 生成汇总图...")
        plot_e1_ablation(e1, out)
        plot_e2_grounder(e1, e2y, e2v, out)
        plot_e8_difficulty(e1, out)
        plot_e4_memory(e4, out)
        plot_e6_disaster(e1, out)
        plot_e3_recheck(e1, out)
        plot_grounding_ablation(out)
        plot_pipeline_diagram(out)
        plot_e1_heatmap_semne(e1, out)
        plot_e4_paired_cold_seed(e4, out)
        plot_e2_full_metrics(e1, e2y, e2v, out)
        plot_e8_semne(e1, out)
        plot_e4_semSR(e4, out)
        plot_b1_scatter(e1, out)

    if not args.summary_only:
        test = json.loads(Path(args.testset).read_text(encoding="utf-8"))
        items = {it["id"]: it for it in test["items"]}
        picks = pick_cases(e1, e4, items, args.max_cases)
        print(f"[e9] 选中 {len(picks)} 个案例")
        run_cases_and_plot(picks, out, out / "cache")

    write_index(out, list(out.glob("*.png")))
    print(f"[e9] 全部完成 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
