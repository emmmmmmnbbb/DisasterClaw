#!/usr/bin/env python3
"""
scripts/benchmarks/calibration_bench.py — P5 校准指标（对应文档 E10 + E10b）

验证 backend/change_perception.py 训练后做的温度标定不是自称的"校准"：在 xBD
change 数据集（scripts/training/gen_xbd_change_dataset.py 产出）的 test / holdout
子集上，对比标定前（T=1，模型原始 softmax）与标定后（T=学到的温度）两版概率分布的：
    - ECE  （Expected Calibration Error，15 bins）
    - Brier Score（多分类版：Σ_k (p_k - y_k)^2 的均值）
    - NLL  （negative log-likelihood）
    - Accuracy（附带看一眼标定没有损害精度——标定只重塑分布形状，不改 argmax）
并画 reliability diagram（confidence vs accuracy）存 PNG。

E10b（调研 baseline）：温度标定只是"后处理重塑一个已训练好的分布"，文献
（Ovadia et al. 2019 / Lakshminarayanan et al. 2017）指出 Deep Ensembles 和
MC-Dropout 是更"重"的不确定性量化方法，通常校准质量更好（但要多份模型/多次
前向的算力代价）。本脚本新增两个可选对比档，复用同一套 ECE/Brier/NLL 计算：
    --mc-dropout --mc-t 30      单模型（须用 change_perception.py train --dropout>0
                                 训练）在推理时保持 Dropout 随机采样，跑 N 次前向，
                                 概率空间取平均（Gal & Ghahramani 2016）。
    --ensemble-ckpts a.pt,b.pt,c.pt
                                 多个独立训练（不同 --seed）的模型，各自用自己学到的
                                 温度 softmax 后在概率空间取平均——即 Ovadia et al.
                                 推荐的 "pool-then-average" Deep Ensemble 做法。

用法：
    python scripts/benchmarks/calibration_bench.py \
        --data-dir /home/lc/datasets/xbd_change \
        --ckpt backend/outputs/change_perception/model.pt \
        --subset test --device cuda:0
    # holdout 子集（跨灾害，若存在）： --subset holdout
    # MC-Dropout： --mc-dropout --mc-t 30
    # Deep Ensemble： --ensemble-ckpts ckA.pt,ckB.pt,ckC.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from change_perception import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    ChangeMultiTaskNet,
    XbdChangeDataset,
    mc_dropout_logits,
    torch,
)
from torch.utils.data import DataLoader  # noqa: E402


def _softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accuracies = (preds == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "acc": None, "conf": None})
            continue
        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (cnt / n) * abs(bin_acc - bin_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt, "acc": bin_acc, "conf": bin_conf})

    onehot = np.eye(NUM_CLASSES)[labels]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    eps = 1e-12
    true_probs = probs[np.arange(n), labels]
    nll = float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))

    acc = float(accuracies.mean())
    order = np.argsort(-confidences)
    risk_coverage = []
    for coverage in np.linspace(0.05, 1.0, 20):
        count = max(1, int(round(n * float(coverage))))
        selective_acc = float(accuracies[order[:count]].mean())
        risk_coverage.append({
            "coverage": float(coverage),
            "count": count,
            "selective_accuracy": selective_acc,
            "selective_risk": 1.0 - selective_acc,
            "confidence_threshold": float(confidences[order[count - 1]]),
        })
    return {
        "ece": float(ece),
        "brier": brier,
        "nll": nll,
        "acc": acc,
        "n": n,
        "bins": bins,
        "risk_coverage": risk_coverage,
    }


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for pre, post, cls_id, _changed in loader:
        pre, post = pre.to(device), post.to(device)
        damage_logits, _ = model(pre, post)
        all_logits.append(damage_logits.cpu().numpy())
        all_labels.append(cls_id.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def collect_mc_dropout_stacked_logits(
    model, loader, device, n_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """跑一遍 n_passes 次随机前向，把 logits（不是概率）缓存下来——T=30 次前向本身
    就很贵，标定前/标定后两个视角只是"同一组 logits 除以不同温度再 softmax"，
    不该为了对比两个温度重新跑一遍 30 次前向（之前的实现有这个浪费，已修正）。
    返回 [N, n_passes, C] logits 和 [N] labels。"""
    all_logits, all_labels = [], []
    with torch.no_grad():
        for pre, post, cls_id, _changed in loader:
            pre, post = pre.to(device), post.to(device)
            stacked = mc_dropout_logits(model, pre, post, n_passes=n_passes)  # [T,B,C]
            all_logits.append(stacked.permute(1, 0, 2).cpu().numpy())  # -> [B,T,C]
            all_labels.append(cls_id.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def mc_dropout_probs_at_temperature(stacked_logits: np.ndarray, temperature: float) -> np.ndarray:
    """给定缓存的 [N,T,C] logits，在某个温度下算平均概率（概率空间取平均，
    与 mc_dropout_logits 文档说明一致：softmax 非线性，必须先各自 softmax 再平均）。"""
    t = max(temperature, 1e-3)
    probs_per_pass = _softmax_np(stacked_logits.reshape(-1, stacked_logits.shape[-1]), t)
    probs_per_pass = probs_per_pass.reshape(stacked_logits.shape)
    return probs_per_pass.mean(axis=1)


def collect_ensemble_probs(
    ckpt_paths: list[Path], loader, device: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Deep Ensemble baseline（Ovadia et al. 2019 的 pool-then-average 做法）：
    每个成员各自用自己训练时学到的温度做 softmax，再在概率空间取平均——不是
    简单平均 logits 再统一标定（那样等价于假设所有成员共享同一套置信度校准，
    而 ensemble 的多样性恰恰来自每个成员自己的训练随机性）。"""
    models, temps = [], []
    for p in ckpt_paths:
        state = torch.load(p, map_location=device)
        m = ChangeMultiTaskNet(
            pretrained=False,
            dropout_p=0.0,
            use_diff_attention=bool(state.get("use_diff_attention", False)),
        ).to(device)
        m.load_state_dict(state["model_state"])
        m.eval()
        models.append(m)
        temps.append(float(state.get("temperature", 1.0)))
    all_probs, all_labels = [], []
    with torch.no_grad():
        for pre, post, cls_id, _changed in loader:
            pre, post = pre.to(device), post.to(device)
            probs_sum = None
            for m, t in zip(models, temps):
                logits, _ = m(pre, post)
                p = torch.softmax(logits / max(t, 1e-3), dim=-1)
                probs_sum = p if probs_sum is None else probs_sum + p
            probs_mean = probs_sum / len(models)
            all_probs.append(probs_mean.cpu().numpy())
            all_labels.append(cls_id.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0), temps


def plot_reliability_diagrams(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), squeeze=False)
    for ax, (name, metrics) in zip(axes[0], results.items()):
        bins = [b for b in metrics["bins"] if b["count"] > 0]
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        accs = [b["acc"] for b in bins]
        confs = [b["conf"] for b in bins]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.bar(centers, accs, width=1.0 / len(metrics["bins"]), alpha=0.6, edgecolor="black", label="accuracy")
        ax.scatter(confs, accs, color="red", zorder=5, s=18, label="bin (conf, acc)")
        ax.set_title(f"{name}\nECE={metrics['ece']:.4f} Brier={metrics['brier']:.4f} NLL={metrics['nll']:.4f}")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/lc/datasets/xbd_change")
    ap.add_argument("--ckpt", default=str(BACKEND / "outputs" / "change_perception" / "model.pt"))
    ap.add_argument("--subset", default="test", choices=["test", "holdout", "val"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "benchmarks" / "calibration"))
    ap.add_argument("--limit", type=int, default=0, help="调试用：截断样本数")
    ap.add_argument("--mc-dropout", action="store_true", help="额外跑 MC-Dropout 基线（--ckpt 须是 dropout_p>0 训练出的模型）")
    ap.add_argument("--mc-t", type=int, default=30, help="MC-Dropout 随机前向次数")
    ap.add_argument("--ensemble-ckpts", default="", help="逗号分隔的多个 checkpoint 路径，额外跑 Deep Ensemble 基线")
    ap.add_argument(
        "--require-event-disjoint",
        action="store_true",
        help="评测前要求 data-dir/manifest.json 声明严格事件级无交集。",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.require_event_disjoint:
        manifest_path = data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = manifest.get("split_audit") or {}
        if (
            manifest.get("split_strategy") != "strict_event"
            or not manifest.get("strict_event_split")
            or not audit.get("event_disjoint")
            or audit.get("overlaps")
        ):
            raise ValueError(f"数据集不是严格事件级无泄漏切分: {manifest_path}")
    jsonl_path = data_dir / f"{args.subset}.jsonl"
    if not jsonl_path.exists():
        print(f"[calib] 子集不存在: {jsonl_path}", file=sys.stderr)
        return 1

    ds = XbdChangeDataset(jsonl_path, augment=False)
    if args.limit:
        ds.records = ds.records[: args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[calib] checkpoint 不存在: {ckpt_path}，先跑 change_perception.py train", file=sys.stderr)
        return 1
    state = torch.load(ckpt_path, map_location=args.device)
    # dropout_p 必须跟训练时一致：nn.Dropout(p) 的 p 在构造时就固定死，用 0.0 构造
    # 出来的模型即使手动切到 train() 也永远不丢弃，MC-Dropout 会静默退化成普通推理。
    ckpt_dropout_p = float(state.get("dropout_p", 0.0))
    ckpt_diff_attention = bool(state.get("use_diff_attention", False))
    model = ChangeMultiTaskNet(
        pretrained=False, dropout_p=ckpt_dropout_p, use_diff_attention=ckpt_diff_attention,
    ).to(args.device)
    model.load_state_dict(state["model_state"])
    learned_temperature = float(state.get("temperature", 1.0))

    print(f"[calib] subset={args.subset} n={len(ds)} device={args.device} T_learned={learned_temperature:.3f}")
    logits, labels = collect_logits(model, loader, args.device)

    results = {}
    for name, t in [("uncalibrated (T=1.0)", 1.0), (f"calibrated (T={learned_temperature:.3f})", learned_temperature)]:
        probs = _softmax_np(logits, t)
        results[name] = compute_calibration_metrics(probs, labels, n_bins=args.n_bins)

    if args.mc_dropout:
        if ckpt_dropout_p <= 0.0:
            print(
                f"[calib] 警告：--ckpt 的 dropout_p={ckpt_dropout_p}（未用 dropout 训练），"
                "MC-Dropout 会退化成 T 次相同前向，结果无意义但仍会跑",
                file=sys.stderr,
            )
        mc_stacked_logits, mc_labels = collect_mc_dropout_stacked_logits(model, loader, args.device, args.mc_t)
        assert np.array_equal(mc_labels, labels)
        mc_probs_raw = mc_dropout_probs_at_temperature(mc_stacked_logits, 1.0)
        results[f"MC-Dropout (T_pass={args.mc_t}, dropout_p={ckpt_dropout_p})"] = compute_calibration_metrics(
            mc_probs_raw, labels, n_bins=args.n_bins,
        )
        mc_probs_cal = mc_dropout_probs_at_temperature(mc_stacked_logits, learned_temperature)
        results[f"MC-Dropout + 温度标定 (T={learned_temperature:.3f})"] = compute_calibration_metrics(
            mc_probs_cal, labels, n_bins=args.n_bins,
        )

    if args.ensemble_ckpts:
        ck_paths = [Path(p.strip()) for p in args.ensemble_ckpts.split(",") if p.strip()]
        missing = [p for p in ck_paths if not p.exists()]
        if missing:
            print(f"[calib] ensemble checkpoint 不存在，跳过: {missing}", file=sys.stderr)
        else:
            ens_probs, ens_labels, ens_temps = collect_ensemble_probs(ck_paths, loader, args.device)
            assert np.array_equal(ens_labels, labels)
            results[f"Deep Ensemble (K={len(ck_paths)}, 各自T={[round(t,2) for t in ens_temps]})"] = (
                compute_calibration_metrics(ens_probs, labels, n_bins=args.n_bins)
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "subset": args.subset,
        "n": len(ds),
        "class_names": CLASS_NAMES,
        "learned_temperature": learned_temperature,
        "results": {
            name: {k: v for k, v in m.items() if k != "bins"} for name, m in results.items()
        },
    }
    (out_dir / f"{args.subset}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{args.subset}_full.json").write_text(
        json.dumps({"subset": args.subset, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[calib] 结果：")
    for name, m in results.items():
        print(f"  {name:28s}: ECE={m['ece']:.4f} Brier={m['brier']:.4f} NLL={m['nll']:.4f} Acc={m['acc']:.3f}")

    try:
        png_path = out_dir / f"{args.subset}_reliability.png"
        plot_reliability_diagrams(results, png_path)
        print(f"[calib] reliability diagram → {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[calib] 画图失败（不影响数值结果）: {exc}", file=sys.stderr)

    print(f"[calib] summary → {out_dir / f'{args.subset}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
