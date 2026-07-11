#!/usr/bin/env python3
"""
scripts/benchmarks/calibration_bench.py — P5 校准指标（对应文档 E10）

验证 backend/change_perception.py 训练后做的温度标定不是自称的"校准"：在 xBD
change 数据集（scripts/training/gen_xbd_change_dataset.py 产出）的 test / holdout
子集上，对比标定前（T=1，模型原始 softmax）与标定后（T=学到的温度）两版概率分布的：
    - ECE  （Expected Calibration Error，15 bins）
    - Brier Score（多分类版：Σ_k (p_k - y_k)^2 的均值）
    - NLL  （negative log-likelihood）
    - Accuracy（附带看一眼标定没有损害精度——标定只重塑分布形状，不改 argmax）
并画 reliability diagram（confidence vs accuracy）存 PNG。

用法：
    python scripts/benchmarks/calibration_bench.py \
        --data-dir /home/lc/datasets/xbd_change \
        --ckpt backend/outputs/change_perception/model.pt \
        --subset test --device cuda:0
    # holdout 子集（跨灾害，若存在）： --subset holdout
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
    return {"ece": float(ece), "brier": brier, "nll": nll, "acc": acc, "n": n, "bins": bins}


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
    args = ap.parse_args()

    jsonl_path = Path(args.data_dir) / f"{args.subset}.jsonl"
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
    model = ChangeMultiTaskNet(pretrained=False).to(args.device)
    model.load_state_dict(state["model_state"])
    learned_temperature = float(state.get("temperature", 1.0))

    print(f"[calib] subset={args.subset} n={len(ds)} device={args.device} T_learned={learned_temperature:.3f}")
    logits, labels = collect_logits(model, loader, args.device)

    results = {}
    for name, t in [("uncalibrated (T=1.0)", 1.0), (f"calibrated (T={learned_temperature:.3f})", learned_temperature)]:
        probs = _softmax_np(logits, t)
        results[name] = compute_calibration_metrics(probs, labels, n_bins=args.n_bins)

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
