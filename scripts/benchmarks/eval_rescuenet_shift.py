#!/usr/bin/env python3
"""RescueNet post-only cross-domain eval of the satellite-trained damage classifier.

RescueNet has no pre-event pair. We duplicate the UAV post crop as the pre stream
(`pre_mode=duplicate_post`) so the change channel is ≈0. This is a domain-shift
diagnostic, not a claim that bitemporal fusion transfers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

from change_perception import CLASS_NAMES, CROP_SIZE, get_change_perception  # noqa: E402
from recheck import entropy_uncertainty  # noqa: E402

MASK_VALUE_TO_CLASS = {
    2: (0, "no-damage"),
    3: (1, "minor-damage"),
    4: (2, "major-damage"),
    5: (3, "destroyed"),
}
MIN_AREA_PX = 900


def _boxes(mask: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    out = []
    for value, (cid, _name) in MASK_VALUE_TO_CLASS.items():
        binary = mask == value
        if not binary.any():
            continue
        labeled, n = ndi.label(binary)
        for comp in range(1, n + 1):
            ys, xs = np.where(labeled == comp)
            if ys.size < MIN_AREA_PX:
                continue
            out.append((cid, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
    return out


def _macro_f1(y, pred) -> float:
    f1s = []
    for c in range(4):
        tp = int(((y == c) & (pred == c)).sum())
        fp = int(((y != c) & (pred == c)).sum())
        fn = int(((y == c) & (pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def _ece(probs, y, n_bins=15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1]) if i else (conf >= bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lc/datasets/rescuenet")
    ap.add_argument("--limit-images", type=int, default=80)
    ap.add_argument("--limit-boxes", type=int, default=2000)
    ap.add_argument("--ckpt", default=str(BACKEND / "outputs/change_perception/strict_diff_attention_seed0_v2.pt"))
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/rescuenet_shift.json"))
    args = ap.parse_args()

    import os
    os.environ["CHANGE_PERCEPTION_CKPT"] = args.ckpt
    model = get_change_perception()

    root = Path(args.root)
    img_dir = root / "test-org-img"
    lab_dir = root / "test-label-img"
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if args.limit_images:
        images = images[: args.limit_images]

    ys, preds, prob_rows = [], [], []
    n_images = 0
    for img_path in images:
        stem = img_path.stem
        mask_path = lab_dir / f"{stem}_lab.png"
        if not mask_path.exists():
            mask_path = lab_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = lab_dir / img_path.name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path))
        rgb = Image.open(img_path).convert("RGB")
        n_images += 1
        for cid, (x1, y1, x2, y2) in _boxes(mask):
            crop = rgb.crop((x1, y1, x2, y2)).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
            pred = model.predict(crop, crop)  # duplicate post
            vec = [float(pred.class_probs[n]) for n in CLASS_NAMES]
            ys.append(cid)
            preds.append(int(np.argmax(vec)))
            prob_rows.append(vec)
            if args.limit_boxes and len(ys) >= args.limit_boxes:
                break
        if args.limit_boxes and len(ys) >= args.limit_boxes:
            break

    y = np.asarray(ys, dtype=np.int64)
    pred = np.asarray(preds, dtype=np.int64)
    P = np.asarray(prob_rows, dtype=np.float64) if prob_rows else np.zeros((0, 4))
    out = {
        "dataset": "RescueNet test (Hurricane Michael UAV, post-only)",
        "license_note": "Use the Scientific Data / figshare CC BY 4.0 release.",
        "pre_mode": "duplicate_post",
        "n_images": n_images,
        "n_boxes": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "macro_f1": _macro_f1(y, pred) if len(y) else None,
        "ece": _ece(P, y) if len(y) else None,
        "mean_entropy": float(np.mean([
            entropy_uncertainty({n: float(p) for n, p in zip(CLASS_NAMES, row)})
            for row in P
        ])) if len(y) else None,
        "per_class": {
            CLASS_NAMES[c]: {
                "n": int((y == c).sum()),
                "recall": float(((pred == c) & (y == c)).sum() / max(1, (y == c).sum())),
            }
            for c in range(4)
        },
        "caveat": (
            "RescueNet 无灾前配对影像，双时相差通道恒为 0；本结果只度量卫星训练分类器"
            "在真实低空外观上的漂移，不能解释为下降动作的目标级配对增益。"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
