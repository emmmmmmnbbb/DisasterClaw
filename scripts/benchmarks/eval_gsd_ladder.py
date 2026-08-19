#!/usr/bin/env python3
"""X1: native vs cruise GSD damage classification + entropy table + conformal qhat."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

from change_perception import (  # noqa: E402
    CLASS_NAMES,
    CONTEXT_MARGIN,
    CROP_SIZE,
    crop_patch,
    get_change_perception,
)
from gsd_ladder import degrade_to_scale, ladder_points  # noqa: E402
from recheck import entropy_uncertainty, fit_conformal_qhat  # noqa: E402


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1]) if i else (conf >= bins[i]) & (conf <= bins[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = 4) -> float:
    f1s = []
    for c in range(k):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def _load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    diffs = []
    idx = np.arange(len(a))
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(float(a[take].mean() - b[take].mean()))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)], "excludes_zero": not (lo <= 0 <= hi)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/lc/datasets/xbd_change_strict_v1")
    ap.add_argument("--split", default="test", choices=["val", "test", "holdout"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--ckpt", default=str(BACKEND / "outputs/change_perception/strict_diff_attention_seed0_v2.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/gsd_ladder.json"))
    ap.add_argument("--table-out", default=str(BACKEND / "data/gsd_entropy_table.json"))
    ap.add_argument("--items-out", default=str(REPO / "runs/benchmarks/gsd_ladder_items.jsonl"))
    ap.add_argument(
        "--scales",
        default="",
        help="comma-separated downsample scales (default: 5-point 10–30 m ladder plus 1 and 4)",
    )
    args = ap.parse_args()

    import os
    os.environ["CHANGE_PERCEPTION_CKPT"] = args.ckpt
    os.environ["PERCEPTION_DEVICE"] = args.device

    jsonl = Path(args.data_dir) / f"{args.split}.jsonl"
    records = _load_jsonl(jsonl, args.limit)
    model = get_change_perception()
    if args.scales.strip():
        scales = sorted({round(float(s), 4) for s in args.scales.split(",") if s.strip()})
    else:
        points = ladder_points(5)
        scales = sorted({round(p["scale"], 4) for p in points} | {1.0, 4.0})

    by_scale: dict[float, dict] = {s: {"probs": [], "y": [], "entropy": [], "pred": []} for s in scales}
    items_fp = Path(args.items_out)
    items_fp.parent.mkdir(parents=True, exist_ok=True)

    with items_fp.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            y = int(rec["class_id"])
            pre = crop_patch(rec["pre_image"], tuple(rec["bbox_pre"]), rec["image_width"], rec["image_height"])
            post = crop_patch(rec["post_image"], tuple(rec["bbox_post"]), rec["image_width"], rec["image_height"])
            item = {"uid": rec.get("uid"), "y": y, "subtype": rec.get("subtype"), "disaster": rec.get("disaster"), "views": {}}
            for scale in scales:
                pre_s = degrade_to_scale(pre, scale)
                post_s = degrade_to_scale(post, scale)
                pred = model.predict(pre_s, post_s)
                probs = [float(pred.class_probs[n]) for n in CLASS_NAMES]
                yhat = int(np.argmax(probs))
                u = entropy_uncertainty(pred.class_probs)
                by_scale[scale]["probs"].append(probs)
                by_scale[scale]["y"].append(y)
                by_scale[scale]["entropy"].append(u)
                by_scale[scale]["pred"].append(yhat)
                item["views"][str(scale)] = {"probs": dict(pred.class_probs), "pred": CLASS_NAMES[yhat], "entropy": u}
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"[gsd] {i + 1}/{len(records)}")

    curve = []
    native_correct = None
    cruise_correct = None
    for scale in scales:
        P = np.asarray(by_scale[scale]["probs"], dtype=np.float64)
        y = np.asarray(by_scale[scale]["y"], dtype=np.int64)
        pred = np.asarray(by_scale[scale]["pred"], dtype=np.int64)
        acc = float((pred == y).mean())
        row = {
            "scale": scale,
            "gsd_m": round(0.5 * scale, 4),
            "n": int(len(y)),
            "accuracy": acc,
            "macro_f1": _macro_f1(y, pred),
            "ece": _ece(P, y),
            "mean_entropy": float(np.mean(by_scale[scale]["entropy"])),
        }
        curve.append(row)
        per_class = {}
        for c, name in enumerate(CLASS_NAMES):
            support = int((y == c).sum())
            pred_n = int((pred == c).sum())
            tp = int(((y == c) & (pred == c)).sum())
            per_class[name] = {
                "support": support,
                "pred_n": pred_n,
                "recall": float(tp / support) if support else 0.0,
            }
        row["per_class"] = per_class
        if abs(scale - 1.0) < 1e-6:
            native_correct = (pred == y).astype(np.float64)
        if abs(scale - 4.0) < 1e-6:
            cruise_correct = (pred == y).astype(np.float64)

    delta = None
    if native_correct is not None and cruise_correct is not None:
        # per-item F1 proxy: correctness delta (paired)
        delta = _paired_bootstrap_delta(native_correct, cruise_correct)

    # entropy table on this split (caller should use val for trigger fitting)
    bins = []
    for scale in scales:
        yhat = by_scale[scale]["pred"]
        ent = by_scale[scale]["entropy"]
        by_cls: dict[str, list[float]] = defaultdict(list)
        for p, u in zip(yhat, ent):
            by_cls[CLASS_NAMES[int(p)]].append(float(u))
        bins.append({
            "scale": scale,
            "gsd_m": round(0.5 * scale, 4),
            "by_pred_class": {
                name: {"mean_entropy": float(np.mean(vs)), "n": len(vs)}
                for name, vs in by_cls.items() if vs
            },
            "all": {"mean_entropy": float(np.mean(ent)), "n": len(ent)},
        })

    conformal_rows = []
    for rec_probs, y in zip(by_scale[min(scales)]["probs"], by_scale[min(scales)]["y"]):
        conformal_rows.append((dict(zip(CLASS_NAMES, rec_probs)), CLASS_NAMES[int(y)]))
    qhat = fit_conformal_qhat(conformal_rows, alpha=0.1)

    native_preds = by_scale[min(scales)]["pred"]
    cruise_key = 4.0 if 4.0 in by_scale else max(scales)
    native_key = 1.0 if 1.0 in by_scale else min(scales)
    n_flip = int(sum(
        int(a) != int(b)
        for a, b in zip(by_scale[native_key]["pred"], by_scale[cruise_key]["pred"])
    ))
    gt_counts = {
        CLASS_NAMES[c]: int(sum(1 for v in by_scale[native_key]["y"] if int(v) == c))
        for c in range(4)
    }
    pred_counts = {
        CLASS_NAMES[c]: int(sum(1 for v in native_preds if int(v) == c))
        for c in range(4)
    }
    out = {
        "split": args.split,
        "n": len(records),
        "ckpt": args.ckpt,
        "scales": scales,
        "curve": curve,
        "gt_counts": gt_counts,
        "pred_counts_native": pred_counts,
        "n_pred_flip_native_vs_cruise": n_flip,
        "native_minus_cruise_accuracy": delta,
        "accept_native_better": bool(delta and delta["excludes_zero"] and delta["mean"] > 0),
        "conformal_qhat_alpha01": qhat,
        "bins": bins,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_out).write_text(
        json.dumps({"source": args.split, "n": len(records), "bins": bins, "conformal_qhat": qhat}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: out[k] for k in (
        "n", "gt_counts", "pred_counts_native", "n_pred_flip_native_vs_cruise",
        "curve", "native_minus_cruise_accuracy", "accept_native_better", "conformal_qhat_alpha01",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
