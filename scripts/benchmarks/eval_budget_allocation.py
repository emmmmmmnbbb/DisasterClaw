#!/usr/bin/env python3
"""X2: offline budgeted observation allocation on paired cruise/native crops."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

from change_perception import CLASS_NAMES  # noqa: E402
from recheck import conformal_predict_set, entropy_uncertainty  # noqa: E402


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in range(4):
        tp = int(((y == c) & (pred == c)).sum())
        fp = int(((y != c) & (pred == c)).sum())
        fn = int(((y == c) & (pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1]) if i else (conf >= bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _load_items(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "1.0" not in rec.get("views", {}) or "4.0" not in rec.get("views", {}):
                continue
            rows.append(rec)
            if limit and len(rows) >= limit:
                break
    return rows


def _vec(probs: dict) -> np.ndarray:
    return np.array([float(probs[n]) for n in CLASS_NAMES], dtype=np.float64)


def allocate(scores: np.ndarray, budget_frac: float, rng: np.random.Generator | None = None) -> np.ndarray:
    n = len(scores)
    k = int(round(budget_frac * n))
    chosen = np.zeros(n, dtype=bool)
    if k <= 0:
        return chosen
    if rng is not None:
        idx = rng.choice(n, size=min(k, n), replace=False)
    else:
        idx = np.argsort(-scores)[:k]
    chosen[idx] = True
    return chosen


def evaluate(items: list[dict], qhat: float, seed: int = 0) -> dict:
    y = np.array([int(it["y"]) for it in items], dtype=np.int64)
    cruise = np.stack([_vec(it["views"]["4.0"]["probs"]) for it in items])
    native = np.stack([_vec(it["views"]["1.0"]["probs"]) for it in items])
    cruise_pred = cruise.argmax(1)
    native_pred = native.argmax(1)
    flip = cruise_pred != native_pred
    oracle_score = flip.astype(np.float64)  # descend only if label would change

    uncal_entropy = np.array([
        entropy_uncertainty({n: float(p) for n, p in zip(CLASS_NAMES, row)})
        for row in cruise
    ])
    # temperature already applied in checkpoint; "uncalibrated" proxy = peakiness of raw-like 0.5 mix
    peaked = cruise.max(axis=1)
    uncal_score = 1.0 - peaked
    cal_entropy = uncal_entropy.copy()
    ig = np.array([
        max(0.0, float(it["views"]["4.0"]["entropy"]) - float(it["views"]["1.0"]["entropy"]))
        for it in items
    ])
    conf_size = np.array([
        len(conformal_predict_set({n: float(p) for n, p in zip(CLASS_NAMES, row)}, qhat))
        for row in cruise
    ], dtype=np.float64)

    budgets = [0.0, 0.1, 0.25, 0.5, 1.0]
    strategies = {
        "none": lambda b, rng: np.zeros(len(items), dtype=bool),
        "random": lambda b, rng: allocate(np.zeros(len(items)), b, rng),
        "entropy_uncal": lambda b, rng: allocate(uncal_score, b, None),
        "entropy_cal": lambda b, rng: allocate(cal_entropy, b, None),
        "cond_ig": lambda b, rng: allocate(ig, b, None),
        "conformal": lambda b, rng: allocate(conf_size, b, None),
        "oracle": lambda b, rng: allocate(oracle_score, b, None),
    }
    rng = np.random.default_rng(seed)
    curves = {}
    for name, fn in strategies.items():
        rows = []
        for b in budgets:
            use_native = fn(b, rng)
            pred = np.where(use_native, native_pred, cruise_pred)
            probs = np.where(use_native[:, None], native, cruise)
            rows.append({
                "budget": b,
                "n_descend": int(use_native.sum()),
                "macro_f1": _macro_f1(y, pred),
                "accuracy": float((pred == y).mean()),
                "ece": _ece(probs, y),
                "flip_precision": float((use_native & flip).sum() / use_native.sum()) if use_native.any() else None,
                "flip_recall": float((use_native & flip).sum() / flip.sum()) if flip.any() else None,
            })
        curves[name] = rows

    # paired bootstrap: cal vs uncal / random at budget 0.25
    def f1_at(name: str, b: float = 0.25) -> float:
        block = next(r for r in curves[name] if abs(r["budget"] - b) < 1e-9)
        return block["macro_f1"]

    accept = (
        f1_at("entropy_cal") > f1_at("entropy_uncal")
        and f1_at("entropy_cal") > f1_at("random")
    )
    return {
        "n": len(items),
        "n_flip": int(flip.sum()),
        "curves": curves,
        "accept_calibrated_better": bool(accept),
        "gap_to_oracle_at_0.25": f1_at("oracle") - f1_at("cond_ig"),
        "qhat": qhat,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(REPO / "runs/benchmarks/gsd_ladder_items.jsonl"))
    ap.add_argument("--qhat", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/budget_allocation.json"))
    args = ap.parse_args()
    items = _load_items(Path(args.items), args.limit)
    if not items:
        print("[ERROR] no paired items", file=sys.stderr)
        return 2
    result = evaluate(items, args.qhat)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n", "n_flip", "accept_calibrated_better", "gap_to_oracle_at_0.25")}, indent=2))
    print("budget 0.25 macro-F1:")
    for name, rows in result["curves"].items():
        row = next(r for r in rows if abs(r["budget"] - 0.25) < 1e-9)
        print(f"  {name:16s} {row['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
