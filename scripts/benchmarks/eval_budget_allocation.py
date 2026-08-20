#!/usr/bin/env python3
"""Leakage-safe offline allocation of paired cruise/native observations.

Deployable policies are functions of the current cruise probabilities only.
Future/native observations are used for scoring and for the explicitly labelled
oracle diagnostic. The conditional expected-gain policy is fitted on a separate
validation-event JSONL file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))

from change_perception import CLASS_NAMES  # noqa: E402
from recheck import conformal_predict_set, entropy_uncertainty  # noqa: E402

DEFAULT_TEST_ITEMS = REPO / "runs/benchmarks/paper_cja_v2/gsd_ladder_test_items.jsonl"
DEFAULT_FIT_ITEMS = REPO / "runs/benchmarks/paper_cja_v2/gsd_ladder_val_items.jsonl"


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
        mask = ((conf > bins[i]) & (conf <= bins[i + 1])) if i else ((conf >= bins[i]) & (conf <= bins[i + 1]))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def _cost_risk(pred: np.ndarray, y: np.ndarray, miss_cost: float) -> float:
    """Cost sensitivity where predicting no-damage for damage costs more."""
    wrong = pred != y
    costs = wrong.astype(np.float64)
    costs[(pred == 0) & (y != 0)] = float(miss_cost)
    return float(costs.mean())


def _aurc(probs: np.ndarray, y: np.ndarray) -> float:
    """Area under selective risk-coverage curve (lower is better)."""
    order = np.argsort(-probs.max(axis=1))
    errors = (probs.argmax(axis=1)[order] != y[order]).astype(np.float64)
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


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


def _entropy_rows(probs: np.ndarray) -> np.ndarray:
    return np.array([
        entropy_uncertainty({n: float(p) for n, p in zip(CLASS_NAMES, row)})
        for row in probs
    ])


def undo_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Recover softmax(logits) from softmax(logits / temperature)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw = np.power(np.clip(probs, 1e-12, 1.0), float(temperature))
    return raw / raw.sum(axis=1, keepdims=True)


def allocate(scores: np.ndarray, budget_frac: float, rng: np.random.Generator | None = None) -> np.ndarray:
    n = len(scores)
    k = int(round(budget_frac * n))
    chosen = np.zeros(n, dtype=bool)
    if k <= 0:
        return chosen
    if rng is not None:
        idx = rng.choice(n, size=min(k, n), replace=False)
    else:
        idx = np.argsort(-scores, kind="stable")[:k]
    chosen[idx] = True
    return chosen


@dataclass(frozen=True)
class ExpectedGainTable:
    entropy_edges: tuple[float, ...]
    by_pred_and_bin: dict[str, float]
    by_pred: dict[str, float]
    global_mean: float
    fit_n: int
    fit_events: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "entropy_edges": list(self.entropy_edges),
            "by_pred_and_bin": dict(self.by_pred_and_bin),
            "by_pred": dict(self.by_pred),
            "global_mean": self.global_mean,
            "fit_n": self.fit_n,
            "fit_events": list(self.fit_events),
        }


def _entropy_bin(value: float, edges: tuple[float, ...]) -> int:
    return int(np.clip(np.digitize([value], edges[1:-1], right=False)[0], 0, len(edges) - 2))


def fit_expected_gain_table(items: list[dict], n_bins: int = 5) -> ExpectedGainTable:
    """Fit E[U_cruise-U_native | current predicted class, entropy bin]."""
    if not items:
        raise ValueError("expected-gain fit set is empty")
    cruise = np.stack([_vec(it["views"]["4.0"]["probs"]) for it in items])
    native = np.stack([_vec(it["views"]["1.0"]["probs"]) for it in items])
    entropy_now = _entropy_rows(cruise)
    gain = entropy_now - _entropy_rows(native)
    pred = cruise.argmax(axis=1)
    bin_idx = np.array([_entropy_bin(v, tuple(np.linspace(0.0, 1.0, n_bins + 1))) for v in entropy_now])
    edges = tuple(float(v) for v in np.linspace(0.0, 1.0, n_bins + 1))
    by_pair: dict[str, float] = {}
    by_pred: dict[str, float] = {}
    for c, name in enumerate(CLASS_NAMES):
        cmask = pred == c
        if cmask.any():
            by_pred[name] = float(gain[cmask].mean())
        for b in range(n_bins):
            mask = cmask & (bin_idx == b)
            if mask.any():
                by_pair[f"{name}|{b}"] = float(gain[mask].mean())
    return ExpectedGainTable(
        entropy_edges=edges,
        by_pred_and_bin=by_pair,
        by_pred=by_pred,
        global_mean=float(gain.mean()),
        fit_n=len(items),
        fit_events=tuple(sorted({str(it.get("disaster") or "unknown") for it in items})),
    )


def score_expected_gain(current_probs: np.ndarray, table: ExpectedGainTable) -> np.ndarray:
    """Score test rows using current observations only.

    This deliberately accepts no native/future argument. The signature is an
    executable leakage guard and is covered by tests.
    """
    entropy_now = _entropy_rows(current_probs)
    pred = current_probs.argmax(axis=1)
    scores = []
    for c, ent in zip(pred, entropy_now):
        name = CLASS_NAMES[int(c)]
        b = _entropy_bin(float(ent), table.entropy_edges)
        score = table.by_pred_and_bin.get(
            f"{name}|{b}", table.by_pred.get(name, table.global_mean),
        )
        scores.append(float(score))
    return np.asarray(scores, dtype=np.float64)


def _paired_bootstrap_metric(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric,
    seed: int,
    n_boot: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(metric(probs_a, y) - metric(probs_b, y))
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = metric(probs_a[idx], y[idx]) - metric(probs_b[idx], y[idx])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": point,
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n_boot": int(n_boot),
    }


def _selection_probs(cruise: np.ndarray, native: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask[:, None], native, cruise)


def evaluate(
    items: list[dict],
    fit_items: list[dict],
    qhat: float,
    temperature: float,
    seed: int = 0,
    n_boot: int = 2000,
) -> dict:
    y = np.array([int(it["y"]) for it in items], dtype=np.int64)
    cruise = np.stack([_vec(it["views"]["4.0"]["probs"]) for it in items])
    native = np.stack([_vec(it["views"]["1.0"]["probs"]) for it in items])
    cruise_pred = cruise.argmax(1)
    native_pred = native.argmax(1)
    flip = cruise_pred != native_pred
    correctable = (cruise_pred != y) & (native_pred == y)
    harmful = (cruise_pred == y) & (native_pred != y)
    oracle_score = correctable.astype(np.float64) - harmful.astype(np.float64)

    raw_cruise = undo_temperature(cruise, temperature)
    uncal_entropy = _entropy_rows(raw_cruise)
    cal_entropy = _entropy_rows(cruise)
    expected_table = fit_expected_gain_table(fit_items)
    expected_score = score_expected_gain(cruise, expected_table)
    conf_size = np.array([
        len(conformal_predict_set({n: float(p) for n, p in zip(CLASS_NAMES, row)}, qhat))
        for row in cruise
    ], dtype=np.float64)

    budgets = [0.0, 0.1, 0.25, 0.5, 1.0]
    strategy_scores = {
        "none": np.zeros(len(items), dtype=np.float64),
        "entropy_uncal": uncal_entropy,
        "entropy_cal": cal_entropy,
        "expected_gain": expected_score,
        "conformal": conf_size,
        "oracle": oracle_score,
    }
    rng = np.random.default_rng(seed)
    curves: dict[str, list[dict]] = {}
    masks: dict[str, dict[str, np.ndarray]] = {}
    names = ["none", "random", *[n for n in strategy_scores if n != "none"]]
    for name in names:
        rows = []
        masks[name] = {}
        for b in budgets:
            if name == "none":
                use_native = np.zeros(len(items), dtype=bool)
            elif name == "random":
                use_native = allocate(np.zeros(len(items)), b, rng)
            else:
                use_native = allocate(strategy_scores[name], b, None)
            pred = np.where(use_native, native_pred, cruise_pred)
            probs = _selection_probs(cruise, native, use_native)
            key = f"{b:.2f}"
            masks[name][key] = use_native
            rows.append({
                "budget": b,
                "n_descend": int(use_native.sum()),
                "macro_f1": _macro_f1(y, pred),
                "accuracy": float((pred == y).mean()),
                "ece": _ece(probs, y),
                "brier": _brier(probs, y),
                "nll": _nll(probs, y),
                "aurc": _aurc(probs, y),
                "cost_risk_3": _cost_risk(pred, y, 3.0),
                "cost_risk_5": _cost_risk(pred, y, 5.0),
                "cost_risk_10": _cost_risk(pred, y, 10.0),
                "n_corrected": int((use_native & correctable).sum()),
                "n_harmed": int((use_native & harmful).sum()),
                "flip_precision": float((use_native & flip).sum() / use_native.sum()) if use_native.any() else None,
                "flip_recall": float((use_native & flip).sum() / flip.sum()) if flip.any() else None,
            })
        curves[name] = rows

    budget_key = "0.25"
    paired: dict[str, dict] = {}
    for comparator in ("entropy_uncal", "random"):
        pa = _selection_probs(cruise, native, masks["entropy_cal"][budget_key])
        pb = _selection_probs(cruise, native, masks[comparator][budget_key])
        paired[f"entropy_cal_minus_{comparator}"] = {
            "macro_f1": _paired_bootstrap_metric(
                y, pa, pb, lambda p, yy: _macro_f1(yy, p.argmax(1)), seed + 11, n_boot,
            ),
            # Lower Brier is better, so report comparator - calibrated entropy.
            "brier_improvement": _paired_bootstrap_metric(
                y, pb, pa, _brier, seed + 23, n_boot,
            ),
        }

    cal_f1_accept = all(
        block["macro_f1"]["ci95"][0] > 0.0 for block in paired.values()
    )
    oracle_f1 = next(r["macro_f1"] for r in curves["oracle"] if r["budget"] == 0.25)
    expected_f1 = next(r["macro_f1"] for r in curves["expected_gain"] if r["budget"] == 0.25)
    test_events = sorted({str(it.get("disaster") or "unknown") for it in items})
    overlap = sorted(set(test_events) & set(expected_table.fit_events))
    return {
        "schema_version": "budget-allocation/2.0",
        "n": len(items),
        "test_events": test_events,
        "fit_events": list(expected_table.fit_events),
        "fit_test_event_overlap": overlap,
        "leakage_check_passed": not overlap,
        "n_flip": int(flip.sum()),
        "n_correctable": int(correctable.sum()),
        "n_harmful": int(harmful.sum()),
        "curves": curves,
        "paired_tests_at_0.25": paired,
        "accept_calibrated_better": bool(cal_f1_accept),
        "gap_to_oracle_at_0.25": float(oracle_f1 - expected_f1),
        "expected_gain_table": expected_table.to_dict(),
        "temperature": float(temperature),
        "qhat": qhat,
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DEFAULT_TEST_ITEMS), help="test-event paired observations")
    ap.add_argument("--fit-items", default=str(DEFAULT_FIT_ITEMS), help="validation-event rows for A5_EXPECTED")
    ap.add_argument("--qhat", type=float, default=0.7950054973805708)
    ap.add_argument("--temperature", type=float, default=1.4183316230773926)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fit-limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/cja_x2_costsens/budget_allocation.json"))
    args = ap.parse_args()
    items = _load_items(Path(args.items), args.limit)
    fit_items = _load_items(Path(args.fit_items), args.fit_limit)
    if not items or not fit_items:
        print("[ERROR] test and fit paired-item files must both be non-empty", file=sys.stderr)
        return 2
    result = evaluate(
        items, fit_items, args.qhat, args.temperature,
        seed=args.seed, n_boot=args.bootstrap,
    )
    if not result["leakage_check_passed"]:
        print(f"[ERROR] fit/test event overlap: {result['fit_test_event_overlap']}", file=sys.stderr)
        return 3
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        k: result[k] for k in (
            "n", "fit_events", "test_events", "n_flip", "n_correctable",
            "n_harmful", "accept_calibrated_better", "gap_to_oracle_at_0.25",
        )
    }, ensure_ascii=False, indent=2))
    print("budget 0.25 macro-F1 / Brier:")
    for name, rows in result["curves"].items():
        row = next(r for r in rows if abs(r["budget"] - 0.25) < 1e-9)
        print(f"  {name:16s} {row['macro_f1']:.4f} / {row['brier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
