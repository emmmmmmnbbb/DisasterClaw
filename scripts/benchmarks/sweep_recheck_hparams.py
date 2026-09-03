#!/usr/bin/env python3
"""Selection-only sweep for τ / min-info-gain / budget on FOV paired rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "backend"))

from change_perception import CLASS_NAMES  # noqa: E402
from eval_budget_allocation import _load_items, _macro_f1, evaluate  # noqa: E402
from recheck import entropy_uncertainty  # noqa: E402


def _select(tau_rows: list[dict], gain_rows: list[dict], curves: dict) -> dict:
    """Freeze τ / min_info_gain / budget on the selection set only."""
    target_budget = 0.25
    best_tau = min(tau_rows, key=lambda row: (abs(row["budget"] - target_budget), -row["macro_f1"]))
    best_gain = max(gain_rows, key=lambda row: (row["macro_f1"], -row["min_info_gain"]))
    cal_rows = curves.get("entropy_cal") or []
    rnd_rows = curves.get("random") or []
    best_budget = target_budget
    best_delta = -1e9
    for cal, rnd in zip(cal_rows, rnd_rows):
        if abs(float(cal["budget"]) - float(rnd["budget"])) > 1e-9:
            continue
        delta = float(cal["macro_f1"]) - float(rnd["macro_f1"])
        if delta > best_delta or (
            abs(delta - best_delta) < 1e-12 and abs(cal["budget"] - target_budget) < abs(best_budget - target_budget)
        ):
            best_delta = delta
            best_budget = float(cal["budget"])
    if best_budget <= 0:
        best_budget = target_budget
    return {
        "entropy_trigger": best_tau["tau"],
        "min_info_gain": best_gain["min_info_gain"],
        "budget": best_budget,
        "conformal_set_size_threshold": 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(REPO / "runs/benchmarks/paper_cja_mech_v1/fov_ladder_test_items.jsonl"))
    ap.add_argument("--fit-items", default=str(REPO / "runs/benchmarks/paper_cja_mech_v1/fov_ladder_val_items.jsonl"))
    ap.add_argument("--qhat", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/paper_cja_mech_v1/recheck_hparam_sweep.json"))
    args = ap.parse_args()
    items = _load_items(Path(args.items), 0)
    fit_items = _load_items(Path(args.fit_items), 0)
    if not items or not fit_items:
        print("[ERROR] run eval_fov_ladder.py for fit and selection first", file=sys.stderr)
        return 2
    base = evaluate(
        items, fit_items, args.qhat, args.temperature,
        n_boot=args.bootstrap,
    )
    y = np.array([int(it["y"]) for it in items])
    cruise = np.stack([[float(it["views"]["cruise"]["probs"][n]) for n in CLASS_NAMES] for it in items])
    native = np.stack([[float(it["views"]["floor"]["probs"][n]) for n in CLASS_NAMES] for it in items])
    U = np.array([entropy_uncertainty({n: float(p) for n, p in zip(CLASS_NAMES, row)}) for row in cruise])
    tau_rows = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        use = U >= tau
        pred = np.where(use[:, None], native, cruise).argmax(1)
        tau_rows.append({
            "tau": tau,
            "n_descend": int(use.sum()),
            "budget": float(use.mean()),
            "macro_f1": _macro_f1(y, pred),
        })
    from eval_budget_allocation import ExpectedGainTable, score_expected_gain
    table = ExpectedGainTable(
        entropy_edges=tuple(base["expected_gain_table"]["entropy_edges"]),
        by_pred_and_bin=base["expected_gain_table"]["by_pred_and_bin"],
        by_pred=base["expected_gain_table"]["by_pred"],
        global_mean=float(base["expected_gain_table"]["global_mean"]),
        fit_n=int(base["expected_gain_table"]["fit_n"]),
        fit_events=tuple(base["expected_gain_table"]["fit_events"]),
    )
    gains = score_expected_gain(cruise, table)
    gain_rows = []
    for min_info_gain in (0.02, 0.05, 0.10, 0.15):
        use = gains > min_info_gain
        pred = np.where(use[:, None], native, cruise).argmax(1)
        gain_rows.append({
            "min_info_gain": min_info_gain,
            "n_descend": int(use.sum()),
            "budget": float(use.mean()),
            "macro_f1": _macro_f1(y, pred),
        })
    selected = _select(tau_rows, gain_rows, base["curves"])
    out = {
        "schema": "recheck-selection-sweep/1.0",
        "selection_only": True,
        "budget_curves": base["curves"],
        "tau_sweep": tau_rows,
        "min_info_gain_sweep": gain_rows,
        "selected": selected,
        "qhat": args.qhat,
        "temperature": args.temperature,
        "n": len(items),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(items), "selected": selected, "tau_sweep": tau_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
