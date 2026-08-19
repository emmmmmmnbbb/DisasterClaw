#!/usr/bin/env python3
"""Hyperparameter sweep for τ / min_info_gain / per-episode budget (X2 companion)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "backend"))

from eval_budget_allocation import _load_items, evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(REPO / "runs/benchmarks/gsd_ladder_items.jsonl"))
    ap.add_argument("--qhat", type=float, default=0.9)
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/recheck_hparam_sweep.json"))
    args = ap.parse_args()
    items = _load_items(Path(args.items), 0)
    if not items:
        print("[ERROR] run eval_gsd_ladder.py first", file=sys.stderr)
        return 2
    # The allocation script already scans budget fractions; here we also scan
    # entropy thresholds by filtering scores.
    base = evaluate(items, args.qhat)
    taus = [0.3, 0.4, 0.5, 0.6, 0.7]
    from change_perception import CLASS_NAMES
    from recheck import entropy_uncertainty
    import numpy as np

    y = np.array([int(it["y"]) for it in items])
    cruise = np.stack([[float(it["views"]["4.0"]["probs"][n]) for n in CLASS_NAMES] for it in items])
    native = np.stack([[float(it["views"]["1.0"]["probs"][n]) for n in CLASS_NAMES] for it in items])
    U = np.array([entropy_uncertainty({n: float(p) for n, p in zip(CLASS_NAMES, row)}) for row in cruise])
    tau_rows = []
    for tau in taus:
        use = U >= tau
        pred = np.where(use[:, None], native, cruise).argmax(1)
        from eval_budget_allocation import _macro_f1
        tau_rows.append({
            "tau": tau,
            "n_descend": int(use.sum()),
            "budget": float(use.mean()),
            "macro_f1": _macro_f1(y, pred),
        })
    out = {"budget_curves": base["curves"], "tau_sweep": tau_rows, "qhat": args.qhat, "n": len(items)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(items), "tau_sweep": tau_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
