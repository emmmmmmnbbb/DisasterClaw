#!/usr/bin/env python3
"""X5: power analysis for the 6-policy Holm-corrected SR comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def z_approx(p: float) -> float:
    # Acklam rational approximation of the standard-normal quantile
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577509590705e+02, -3.066479806614736e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163469594518e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def n_per_arm(p: float, delta: float, alpha: float, power: float) -> int:
    za = z_approx(1 - alpha / 2)
    zb = z_approx(power)
    q = 1.0 - p
    n = ((za + zb) ** 2) * 2 * p * q / max(delta ** 2, 1e-12)
    return int(math.ceil(n))


def min_detectable(p: float, n: int, alpha: float, power: float) -> float:
    za = z_approx(1 - alpha / 2)
    zb = z_approx(power)
    q = 1.0 - p
    return math.sqrt(((za + zb) ** 2) * 2 * p * q / max(n, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-sr", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05, help="absolute SR lift to detect")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--n-policies", type=int, default=6)
    ap.add_argument("--current-n", type=int, default=40)
    ap.add_argument("--current-seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    holm_alpha = args.alpha / max(args.n_policies - 1, 1)
    n_need = n_per_arm(args.baseline_sr, args.delta, holm_alpha, args.power)
    current_n = args.current_n * args.current_seeds
    mde = min_detectable(args.baseline_sr, current_n, holm_alpha, args.power)
    out = {
        "baseline_sr": args.baseline_sr,
        "target_delta": args.delta,
        "alpha": args.alpha,
        "holm_alpha": holm_alpha,
        "power": args.power,
        "n_items_per_policy_needed": n_need,
        "current_design": {
            "items": args.current_n,
            "seeds": args.current_seeds,
            "episodes_per_policy": current_n,
            "min_detectable_sr": round(mde, 4),
        },
        "zero_trigger_rule": (
            "A policy/run with zero eligible evidence-positive episodes is missing "
            "for conditional ΔU; its trigger rate is reported as zero and is never "
            "imputed as a successful reduction."
        ),
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
