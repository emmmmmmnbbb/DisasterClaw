"""Altitude → effective GSD ladder for X1.

xBD tiles are a single native GSD (≈0.5 m/px). A naive descend that merely
crops a smaller window and bilinear-resizes the classifier crop back to a
fixed 96×96 does not add information. This module simulates a resolution
ladder:

    cruise altitude (default 30 m) → 4× downsample + blur  (coarse GSD)
    minimum altitude (default 10 m) → native pixels         (fine GSD)

Intermediate altitudes interpolate the downsample scale linearly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

NATIVE_GSD_M = 0.5
CRUISE_ALT_M = 30.0
MIN_ALT_M = 10.0
CRUISE_SCALE = 4.0
NATIVE_SCALE = 1.0

CLASS_NAMES = ("no-damage", "minor-damage", "major-damage", "destroyed")


def effective_scale(
    alt_m: float,
    cruise_alt_m: float = CRUISE_ALT_M,
    min_alt_m: float = MIN_ALT_M,
    cruise_scale: float = CRUISE_SCALE,
    native_scale: float = NATIVE_SCALE,
) -> float:
    """Linear map from altitude to downsample scale in [native, cruise]."""
    if cruise_alt_m <= min_alt_m:
        return native_scale
    t = (float(alt_m) - min_alt_m) / (cruise_alt_m - min_alt_m)
    t = max(0.0, min(1.0, t))
    return native_scale + t * (cruise_scale - native_scale)


def effective_gsd_m(alt_m: float, native_gsd_m: float = NATIVE_GSD_M, **kwargs) -> float:
    return float(native_gsd_m) * effective_scale(alt_m, **kwargs)


def degrade_to_scale(image: Image.Image, scale: float) -> Image.Image:
    """Downsample by `scale` then upsample back, with scale-dependent blur.

    scale=1 is a no-op. scale=4 is a 4× coarser GSD.
    """
    if scale <= 1.01 or image.width < 2 or image.height < 2:
        return image
    w, h = image.size
    small_w = max(1, int(round(w / scale)))
    small_h = max(1, int(round(h / scale)))
    radius = max(0.0, 0.5 * (scale - 1.0))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius)) if radius > 0.15 else image
    low = blurred.resize((small_w, small_h), Image.BILINEAR)
    return low.resize((w, h), Image.BILINEAR)


def ladder_points(
    n: int = 5,
    cruise_alt_m: float = CRUISE_ALT_M,
    min_alt_m: float = MIN_ALT_M,
) -> list[dict]:
    if n < 2:
        n = 2
    alts = [min_alt_m + i * (cruise_alt_m - min_alt_m) / (n - 1) for i in range(n)]
    out = []
    for alt in alts:
        scale = effective_scale(alt, cruise_alt_m=cruise_alt_m, min_alt_m=min_alt_m)
        out.append({
            "alt_m": round(alt, 3),
            "scale": round(scale, 4),
            "gsd_m": round(NATIVE_GSD_M * scale, 4),
        })
    return out


class ExpectedEntropyTable:
    """Lookup \(\hat E[U | GSD, \hat y]\) fitted offline on event-disjoint crops."""

    def __init__(self, payload: Optional[dict] = None):
        self.payload = payload or {}
        self._rows: list[dict] = list(self.payload.get("bins") or [])
        self._rows.sort(key=lambda r: float(r.get("gsd_m") or r.get("scale") or 0.0))

    @classmethod
    def load(cls, path: str | Path) -> "ExpectedEntropyTable":
        p = Path(path)
        if not p.is_file():
            return cls({})
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def expected_entropy(self, gsd_m: float, pred_class: str) -> Optional[float]:
        if not self._rows:
            return None
        gsd = float(gsd_m)
        pred = pred_class if pred_class in CLASS_NAMES else "no-damage"
        # nearest GSD bin, then that class mean entropy
        row = min(self._rows, key=lambda r: abs(float(r.get("gsd_m", 0.0)) - gsd))
        by_cls = row.get("by_pred_class") or {}
        stats = by_cls.get(pred) or by_cls.get("all") or {}
        mean_u = stats.get("mean_entropy")
        return float(mean_u) if mean_u is not None else None

    def info_gain(
        self,
        entropy_now: float,
        alt_now_m: float,
        descend_step_m: float,
        alt_min_m: float,
        pred_class: str,
    ) -> Optional[float]:
        alt_after = max(float(alt_now_m) - float(descend_step_m), float(alt_min_m))
        gsd_after = effective_gsd_m(alt_after)
        expected_after = self.expected_entropy(gsd_after, pred_class)
        if expected_after is None:
            return None
        return max(0.0, float(entropy_now) - float(expected_after))
