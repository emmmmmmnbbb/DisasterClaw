"""backend/detectors — 可切换的双时相建筑损伤检测后端 (计划 §3.4)。

`DETECTOR_BACKEND` 默认 `legacy_unet`，保持既有可复现产物不变；
切到 `xview2_first` 得到 SOTA 参照上界（**leaky，见该模块 docstring**）。
"""
from __future__ import annotations

import os

from .base import DAMAGE_SUBTYPES, SUBTYPE_TO_ZH, Detection, DetectorBackend

DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "legacy_unet").strip().lower()

__all__ = [
    "DAMAGE_SUBTYPES", "SUBTYPE_TO_ZH", "Detection", "DetectorBackend",
    "DETECTOR_BACKEND", "get_detector",
]


def get_detector(name: str | None = None, **kwargs):
    """按名字取后端实例。"""
    key = (name or DETECTOR_BACKEND).strip().lower()
    if key in {"xview2_first", "xview2", "sota"}:
        from .xview2_first import XView2FirstDetector

        archs = tuple(
            a for a in (os.getenv("XVIEW2_ARCHS", "res34").split(",")) if a.strip()
        )
        seeds = tuple(
            int(s) for s in os.getenv("XVIEW2_SEEDS", "0").split(",") if s.strip()
        )
        kwargs.setdefault("archs", tuple(a.strip() for a in archs))
        kwargs.setdefault("seeds", seeds)
        kwargs.setdefault("device", os.getenv("PERCEPTION_DEVICE", "cuda"))
        return XView2FirstDetector(**kwargs)
    if key in {"legacy_unet", "legacy", "unet"}:
        from .legacy_unet import LegacyUnetDetector

        kwargs.setdefault("device", os.getenv("PERCEPTION_DEVICE", "cuda"))
        return LegacyUnetDetector(**kwargs)
    if key in {"xview2_eventdisjoint", "xview2_ed", "eventdisjoint", "trackb"}:
        from .xview2_eventdisjoint import XView2EventDisjointDetector

        kwargs.setdefault("device", os.getenv("PERCEPTION_DEVICE", "cuda"))
        return XView2EventDisjointDetector(**kwargs)
    raise ValueError(f"unknown DETECTOR_BACKEND: {key!r}")
