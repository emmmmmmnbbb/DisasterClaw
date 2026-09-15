"""backend/detectors — 可切换的双时相建筑损伤检测后端 (计划 §3.4)。

`DETECTOR_BACKEND` 默认 `legacy_unet`，保持既有可复现产物不变；
也可切到 ChangeOS 或 xView2 后端。各后端的训练状态和实验角色由
``describe()`` 显式报告。
"""
from __future__ import annotations

import os

from .base import (
    BINARY_DAMAGE_SUBTYPES,
    DAMAGE_SUBTYPES,
    SUBTYPE_TO_ZH,
    Detection,
    DetectorBackend,
)

DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "legacy_unet").strip().lower()

__all__ = [
    "BINARY_DAMAGE_SUBTYPES", "DAMAGE_SUBTYPES", "SUBTYPE_TO_ZH", "Detection", "DetectorBackend",
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
    if key in {"changeos", "changeos_r34", "binary_changeos"}:
        from .changeos import ChangeOSDetector

        kwargs.setdefault("device", os.getenv("PERCEPTION_DEVICE", "cuda"))
        kwargs.setdefault("weights", os.getenv("CHANGEOS_WEIGHTS", "") or None)
        kwargs.setdefault("input_size", int(os.getenv("CHANGEOS_INPUT_SIZE", "1024")))
        kwargs.setdefault("loc_threshold", float(os.getenv("CHANGEOS_LOC_THRESHOLD", "0.5")))
        kwargs.setdefault("damage_threshold", float(os.getenv("CHANGEOS_DAMAGE_THRESHOLD", "0.5")))
        kwargs.setdefault("temperature", float(os.getenv("CHANGEOS_TEMPERATURE", "1.0")))
        return ChangeOSDetector(**kwargs)
    raise ValueError(f"unknown DETECTOR_BACKEND: {key!r}")
