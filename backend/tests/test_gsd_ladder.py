"""backend/tests/test_gsd_ladder.py"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_split import TRAIN_EVENTS, assert_eval_only, event_partition  # noqa: E402
from gsd_ladder import (  # noqa: E402
    CRUISE_SCALE,
    NATIVE_SCALE,
    degrade_to_scale,
    effective_gsd_m,
    effective_scale,
)


def test_scale_endpoints() -> None:
    assert abs(effective_scale(10.0) - NATIVE_SCALE) < 1e-6
    assert abs(effective_scale(30.0) - CRUISE_SCALE) < 1e-6
    assert effective_gsd_m(30.0) > effective_gsd_m(10.0)
    print("[OK] GSD endpoints")


def test_degrade_keeps_size() -> None:
    img = Image.new("RGB", (64, 64), (80, 80, 80))
    out = degrade_to_scale(img, 4.0)
    assert out.size == (64, 64)
    print("[OK] degrade keeps size")


def test_degrade_changes_pixels() -> None:
    img = Image.new("RGB", (96, 96), (0, 0, 0))
    for x in range(0, 96, 2):
        img.putpixel((x, 48), (255, 255, 255))
    out = degrade_to_scale(img, 4.0)
    assert out.size == (96, 96)
    assert list(out.getdata()) != list(img.getdata())
    print("[OK] degrade changes high-frequency content")


def test_event_split() -> None:
    assert event_partition("hurricane-michael") == "test"
    assert event_partition("hurricane-harvey") == "val"
    assert "guatemala-volcano" in TRAIN_EVENTS
    try:
        assert_eval_only("hurricane-harvey")
    except ValueError:
        print("[OK] leak assertion")
        return
    raise AssertionError("val event should be rejected")


if __name__ == "__main__":
    test_scale_endpoints()
    test_degrade_keeps_size()
    test_degrade_changes_pixels()
    test_event_split()
    print("all passed")
