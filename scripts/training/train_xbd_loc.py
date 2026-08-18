#!/usr/bin/env python3
"""Thin wrapper: train building localization U-Net (see backend/building_localization.py)."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from building_localization import train_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(train_main())
