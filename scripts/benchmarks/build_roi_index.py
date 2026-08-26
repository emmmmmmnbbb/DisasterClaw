#!/usr/bin/env python3
"""预计算每张 xBD post 瓦片在巡航视场下的真实影像覆盖率，写 roi_index.json。

作者决策（2026-08-24）：ROI 场景池要求巡航视场（3×3 瓦片 / 1536 m）的真实
xBD 覆盖 >= 0.80，其余 <=20% 由 Esri 底图补齐。实测全量均值仅 0.50，
不设门槛会让平均半个画面是异日期灾前影像。

用法::
    python scripts/benchmarks/build_roi_index.py [--min-coverage 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import fov_ladder as FL  # noqa: E402
import xbd_map  # noqa: E402
import mosaic as mosaic_mod  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "backend/data/xbd/manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "backend/data/xbd/roi_index.json"))
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--grid", type=int, default=64)
    args = ap.parse_args()

    manifest = xbd_map.load_manifest(args.manifest)
    # 只做几何计算，不抓底图
    mo = mosaic_mod.TileMosaic(
        entries=manifest.get("items") or [],
        dataset_root=manifest.get("dataset_root") or "",
        basemap=None,
    )
    span = FL.span_m_for_alt(FL.alt_cruise_m())
    cov = mo.build_roi_index(span_m=span, grid=args.grid)

    kept = {k: v for k, v in cov.items() if v >= args.min_coverage}
    by_event: Counter = Counter()
    for tid in kept:
        e = mo.get_entry(tid)
        by_event[(e or {}).get("disaster") or "unknown"] += 1

    payload = {
        "schema": "xbd-roi-index/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cruise_span_m": round(span, 3),
        "cruise_span_tiles": FL.SPAN_TILES_MAX,
        "cruise_alt_m": round(FL.alt_cruise_m(), 3),
        "floor_alt_m": round(FL.alt_min_m(), 3),
        "grid": args.grid,
        "min_coverage": args.min_coverage,
        "n_total": len(cov),
        "n_kept": len(kept),
        "by_event": dict(sorted(by_event.items(), key=lambda kv: -kv[1])),
        "coverage": cov,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total post tiles : {len(cov)}")
    print(f"kept (>= {args.min_coverage}) : {len(kept)}")
    for ev, n in payload["by_event"].items():
        print(f"  {ev:24s} {n:4d}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
