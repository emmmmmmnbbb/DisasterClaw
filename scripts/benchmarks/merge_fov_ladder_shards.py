#!/usr/bin/env python3
"""Merge deterministic ROI shards produced by eval_fov_ladder.py."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import fov_ladder as FL
from eval_fov_ladder import CLASS_NAMES, VIEW_ALTS, _items_hash, _metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    args = ap.parse_args()

    rows = []
    for raw in args.inputs:
        path = Path(raw)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by_uid = {row["uid"]: row for row in rows}
    if len(by_uid) != len(rows):
        raise ValueError("duplicate building uid across FOV shards")
    rows = [by_uid[key] for key in sorted(by_uid)]
    labels = np.asarray([int(row["y"]) for row in rows], dtype=np.int64)
    curve = []
    for view_name in VIEW_ALTS:
        probs = np.stack([
            [float(row["views"][view_name]["probs"][name]) for name in CLASS_NAMES]
            for row in rows
        ])
        curve.append({
            "view": view_name,
            "alt_m": round(VIEW_ALTS[view_name], 3),
            "gsd_m": round(FL.eff_gsd_for_alt(VIEW_ALTS[view_name]), 6),
            **_metrics(probs, labels),
        })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"fov_ladder_{args.split}_items.jsonl"
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema": "fov-ladder-eval/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "events": sorted({row.get("disaster") for row in rows}),
        "n_rois": len({row["tile_id"] for row in rows}),
        "n_buildings": len(rows),
        "observation_model": "mosaic_fov",
        "temperature": args.temperature,
        "items_sha256": _items_hash(rows),
        "curve": curve,
        "merged_from": [str(Path(raw)) for raw in args.inputs],
    }
    (out_dir / f"fov_ladder_{args.split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
