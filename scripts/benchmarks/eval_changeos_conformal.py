#!/usr/bin/env python3
"""Fit the binary conformal (APS) qhat for the ChangeOS reobservation trigger.

The A4_CONFORMAL reobservation trigger applies ``conformal_predict_set`` to the
binary ``{"no-damage", "damaged"}`` class probabilities that ChangeOS emits per
building.  The historical qhat was fitted on the four-class ``change_perception``
space; reusing it on a two-class prediction set silently changes the coverage
semantics (it becomes a bare "top-prob < qhat" threshold, no longer a coverage
guarantee).  This script re-fits qhat on the same development events used by the
four-class pipeline (``VAL_EVENTS``), but on the binary label space.

Calibration rows are the matched building detections only — a building that
ChangeOS fails to localise never enters the reobservation loop, so it has no
prediction set to calibrate.  ``n_matched`` / ``n_unmatched`` are reported
separately so the fitted coverage is not silently claimed over missed buildings.

Output is a small JSON consumed by ``freeze_recheck_manifest.py`` (which records
``label_mode`` so ``bench_agent_vqa.py`` can fail-closed on any binary/four-class
qhat mismatch).

Usage::

    python scripts/benchmarks/eval_changeos_conformal.py \
        --split val --backend changeos --device cuda:0 \
        --out runs/benchmarks/changeos_conformal/val_binary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import fov_ladder as FL  # noqa: E402
import mosaic as mosaic_mod  # noqa: E402
import xbd_map  # noqa: E402
from eval_identifiability import _gt_buildings, _match  # noqa: E402
from event_split import HOLDOUT_EVENTS, TEST_EVENTS, VAL_EVENTS  # noqa: E402
from recheck import fit_conformal_qhat  # noqa: E402

BINARY_CLASS_ORDER = ["no-damage", "damaged"]


def _binary_label(subtype: str) -> str:
    return "no-damage" if subtype == "no-damage" else "damaged"


def _view_to_geo(x: float, y: float, window: dict, out_px: int) -> tuple[float, float]:
    sw = (window["east"] - window["west"]) / out_px
    sh = (window["north"] - window["south"]) / out_px
    return window["west"] + x * sw, window["north"] - y * sh


def _observe_changeos(mo, det, entry, alt) -> tuple[list[dict], object]:
    """在给定高度渲染一次 ChangeOS 双时相视场，返回 ROI 内的检测（带经纬质心）。

    与 ``perception.py`` 的 ChangeOS 路径一致：pre/post 都渲染到
    ``fov_ladder.SENSOR_PX``（1024×1024）同一地理窗口，交给 ChangeOS 固定
    1024 输入。不使用 ``eval_identifiability._observe``——那个函数把 pre 渲染到
    原生分辨率（>3072px）再上采样 post，既慢又不匹配 ChangeOS 的真实观测。
    """
    b = entry["bounds"]
    clat = (b["north"] + b["south"]) / 2
    clon = (b["east"] + b["west"]) / 2
    post_img, meta = mo.render_for_alt(
        clat, clon, alt, stage="post", out_px=FL.SENSOR_PX,
        roi_tile_id=entry["tile_id"], enforce_roi=False,
    )
    pre_img, pre_meta = mo.render_for_alt(
        clat, clon, alt, stage="pre", out_px=FL.SENSOR_PX, enforce_roi=False,
    )
    if pre_meta.window != meta.window:
        raise RuntimeError("pre/post 窗口不一致")
    dets = det.detect(pre_img, post_img)
    kept: list[dict] = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        lon, lat = _view_to_geo((x1 + x2) / 2, (y1 + y2) / 2, meta.window, FL.SENSOR_PX)
        if not (b["west"] <= lon <= b["east"] and b["south"] <= lat <= b["north"]):
            continue
        kept.append({
            "lon": lon, "lat": lat, "subtype": d.raw_class_name,
            "conf": float(d.conf), "class_probs": dict(d.class_probs),
        })
    return kept, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="拟合 ChangeOS 二分类 conformal qhat")
    ap.add_argument("--split", choices=["val", "test", "holdout"], default="val")
    ap.add_argument("--device", default=os.getenv("PERCEPTION_DEVICE", "cuda:0"))
    ap.add_argument("--backend", default="changeos")
    ap.add_argument("--limit", type=int, default=0, help="ROI 上限，0=全部")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--manifest", default=str(ROOT / "backend/data/xbd/manifest.json"))
    ap.add_argument("--roi-index", default=str(ROOT / "backend/data/xbd/roi_index.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    events = {"val": set(VAL_EVENTS), "test": set(TEST_EVENTS),
              "holdout": set(HOLDOUT_EVENTS)}[args.split]
    manifest = xbd_map.load_manifest(args.manifest)
    root = Path(manifest["dataset_root"])
    cov = json.loads(Path(args.roi_index).read_text(encoding="utf-8"))["coverage"]
    mo = mosaic_mod.from_manifest(manifest)

    cands = [
        e for e in manifest["items"]
        if e.get("stage") == "post" and e.get("disaster") in events
        and e.get("label_relpath") and e.get("paired_tile_id")
        and float(cov.get(e["tile_id"], 0.0)) >= args.min_coverage
    ]
    cands.sort(key=lambda e: (e["disaster"], e["tile_id"]))
    if args.limit:
        cands = cands[: args.limit]
    if not cands:
        raise RuntimeError("没有符合 split/coverage 条件的 ROI")

    from detectors import get_detector

    det = get_detector(args.backend, device=args.device)
    if det is None or not det.is_available():
        print(f"backend {args.backend!r} 权重不可用", file=sys.stderr)
        return 2
    desc = det.describe()
    if desc.get("label_mode") != "binary":
        raise RuntimeError(f"本脚本只支持二分类后端，got label_mode={desc.get('label_mode')!r}")

    alt_cruise = FL.alt_cruise_m()
    rows: list[tuple[dict[str, float], str]] = []
    n_matched = 0
    n_unmatched = 0
    for i, entry in enumerate(cands, 1):
        try:
            gt = _gt_buildings(root / entry["label_relpath"], entry)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {entry['tile_id']}: {exc}", flush=True)
            continue
        if not gt:
            continue
        rb = entry["bounds"]
        try:
            d_cruise, _meta = _observe_changeos(mo, det, entry, alt_cruise)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {entry['tile_id']} (render/detect): {exc}", flush=True)
            continue
        matched = _match(gt, d_cruise)
        for g, d in zip(gt, matched):
            if d is None:
                n_unmatched += 1
                continue
            probs = d.get("class_probs") or {}
            if set(probs) != {"no-damage", "damaged"}:
                print(f"  skip {entry['tile_id']}: 非二分类 class_probs {sorted(probs)}", flush=True)
                continue
            rows.append(({k: float(probs[k]) for k in BINARY_CLASS_ORDER},
                         _binary_label(g["subtype"])))
            n_matched += 1
        if i % 5 == 0:
            print(f"  {i}/{len(cands)} ROIs matched={n_matched} unmatched={n_unmatched}",
                  flush=True)

    if not rows:
        raise RuntimeError("没有可拟合的匹配样本")
    qhat = fit_conformal_qhat(rows, alpha=0.1, class_order=BINARY_CLASS_ORDER)

    report = {
        "schema": "changeos-conformal/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "label_mode": "binary",
        "class_order": BINARY_CLASS_ORDER,
        "split": args.split,
        "events": sorted(events),
        "n_rois": len(cands),
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "conformal_qhat_alpha01": qhat,
        "conformal_alpha": 0.1,
        "detector": desc,
    }
    out = Path(args.out) if args.out else (
        ROOT / f"runs/benchmarks/changeos_conformal/{args.split}_{args.backend}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== ChangeOS 二分类 conformal / {args.split} / {args.backend} ===")
    print(f"ROI={report['n_rois']}  matched={n_matched}  unmatched={n_unmatched}")
    print(f"conformal_qhat_alpha01 = {qhat}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
