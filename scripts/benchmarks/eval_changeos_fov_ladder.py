#!/usr/bin/env python3
"""Evaluate frozen ChangeOS at cruise/mid/floor on the same xBD ROIs.

This is the P4 binary evidence-chain evaluator.  It reports localization,
matched-only binary classification and probability quality, while retaining
unmatched buildings and correction/harm transitions instead of hiding them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import fov_ladder as FL  # noqa: E402
import mosaic as mosaic_mod  # noqa: E402
import xbd_map  # noqa: E402
from eval_changeos_conformal import _observe_changeos  # noqa: E402
from eval_identifiability import _gt_buildings, _match  # noqa: E402

CLASS_NAMES = ("no-damage", "damaged")


def binary_label(subtype: str) -> str:
    return "no-damage" if subtype == "no-damage" else "damaged"


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float | None:
    if not len(conf):
        return None
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = ((conf >= edges[i]) & (conf <= edges[i + 1])) if i == 0 else (
            (conf > edges[i]) & (conf <= edges[i + 1])
        )
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return total


def summarize_view(rows: list[dict], n_predictions: int) -> dict:
    """Summarize one altitude; probability metrics use matched buildings only."""
    n_gt = len(rows)
    matched = [row for row in rows if row.get("matched")]
    correct_all = sum(bool(row.get("correct")) for row in rows)
    correct_matched = sum(bool(row.get("correct")) for row in matched)
    loc_tp = len(matched)
    loc_fp = max(0, int(n_predictions) - loc_tp)
    loc_fn = n_gt - loc_tp

    per_class = {}
    f1s = []
    for name in CLASS_NAMES:
        tp = sum(r["gt"] == name and r.get("pred") == name for r in matched)
        fp = sum(r["gt"] != name and r.get("pred") == name for r in matched)
        fn = sum(r["gt"] == name and r.get("pred") != name for r in matched)
        value = _f1(tp, fp, fn)
        f1s.append(value)
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(value, 6)}

    probs = np.asarray([float(r["p_damage"]) for r in matched], dtype=np.float64)
    labels = np.asarray([1.0 if r["gt"] == "damaged" else 0.0 for r in matched])
    if len(probs):
        clipped = np.clip(probs, 1e-9, 1 - 1e-9)
        confidence = np.maximum(clipped, 1 - clipped)
        correct = np.asarray([bool(r["correct"]) for r in matched], dtype=np.float64)
        brier = float(np.mean((probs - labels) ** 2))
        nll = float(np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))))
        ece = _ece(confidence, correct)
    else:
        brier = nll = ece = None

    return {
        "n_gt": n_gt,
        "n_predictions": int(n_predictions),
        "n_matched": loc_tp,
        "n_unmatched_gt": loc_fn,
        "n_unmatched_predictions": loc_fp,
        "localization_precision": round(loc_tp / (loc_tp + loc_fp), 6) if loc_tp + loc_fp else 0.0,
        "localization_recall": round(loc_tp / n_gt, 6) if n_gt else 0.0,
        "localization_f1": round(_f1(loc_tp, loc_fp, loc_fn), 6),
        "binary_accuracy_all_gt": round(correct_all / n_gt, 6) if n_gt else None,
        "matched_only": {
            "n": len(matched),
            "accuracy": round(correct_matched / len(matched), 6) if matched else None,
            "macro_f1": round(float(np.mean(f1s)), 6) if matched else None,
            "per_class": per_class,
            "brier": round(brier, 6) if brier is not None else None,
            "nll": round(nll, 6) if nll is not None else None,
            "ece10": round(ece, 6) if ece is not None else None,
        },
    }


def transition(rows: list[dict], before: str, after: str) -> dict:
    pairs = [(r["views"][before], r["views"][after]) for r in rows]
    corrected = sum((not a["correct"]) and b["correct"] for a, b in pairs)
    harmed = sum(a["correct"] and (not b["correct"]) for a, b in pairs)
    flipped = sum(a.get("pred") != b.get("pred") for a, b in pairs)
    return {
        "n": len(pairs), "n_prediction_flip": flipped,
        "n_corrected": corrected, "n_harmed": harmed,
        "net_corrected": corrected - harmed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="changeos")
    parser.add_argument("--device", default=os.getenv("PERCEPTION_DEVICE", "cuda:0"))
    parser.add_argument("--manifest", default=str(ROOT / "backend/data/xbd/manifest.json"))
    parser.add_argument("--tiles-from", default=str(ROOT / "backend/data/benchmarks/agent_vqa_v2_binary.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--items-out", required=True)
    args = parser.parse_args()

    manifest = xbd_map.load_manifest(args.manifest)
    data_root = Path(manifest["dataset_root"])
    allowed = json.loads(Path(args.tiles_from).read_text(encoding="utf-8"))
    tile_ids = {str(row.get("tile_id")) for row in allowed.get("items", [])}
    entries = [
        row for row in manifest["items"]
        if row.get("stage") == "post" and row.get("tile_id") in tile_ids
        and row.get("label_relpath") and row.get("paired_tile_id")
    ]
    entries.sort(key=lambda row: (str(row.get("disaster")), str(row.get("tile_id"))))
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise RuntimeError("no matching post-disaster ROIs")

    from detectors import get_detector

    detector = get_detector(args.backend, device=args.device)
    if detector is None or not detector.is_available():
        raise RuntimeError(f"detector unavailable: {args.backend}")
    description = detector.describe()
    if description.get("label_mode") != "binary":
        raise RuntimeError(f"P4 requires binary detector, got {description.get('label_mode')!r}")

    mosaic = mosaic_mod.from_manifest(manifest)
    points = FL.ladder_points(3)
    names = ("cruise", "intermediate", "floor")
    ladder = {name: point for name, point in zip(names, points)}
    building_rows = []
    prediction_counts = defaultdict(int)
    scene_rows = []
    for index, entry in enumerate(entries, 1):
        gt = _gt_buildings(data_root / entry["label_relpath"], entry)
        if not gt:
            continue
        per_view = {}
        for name in names:
            detections, meta = _observe_changeos(mosaic, detector, entry, ladder[name]["alt_m"])
            matched = _match(gt, detections)
            prediction_counts[name] += len(detections)
            per_view[name] = (detections, matched, meta)
        for gt_index, building in enumerate(gt):
            record = {
                "tile_id": entry["tile_id"], "event": entry.get("disaster", ""),
                "gt_index": gt_index, "gt_subtype": building["subtype"],
                "gt_binary": binary_label(building["subtype"]), "views": {},
            }
            for name in names:
                detection = per_view[name][1][gt_index]
                pred = detection.get("subtype") if detection else None
                probs = detection.get("class_probs") if detection else {}
                p_damage = float(probs.get("damaged")) if "damaged" in probs else None
                record["views"][name] = {
                    "matched": detection is not None,
                    "pred": pred,
                    "p_damage": p_damage,
                    "correct": detection is not None and pred == record["gt_binary"],
                }
            building_rows.append(record)
        scene_rows.append({
            "tile_id": entry["tile_id"], "event": entry.get("disaster", ""),
            "n_gt": len(gt),
            "n_predictions": {name: len(per_view[name][0]) for name in names},
            "xbd_fraction": {name: round(float(per_view[name][2].xbd_fraction), 6) for name in names},
        })
        print(f"  {index}/{len(entries)} {entry['tile_id']} gt={len(gt)}", flush=True)

    views = {}
    for name in names:
        view_rows = []
        for row in building_rows:
            rec = dict(row["views"][name])
            rec["gt"] = row["gt_binary"]
            view_rows.append(rec)
        views[name] = summarize_view(view_rows, prediction_counts[name])
    transitions = {
        "cruise_to_intermediate": transition(building_rows, "cruise", "intermediate"),
        "intermediate_to_floor": transition(building_rows, "intermediate", "floor"),
        "cruise_to_floor": transition(building_rows, "cruise", "floor"),
    }
    report = {
        "schema": "changeos-binary-fov-ladder/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "detector": description,
        "tiles_from": args.tiles_from,
        "n_rois": len(scene_rows), "n_buildings": len(building_rows),
        "ladder": ladder, "views": views, "transitions": transitions,
        "scenes": scene_rows,
        "metric_scope": {
            "localization": "all GT and predicted instances",
            "probability_quality": "matched GT buildings only",
            "binary_accuracy_all_gt": "unmatched GT buildings counted incorrect",
        },
    }
    out = Path(args.out)
    items_out = Path(args.items_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with items_out.open("w", encoding="utf-8") as fp:
        for row in building_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n_rois": report["n_rois"], "n_buildings": report["n_buildings"],
                      "views": views, "transitions": transitions}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
