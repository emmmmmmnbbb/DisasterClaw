#!/usr/bin/env python3
"""Validate and approve an evidence-rich VLN test set after visual review.

Approval combines deterministic annotation/geometry checks with an explicit
reviewer attestation that the contact sheets produced by
``render_vln_review_sheet.py`` were inspected.  It does not claim an
independent damage relabeling; xBD remains the severity ground truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageStat

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from geo import latlon_to_meters  # noqa: E402
from xbd_map import geo_to_pixel  # noqa: E402

ALLOWED_SUBTYPES = {"major-damage", "destroyed"}


def _distance_m(a: dict, b: dict) -> float:
    north, east = latlon_to_meters(a["lat"], a["lon"], b["lat"], b["lon"])
    return math.hypot(north, east)


def _quality_checks(item: dict, entry: dict, dataset_root: Path) -> tuple[dict, list[str]]:
    reasons: list[str] = []
    goals = item.get("goals") or []
    if len(goals) != 1:
        reasons.append("requires exactly one goal")
        return {}, reasons
    goal = goals[0]
    start = item.get("start") or {}
    if goal.get("subtype") not in ALLOWED_SUBTYPES:
        reasons.append("goal is not major-damage/destroyed")
    distance = _distance_m(start, goal)
    if not 28.0 <= distance <= 55.0:
        reasons.append(f"start-goal distance {distance:.1f}m outside [28,55]m")

    transform = {
        "pixel_to_geo": entry.get("pixel_to_geo"),
        "geo_to_pixel": entry.get("geo_to_pixel"),
    }
    if not transform["geo_to_pixel"]:
        reasons.append("missing georeference")
        return {}, reasons
    gx, gy = geo_to_pixel(transform, goal["lon"], goal["lat"])
    sx, sy = geo_to_pixel(transform, start["lon"], start["lat"])
    width = int(entry.get("width") or 1024)
    height = int(entry.get("height") or 1024)
    margin = 32
    if not (margin <= gx < width - margin and margin <= gy < height - margin):
        reasons.append("goal too close to image boundary")
    if not (0 <= sx < width and 0 <= sy < height):
        reasons.append("start outside image")

    image_path = dataset_root / entry["image_relpath"]
    if not image_path.exists():
        reasons.append("post image missing")
        return {}, reasons
    with Image.open(image_path).convert("L") as image:
        r = 24
        crop = image.crop((
            max(0, int(gx) - r),
            max(0, int(gy) - r),
            min(width, int(gx) + r),
            min(height, int(gy) + r),
        ))
        stat = ImageStat.Stat(crop)
        mean_luma = float(stat.mean[0]) if stat.mean else 0.0
        if mean_luma < 3.0:
            reasons.append("goal crop is effectively black/no-data")

    checks = {
        "target_visible_in_post_image": not any(
            "boundary" in reason or "black" in reason or "missing" in reason for reason in reasons
        ),
        "damage_label_unambiguous": goal.get("subtype") in ALLOWED_SUBTYPES,
        "instruction_unambiguous": len(goals) == 1 and bool(item.get("instruction")),
        "start_and_goal_in_bounds": not any(
            "outside" in reason or "boundary" in reason for reason in reasons
        ),
        "start_goal_distance_m": round(distance, 2),
        "goal_pixel": [round(gx, 1), round(gy, 1)],
        "goal_crop_mean_luma": round(mean_luma, 2),
    }
    return checks, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--dataset-root", default="/home/lc/datasets/xbd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument(
        "--contact-sheets-reviewed",
        action="store_true",
        help="Attest that every rendered contact-sheet panel was visually inspected.",
    )
    args = ap.parse_args()
    if not args.contact_sheets_reviewed:
        print("Refusing approval without --contact-sheets-reviewed", file=sys.stderr)
        return 2

    data = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = {entry["tile_id"]: entry for entry in manifest.get("items", [])}
    approved: list[dict] = []
    rejected: list[dict] = []
    reviewed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for item in data.get("items", []):
        entry = entries.get(item.get("tile_id"))
        if entry is None:
            checks, reasons = {}, ["tile missing from manifest"]
        else:
            checks, reasons = _quality_checks(item, entry, Path(args.dataset_root))
        review = {
            "status": "approved" if not reasons else "rejected",
            "reviewer": args.reviewer,
            "reviewed_at": reviewed_at,
            "checks": checks,
            "notes": (
                "Contact-sheet visual inspection plus deterministic geometry/no-data checks; "
                "damage severity follows the xBD annotation and was not independently relabeled."
                if not reasons else "; ".join(reasons)
            ),
        }
        item["review"] = review
        (approved if not reasons else rejected).append(item)

    data["items"] = approved
    data["stratification"] = {
        "total": len(approved),
        "by_disaster": {
            disaster: sum(1 for item in approved if item.get("disaster") == disaster)
            for disaster in sorted({item.get("disaster") for item in approved})
        },
        "by_difficulty": {
            difficulty: sum(1 for item in approved if item.get("difficulty") == difficulty)
            for difficulty in ("easy", "medium", "hard")
        },
        "with_direction": sum(1 for item in approved if item.get("with_direction")),
        "multi_landmark": sum(1 for item in approved if item.get("multi")),
    }
    data["review_summary"] = {
        "reviewer": args.reviewer,
        "reviewed_at": reviewed_at,
        "contact_sheets_reviewed": True,
        "approved": len(approved),
        "rejected": len(rejected),
        "rejected_items": [
            {"id": item["id"], "reason": item["review"]["notes"]} for item in rejected
        ],
        "severity_ground_truth": "xBD annotations; not independently relabeled",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"approved={len(approved)} rejected={len(rejected)} -> {out}")
    return 0 if approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
