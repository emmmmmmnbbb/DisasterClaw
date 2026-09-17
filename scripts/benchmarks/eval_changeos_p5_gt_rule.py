#!/usr/bin/env python3
"""Offline P5 GT-structure + rule-answerer taskset self-check.

This is not an online policy or a Qwen oracle. GT labels are read only here,
after task construction, and never supplied to the benchmark controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import xbd_map  # noqa: E402

DAMAGED = {"minor-damage", "major-damage", "destroyed"}
SUBTYPES = DAMAGED | {"no-damage"}
BEARINGS = ("北", "东北", "东", "东南", "南", "西南", "西", "西北")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gt_buildings(entry: dict, dataset_root: Path) -> list[dict]:
    label = dataset_root / str(entry.get("label_relpath") or "")
    if not entry.get("label_relpath") or not label.is_file():
        raise ValueError(f"missing GT label for {entry.get('tile_id')}: {label}")
    data = json.loads(label.read_text(encoding="utf-8"))
    result = []
    for feature in ((data.get("features") or {}).get("lng_lat") or []):
        props = feature.get("properties") or {}
        subtype = str(props.get("subtype") or "")
        if subtype not in SUBTYPES:
            continue
        ring = xbd_map._parse_polygon_wkt(feature.get("wkt") or "")
        if not ring:
            continue
        lon, lat = xbd_map._polygon_centroid(ring)
        result.append({
            "uid": str(props.get("uid") or ""), "subtype": subtype,
            "lat": float(lat), "lon": float(lon),
        })
    return result


def in_bounds(building: dict, bounds: dict) -> bool:
    return (bounds["south"] <= building["lat"] <= bounds["north"]
            and bounds["west"] <= building["lon"] <= bounds["east"])


def north_east_m(origin: dict, point: dict) -> tuple[float, float]:
    north = (point["lat"] - origin["lat"]) * 110540.0
    east = (point["lon"] - origin["lon"]) * 111320.0 * math.cos(math.radians(origin["lat"]))
    return north, east


def nearest_damaged(buildings: list[dict], center: dict) -> dict | None:
    candidates = [building for building in buildings if building["subtype"] in DAMAGED]
    if not candidates:
        return None
    return min(candidates, key=lambda building: sum(
        component ** 2 for component in north_east_m(center, building)
    ))


def answer_from_gt(item: dict, buildings: list[dict]) -> tuple[str | None, list[str]]:
    """Derive one answer from full-ROI GT, with explicit identity checks."""
    qtype = str(item.get("question_type") or "")
    center = item["roi"]["center"]
    target = item.get("target") or {}
    damaged = [building for building in buildings if building["subtype"] in DAMAGED]
    issues = []
    if qtype == "presence":
        return ("是" if damaged else "否"), issues
    if qtype == "count":
        count = len(damaged)
        return ("3+" if count >= 3 else str(count)), issues
    if qtype == "damage":
        ref_id = str(target.get("ref_id") or "")
        matches = [building for building in buildings if building["uid"] == ref_id]
        if len(matches) != 1:
            return None, [f"target_uid_matches={len(matches)}"]
        building = matches[0]
        if str(target.get("subtype") or "") != building["subtype"]:
            issues.append("target_subtype_mismatch")
        if "lat" not in target or "lon" not in target:
            issues.append("target_coordinate_missing")
        else:
            north, east = north_east_m(target, building)
            if math.hypot(north, east) > 1.0:
                issues.append("target_coordinate_mismatch_gt_1m")
        return ("无损伤" if building["subtype"] == "no-damage" else "损伤"), issues
    if qtype == "spatial":
        building = nearest_damaged(buildings, center)
        if building is None:
            return None, ["no_damaged_building_for_spatial"]
        north, east = north_east_m(center, building)
        angle = math.degrees(math.atan2(east, north)) % 360.0
        direction = BEARINGS[int((angle + 22.5) // 45) % 8]
        if "lat" not in target or "lon" not in target:
            issues.append("nearest_target_coordinate_missing")
        else:
            target_north, target_east = north_east_m(target, building)
            if math.hypot(target_north, target_east) > 1.0:
                issues.append("nearest_target_coordinate_mismatch_gt_1m")
        return direction, issues
    return None, [f"unsupported_question_type={qtype}"]


def evaluate(taskset: dict, manifest: dict, dataset_root: Path) -> dict:
    entries = {str(entry.get("tile_id")): entry for entry in manifest.get("items", [])
               if entry.get("stage") == "post"}
    cache: dict[str, list[dict]] = {}
    rows = []
    by_type = Counter()
    correct_by_type = Counter()
    issue_counts = Counter()
    for item in taskset.get("items", []):
        tile_id = str(item.get("tile_id") or "")
        qtype = str(item.get("question_type") or "")
        issues = []
        entry = entries.get(tile_id)
        if entry is None:
            answer, issues = None, ["tile_missing_from_manifest"]
            count = 0
        else:
            if tile_id not in cache:
                cache[tile_id] = load_gt_buildings(entry, dataset_root)
            buildings = [building for building in cache[tile_id]
                         if in_bounds(building, item["roi"]["bounds"])]
            answer, issues = answer_from_gt(item, buildings)
            count = len(buildings)
        expected = str(item.get("answer") or "")
        choices = item.get("choices") or []
        if answer not in choices:
            issues.append("derived_answer_not_in_choices")
        if expected not in choices:
            issues.append("taskset_answer_not_in_choices")
        match = answer == expected and not issues
        by_type[qtype] += 1
        correct_by_type[qtype] += int(match)
        issue_counts.update(issues)
        rows.append({
            "qid": item.get("id"), "tile_id": tile_id, "question_type": qtype,
            "taskset_answer": expected, "gt_rule_answer": answer,
            "n_gt_buildings_in_roi": count, "answer_matches": match,
            "issues": issues,
        })
    n = len(rows)
    report = {
        "schema": "changeos-p5-gt-rule-selfcheck/1.0",
        "definition": "Full ROI xBD GT structured buildings + deterministic binary task answerer; offline taskset workflow check, not Qwen performance",
        "n_questions": n, "n_unique_roi": len(cache),
        "n_answer_matches": sum(bool(row["answer_matches"]) for row in rows),
        "valid_for_taskset_selfcheck": bool(n) and all(row["answer_matches"] for row in rows),
        "by_question_type": {
            qtype: {"n": by_type[qtype], "matches": correct_by_type[qtype],
                    "match_rate": correct_by_type[qtype] / by_type[qtype]}
            for qtype in sorted(by_type)
        },
        "issue_counts": dict(issue_counts), "rows": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    taskset = json.loads(args.testset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if taskset.get("damage_label_mode") != "binary":
        raise SystemExit("P5 GT rule self-check requires a binary damage taskset")
    manifest_hash = sha256(args.manifest)
    declared_hash = str(taskset.get("dataset_manifest_sha256") or "")
    if declared_hash and declared_hash != manifest_hash:
        raise SystemExit("taskset dataset_manifest_sha256 differs from supplied manifest")
    dataset_root = Path(str(manifest["dataset_root"]))
    report = evaluate(taskset, manifest, dataset_root)
    report["inputs"] = {
        "testset": str(args.testset), "testset_sha256": sha256(args.testset),
        "manifest": str(args.manifest), "manifest_sha256": manifest_hash,
        "dataset_root": str(dataset_root),
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "n_questions", "n_unique_roi", "n_answer_matches",
        "valid_for_taskset_selfcheck", "by_question_type", "issue_counts",
    )}, ensure_ascii=False, indent=2))
    return 0 if report["valid_for_taskset_selfcheck"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
