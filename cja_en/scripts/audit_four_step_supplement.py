#!/usr/bin/env python3
"""Recompute the matched-call, spatial-overlap and detection audits.

Uses frozen local artifacts only; no model inference or manuscript editing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/benchmarks/changeos_binary"
DEV = ROOT / "backend/data/benchmarks/agent_vqa_v2_binary.json"
FINAL = ROOT / "backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json"
MANIFEST = ROOT / "backend/data/xbd/manifest.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def bootstrap(forks: list[dict], draws: int = 10000) -> list[float]:
    clusters = defaultdict(list)
    for row in forks:
        clusters[row["roi_id"]].append(int(row["changed_correct"]) - int(row["fixed_correct"]))
    ids = sorted(clusters)
    rng = random.Random(20260920)
    values = []
    for _ in range(draws):
        sample = [v for roi in rng.choices(ids, k=len(ids)) for v in clusters[roi]]
        values.append(sum(sample) / len(sample))
    values.sort()
    return [values[int(.025 * draws)], values[int(.975 * draws) - 1]]


def summarize_forks(episodes: list[dict]) -> dict:
    forks = [f for e in episodes for f in e["forks"]]
    if not all(f["same_fixed_image_as_pre"] and f["matched_seed"] and
               f["matched_template"] and f["changed_observation_id"] for f in forks):
        raise ValueError("a matched-call fork failed the strict validity gate")
    cells = Counter((bool(f["fixed_correct"]), bool(f["changed_correct"])) for f in forks)
    return {
        "episodes": len(episodes), "pairs": len(forks),
        "fixed_correct": sum(bool(f["fixed_correct"]) for f in forks),
        "changed_correct": sum(bool(f["changed_correct"]) for f in forks),
        "fixed_accuracy": sum(bool(f["fixed_correct"]) for f in forks) / len(forks),
        "changed_accuracy": sum(bool(f["changed_correct"]) for f in forks) / len(forks),
        "paired_difference": (cells[(False, True)] - cells[(True, False)]) / len(forks),
        "roi_cluster_95pct_ci": bootstrap(forks),
        "help": cells[(False, True)], "harm": cells[(True, False)],
        "wrong_wrong": cells[(False, False)], "correct_correct": cells[(True, True)],
        "old_view_byte_identical": sum(f["pre_call"]["image_sha256"] == f["fixed_call"]["image_sha256"] for f in forks),
        "old_view_answer_agreement": sum(f["pre_answer"] == f["fixed_answer"] for f in forks),
        "old_view_prompt_identical": sum(f["pre_call"]["prompt_sha256"] == f["fixed_call"]["prompt_sha256"] for f in forks),
        "old_view_seed_identical": sum(f["pre_call"]["generation_seed"] == f["fixed_call"]["generation_seed"] for f in forks),
    }


def matched_audit() -> dict:
    paths = sorted(RUN.glob("revision_matched_view_repeat*_shard*of4_full_20260920/episodes.jsonl"))
    if len(paths) != 12:
        raise ValueError(f"expected 12 matched shards, found {len(paths)}")
    episodes = [e for path in paths for e in read_jsonl(path)]
    references = {}
    for path in sorted(RUN.glob("p6_final_three_event_v1_20260915_repeat*_shard*of4/episodes.jsonl")):
        for row in read_jsonl(path):
            if row["config"] == "T1_TASK":
                references[(row["generation_repeat"], row["qid"])] = [
                    i for i, step in enumerate(row["trajectory"])
                    if step["decision"] == "reobserve"
                ]
    mismatch = []
    retained = []
    for row in episodes:
        actual = [i for i, step in enumerate(row["trajectory"])
                  if step["decision"] == "reobserve"]
        expected = references[(row["repeat"], row["qid"])]
        if actual == expected:
            retained.append(row)
        else:
            mismatch.append({"qid": row["qid"], "repeat": row["repeat"],
                             "actual_steps": actual, "reference_steps": expected,
                             "n_forks": len(row["forks"])})
    return {"all": summarize_forks(episodes), "schedule_matched": summarize_forks(retained),
            "schedule_mismatch": mismatch,
            "old_view_agreement_scope": "pre-answer versus fixed branch on byte-identical old image; prompts and generation seeds may differ"}


def area(box: dict) -> float:
    return (box["east"] - box["west"]) * (box["north"] - box["south"])


def overlap(a: dict, b: dict) -> tuple[float, float]:
    dx = max(0., min(a["east"], b["east"]) - max(a["west"], b["west"]))
    dy = max(0., min(a["north"], b["north"]) - max(a["south"], b["south"]))
    intersection = dx * dy
    if intersection == 0:
        return 0., 0.
    return intersection / (area(a) + area(b) - intersection), intersection / min(area(a), area(b))


def footprint(position: dict) -> dict:
    lat, lon = float(position["lat"]), float(position["lon"])
    alt = max(443.405, min(1330.215, float(position["alt"])))
    half = alt * math.tan(math.radians(60.) / 2.)
    dlat = half / 110540.
    dlon = half / (111320. * max(math.cos(math.radians(lat)), 1e-6))
    return {"west": lon - dlon, "south": lat - dlat,
            "east": lon + dlon, "north": lat + dlat}


def pair_audit(left: dict[str, dict], right: dict[str, dict]) -> dict:
    max_iou = (0., "", "")
    max_small = (0., "", "")
    positive = set()
    total = 0
    for lid, a in left.items():
        for rid, b in right.items():
            total += 1
            iou, smaller = overlap(a, b)
            if iou > 0:
                positive.add((lid, rid))
            if iou > max_iou[0]:
                max_iou = (iou, lid, rid)
            if smaller > max_small[0]:
                max_small = (smaller, lid, rid)
    return {"pair_count": total, "overlapping_pairs": len(positive),
            "max_iou": max_iou[0], "max_iou_pair": list(max_iou[1:]),
            "max_overlap_of_smaller": max_small[0],
            "max_overlap_pair": list(max_small[1:])}


def building_ids(items: dict[str, dict], manifest: dict) -> set[tuple[str, str]]:
    by_tile = {x["tile_id"]: x for x in manifest["items"]}
    root = Path(manifest["dataset_root"])
    ids = set()
    for tile, item in items.items():
        entry = by_tile[tile]
        label = read_json(root / entry["label_relpath"])
        bounds = item["roi"]["bounds"]
        transform = entry["pixel_to_geo"]
        for feature in label["features"]["xy"]:
            properties = feature.get("properties", {})
            uid = properties.get("uid")
            if properties.get("subtype") not in {"no-damage", "minor-damage", "major-damage", "destroyed"}:
                continue
            points = re.search(r"POLYGON\s*\(\((.*?)\)\)", feature.get("wkt", ""), re.S)
            if not uid or not points:
                continue
            xy = [[float(v) for v in p.strip().split()[:2]] for p in points.group(1).split(",")]
            x = sum(p[0] for p in xy) / len(xy)
            y = sum(p[1] for p in xy) / len(xy)
            lon = transform["lon"][0] * x + transform["lon"][1] * y + transform["lon"][2]
            lat = transform["lat"][0] * x + transform["lat"][1] * y + transform["lat"][2]
            if bounds["west"] <= lon <= bounds["east"] and bounds["south"] <= lat <= bounds["north"]:
                ids.add((item["disaster"], uid))
    return ids


def spatial_audit() -> dict:
    dev_items = {x["tile_id"]: x for x in read_json(DEV)["items"]}
    final_items = {x["tile_id"]: x for x in read_json(FINAL)["items"]}
    manifest = read_json(MANIFEST)
    dev_ids = building_ids(dev_items, manifest)
    final_ids = building_ids(final_items, manifest)
    dev_rois = {k: v["roi"]["bounds"] for k, v in dev_items.items()}
    final_rois = {k: v["roi"]["bounds"] for k, v in final_items.items()}
    # The development diagnostic observes every ROI at all three ladder heights.
    ladder = (1330.215, 886.81, 443.405)
    dev_footprints = {f"{k}@{int(h)}": footprint({**v["roi"]["center"], "alt": h})
                      for k, v in dev_items.items() for h in ladder}
    # Every actual observation recorded under the frozen T1 final policy, all repeats.
    final_footprints = {}
    final_episodes = 0
    for path in sorted(RUN.glob("p6_final_three_event_v1_20260915_repeat*_shard*of4/episodes.jsonl")):
        for row in read_jsonl(path):
            if row["config"] != "T1_TASK":
                continue
            final_episodes += 1
            for i, step in enumerate(row["trajectory"]):
                pos = step.get("position")
                if pos:
                    final_footprints[f"{row['qid']}#r{row['generation_repeat']}#s{i}"] = footprint(pos)
    if final_episodes != 480:
        raise ValueError(f"expected 480 final T1 episodes, found {final_episodes}")
    return {"dev_rois": len(dev_items), "final_rois": len(final_items),
            "shared_tile_ids": len(set(dev_items) & set(final_items)),
            "dev_building_ids": len(dev_ids), "final_building_ids": len(final_ids),
            "shared_building_ids": len(dev_ids & final_ids),
            "building_id_definition": "(disaster, xBD feature UID), centroid inside question ROI",
            "roi": pair_audit(dev_rois, final_rois),
            "dev_ladder_footprints": len(dev_footprints),
            "final_t1_observation_footprints": len(final_footprints),
            "observation_footprint": pair_audit(dev_footprints, final_footprints),
            "footprint_definition": "axis-aligned geographic window from recorded pose and 60-degree FOV; all T1 final observations, three repeats"}


def detection_audit() -> dict:
    report = read_json(RUN / "p4_fov_ladder_dev_20260914/report.json")
    out = {"n_rois": report["n_rois"], "n_gt": report["n_buildings"], "views": {}}
    for name, view in report["views"].items():
        tp = view["n_matched"]
        fp = view["n_unmatched_predictions"]
        fn = view["n_unmatched_gt"]
        if tp + fp != view["n_predictions"] or tp + fn != report["n_buildings"]:
            raise ValueError(f"inconsistent detection counts at {name}")
        out["views"][name] = {"tp": tp, "fp": fp, "fn": fn,
                               "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                               "fp_per_roi": fp / report["n_rois"],
                               "all_gt_binary_accuracy": view["binary_accuracy_all_gt"],
                               "matched_binary_accuracy": view["matched_only"]["accuracy"],
                               "matched_macro_f1": view["matched_only"]["macro_f1"]}
    out["matching_rule"] = "one-to-one nearest-centroid match within 12 m; predictions restricted to evaluated ROI"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "cja_en/review/four_step_audit.json")
    args = parser.parse_args()
    report = {"schema": "disasterclaw-four-step-audit/1.0",
              "matched_call": matched_audit(), "spatial": spatial_audit(),
              "detection": detection_audit()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched_call": report["matched_call"], "spatial": report["spatial"],
                      "detection": report["detection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
