#!/usr/bin/env python3
"""X0: single-item geodesic trace + E1/E11 item-id check + pipeline counters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BACKEND))


def _ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" in data:
        return [str(it.get("id")) for it in data["items"]]
    rows = data.get("configs") or {}
    ids = []
    for blk in rows.values():
        for row in blk.get("rows") or []:
            ids.append(str(row.get("id")))
    return ids


def compare_item_sets(e1: Path, e11: Path) -> dict:
    a, b = set(_ids(e1)), set(_ids(e11))
    return {
        "e1_n": len(a),
        "e11_n": len(b),
        "same_set": a == b,
        "only_e1": sorted(a - b)[:8],
        "only_e11": sorted(b - a)[:8],
        "note": (
            "E1 and E11 share the evidence-rich set by design; identical NE "
            "is expected when reinspection never triggers."
            if a == b
            else "item id sets differ"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(BACKEND / "data/benchmarks/vln_recheck_testset.json"))
    ap.add_argument("--item-index", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="run one headless episode and print geodesic")
    ap.add_argument("--e1-results", default="")
    ap.add_argument("--e11-results", default="")
    ap.add_argument("--out", default=str(REPO / "runs/benchmarks/x0_diag.json"))
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    item = testset["items"][args.item_index]
    start, goal = item["start"], item["goals"][0]
    from geo import latlon_to_meters

    n, e = latlon_to_meters(start["lat"], start["lon"], goal["lat"], goal["lon"])
    start_goal = (n * n + e * e) ** 0.5
    report: dict = {
        "item_id": item["id"],
        "disaster": item.get("disaster"),
        "instruction": item["instruction"],
        "start": start,
        "goal": goal,
        "start_goal_m": round(start_goal, 2),
        "lat_delta": goal["lat"] - start["lat"],
        "lon_delta": goal["lon"] - start["lon"],
        "north_m": round(n, 2),
        "east_m": round(e, 2),
    }
    if args.e1_results and args.e11_results:
        report["item_set_check"] = compare_item_sets(Path(args.e1_results), Path(args.e11_results))

    if args.trace:
        import app

        app.VLN_ORACLE_GOAL = goal
        rec = app.run_vln_episode_headless(item["instruction"], start, source="x0")
        app.VLN_ORACLE_GOAL = None
        traj = rec.get("trajectory") or []
        geodesics = []
        for step in traj:
            nn, ee = latlon_to_meters(step["lat"], step["lon"], goal["lat"], goal["lon"])
            geodesics.append({
                "lat": step["lat"],
                "lon": step["lon"],
                "alt": step.get("alt"),
                "goal_dist_m": round((nn * nn + ee * ee) ** 0.5, 2),
                "start_dist_m": step.get("start_dist_m"),
                "pipeline": step.get("pipeline"),
                "effective_gsd_m": step.get("effective_gsd_m"),
            })
        report["episode"] = {
            "error": rec.get("error"),
            "arrived": rec.get("arrived"),
            "final_pos": rec.get("final_pos"),
            "steps": rec.get("steps_executed"),
            "evidence_observations": rec.get("evidence_observations"),
            "recheck": rec.get("recheck"),
            "geodesics": geodesics,
            "distance_monotone_up": bool(
                geodesics
                and all(geodesics[i]["goal_dist_m"] <= geodesics[i + 1]["goal_dist_m"] + 1e-6
                        for i in range(len(geodesics) - 1))
            ),
        }
        if geodesics:
            report["episode"]["final_ne_m"] = geodesics[-1]["goal_dist_m"]
            report["x0_accept_ne_lt_start"] = geodesics[-1]["goal_dist_m"] < start_goal

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
