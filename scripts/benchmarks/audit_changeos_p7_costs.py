#!/usr/bin/env python3
"""Audit full trajectory-state motion and cost observability in ChangeOS VQA.

State-derived distances are geometric estimates, not measured UAV flight time.
No controller, model, or basemap source is modified by this offline audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motion_between(a: dict, b: dict) -> tuple[float, float]:
    north = (float(b["lat"]) - float(a["lat"])) * 110540.0
    east = (float(b["lon"]) - float(a["lon"])) * 111320.0 * math.cos(math.radians(float(a["lat"])))
    vertical = float(b["alt"]) - float(a["alt"])
    return math.hypot(north, east), abs(vertical)


def audit_row(row: dict, start: dict) -> dict:
    steps = list(row.get("trajectory") or [])
    first = (steps[0].get("position") or {}) if steps else {}
    initial_gap_h, initial_gap_v = motion_between(start, first) if first else (0.0, 0.0)
    total_h = total_v = search_h = search_v = reobserve_h = reobserve_v = 0.0
    n_search_moves = n_reobserve_moves = n_unmapped_moves = 0
    requested_rechecks = allocated_rechecks = executed_rechecks = 0
    for index, step in enumerate(steps):
        kind = str(step.get("reobserve_kind") or "")
        if kind == "recheck":
            requested_rechecks += int(bool(step.get("reobserve_params")))
            before, after = step.get("budget_before"), step.get("budget_after")
            allocated_rechecks += int(isinstance(before, int) and isinstance(after, int)
                                      and after < before)
            executed_rechecks += int(bool(step.get("reobserve_executed")))
        if index + 1 >= len(steps):
            continue
        a, b = step.get("position") or {}, steps[index + 1].get("position") or {}
        if not a or not b:
            continue
        horizontal, vertical = motion_between(a, b)
        total_h += horizontal
        total_v += vertical
        if str(step.get("action") or "") == "fly_relative":
            if kind == "recheck":
                reobserve_h += horizontal
                reobserve_v += vertical
                n_reobserve_moves += int(horizontal > 0.01 or vertical > 0.01)
            else:
                search_h += horizontal
                search_v += vertical
                n_search_moves += int(horizontal > 0.01 or vertical > 0.01)
        elif horizontal > 0.01 or vertical > 0.01:
            n_unmapped_moves += 1
    last_action = str((steps[-1] if steps else {}).get("action") or "")
    # A final fly action has no following state in the recorded trajectory.
    complete = bool(steps) and last_action != "fly_relative" and first != {}
    return {
        "qid": row.get("qid"), "config": row.get("config"),
        "question_type": row.get("question_type"), "event": row.get("disaster"),
        "reason_code": row.get("reason_code"),
        "correct": bool(row.get("correct")),
        "trajectory_motion_complete": complete,
        "initial_start_to_first_state_horizontal_m": initial_gap_h,
        "initial_start_to_first_state_vertical_m": initial_gap_v,
        "total_horizontal_m": total_h, "total_vertical_m": total_v,
        "search_horizontal_m": search_h, "search_vertical_m": search_v,
        "reobserve_horizontal_m": reobserve_h,
        "reobserve_vertical_m": reobserve_v,
        "n_search_moves": n_search_moves,
        "n_reobserve_moves": n_reobserve_moves,
        "n_unmapped_state_moves": n_unmapped_moves,
        "requested_rechecks_logged": requested_rechecks,
        "allocated_rechecks_inferred_from_budget": allocated_rechecks,
        "executed_rechecks_logged": executed_rechecks,
        "reported_n_reobservations": int(row.get("n_reobservations") or 0),
        "end_to_end_episode_wall_s": float(row.get("wall_s") or 0.0),
    }


def summarize(rows: list[dict]) -> dict:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config"])].append(row)
    metrics = (
        "total_horizontal_m", "total_vertical_m",
        "search_horizontal_m", "search_vertical_m",
        "reobserve_horizontal_m", "reobserve_vertical_m",
        "requested_rechecks_logged", "allocated_rechecks_inferred_from_budget",
        "executed_rechecks_logged", "reported_n_reobservations",
        "end_to_end_episode_wall_s",
    )
    config_summaries = {}
    for config, items in sorted(by_config.items()):
        totals = {metric: sum(float(item[metric]) for item in items) for metric in metrics}
        config_summaries[config] = {
            "n": len(items),
            "n_motion_complete": sum(item["trajectory_motion_complete"] for item in items),
            "n_unmapped_state_moves": sum(item["n_unmapped_state_moves"] for item in items),
            "n_nonzero_initial_gap": sum(
                item["initial_start_to_first_state_horizontal_m"] > 0.01
                or item["initial_start_to_first_state_vertical_m"] > 0.01
                for item in items
            ),
            "totals": totals,
            "means": {metric: totals[metric] / len(items) for metric in metrics},
        }
    return {
        "schema": "changeos-p7-cost-observability/1.0",
        "measurement_scope": {
            "motion": "geometric state-to-state distances from recorded trajectory positions; all search and reobservation moves; incomplete terminal fly actions flagged",
            "latency": "per-episode wall_s only; render, ChangeOS, controller, and Qwen component timing not recorded here",
            "flight_time": "not measured; no UAV airframe dynamics or PX4/Gazebo mapping",
            "requested_allocated_executed": "recheck request and execution logged; allocation inferred only from budget decrease; search action request/allocation detail absent",
        },
        "n_rows": len(rows), "by_config": config_summaries, "rows": rows,
    }


def load(taskset: Path, runs: list[Path]) -> tuple[list[dict], list[dict]]:
    task = json.loads(taskset.read_text(encoding="utf-8"))
    starts = {str(item["id"]): item["start"] for item in task.get("items", [])}
    task_hash = sha256(taskset)
    identity = None
    seen = set()
    audited = []
    inputs = []
    for run in runs:
        result_path = run / "results.json"
        episodes_path = run / "episodes.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("valid_for_analysis") is not True or result.get("n_execution_errors") != 0:
            raise ValueError(f"invalid run refused: {run}")
        if str(result.get("testset_sha256_16") or "") != task_hash:
            raise ValueError(f"taskset hash differs for {run}")
        fp = str((result.get("env") or {}).get("source_fingerprint") or "")
        if not fp:
            raise ValueError(f"source fingerprint missing for {run}")
        if identity is None:
            identity = fp
        elif fp != identity:
            raise ValueError(f"source fingerprint differs for {run}")
        inputs.append({"run": str(run), "results_sha256": sha256(result_path),
                       "episodes_sha256": sha256(episodes_path)})
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            key = (str(row.get("config") or ""), qid)
            if key in seen or qid not in starts:
                raise ValueError(f"duplicate key or qid missing from taskset: {key}")
            seen.add(key)
            audited.append(audit_row(row, starts[qid]))
    return audited, inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, inputs = load(args.testset, args.runs)
    report = summarize(rows)
    report["inputs"] = {"testset": str(args.testset),
                        "testset_sha256": sha256(args.testset), "runs": inputs}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_rows": report["n_rows"],
        "motion_complete_by_config": {
            config: f"{data['n_motion_complete']}/{data['n']}"
            for config, data in report["by_config"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
