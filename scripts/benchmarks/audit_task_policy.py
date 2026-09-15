#!/usr/bin/env python3
"""Audit P2 task-conditioned actions and cross-view evidence on development runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for path in paths:
        episode_file = path / "episodes.jsonl" if path.is_dir() else path
        for line in episode_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            latest[(str(row.get("config") or ""), str(row.get("qid") or ""))] = row
    return list(latest.values())


def audit(rows: list[dict], min_roi_coverage: float = 0.98) -> dict:
    violations: list[dict] = []
    by_type: dict[str, Counter] = defaultdict(Counter)
    association = Counter()

    def fail(row: dict, step_index: int, code: str, detail: str = "") -> None:
        violations.append({
            "qid": row.get("qid"), "question_type": row.get("question_type"),
            "step": step_index, "code": code, "detail": detail,
        })

    for row in rows:
        qtype = str(row.get("question_type") or "")
        fixed_ablation = str(row.get("config") or "").startswith("AB_")
        stats = by_type[qtype]
        stats["questions"] += 1
        stats["correct"] += int(bool(row.get("correct")))
        stats["execution_errors"] += int(not bool(row.get("ok")))
        stats["reobservations"] += int(row.get("n_reobservations") or 0)
        previous_history: list[str] = []

        for index, step in enumerate(row.get("trajectory") or []):
            evidence = step.get("evidence") or {}
            metrics = step.get("policy_metrics") or {}
            kind = str(step.get("reobserve_kind") or "")
            params = step.get("reobserve_params") or {}
            history = list(evidence.get("history_observation_ids") or [])

            if len(history) != len(set(history)):
                fail(row, index, "duplicate_history_observation")
            if previous_history and history[: len(previous_history)] != previous_history:
                fail(row, index, "non_monotonic_evidence_history")
            previous_history = history

            for obj in evidence.get("objects") or []:
                association["tracks"] += 1
                sightings = int(obj.get("n_sightings") or 1)
                association["sightings"] += sightings
                association["multi_view_tracks"] += int(sightings > 1)

            if kind == "recheck":
                if fixed_ablation:
                    # Motion arms intentionally do not carry T1's utility
                    # metrics. They are judged by schedule/motion audits below.
                    if not all(key in params for key in ("north_m", "east_m", "up_m")):
                        fail(row, index, "fixed_ablation_missing_motion_params")
                    continue
                if not metrics or not metrics.get("allow"):
                    fail(row, index, "recheck_without_positive_utility")
                if float(metrics.get("utility", -math.inf)) < 0.0:
                    fail(row, index, "negative_utility_recheck", str(metrics.get("utility")))
                if qtype == "presence":
                    fail(row, index, "presence_recheck")
                if qtype in {"count", "spatial"}:
                    if abs(float(params.get("north_m", 0.0))) > 1e-6 or abs(float(params.get("east_m", 0.0))) > 1e-6:
                        fail(row, index, "context_task_recenters")
                    if float(metrics.get("predicted_roi_coverage", 0.0)) < min_roi_coverage:
                        fail(row, index, "context_roi_below_minimum")
                if qtype == "damage":
                    if not evidence.get("target_visible"):
                        fail(row, index, "damage_recheck_target_not_visible")
                    target = evidence.get("target_norm_xy")
                    if isinstance(target, list) and len(target) == 2:
                        expected_east = float(target[0]) - 0.5
                        expected_north = 0.5 - float(target[1])
                        if expected_east * float(params.get("east_m", 0.0)) < -1e-6:
                            fail(row, index, "damage_east_direction_wrong")
                        if expected_north * float(params.get("north_m", 0.0)) < -1e-6:
                            fail(row, index, "damage_north_direction_wrong")
            elif kind == "skip" and metrics and not str(step.get("reobserve_reason") or ""):
                fail(row, index, "unexplained_task_policy_stop")

    type_report = {}
    for qtype, stats in sorted(by_type.items()):
        n = stats["questions"]
        type_report[qtype] = {
            **dict(stats),
            "accuracy": stats["correct"] / n if n else 0.0,
            "reobservations_per_question": stats["reobservations"] / n if n else 0.0,
        }
    return {
        "schema": "task-policy-audit/1.0",
        "n_questions": len(rows),
        "n_violations": len(violations),
        "valid_for_analysis": bool(rows) and not violations
            and all(bool(row.get("ok")) for row in rows),
        "by_question_type": type_report,
        "association_diagnostics": dict(association),
        "violations": violations,
    }


def markdown(report: dict) -> str:
    lines = [
        "# P2 task-conditioned policy audit", "",
        f"- Questions: {report['n_questions']}",
        f"- Violations: {report['n_violations']}",
        f"- Valid for analysis: `{str(report['valid_for_analysis']).lower()}`", "",
        "| question type | n | accuracy | reobservations | per question | execution errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for qtype, row in report["by_question_type"].items():
        lines.append(
            f"| {qtype} | {row['questions']} | {row['accuracy']:.4f} | "
            f"{row['reobservations']} | {row['reobservations_per_question']:.3f} | "
            f"{row['execution_errors']} |"
        )
    lines += ["", "## Geographic association diagnostics", ""]
    for key, value in report["association_diagnostics"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Violations", ""]
    if not report["violations"]:
        lines.append("None.")
    else:
        for item in report["violations"]:
            lines.append(
                f"- `{item['code']}` qid={item['qid']} step={item['step']} {item['detail']}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-roi-coverage", type=float, default=0.98)
    args = parser.parse_args()
    rows = load_rows(args.runs)
    report = audit(rows, args.min_roi_coverage)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "audit.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid_for_analysis"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
