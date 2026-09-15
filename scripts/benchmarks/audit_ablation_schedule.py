#!/usr/bin/env python3
"""Verify paired action-ablation schedules and generation-call identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[(str(row.get("config")), str(row.get("qid")))] = row
    return out


def calls(row: dict) -> list[tuple]:
    return [
        (step.get("generation_seed"), step.get("generation_call_role"))
        for step in row.get("trajectory") or []
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", default="T1_TASK")
    parser.add_argument("--configs", default="AB_NOOP,AB_CENTER,AB_DESCEND,AB_FULL,AB_WIDE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = rows(args.run)
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    qids = sorted({qid for cfg, qid in data if cfg == args.reference})
    action_violations = []
    generation_mismatches = []
    records = []
    for qid in qids:
        ref = data.get((args.reference, qid))
        if ref is None:
            continue
        ref_calls = calls(ref)
        ref_n = int(ref.get("n_reobservations") or 0)
        for cfg in configs:
            row = data.get((cfg, qid))
            if row is None:
                action_violations.append({"qid": qid, "config": cfg, "code": "missing_row"})
                continue
            c = calls(row)
            if int(row.get("n_reobservations") or 0) != ref_n:
                action_violations.append({"qid": qid, "config": cfg, "code": "reobserve_count_mismatch",
                                          "reference": ref_n, "actual": row.get("n_reobservations")})
            if c != ref_calls:
                generation_mismatches.append({
                    "qid": qid, "config": cfg, "code": "generation_call_key_mismatch",
                })
            records.append({"qid": qid, "config": cfg, "reference_reobservations": ref_n,
                            "actual_reobservations": row.get("n_reobservations"),
                            "generation_calls_equal": c == ref_calls})
    report = {
        "schema": "agent-vqa-ablation-schedule-audit/1.1",
        "reference": args.reference, "configs": configs, "n_questions": len(qids),
        "n_records": len(records),
        "n_action_violations": len(action_violations),
        "action_schedule_valid": bool(qids) and not action_violations,
        "n_generation_call_mismatches": len(generation_mismatches),
        "generation_call_schedule_identical": not generation_mismatches,
        # End-to-end policy analysis requires the paired action budget. Extra
        # answer retries remain disclosed separately as a compute-cost outcome.
        "valid_for_analysis": bool(qids) and not action_violations,
        "records": records,
        "action_violations": action_violations,
        "generation_call_mismatches": generation_mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "n_questions", "n_records", "n_action_violations", "action_schedule_valid",
        "n_generation_call_mismatches", "generation_call_schedule_identical",
        "valid_for_analysis",
    )}, ensure_ascii=False, indent=2))
    return 0 if report["valid_for_analysis"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
