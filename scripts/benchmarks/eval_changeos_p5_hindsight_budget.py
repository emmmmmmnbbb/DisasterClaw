#!/usr/bin/env python3
"""Offline, budget-limited hindsight outcome selection for paired A0/A2 runs.

This is a GT-informed diagnostic upper bound over two *observed outcomes*, not
an executable policy and not a causal estimate of the effect of an action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

DEFAULT_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_paired(runs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows: dict[tuple[str, str], dict] = {}
    inputs = []
    identity: tuple[str, str] | None = None
    for run in runs:
        result_path = run / "results.json"
        episodes_path = run / "episodes.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("valid_for_analysis") is not True or result.get("n_execution_errors") != 0:
            raise ValueError(f"invalid run, refusing hindsight budget analysis: {run}")
        current = (str(result.get("testset_sha256_16") or ""),
                   str((result.get("env") or {}).get("source_fingerprint") or ""))
        if not all(current):
            raise ValueError(f"missing testset or executable fingerprint: {run}")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(f"run identity differs: {run}")
        inputs.append({"run": str(run), "results_sha256": sha256(result_path),
                       "episodes_sha256": sha256(episodes_path)})
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            config = str(row.get("config") or "")
            if config not in {"A0_HOLD", "A2_ALWAYS"}:
                continue
            qid = str(row.get("qid") or "")
            key = (config, qid)
            if not qid or key in rows:
                raise ValueError(f"missing or duplicate paired outcome key: {key}")
            if row.get("ok") is not True and row.get("reason_code") != "out_of_coverage":
                raise ValueError(f"execution failure in paired outcome: {key}")
            rows[key] = row
    hold_qids = {qid for config, qid in rows if config == "A0_HOLD"}
    always_qids = {qid for config, qid in rows if config == "A2_ALWAYS"}
    if not hold_qids or hold_qids != always_qids:
        raise ValueError("A0_HOLD and A2_ALWAYS must have exactly the same qids")
    pairs = []
    for qid in sorted(hold_qids):
        hold = rows[("A0_HOLD", qid)]
        always = rows[("A2_ALWAYS", qid)]
        for field in ("gt_answer", "question_type", "disaster", "tile_id"):
            if hold.get(field) != always.get(field):
                raise ValueError(f"paired field {field} differs for {qid}")
        cost = int(always.get("n_reobservations") or 0)
        if cost < 0:
            raise ValueError(f"negative action cost for {qid}")
        pairs.append({
            "qid": qid, "question_type": hold.get("question_type"),
            "event": hold.get("disaster"), "tile_id": hold.get("tile_id"),
            "hold_correct": bool(hold.get("correct")),
            "always_correct": bool(always.get("correct")),
            "always_action_cost": cost,
        })
    return pairs, inputs


def curve(pairs: list[dict], fractions: tuple[float, ...] = DEFAULT_FRACTIONS) -> dict:
    if not pairs:
        raise ValueError("no paired outcomes")
    if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError("budget fractions must lie in [0, 1]")
    n = len(pairs)
    total_always_actions = sum(pair["always_action_cost"] for pair in pairs)
    baseline_correct = sum(pair["hold_correct"] for pair in pairs)
    correctable = [pair for pair in pairs
                   if not pair["hold_correct"] and pair["always_correct"]
                   and pair["always_action_cost"] > 0]
    correctable.sort(key=lambda pair: (pair["always_action_cost"], pair["qid"]))
    harmful = [pair for pair in pairs
               if pair["hold_correct"] and not pair["always_correct"]
               and pair["always_action_cost"] > 0]
    zero_cost_differences = [pair for pair in pairs
                             if pair["always_action_cost"] == 0
                             and pair["hold_correct"] != pair["always_correct"]]
    by_type = Counter(pair["question_type"] for pair in pairs)
    baseline_by_type = Counter(pair["question_type"] for pair in pairs if pair["hold_correct"])
    points = []
    for fraction in fractions:
        budget = math.floor(fraction * total_always_actions)
        selected = []
        spent = 0
        for pair in correctable:
            cost = pair["always_action_cost"]
            if spent + cost > budget:
                continue
            selected.append(pair)
            spent += cost
        selected_by_type = Counter(pair["question_type"] for pair in selected)
        points.append({
            "budget_fraction_of_always_actions": fraction,
            "action_budget": budget, "actions_spent": spent,
            "selected_corrections": len(selected),
            "task_accuracy": (baseline_correct + len(selected)) / n,
            "by_question_type": {
                qtype: {"n": by_type[qtype],
                        "accuracy": (baseline_by_type[qtype] + selected_by_type[qtype])
                        / by_type[qtype]}
                for qtype in sorted(by_type)
            },
            "selected_qids": [pair["qid"] for pair in selected],
        })
    return {
        "schema": "changeos-p5-hindsight-budget/1.0",
        "definition": "Offline GT-informed selection between observed A0_HOLD and A2_ALWAYS outcomes; A2 costs its executed reobservations; only positive-cost corrections selectable",
        "causal_policy_claim": False, "deployable": False,
        "n_paired_questions": n, "total_always_actions": total_always_actions,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / n,
        "n_positive_cost_correctable": len(correctable),
        "n_positive_cost_harmful": len(harmful),
        "n_zero_cost_outcome_differences_excluded": len(zero_cost_differences),
        "points": points,
        "paired_cases": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fractions", default=",".join(str(x) for x in DEFAULT_FRACTIONS))
    args = parser.parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(",") if value.strip())
    pairs, inputs = load_paired(args.runs)
    report = curve(pairs, fractions)
    report["inputs"] = inputs
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "n_paired_questions", "total_always_actions", "baseline_accuracy",
        "n_positive_cost_correctable", "n_positive_cost_harmful",
        "n_zero_cost_outcome_differences_excluded",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
