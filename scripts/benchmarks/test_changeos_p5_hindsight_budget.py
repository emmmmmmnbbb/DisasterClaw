"""Tests for offline, budget-limited hindsight selection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("eval_changeos_p5_hindsight_budget.py")
SPEC = importlib.util.spec_from_file_location("p5_hindsight_budget", MODULE)
assert SPEC and SPEC.loader
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def test_budget_selects_cheapest_correctable_and_avoids_harm() -> None:
    pairs = [
        {"qid": "cheap", "question_type": "count", "hold_correct": False,
         "always_correct": True, "always_action_cost": 1},
        {"qid": "expensive", "question_type": "count", "hold_correct": False,
         "always_correct": True, "always_action_cost": 2},
        {"qid": "harm", "question_type": "damage", "hold_correct": True,
         "always_correct": False, "always_action_cost": 2},
        {"qid": "same", "question_type": "damage", "hold_correct": True,
         "always_correct": True, "always_action_cost": 2},
    ]
    report = oracle.curve(pairs, (0.0, 1.0 / 7.0, 1.0))
    assert report["baseline_accuracy"] == 0.5
    assert report["n_positive_cost_correctable"] == 2
    assert report["n_positive_cost_harmful"] == 1
    assert report["points"][0]["task_accuracy"] == 0.5
    assert report["points"][1]["selected_qids"] == ["cheap"]
    assert report["points"][2]["task_accuracy"] == 1.0
    assert "harm" not in report["points"][2]["selected_qids"]


def test_zero_cost_answer_difference_is_not_action_headroom() -> None:
    pairs = [{"qid": "free", "question_type": "presence", "hold_correct": False,
              "always_correct": True, "always_action_cost": 0}]
    report = oracle.curve(pairs)
    assert report["n_zero_cost_outcome_differences_excluded"] == 1
    assert report["points"][-1]["task_accuracy"] == 0.0


def test_invalid_run_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "bad"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({
        "valid_for_analysis": False, "n_execution_errors": 1,
    }), encoding="utf-8")
    (run / "episodes.jsonl").write_text("", encoding="utf-8")
    try:
        oracle.load_paired([run])
    except ValueError as exc:
        assert "invalid run" in str(exc)
    else:
        raise AssertionError("invalid run was accepted")


def test_policy_out_of_coverage_is_scored_but_not_called_execution_crash(tmp_path: Path) -> None:
    run = tmp_path / "valid"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({
        "valid_for_analysis": True, "n_execution_errors": 0,
        "testset_sha256_16": "test", "env": {"source_fingerprint": "source"},
    }), encoding="utf-8")
    rows = [
        {"config": "A0_HOLD", "qid": "q", "ok": False,
         "reason_code": "out_of_coverage", "correct": False,
         "gt_answer": "是", "question_type": "presence",
         "disaster": "event", "tile_id": "tile"},
        {"config": "A2_ALWAYS", "qid": "q", "ok": True,
         "correct": True, "n_reobservations": 2,
         "gt_answer": "是", "question_type": "presence",
         "disaster": "event", "tile_id": "tile"},
    ]
    (run / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    pairs, _ = oracle.load_paired([run])
    assert len(pairs) == 1
    assert not pairs[0]["hold_correct"]
    assert pairs[0]["always_correct"]
