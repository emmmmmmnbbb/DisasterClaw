from __future__ import annotations

import pytest

from eval_changeos_fov_ladder import summarize_view, transition


def test_summary_separates_localization_and_matched_classification() -> None:
    rows = [
        {"gt": "damaged", "matched": True, "pred": "damaged", "p_damage": 0.8, "correct": True},
        {"gt": "no-damage", "matched": True, "pred": "damaged", "p_damage": 0.7, "correct": False},
        {"gt": "damaged", "matched": False, "pred": None, "p_damage": None, "correct": False},
    ]
    result = summarize_view(rows, n_predictions=3)
    assert result["n_matched"] == 2
    assert result["n_unmatched_gt"] == 1
    assert result["n_unmatched_predictions"] == 1
    assert result["binary_accuracy_all_gt"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["matched_only"]["accuracy"] == 0.5
    assert result["matched_only"]["brier"] is not None


def test_transition_counts_correction_and_harm() -> None:
    rows = [
        {"views": {"cruise": {"correct": False, "pred": None},
                   "floor": {"correct": True, "pred": "damaged"}}},
        {"views": {"cruise": {"correct": True, "pred": "no-damage"},
                   "floor": {"correct": False, "pred": "damaged"}}},
    ]
    result = transition(rows, "cruise", "floor")
    assert result == {"n": 2, "n_prediction_flip": 2, "n_corrected": 1,
                      "n_harmed": 1, "net_corrected": 0}
