"""Tests for state-derived total-motion accounting."""
from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).with_name("audit_changeos_p7_costs.py")
SPEC = importlib.util.spec_from_file_location("p7_costs", MODULE)
assert SPEC and SPEC.loader
costs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(costs)


def test_total_motion_includes_search_and_reobserve() -> None:
    start = {"lat": 0.0, "lon": 0.0, "alt": 100.0}
    row = {"qid": "q", "config": "A2_ALWAYS", "n_reobservations": 1,
           "trajectory": [
               {"action": "fly_relative", "position": start,
                "reobserve_kind": "", "budget_before": 3, "budget_after": 2},
               {"action": "fly_relative",
                "position": {"lat": 0.001, "lon": 0.0, "alt": 100.0},
                "reobserve_kind": "recheck",
                "reobserve_params": {"up_m": -10.0},
                "reobserve_executed": {"up_m": -10.0},
                "budget_before": 2, "budget_after": 1},
               {"action": "report_observation",
                "position": {"lat": 0.001, "lon": 0.0, "alt": 90.0}},
           ]}
    audit = costs.audit_row(row, start)
    assert audit["trajectory_motion_complete"]
    assert abs(audit["search_horizontal_m"] - 110.54) < 0.01
    assert audit["reobserve_vertical_m"] == 10.0
    assert audit["total_vertical_m"] == 10.0
    assert audit["requested_rechecks_logged"] == 1
    assert audit["allocated_rechecks_inferred_from_budget"] == 1
    assert audit["executed_rechecks_logged"] == 1


def test_final_fly_without_following_state_is_flagged_incomplete() -> None:
    start = {"lat": 0.0, "lon": 0.0, "alt": 100.0}
    row = {"qid": "q", "config": "A0_HOLD", "trajectory": [
        {"action": "fly_relative", "position": start},
    ]}
    audit = costs.audit_row(row, start)
    assert not audit["trajectory_motion_complete"]
