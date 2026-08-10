from __future__ import annotations

import math

import numpy as np

from bench_report import (
    damage_macro_f1,
    holm_adjust,
    paired_wilcoxon,
    table_recheck_significance,
    table_significance,
)
import bench_vln_navigation as navigation_bench
from bench_vln_navigation import noisy_start
from calibration_bench import compute_calibration_metrics


def test_holm_adjust_monotone_and_ordered() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, None])
    assert adjusted[0] == 0.03
    assert adjusted[1] == 0.06
    assert adjusted[2] == 0.06
    assert adjusted[3] is None
    assert paired_wilcoxon([1, 1, 1, 1], [0, 0, 0, 0]) is not None


def test_significance_aggregates_repeats_by_item() -> None:
    rows = []
    for item_id in ("a", "b", "c"):
        for repeat in range(3):
            rows.append({"id": item_id, "config": "B0", "success": item_id != "c", "repeat": repeat})
            rows.append({"id": item_id, "config": "B1", "success": item_id == "a", "repeat": repeat})
    text = "\n".join(table_significance(rows))
    assert "n=3" in text
    assert "Holm p" in text


def test_noisy_start_is_deterministic_and_meter_scaled() -> None:
    start = {"lat": 31.2304, "lon": 121.4737, "alt": 30.0}
    a = noisy_start(start, sigma_m=5.0, seed=7)
    b = noisy_start(start, sigma_m=5.0, seed=7)
    assert a == b
    assert a != start
    # This guards against accidentally treating meters as degrees.
    approx_m = math.hypot(
        (a["lat"] - start["lat"]) * 111_000,
        (a["lon"] - start["lon"]) * 111_000 * math.cos(math.radians(start["lat"])),
    )
    assert 0.0 < approx_m < 30.0


def test_damage_macro_f1_counts_missing_predictions_as_false_negative() -> None:
    rows = [
        {"goal_class": "major", "pred_class": "major"},
        {"goal_class": "major", "pred_class": None},
        {"goal_class": "destroyed", "pred_class": "major"},
        {"goal_class": "destroyed", "pred_class": "destroyed"},
    ]
    assert damage_macro_f1(rows) == 0.583


def test_calibration_metrics_include_risk_coverage() -> None:
    probs = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.6, 0.4, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.3, 0.7, 0.0, 0.0],
    ])
    metrics = compute_calibration_metrics(probs, np.array([0, 0, 0, 1]), n_bins=4)
    curve = metrics["risk_coverage"]
    assert len(curve) == 20
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["selective_accuracy"] == 0.75


def test_no_recheck_policy_still_gets_evidence_stratum(monkeypatch) -> None:
    monkeypatch.setattr(navigation_bench, "semantic_ne_m", lambda *_args: 0.0)
    report = {
        "final_pos": {"lat": 31.0, "lon": 121.0},
        "path_len_m": 10.0,
        "steps_executed": 1,
        "trajectory": [{"labels": {"完全损毁建筑": 1}}],
        "evidence_observations": 1,
        "recheck": None,
    }
    item = {
        "id": "x",
        "tile_id": "tile",
        "goals": [{
            "lat": 31.0,
            "lon": 121.0,
            "class": "完全损毁建筑",
        }],
        "success_radius_m": 25,
        "shortest_path_m": 10,
    }
    row = navigation_bench.eval_episode(report, item)
    assert row["evidence_stratum"] == "evidence"
    assert row["recheck_triggered"] is None


def test_recheck_significance_uses_evidence_rows_and_item_pairs() -> None:
    rows = []
    for item_id in ("a", "b", "c"):
        for repeat in range(2):
            rows.extend([
                {
                    "id": item_id,
                    "config": "E11_ENTROPY",
                    "evidence_stratum": "evidence",
                    "delta_u": 0.2,
                    "judge_ok": True,
                    "repeat": repeat,
                },
                {
                    "id": item_id,
                    "config": "E11_INFOGAIN",
                    "evidence_stratum": "evidence",
                    "delta_u": 0.1,
                    "judge_ok": item_id == "a",
                    "repeat": repeat,
                },
            ])
    text = "\n".join(table_recheck_significance(rows))
    assert "ΔU" in text
    assert "damage correctness" in text
    assert "| 3 |" in text
