"""Unit tests for the offline P5 GT-structure taskset self-check."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).with_name("eval_changeos_p5_gt_rule.py")
SPEC = importlib.util.spec_from_file_location("p5_gt_rule", MODULE)
assert SPEC and SPEC.loader
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def make_item(qtype: str, answer: str, target=None) -> dict:
    choices = {
        "presence": ["否", "是"], "damage": ["无损伤", "损伤"],
        "count": ["0", "1", "2", "3+"], "spatial": list(oracle.BEARINGS),
    }
    return {
        "id": f"q_{qtype}", "tile_id": "tile", "question_type": qtype,
        "answer": answer, "choices": choices[qtype], "target": target,
        "roi": {"center": {"lat": 0.0, "lon": 0.0},
                "bounds": {"south": -0.01, "north": 0.01,
                           "west": -0.01, "east": 0.01}},
    }


def test_rule_answers_four_binary_question_types() -> None:
    buildings = [
        {"uid": "a", "subtype": "minor-damage", "lat": 0.001, "lon": 0.0},
        {"uid": "b", "subtype": "no-damage", "lat": 0.0, "lon": 0.001},
    ]
    cases = [
        make_item("presence", "是"), make_item("count", "1"),
        make_item("damage", "无损伤", {"ref_id": "b", "subtype": "no-damage",
                                            "lat": 0.0, "lon": 0.001}),
        make_item("spatial", "北", {"lat": 0.001, "lon": 0.0}),
    ]
    for item in cases:
        answer, issues = oracle.answer_from_gt(item, buildings)
        assert answer == item["answer"]
        assert not issues


def test_missing_or_conflicting_target_is_not_silently_scored() -> None:
    buildings = [{"uid": "a", "subtype": "destroyed", "lat": 0.001, "lon": 0.0}]
    missing = make_item("damage", "损伤", {"ref_id": "missing"})
    answer, issues = oracle.answer_from_gt(missing, buildings)
    assert answer is None and issues == ["target_uid_matches=0"]
    conflict = make_item("damage", "损伤", {"ref_id": "a", "subtype": "no-damage"})
    answer, issues = oracle.answer_from_gt(conflict, buildings)
    assert answer == "损伤"
    assert "target_subtype_mismatch" in issues
    assert "target_coordinate_missing" in issues


def test_evaluate_flags_wrong_taskset_answer(tmp_path: Path) -> None:
    label = tmp_path / "tile.json"
    label.write_text(json.dumps({"features": {"lng_lat": [{
        "wkt": "POLYGON ((0 0.001, 0.0001 0.001, 0.0001 0.0011, 0 0.0011, 0 0.001))",
        "properties": {"uid": "a", "subtype": "destroyed"},
    }]}}), encoding="utf-8")
    taskset = {"items": [make_item("presence", "否")]}
    manifest = {"items": [{"tile_id": "tile", "stage": "post",
                           "label_relpath": "tile.json"}]}
    report = oracle.evaluate(taskset, manifest, tmp_path)
    assert report["n_questions"] == 1
    assert report["n_answer_matches"] == 0
    assert not report["valid_for_taskset_selfcheck"]
