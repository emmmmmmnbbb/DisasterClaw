from __future__ import annotations

import importlib.util
from pathlib import Path


def load(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shortcut = load("eval_revision_shortcuts.py")
matched = load("eval_revision_matched_view.py")
report = load("report_revision_controls.py")


def test_shortcut_input_whitelist_excludes_scoring_fields() -> None:
    item = {
        "id": "q", "question_type": "damage",
        "question": "标记区域内标记建筑 abc-def 的损伤状态是什么？",
        "answer": "损伤", "choices": ["无损伤", "损伤"],
        "roi": {"tile_id": "tile", "bounds": {"west": 1, "south": 2,
                                             "east": 3, "north": 4}},
        "start": {"lat": 2.5, "lon": 1.5, "alt": 100},
        "target": {"ref_id": "abc-def", "lat": 2.6, "lon": 1.6,
                   "subtype": "destroyed"},
        "gt_polygon": [1, 2, 3], "detector_output": {"damaged": True},
    }
    language = shortcut.allowed_payload(item, "S1_LANGUAGE")
    metadata = shortcut.allowed_payload(item, "S2_METADATA")
    assert set(language) == {"question", "choices", "question_type"}
    assert metadata["marked_target"] == {"ref_id": "abc-def", "lat": 2.6, "lon": 1.6}
    assert "损伤" not in str(metadata["marked_target"])
    assert "destroyed" not in str(metadata)
    assert "gt_polygon" not in str(metadata)
    assert "detector_output" not in str(metadata)


def test_shortcut_spatial_excludes_true_target_coordinate() -> None:
    item = {
        "id": "s", "question_type": "spatial",
        "question": "标记区域内距离中心最近的受损建筑位于哪个方向？",
        "roi": {"tile_id": "tile", "bounds": {"west": 1, "south": 2,
                                             "east": 3, "north": 4}},
        "start": {"lat": 2.5, "lon": 1.5, "alt": 100},
        "target": {"lat": 99.123456, "lon": 88.123456, "subtype": "damaged"},
    }
    payload = shortcut.allowed_payload(item, "S2_METADATA")
    assert "marked_target" not in payload
    assert "99.123456" not in str(payload)
    assert "88.123456" not in str(payload)


def test_shortcut_invalid_output_is_not_a_correct_answer() -> None:
    assert shortcut.parse_answer('{"answer":"损伤","abstain":false}', ["损伤"])
    assert shortcut.parse_answer('{"answer":"损伤","abstain":"false"}', ["损伤"])[2] == "invalid_abstain"
    assert shortcut.parse_answer('{"answer":"销毁","abstain":false}', ["损伤"])[2] == "out_of_vocabulary"


def test_paired_report_cells() -> None:
    sample = [
        {"roi_id": "r1", "question_type": "damage", "fixed_correct": False,
         "changed_correct": True, "fixed_answer": "无损伤", "changed_answer": "损伤",
         "changed_observation_id": "v2", "matched_seed": True,
         "matched_template": True, "same_fixed_image_as_pre": True},
        {"roi_id": "r2", "question_type": "damage", "fixed_correct": True,
         "changed_correct": False, "fixed_answer": "损伤", "changed_answer": "无损伤",
         "changed_observation_id": "v4", "matched_seed": True,
         "matched_template": True, "same_fixed_image_as_pre": True},
    ]
    summary = report.matched_summary([{"ok": True, "forks": sample}], 2)
    assert summary["paired_delta"] == 0
    assert summary["four_cells"]["help_wrong_correct"] == 1
    assert summary["four_cells"]["harm_correct_wrong"] == 1


def test_semantic_map_fork_does_not_share_mutable_layers_or_lock() -> None:
    from semantic_map import SemanticMap

    live = SemanticMap(30.0, -85.0, cell_size_m=5.0)
    live.mark_observation(30.0, -85.0, 10.0)
    fixed = matched.clone_semantic_map(live)
    assert fixed is not live
    assert fixed.snapshot() == live.snapshot()
    fixed.mark_observation(30.0002, -85.0, 10.0)
    assert fixed.step_count == live.step_count + 1
    assert fixed._explored is not live._explored


def test_shortcut_gap_is_pairwise_and_clustered() -> None:
    reference = {("T1_TASK", 0, "a"): {"correct": True},
                 ("T1_TASK", 0, "b"): {"correct": False}}
    predictions = [
        {"qid": "a", "repeat": 0, "arm": "S1_LANGUAGE", "correct": False,
         "roi_id": "r1"},
        {"qid": "b", "repeat": 0, "arm": "S1_LANGUAGE", "correct": False,
         "roi_id": "r2"},
    ]
    result = report.paired_gap(reference, predictions, "T1_TASK", "S1_LANGUAGE", draws=100)
    assert result["paired_accuracy_gap"] == 0.5
    assert result["n"] == 2
