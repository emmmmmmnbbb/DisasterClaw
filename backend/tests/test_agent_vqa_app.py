"""Integration tests for the app-level Agent-VQA dependency wiring."""
from __future__ import annotations

from io import BytesIO
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
from agent_vqa import EvidenceBundle, parse_question
from vln_navigator import Decision, Observation


class _Perception:
    patch_width = 100
    patch_height = 100
    patch_radius_m = 60.0
    patch_path = ""
    patch_id = "obs"
    risk_level = "high"
    degraded = False
    scene_text = ""
    detection = {"detections": []}


def test_hspm_state_is_reused_and_receives_real_observation(monkeypatch) -> None:
    class Nav:
        def __init__(self):
            self.resets = 0
            self.steps = []

        def reset(self, question):
            self.resets += 1

        def step(self, observation, snapshot):
            self.steps.append((observation, snapshot))
            return Decision(action="fly_relative", params={"north_m": 1.0, "east_m": 2.0})

    nav = Nav()
    monkeypatch.setattr(app, "_make_hspm_navigator", lambda: nav)
    monkeypatch.setattr(app, "execute_action", lambda action, params, source="": {"success": True})
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())

    ctl = app._make_agent_vqa_controller("test")
    spec = parse_question("视场中心十字标记建筑的损伤等级是什么？")
    ctl._search(spec, 0, _Perception())
    ctl._search(spec, 1, _Perception())

    assert nav.resets == 1
    assert len(nav.steps) == 2
    assert all(isinstance(obs, Observation) for obs, _ in nav.steps)


def test_recheck_factory_receives_all_policy_switches(monkeypatch) -> None:
    captured = {}

    class Rechecker:
        def __init__(self, config):
            captured.update(vars(config))

        def assess(self, **kwargs):
            return type("Outcome", (), {"kind": "skip", "params": None, "reason": "test"})()

    monkeypatch.setattr(app, "RecheckController", Rechecker)
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())
    monkeypatch.setattr(app, "VLN_RECHECK_TRIGGER", "conformal")
    monkeypatch.setattr(app, "VLN_UNCERTAINTY_MODE", "entropy")
    monkeypatch.setattr(app, "VLN_RECHECK_RANDOM_PROB", 0.37)
    monkeypatch.setattr(app, "VLN_RECHECK_RANDOM_SEED", 13)
    monkeypatch.setattr(app, "VLN_RECHECK_MIN_INFO_GAIN", 0.12)
    monkeypatch.setattr(app, "VLN_ENTROPY_TABLE", "entropy.json")
    monkeypatch.setattr(app, "VLN_CONFORMAL_QHAT", 0.73)
    monkeypatch.setattr(app, "VLN_CONFORMAL_ALPHA", 0.2)

    app._make_agent_vqa_controller("test")
    assert captured["trigger_mode"] == "conformal"
    assert captured["uncertainty_mode"] == "entropy"
    assert captured["random_prob"] == 0.37
    assert captured["random_seed"] == 13
    assert captured["min_info_gain"] == 0.12
    assert captured["entropy_table_path"] == "entropy.json"
    assert captured["conformal_qhat"] == 0.73
    assert captured["conformal_alpha"] == 0.2


def test_damage_marker_uses_projected_target_not_image_center() -> None:
    image = Image.new("RGB", (128, 128), "black")
    raw = BytesIO()
    image.save(raw, format="JPEG")
    marked = Image.open(BytesIO(app._mark_agent_vqa_target(
        raw.getvalue(), [0.25, 0.75], [0.1, 0.1, 0.9, 0.9]
    ))).convert("RGB")
    target = marked.getpixel((32, 96))
    center = marked.getpixel((64, 64))
    assert max(target) > 100
    assert max(center) < 40


def test_damage_evidence_reaches_the_vlm_prompt_and_marked_image(monkeypatch) -> None:
    """损坏目标坐标必须同时进提示词载荷和标记图，两者都不得退回图像中心。"""
    captured = {}

    class _Analyzer:
        def answer_image_question(self, image_bytes, question, choices, evidence_text,
                                  max_tokens, generation_seed=None):
            captured["image"] = image_bytes
            captured["evidence_text"] = evidence_text
            captured["generation_seed"] = generation_seed
            return {"raw": '{"answer": "无损伤", "confidence": 0.5, '
                           '"decision": "answer", "reason_code": "sufficient_evidence", '
                           '"abstain": false, "evidence": {"source": "image"}}'}

    monkeypatch.setattr(app, "VLMAnalyzer", _Analyzer)
    monkeypatch.setattr(app, "AGENT_VQA_EVIDENCE_LEVEL", "struct")
    monkeypatch.setattr(app.state, "semantic_map", None, raising=False)

    ctl = app._make_agent_vqa_controller("test")
    image = Image.new("RGB", (128, 128), "black")
    raw = BytesIO()
    image.save(raw, format="JPEG")

    spec = parse_question("标记区域内标记建筑 b-5 的损伤等级是什么？")
    ev = EvidenceBundle(
        observation_id="obs0", source="detector", target_label="轻微损伤建筑",
        target_subtype="minor-damage", target_conf=0.8, matching_count=1,
        target_norm_xy=[0.25, 0.75], roi_norm_bbox=[0.1, 0.1, 0.9, 0.9],
        target_ref_id="b-5", target_visible=True, target_matched=True,
        match_method="target_point_in_bbox",
    )
    from agent_vqa import GenerationContext
    context = GenerationContext("q", 0, 0, "candidate_answer", 123)
    ctl._vlm(raw.getvalue(), _Perception(), spec, "q", ev, context)

    payload = json.loads(captured["evidence_text"])
    assert payload["target_evidence"]["question_target_norm_xy"] == [0.25, 0.75]
    assert payload["target_evidence"]["predicted_target_subtype"] == "minor-damage"
    assert payload["target_evidence"]["target_matched"] is True
    assert payload["target_evidence"]["match_method"] == "target_point_in_bbox"
    assert captured["generation_seed"] == 123

    marked = Image.open(BytesIO(captured["image"])).convert("RGB")
    assert max(marked.getpixel((32, 96))) > 100      # crosshair on the target
    assert max(marked.getpixel((64, 64))) < 40       # centre untouched


def test_fixed_policy_forces_center_descent_when_rechecker_skips(monkeypatch) -> None:
    class Rechecker:
        def __init__(self, config):
            pass

        def assess(self, **kwargs):
            return type("Outcome", (), {
                "kind": "skip", "params": None, "reason": "no evidence",
                "uncertainty": 0.0, "label": "",
            })()

    actions = []
    monkeypatch.setattr(app, "RecheckController", Rechecker)
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())
    monkeypatch.setattr(app, "VLN_RECHECK_TRIGGER", "fixed")
    monkeypatch.setattr(app, "VLN_RECHECK_DESCEND_M", 10.0)
    monkeypatch.setattr(app, "VLN_RECHECK_ALT_MIN_M", 10.0)
    monkeypatch.setattr(
        app.state.adapter, "snapshot",
        lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    monkeypatch.setattr(
        app, "execute_action",
        lambda action, params, source="": actions.append((action, params)) or {"success": True},
    )

    ctl = app._make_agent_vqa_controller("test")
    out = ctl._reobserve(
        _Perception(), parse_question("当前视场是否存在完全损毁建筑？"),
        EvidenceBundle(observation_id="obs"),
    )

    assert out["kind"] == "recheck"
    assert out["params"] == {
        "north_m": 0.0, "east_m": 0.0, "up_m": -10.0, "speed": 10.0,
    }
    assert actions == [("fly_relative", out["params"])]


def test_fixed_ablation_uses_public_marker_and_labels_zero_motion(monkeypatch) -> None:
    class Rechecker:
        def __init__(self, config):
            pass

        def assess(self, **kwargs):
            return type("Outcome", (), {
                "kind": "skip", "params": None, "reason": "fixed",
                "uncertainty": 0.0, "label": "",
            })()

    actions = []
    monkeypatch.setattr(app, "RecheckController", Rechecker)
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())
    monkeypatch.setattr(app, "VLN_RECHECK_TRIGGER", "fixed")
    monkeypatch.setattr(app, "VLN_RECHECK_DESCEND_M", 10.0)
    monkeypatch.setattr(app, "VLN_RECHECK_ALT_MIN_M", 10.0)
    monkeypatch.setattr(app.state.adapter, "snapshot",
                        lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0})
    monkeypatch.setattr(
        app, "execute_action",
        lambda action, params, source="": actions.append(dict(params)) or {"success": True},
    )
    evidence = EvidenceBundle(
        observation_id="obs", target_norm_xy=[0.7, 0.45],
        roi_norm_bbox=[0.0, 0.0, 1.0, 1.0],
    )

    monkeypatch.setattr(app, "VLN_RECHECK_MOTION_MODE", "no_op")
    noop = app._make_agent_vqa_controller("test")._reobserve(
        _Perception(), parse_question("标记建筑 b-1 的损伤状态是什么？"), evidence,
    )
    assert noop["params"] == {
        "north_m": 0.0, "east_m": 0.0, "up_m": 0.0, "speed": 10.0,
    }
    assert noop["zero_motion"] is True and actions == []

    monkeypatch.setattr(app, "VLN_RECHECK_MOTION_MODE", "center_only")
    center = app._make_agent_vqa_controller("test")._reobserve(
        _Perception(), parse_question("标记建筑 b-1 的损伤状态是什么？"), evidence,
    )
    assert center["params"]["north_m"] == 6.0
    assert center["params"]["east_m"] == 24.0
    assert center["params"]["up_m"] == 0.0
    assert actions[-1] == center["params"]

    monkeypatch.setattr(app, "VLN_RECHECK_MOTION_MODE", "wide_roi")
    wide = app._make_agent_vqa_controller("test")._reobserve(
        _Perception(), parse_question("标记区域内有多少栋受损建筑？"), evidence,
    )
    assert wide["params"]["up_m"] == 0.0
    assert wide["zero_motion"] is True


def test_task_policy_keeps_binary_negative_and_descends_without_recentering(monkeypatch) -> None:
    captured = {}

    class Rechecker:
        def __init__(self, config):
            pass

        entropy_table_loaded = False

        def assess(self, **kwargs):
            captured.update(kwargs)
            return type("Outcome", (), {
                "kind": "recheck",
                "params": {"north_m": 9.0, "east_m": -7.0, "up_m": -10.0, "speed": 10.0},
                "reason": "uncertain", "uncertainty": 1.0,
                "label": "无损伤建筑",
            })()

    perception = _Perception()
    perception.risk_level = "none"
    perception.detection = {"detections": [{
        "class_name": "无损伤建筑", "conf": 0.51,
        "bbox": [10, 10, 30, 30],
        "class_probs": {"no-damage": 0.51, "damaged": 0.49},
    }]}
    actions = []
    monkeypatch.setattr(app, "RecheckController", Rechecker)
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())
    monkeypatch.setattr(app, "VLN_RECHECK_TRIGGER", "task_conditioned")
    monkeypatch.setattr(app, "VLN_UNCERTAINTY_MODE", "entropy")
    monkeypatch.setattr(app, "VLN_RECHECK_DESCEND_M", 10.0)
    monkeypatch.setattr(app, "VLN_RECHECK_ALT_MIN_M", 10.0)
    monkeypatch.setattr(app.state.adapter, "snapshot",
                        lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0})
    monkeypatch.setattr(
        app, "execute_action",
        lambda action, params, source="": actions.append((action, dict(params))) or {"success": True},
    )

    ctl = app._make_agent_vqa_controller("test")
    out = ctl._reobserve(
        perception, parse_question("当前视场有多少栋受损建筑？"),
        EvidenceBundle(
            observation_id="obs", roi_norm_bbox=[1 / 3, 1 / 3, 2 / 3, 2 / 3],
        ),
    )

    assert captured["detections"] == perception.detection["detections"]
    assert out["kind"] == "recheck"
    assert out["motion_mode"] == "descend_only"
    assert out["params"]["north_m"] == out["params"]["east_m"] == 0.0
    assert out["policy_metrics"]["predicted_roi_coverage"] == 1.0
    assert actions[0][1] == out["params"]


def test_task_policy_never_zooms_presence_answer(monkeypatch) -> None:
    class Rechecker:
        def __init__(self, config):
            pass

        def assess(self, **kwargs):
            raise AssertionError("presence task should be rejected by task gate first")

    monkeypatch.setattr(app, "RecheckController", Rechecker)
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())
    monkeypatch.setattr(app, "VLN_RECHECK_TRIGGER", "task_conditioned")
    monkeypatch.setattr(app, "VLN_RECHECK_DESCEND_M", 10.0)
    monkeypatch.setattr(app, "VLN_RECHECK_ALT_MIN_M", 10.0)
    monkeypatch.setattr(app.state.adapter, "snapshot",
                        lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0})

    ctl = app._make_agent_vqa_controller("test")
    out = ctl._reobserve(
        _Perception(), parse_question("当前视场是否存在完全损毁建筑？"),
        EvidenceBundle(observation_id="obs"),
    )
    assert out["kind"] == "skip"
    assert out["motion_mode"] == "hold"
    assert "wide view" in out["reason"]


def test_task_policy_damage_uses_target_entropy_and_center_motion(monkeypatch) -> None:
    captured = {}

    class Rechecker:
        def __init__(self, config):
            pass

        entropy_table_loaded = False

        def assess(self, **kwargs):
            captured.update(kwargs)
            return type("Outcome", (), {
                "kind": "recheck",
                "params": {"north_m": -5.0, "east_m": 8.0, "up_m": -10.0, "speed": 10.0},
                "reason": "uncertain target", "uncertainty": 1.0,
                "label": "无损伤建筑",
            })()

    perception = _Perception()
    perception.risk_level = "moderate"
    perception.detection = {"detections": [{
        "class_name": "无损伤建筑", "conf": 0.51,
        "bbox": [65, 40, 75, 50],
        "class_probs": {"no-damage": 0.51, "damaged": 0.49},
    }]}
    monkeypatch.setattr(app, "RecheckController", Rechecker)
    monkeypatch.setattr(app, "VLMAnalyzer", lambda: object())
    monkeypatch.setattr(app, "VLN_RECHECK_TRIGGER", "task_conditioned")
    monkeypatch.setattr(app, "VLN_UNCERTAINTY_MODE", "entropy")
    monkeypatch.setattr(app, "VLN_RECHECK_DESCEND_M", 10.0)
    monkeypatch.setattr(app, "VLN_RECHECK_ALT_MIN_M", 10.0)
    monkeypatch.setattr(app.state.adapter, "snapshot",
                        lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0})
    monkeypatch.setattr(app, "execute_action", lambda *args, **kwargs: {"success": True})

    evidence = EvidenceBundle(
        observation_id="obs", target_bbox=[65, 40, 75, 50],
        norm_xy=[0.7, 0.45], target_norm_xy=[0.7, 0.45],
        target_visible=True, target_matched=True,
        target_conf=0.51, class_probs={"no-damage": 0.51, "damaged": 0.49},
    )
    ctl = app._make_agent_vqa_controller("test")
    out = ctl._reobserve(
        perception, parse_question("标记建筑 b-1 的损伤状态是什么？"), evidence,
    )
    assert out["kind"] == "recheck"
    assert out["motion_mode"] == "descend_center"
    assert out["params"]["north_m"] == 6.0 and out["params"]["east_m"] == 24.0
    assert out["policy_metrics"]["uncertainty_source"] == "damage_target_entropy"
    assert captured["detections"] == perception.detection["detections"]
