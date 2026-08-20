"""Integration tests for the app-level Agent-VQA dependency wiring."""
from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
from agent_vqa import parse_question
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


def test_damage_marker_is_visible_at_image_center() -> None:
    image = Image.new("RGB", (128, 128), "black")
    raw = BytesIO()
    image.save(raw, format="JPEG")
    marked = Image.open(BytesIO(app._mark_agent_vqa_target(raw.getvalue()))).convert("RGB")
    center = marked.getpixel((64, 64))
    assert max(center) > 100
    assert center != (0, 0, 0)
