from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("fit_language_baselines.py")
SPEC = importlib.util.spec_from_file_location("fit_language_baselines", PATH)
baseline = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(baseline)


def test_question_only_removes_uuid_and_ignores_hidden_fields() -> None:
    a = {
        "id": "a", "question_type": "damage",
        "question": "标记建筑 11111111-1111-1111-1111-111111111111 的损伤状态是什么？",
        "answer": "损伤", "target": {"subtype": "destroyed", "lat": 1, "lon": 2},
    }
    b = {
        "id": "b", "question_type": "damage",
        "question": "标记建筑 22222222-2222-2222-2222-222222222222 的损伤状态是什么？",
        "answer": "无损伤", "target": {"subtype": "no-damage", "lat": 9, "lon": 8},
    }
    assert baseline.normalize_question(a["question"]) == baseline.normalize_question(b["question"])
    model = baseline.fit([a, a, b])
    evaluated = baseline.evaluate([b], model)
    assert evaluated["question_only"]["accuracy"] == 0.0
    assert evaluated["rows"][0]["question_only_answer"] == "损伤"


def test_unseen_template_falls_back_to_fitted_type_majority() -> None:
    fit_item = {"id": "a", "question_type": "presence", "question": "是否有受损建筑？", "answer": "是"}
    eval_item = {"id": "b", "question_type": "presence", "question": "能看到受损建筑吗？", "answer": "是"}
    result = baseline.evaluate([eval_item], baseline.fit([fit_item]))
    assert result["question_only_fallbacks"] == 1
    assert result["question_only"]["accuracy"] == 1.0
