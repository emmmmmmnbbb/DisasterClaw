"""backend/tests/test_agent_vqa_schema.py — Agent-VQA schema 校验测试 (D3, 计划 7.8).

验收点:
  1. 四种问题类型解析正确 (含中英文问句)。
  2. 合法 JSON 回答解析成功。
  3. 非法答案、非法 confidence、缺字段被拒绝。
  4. evidence.norm_xy 仅在定位到目标时填写; 非法 norm_xy 被拒绝。
  5. 非法 decision / reason_code / evidence_source 被拒绝。
  6. VLM 输出解析失败返回显式 invalid_output, 不静默猜测。
  7. 封闭集合常量完备。

运行: python backend/tests/test_agent_vqa_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_vqa import (  # noqa: E402
    BEARING_CHOICES, COUNT_CHOICES, DAMAGE_CHOICES, DECISIONS, EVIDENCE_SOURCES,
    PRESENCE_CHOICES, QUESTION_TYPES, REASON_CODES, SEVERE_SUBTYPES,
    VqaAnswer, build_evidence_from_perception, parse_question, parse_vlm_json_output,
    validate_answer_dict,
)
from vlm_analyzer import AGENT_VQA_SYSTEM_PROMPT  # noqa: E402


# ── 问题解析 ──────────────────────────────────────────────────────────────────

def test_parse_four_question_types() -> None:
    assert parse_question("当前视场是否存在完全损毁建筑？").question_type == "presence"
    assert parse_question("标记建筑 abc-1 的损伤等级是什么？").question_type == "damage"
    assert parse_question("当前视场有多少栋严重或完全损毁建筑？").question_type == "count"
    assert parse_question("最近的完全损毁建筑位于无人机哪个方向？").question_type == "spatial"
    assert parse_question("hello world").question_type == "invalid_question"
    print("[OK] 四类问题 + invalid 解析")


def test_parse_chinese_and_english_punct() -> None:
    # 中文问号与英文问号都应识别
    assert parse_question("当前视场是否存在严重损伤建筑?").question_type == "presence"
    assert parse_question("当前视场有多少栋完全损毁建筑?").question_type == "count"
    print("[OK] 中英文标点兼容")


def test_parse_presence_severe_combo() -> None:
    spec = parse_question("当前视场是否存在严重或完全损毁建筑？")
    assert spec.question_type == "presence"
    assert set(spec.target_subtypes) == set(SEVERE_SUBTYPES), spec.target_subtypes
    print("[OK] presence 严重或完全损毁 -> severe 组合")


def test_parse_damage_ref_id() -> None:
    spec = parse_question("标记建筑 fdf000ba-cecf-4cd7 的损伤等级是什么？")
    assert spec.question_type == "damage"
    assert spec.ref_id == "fdf000ba-cecf-4cd7", spec.ref_id
    assert spec.needs_target_location
    print(f"[OK] damage ref_id 提取: {spec.ref_id}")


# ── schema 校验 ──────────────────────────────────────────────────────────────

def test_valid_answer_passes() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    d = {"answer": "是", "confidence": 0.8, "abstain": False, "decision": "answer",
         "reason_code": "sufficient_evidence",
         "evidence": {"source": "detector", "norm_xy": [0.1, 0.05]}}
    assert validate_answer_dict(d, spec) == [], validate_answer_dict(d, spec)
    print("[OK] 合法回答通过校验")


def test_missing_answer_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    assert "missing_answer" in validate_answer_dict({"confidence": 0.5, "decision": "answer"}, spec)
    assert "missing_answer" in validate_answer_dict({"answer": "  ", "confidence": 0.5}, spec)
    print("[OK] 缺/空 answer 被拒绝")


def test_answer_outside_closed_choices_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    d = {"answer": "可能", "confidence": 0.8, "abstain": False,
         "decision": "answer", "reason_code": "sufficient_evidence",
         "evidence": {"source": "image"}}
    errs = validate_answer_dict(d, spec)
    assert any(e.startswith("answer_not_in_choices:") for e in errs), errs


def test_confidence_out_of_range_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    assert "missing_confidence" in validate_answer_dict({"answer": "是", "decision": "answer"}, spec)
    errs = validate_answer_dict({"answer": "是", "confidence": 1.5, "decision": "answer"}, spec)
    assert any("confidence_out_of_range" in e for e in errs), errs
    errs = validate_answer_dict({"answer": "是", "confidence": "high", "decision": "answer"}, spec)
    assert "confidence_not_number" in errs, errs
    print("[OK] 非法 confidence 被拒绝")


def test_invalid_decision_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    errs = validate_answer_dict({"answer": "是", "confidence": 0.5, "decision": "guess"}, spec)
    assert any("invalid_decision" in e for e in errs), errs
    print("[OK] 非法 decision 被拒绝")


def test_invalid_reason_code_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    errs = validate_answer_dict({"answer": "是", "confidence": 0.5, "decision": "answer",
                                   "reason_code": "because_i_said_so"}, spec)
    assert any("invalid_reason_code" in e for e in errs), errs
    print("[OK] 非法 reason_code 被拒绝")


def test_invalid_norm_xy_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    errs = validate_answer_dict({"answer": "是", "confidence": 0.5, "decision": "answer",
                                   "evidence": {"norm_xy": [0.1]}}, spec)
    assert "invalid_norm_xy" in errs, errs
    errs = validate_answer_dict({"answer": "是", "confidence": 0.5, "abstain": False,
                                   "decision": "answer", "reason_code": "sufficient_evidence",
                                   "evidence": {"source": "image", "norm_xy": [-0.1, 1.2]}}, spec)
    assert "invalid_norm_xy" in errs, errs
    errs = validate_answer_dict({"answer": "是", "confidence": 0.5, "decision": "answer",
                                   "evidence": {"norm_xy": ["a", "b"]}}, spec)
    assert "invalid_norm_xy" in errs, errs
    print("[OK] 非法 norm_xy 被拒绝")


def test_invalid_evidence_source_rejected() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    errs = validate_answer_dict({"answer": "是", "confidence": 0.5, "decision": "answer",
                                   "evidence": {"source": "telepathy"}}, spec)
    assert any("invalid_evidence_source" in e for e in errs), errs
    print("[OK] 非法 evidence_source 被拒绝")


def test_abstain_and_evidence_are_required_and_consistent() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    base = {"answer": "是", "confidence": 0.5, "decision": "answer",
            "reason_code": "sufficient_evidence", "evidence": {"source": "image"}}
    assert "invalid_abstain" in validate_answer_dict(base, spec)
    mismatch = dict(base, abstain=True)
    assert "abstain_decision_mismatch" in validate_answer_dict(mismatch, spec)
    missing_ev = dict(base, abstain=False)
    missing_ev.pop("evidence")
    assert "missing_evidence" in validate_answer_dict(missing_ev, spec)
    bad_reason = dict(base, abstain=False, reason_code="target_missing")
    assert "decision_reason_mismatch" in validate_answer_dict(bad_reason, spec)


# ── VLM 输出解析 ──────────────────────────────────────────────────────────────

def test_parse_valid_json_output() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    text = '{"answer": "是", "confidence": 0.9, "abstain": false, "decision": "answer", "reason_code": "sufficient_evidence", "evidence": {"source": "image", "norm_xy": [0.5, 0.5]}}'
    ans = parse_vlm_json_output(text, spec, "q1")
    assert ans.answer == "是" and abs(ans.confidence - 0.9) < 1e-6
    assert ans.decision == "answer" and ans.reason_code == "sufficient_evidence"
    print("[OK] 合法 JSON 解析")


def test_parse_fenced_json_output() -> None:
    spec = parse_question("标记建筑 abc 的损伤等级是什么？")
    text = '分析如下:\n```json\n{"answer": "完全损毁", "confidence": 0.85, "abstain": false, "decision": "answer", "reason_code": "sufficient_evidence", "evidence": {"source": "image", "norm_xy": [0.5, 0.5]}}\n```'
    ans = parse_vlm_json_output(text, spec, "q2")
    assert ans.answer == "完全损毁" and ans.decision == "answer"
    print("[OK] ```json 包裹的输出解析")


def test_invalid_output_returns_explicit_invalid() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    # 非 JSON 文本
    ans = parse_vlm_json_output("我觉得是有的", spec, "q3")
    assert ans.decision == "abstain" and ans.reason_code == "invalid_output", ans
    assert ans.raw_model_output == "我觉得是有的"
    assert ans.schema_errors and ans.schema_errors[0].startswith("invalid_json:")
    # 非法 confidence
    bad = '{"answer": "是", "confidence": 2.0, "decision": "answer"}'
    ans = parse_vlm_json_output(bad, spec, "q4")
    assert ans.reason_code == "invalid_output", ans
    assert any("confidence_out_of_range" in e for e in ans.schema_errors)
    # 缺 answer
    bad = '{"confidence": 0.5, "decision": "answer"}'
    ans = parse_vlm_json_output(bad, spec, "q5")
    assert ans.reason_code == "invalid_output", ans
    assert "missing_answer" in ans.schema_errors
    print("[OK] 非法输出返回显式 invalid_output (不静默猜测)")


# ── 证据束 ────────────────────────────────────────────────────────────────────

class _FakePerception:
    def __init__(self, dets, pw=100, ph=100):
        self.detection = {"detections": dets}
        self.patch_width = pw
        self.patch_height = ph
        self.risk_level = "high"
        self.scene_text = "scene summary"
        self.degraded = False
        self.degraded_reason = ""


def test_build_evidence_picks_matching_target() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    dets = [
        {"class_name": "无损伤建筑", "conf": 0.95, "bbox": [10, 10, 30, 30]},
        {"class_name": "完全损毁建筑", "conf": 0.7, "bbox": [60, 60, 80, 80]},
    ]
    ev = build_evidence_from_perception(_FakePerception(dets), spec, "obs1")
    assert ev.target_subtype == "destroyed", ev.target_subtype
    assert ev.target_label == "完全损毁建筑"
    assert abs(ev.target_conf - 0.7) < 1e-6
    assert ev.norm_xy is not None and len(ev.norm_xy) == 2
    assert ev.norm_xy == [0.7, 0.7]
    assert ev.detection_source == "detector"
    print(f"[OK] 证据束选取匹配目标: {ev.target_label}@{ev.target_conf} norm_xy={ev.norm_xy}")


def test_build_evidence_no_match() -> None:
    spec = parse_question("当前视场是否存在完全损毁建筑？")
    dets = [{"class_name": "无损伤建筑", "conf": 0.9, "bbox": [10, 10, 30, 30]}]
    ev = build_evidence_from_perception(_FakePerception(dets), spec, "obs2")
    assert ev.target_subtype == "" and ev.norm_xy is None
    print("[OK] 无匹配目标时证据束为空")


def test_evidence_does_not_use_scene_text_as_label() -> None:
    """scene_text 是未经验证的自由文本, 不得当作事实标签 (计划 7.5)。"""
    spec = parse_question("标记建筑 abc 的损伤等级是什么？")
    dets = []  # 无检测
    ev = build_evidence_from_perception(_FakePerception(dets), spec, "obs3")
    assert ev.target_subtype == ""  # 即使 scene_text 有内容也不当标签
    assert ev.scene_text == "scene summary"  # 仍记录, 但不作 label
    print("[OK] scene_text 不被当作事实标签")


# ── 常量完备性 ────────────────────────────────────────────────────────────────

def test_closed_sets_complete() -> None:
    assert set(QUESTION_TYPES) == {"presence", "damage", "count", "spatial"}
    assert set(DECISIONS) == {"answer", "continue_search", "reobserve", "abstain"}
    assert "sufficient_evidence" in REASON_CODES and "invalid_output" in REASON_CODES
    assert set(EVIDENCE_SOURCES) == {"image", "detector", "change_classifier",
                                      "semantic_map", "history"}
    assert len(BEARING_CHOICES) == 8
    assert set(DAMAGE_CHOICES) == {"无损伤", "轻微损伤", "严重损伤", "完全损毁"}
    assert set(COUNT_CHOICES) == {"0", "1", "2", "3+"}
    assert set(PRESENCE_CHOICES) == {"是", "否"}
    print("[OK] 封闭集合常量完备")


def test_vlm_prompt_names_machine_readable_evidence_sources() -> None:
    for source in EVIDENCE_SOURCES:
        assert source in AGENT_VQA_SYSTEM_PROMPT
    assert "绝不能填写证据描述句" in AGENT_VQA_SYSTEM_PROMPT


def _run_all() -> int:
    tests = [
        test_parse_four_question_types, test_parse_chinese_and_english_punct,
        test_parse_presence_severe_combo, test_parse_damage_ref_id,
        test_valid_answer_passes, test_missing_answer_rejected,
        test_answer_outside_closed_choices_rejected,
        test_confidence_out_of_range_rejected, test_invalid_decision_rejected,
        test_invalid_reason_code_rejected, test_invalid_norm_xy_rejected,
        test_invalid_evidence_source_rejected,
        test_abstain_and_evidence_are_required_and_consistent,
        test_parse_valid_json_output,
        test_parse_fenced_json_output, test_invalid_output_returns_explicit_invalid,
        test_build_evidence_picks_matching_target, test_build_evidence_no_match,
        test_evidence_does_not_use_scene_text_as_label, test_closed_sets_complete,
        test_vlm_prompt_names_machine_readable_evidence_sources,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"[ERROR] {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{'=' * 48}\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
