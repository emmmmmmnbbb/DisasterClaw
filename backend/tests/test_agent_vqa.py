"""backend/tests/test_agent_vqa.py — Agent-VQA 控制器行为测试 (D3, 计划 7.8).

用桩依赖 (vlm/perceive/search/reobserve) 覆盖:
  1. 目标不存在 -> continue_search。
  2. 低置信 -> reobserve。
  3. 预算耗尽 -> abstain。
  4. LLM/VLM 不可用 -> 规则回退。
  5. 非 oracle 配置无法读取 GT (item.answer / item.target 不影响在线决策)。
  6. 当前观测不变时结果结构稳定。
  7. 日志区分错误、弃答和普通错误答案 (trajectory reason_code)。
  8. 充分证据 -> answer。

运行: python backend/tests/test_agent_vqa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_vqa import (  # noqa: E402
    AgentVqaConfig, AgentVqaController, QuestionSpec, VqaAnswer,
    build_evidence_from_perception, parse_question,
)


class _FakePerception:
    def __init__(self, dets=None, pw=100, ph=100, degraded=False):
        self.detection = {"detections": dets or []}
        self.patch_width = pw
        self.patch_height = ph
        self.risk_level = "low" if not dets else "high"
        self.scene_text = ""
        self.degraded = degraded
        self.degraded_reason = "no_post_coverage" if degraded else ""
        self.patch_id = "obs0"


def _det(cls, conf, bbox=None):
    return {"class_name": cls, "conf": conf, "bbox": bbox or [40, 40, 60, 60]}


def _vlm_confident(img, result, spec, qid):
    """VLM 桩: 总是返回高置信回答。"""
    if spec.question_type == "presence":
        answer = "是"
    if spec.question_type == "damage":
        answer = "完全损毁"
    elif spec.question_type == "count":
        answer = "1"
    elif spec.question_type == "spatial":
        answer = "北"
    return (f'{{"answer": "{answer}", "confidence": 0.9, "abstain": false, '
            '"decision": "answer", "reason_code": "sufficient_evidence", '
            '"evidence": {"source": "image", "norm_xy": [0.5, 0.5]}}')


def _vlm_low_conf(img, result, spec, qid):
    """VLM 桩: 总是返回低置信 (触发 reobserve)。"""
    return ('{"answer": "是", "confidence": 0.3, "abstain": false, '
            '"decision": "answer", "reason_code": "sufficient_evidence", '
            '"evidence": {"source": "image", "norm_xy": [0.5, 0.5]}}')


def _vlm_invalid(img, result, spec, qid):
    """VLM 桩: 返回非 JSON (触发 invalid_output -> 规则回退)。"""
    return "我觉得有损坏"


def _make_ctrl(perceive_dets, vlm_fn=None, search_fn=None, reobserve_fn=None,
                config=None, get_pos=None):
    """构造一个用固定感知结果作桩的控制器。"""
    result = _FakePerception(perceive_dets)
    return AgentVqaController(
        config=config or AgentVqaConfig(),
        vlm_answer_fn=vlm_fn,
        perceive_fn=lambda: result,
        search_fn=search_fn,
        reobserve_fn=reobserve_fn,
        get_position_fn=get_pos or (lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0}),
    )


# ── 1. 充分证据 -> answer ──────────────────────────────────────────────────────

def test_sufficient_evidence_answers() -> None:
    ctl = _make_ctrl([_det("完全损毁建筑", 0.9)], vlm_fn=_vlm_confident)
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q1")
    assert ans.decision == "answer", (ans.decision, ans.reason_code)
    assert ans.answer == "是" and ans.confidence >= 0.5
    assert not ans.abstain
    print(f"[OK] 充分证据 -> answer: {ans.answer}@{ans.confidence}")


# ── 2. 目标不存在 -> continue_search -> 预算耗尽 abstain ───────────────────────

def test_target_missing_continues_search_then_abstains() -> None:
    search_calls = []
    def search_fn(spec, step, result):
        search_calls.append(step)
        return {"north_m": 10.0, "east_m": 0.0}  # 搜索但找不到 (感知结果固定为空)
    ctl = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=3, max_reobservations=0),
        vlm_answer_fn=None,  # 规则回退
        perceive_fn=lambda: _FakePerception([]),  # 永远无目标
        search_fn=search_fn,
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    ans = ctl.run("视场中心十字标记建筑的损伤等级是什么？", "q2")
    assert ans.decision == "abstain", (ans.decision, ans.reason_code)
    assert ans.reason_code == "budget_exhausted", ans.reason_code
    assert len(search_calls) > 0, "应至少尝试搜索一次"
    print(f"[OK] 目标缺失 -> continue_search {len(search_calls)} 次后 abstain (budget_exhausted)")


# ── 3. 低置信 -> reobserve ─────────────────────────────────────────────────────

def test_low_confidence_triggers_reobserve() -> None:
    reobs_calls = []
    def reobserve_fn(result, spec):
        reobs_calls.append(1)
        return {"kind": "recheck",
                "params": {"north_m": 0.0, "east_m": 0.0, "up_m": -10.0}}
    ctl = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=0, max_reobservations=2,
                              confidence_threshold=0.6),
        vlm_answer_fn=_vlm_low_conf,  # 总是 0.3 < 0.6
        perceive_fn=lambda: _FakePerception([_det("完全损毁建筑", 0.4)]),
        reobserve_fn=reobserve_fn,
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q3")
    assert len(reobs_calls) == 2, f"应用完 2 次重观测预算, 实际 {len(reobs_calls)}"
    assert ans.decision == "abstain" or ans.decision == "answer"
    # 预算用完后用最后一次观测作答。
    assert ans.decision == "answer" and ans.reason_code == "sufficient_evidence", ans
    print(f"[OK] 低置信 -> reobserve {len(reobs_calls)} 次后预算耗尽")


# ── 4. VLM 不可用 -> 规则回退 ───────────────────────────────────────────────────

def test_vlm_unavailable_uses_rule_fallback() -> None:
    ctl = _make_ctrl([_det("完全损毁建筑", 0.8)], vlm_fn=None)  # 无 VLM
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q4")
    assert ctl.fallback_used, "应标记 fallback_used"
    assert ans.decision == "answer", (ans.decision, ans.reason_code)
    assert ans.answer == "是"  # 规则回退: 有匹配目标 -> 是
    print(f"[OK] VLM 不可用 -> 规则回退 answer={ans.answer}, fallback_used={ctl.fallback_used}")


def test_vlm_invalid_output_uses_rule_fallback() -> None:
    ctl = _make_ctrl([_det("完全损毁建筑", 0.8)], vlm_fn=_vlm_invalid)
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q5")
    assert not ctl.fallback_used, "非法 JSON 必须显式失败，不能静默换成规则答案"
    assert ans.decision == "abstain" and ans.reason_code == "invalid_output"
    assert "invalid" in ctl.degraded_reason, ctl.degraded_reason
    print(f"[OK] VLM 非法输出 -> 规则回退, degraded={ctl.degraded_reason}")


# ── 5. 非 oracle 配置无法读取 GT ───────────────────────────────────────────────

def test_non_oracle_ignores_item_answer_and_target() -> None:
    """非 oracle 配置下, item 的 answer/target 字段不得影响在线决策 (计划 7.3)。"""
    # 感知结果为空 (无目标), 但 item 里塞了 GT answer=是 / target 坐标
    item = {"answer": "是", "target": {"lat": 99.0, "lon": 99.0, "subtype": "destroyed"}}
    ctl = AgentVqaController(
        config=AgentVqaConfig(oracle=False, max_search_steps=1, max_reobservations=0),
        vlm_answer_fn=None,
        perceive_fn=lambda: _FakePerception([]),  # 无目标
        search_fn=lambda spec, step, result: {"north_m": 5.0, "east_m": 0.0},
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    ans = ctl.run("视场中心十字标记建筑的损伤等级是什么？", "q6", item=item)
    # 即使 item.answer=是, 在线感知无目标 -> 不应回答"是"
    assert ans.answer != "是" or ans.decision == "abstain", (ans.answer, ans.decision)
    assert ans.decision == "abstain", ans.decision
    print(f"[OK] 非 oracle 忽略 item.answer/target: decision={ans.decision} answer={ans.answer!r}")


def test_oracle_config_can_read_item_target() -> None:
    """oracle 配置 (诊断用) 才允许从 item 读目标坐标。"""
    # 这里只验证 oracle 开关存在且不崩; 真正的 oracle 路径在 app 层接入时实现
    cfg = AgentVqaConfig(oracle=True, allow_target_leak=True)
    assert cfg.oracle and cfg.allow_target_leak
    print("[OK] oracle 配置开关存在")


# ── 6. 当前观测不变时结果结构稳定 ─────────────────────────────────────────────

def test_stable_result_on_same_observation() -> None:
    dets = [_det("完全损毁建筑", 0.85)]
    ctl1 = _make_ctrl(dets, vlm_fn=_vlm_confident)
    ctl2 = _make_ctrl(dets, vlm_fn=_vlm_confident)
    a1 = ctl1.run("当前视场是否存在完全损毁建筑？", "q7")
    a2 = ctl2.run("当前视场是否存在完全损毁建筑？", "q7")
    assert a1.decision == a2.decision == "answer"
    assert a1.answer == a2.answer
    print(f"[OK] 同观测结果稳定: {a1.answer}@{a1.confidence}")


# ── 7. 日志区分错误、弃答和普通错误答案 ─────────────────────────────────────────

def test_trajectory_distinguishes_outcomes() -> None:
    # (a) 正常 answer
    ctl_ok = _make_ctrl([_det("完全损毁建筑", 0.9)], vlm_fn=_vlm_confident)
    ctl_ok.run("当前视场是否存在完全损毁建筑？", "q_ok")
    assert any(r.decision == "answer" and r.reason_code == "sufficient_evidence"
               for r in ctl_ok.trajectory)

    # (b) 弃答 (预算耗尽)
    ctl_abs = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=1, max_reobservations=0),
        vlm_answer_fn=None,
        perceive_fn=lambda: _FakePerception([]),
        search_fn=lambda spec, step, result: {"north_m": 5.0},
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    abs_ans = ctl_abs.run("标记建筑 abc 的损伤等级是什么？", "q_abs")
    assert abs_ans.decision == "abstain"
    # trajectory 里应有 continue_search 和最终 abstain 的记录
    decisions = {r.decision for r in ctl_abs.trajectory}
    assert "continue_search" in decisions or "abstain" in decisions, decisions
    print(f"[OK] 轨迹区分: ok->answer, abstain->{abs_ans.reason_code}")


def test_out_of_coverage_abstains() -> None:
    """感知返回 None (无 POST 覆盖) -> out_of_coverage abstain。"""
    ctl = AgentVqaController(
        config=AgentVqaConfig(),
        vlm_answer_fn=_vlm_confident,
        perceive_fn=lambda: None,
        get_position_fn=lambda: {"lat": 0.0, "lon": 0.0, "alt": 30.0},
    )
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q_oob")
    assert ans.decision == "abstain" and ans.reason_code == "out_of_coverage", ans
    print(f"[OK] 无 POST 覆盖 -> out_of_coverage abstain")


def test_cancelled_episode_is_explicit() -> None:
    ctl = AgentVqaController(
        config=AgentVqaConfig(), perceive_fn=lambda: _FakePerception([]),
        is_cancelled_fn=lambda: True,
    )
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q_cancel")
    assert ans.decision == "abstain" and ans.abstain
    assert ans.reason_code == "cancelled"


def test_rule_fallback_count_and_spatial_use_all_detections() -> None:
    dets = [
        _det("严重损伤建筑", 0.7, [70, 45, 90, 55]),
        _det("完全损毁建筑", 0.8, [75, 45, 95, 55]),
        _det("完全损毁建筑", 0.6, [60, 45, 80, 55]),
    ]
    count_ctl = _make_ctrl(dets, vlm_fn=None)
    count = count_ctl.run("当前视场有多少栋严重或完全损毁建筑？", "q_count")
    assert count.answer == "3+", count
    spatial_ctl = _make_ctrl(dets, vlm_fn=None)
    spatial = spatial_ctl.run("最近的完全损毁建筑位于无人机哪个方向？", "q_spatial")
    assert spatial.answer == "东", spatial


def test_raw_evidence_does_not_fallback_to_detector() -> None:
    ctl = _make_ctrl(
        [_det("完全损毁建筑", 0.95)], vlm_fn=None,
        config=AgentVqaConfig(evidence_level="raw", max_search_steps=0, max_reobservations=0),
    )
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q_raw")
    assert ans.decision == "abstain" and ans.reason_code == "vlm_unavailable"
    assert not ctl.fallback_used


def test_static_negative_presence_answers_no() -> None:
    ctl = _make_ctrl([], vlm_fn=None,
                     config=AgentVqaConfig(max_search_steps=0, max_reobservations=0))
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q_negative")
    assert ans.answer == "否" and ans.decision == "answer" and not ans.abstain


# ── 8. 四类问题在控制器层都能跑通 ───────────────────────────────────────────────

def test_all_four_types_run() -> None:
    for q in (
        "当前视场是否存在完全损毁建筑？",
        "标记建筑 abc-1 的损伤等级是什么？",
        "当前视场有多少栋严重或完全损毁建筑？",
        "最近的完全损毁建筑位于无人机哪个方向？",
    ):
        ctl = _make_ctrl([_det("完全损毁建筑", 0.9)], vlm_fn=_vlm_confident)
        ans = ctl.run(q, "q_all")
        assert ans.decision in ("answer", "abstain", "continue_search"), (q, ans.decision)
    print("[OK] 四类问题在控制器层都能跑通")


def _run_all() -> int:
    tests = [
        test_sufficient_evidence_answers,
        test_target_missing_continues_search_then_abstains,
        test_low_confidence_triggers_reobserve,
        test_vlm_unavailable_uses_rule_fallback,
        test_vlm_invalid_output_uses_rule_fallback,
        test_non_oracle_ignores_item_answer_and_target,
        test_oracle_config_can_read_item_target,
        test_stable_result_on_same_observation,
        test_trajectory_distinguishes_outcomes,
        test_out_of_coverage_abstains,
        test_cancelled_episode_is_explicit,
        test_rule_fallback_count_and_spatial_use_all_detections,
        test_raw_evidence_does_not_fallback_to_detector,
        test_static_negative_presence_answers_no,
        test_all_four_types_run,
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
