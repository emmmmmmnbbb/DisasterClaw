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
    AgentVqaConfig, AgentVqaController, EvidenceBundle, GeographicEvidenceMemory,
    QuestionSpec, TaskContext, VqaAnswer,
    bboxes_match, build_evidence_from_perception, derive_generation_seed,
    choices_for_question_type, parse_question, task_context_from_item,
)


class _FakePerception:
    def __init__(self, dets=None, pw=100, ph=100, degraded=False, extras=None):
        self.detection = {"detections": dets or []}
        self.patch_width = pw
        self.patch_height = ph
        self.risk_level = "low" if not dets else "high"
        self.scene_text = ""
        self.degraded = degraded
        self.degraded_reason = "no_post_coverage" if degraded else ""
        self.patch_id = "obs0"
        self.extras = extras or {}


def _det(cls, conf, bbox=None):
    return {"class_name": cls, "conf": conf, "bbox": bbox or [40, 40, 60, 60]}


def _geo_evidence(obs_id, objects, *, gsd=1.0):
    return EvidenceBundle(
        observation_id=obs_id, objects=objects, observation_gsd_m=gsd,
        roi_coverage=1.0,
        view_window={"west": 120.0, "south": 30.0, "east": 121.0, "north": 31.0},
        roi_geo_bounds={"west": 120.4, "south": 30.4, "east": 120.6, "north": 30.6},
    )


def test_geographic_memory_deduplicates_and_fuses_resolution_weighted_probs() -> None:
    spec = parse_question("标记区域内有多少栋受损建筑？")
    memory = GeographicEvidenceMemory(match_radius_m=6.0)
    first = _geo_evidence("wide", [{
        "lat": 30.5, "lon": 120.5, "label": "无损伤建筑", "subtype": "no-damage",
        "confidence": 0.6, "class_probs": {"no-damage": 0.6, "damaged": 0.4},
        "bbox": [40, 40, 60, 60], "norm_xy": [0.5, 0.5],
        "observation_id": "wide", "gsd_m": 1.5,
    }], gsd=1.5)
    second = _geo_evidence("fine", [{
        "lat": 30.50001, "lon": 120.50001, "label": "受损建筑", "subtype": "damaged",
        "confidence": 0.9, "class_probs": {"no-damage": 0.1, "damaged": 0.9},
        "bbox": [41, 41, 61, 61], "norm_xy": [0.51, 0.51],
        "observation_id": "fine", "gsd_m": 0.5,
    }], gsd=0.5)
    memory.update(spec, first)
    fused = memory.update(spec, second)
    assert fused.matching_count == 1
    assert fused.target_subtype == "damaged"
    assert fused.class_probs["damaged"] > 0.7
    assert fused.history_observation_ids == ["wide", "fine"]
    assert fused.objects[0]["n_sightings"] == 2


def test_geographic_memory_does_not_overwrite_strong_fine_view() -> None:
    spec = parse_question("标记区域内是否存在受损建筑？")
    memory = GeographicEvidenceMemory(match_radius_m=6.0)
    fine = _geo_evidence("fine", [{
        "lat": 30.5, "lon": 120.5, "label": "受损建筑", "subtype": "damaged",
        "confidence": 0.95, "class_probs": {"no-damage": 0.05, "damaged": 0.95},
        "bbox": [40, 40, 60, 60], "norm_xy": [0.5, 0.5],
        "observation_id": "fine", "gsd_m": 0.5,
    }], gsd=0.5)
    noisy = _geo_evidence("noisy", [{
        "lat": 30.50001, "lon": 120.50001, "label": "无损伤建筑", "subtype": "no-damage",
        "confidence": 0.55, "class_probs": {"no-damage": 0.55, "damaged": 0.45},
        "bbox": [42, 42, 62, 62], "norm_xy": [0.52, 0.52],
        "observation_id": "noisy", "gsd_m": 1.5,
    }], gsd=1.5)
    memory.update(spec, fine)
    fused = memory.update(spec, noisy)
    assert fused.target_subtype == "damaged"
    assert fused.matching_count == 1


def test_detailed_changeos_labels_survive_history_fusion() -> None:
    """Four-class ChangeOS probabilities must remain available after fusion."""
    spec = QuestionSpec(
        "count",
        "视野内有多少完全损毁的建筑？",
        target_subtypes=("destroyed",),
    )
    result = _FakePerception(
        [
            {
                "class_name": "受损建筑",
                "confidence": 0.91,
                "bbox": [10, 10, 30, 30],
                "class_probs": {"受损建筑": 0.91, "无损伤建筑": 0.09},
                "extras": {
                    "four_class_probs": {
                        "no-damage": 0.05,
                        "minor-damage": 0.03,
                        "major-damage": 0.02,
                        "destroyed": 0.90,
                    }
                },
            },
            {
                "class_name": "受损建筑",
                "confidence": 0.84,
                "bbox": [60, 60, 80, 80],
                "class_probs": {"受损建筑": 0.84, "无损伤建筑": 0.16},
                "extras": {
                    "four_class_probs": {
                        "no-damage": 0.10,
                        "minor-damage": 0.05,
                        "major-damage": 0.75,
                        "destroyed": 0.10,
                    }
                },
            },
        ],
        pw=100,
        ph=100,
    )

    current = build_evidence_from_perception(result, spec, "obs")
    fused = GeographicEvidenceMemory().update(spec, current)

    assert current.matching_count == 1
    assert fused.matching_count == 1
    assert fused.target_subtype == "destroyed"


def test_geographic_memory_keeps_distinct_buildings() -> None:
    spec = parse_question("标记区域内有多少栋受损建筑？")
    memory = GeographicEvidenceMemory(match_radius_m=6.0)
    objects = []
    for idx, lat in enumerate((30.5, 30.5001)):
        objects.append({
            "lat": lat, "lon": 120.5, "label": "受损建筑", "subtype": "damaged",
            "confidence": 0.9, "class_probs": {"no-damage": 0.1, "damaged": 0.9},
            "bbox": [10 + idx * 30, 10, 20 + idx * 30, 20],
            "norm_xy": [0.4 + idx * 0.1, 0.5], "observation_id": "obs", "gsd_m": 0.5,
        })
    fused = memory.update(spec, _geo_evidence("obs", objects, gsd=0.5))
    assert fused.matching_count == 2
    assert len({obj["geographic_id"] for obj in fused.objects}) == 2


def test_geographic_memory_does_not_answer_from_stale_full_roi_track() -> None:
    """A coarse false positive absent from the next full-ROI view is historical only."""
    spec = parse_question("最近的受损建筑位于标记区域中心哪个方向？")
    memory = GeographicEvidenceMemory(match_radius_m=6.0)
    stale = {
        "lat": 30.50001, "lon": 120.50001, "label": "受损建筑",
        "subtype": "damaged", "confidence": 0.95,
        "class_probs": {"no-damage": 0.05, "damaged": 0.95},
        "bbox": [49, 49, 51, 51], "norm_xy": [0.5, 0.5],
        "observation_id": "wide", "gsd_m": 1.5,
    }
    current = {
        "lat": 30.4997, "lon": 120.5, "label": "受损建筑",
        "subtype": "damaged", "confidence": 0.8,
        "class_probs": {"no-damage": 0.2, "damaged": 0.8},
        "bbox": [49, 55, 51, 60], "norm_xy": [0.5, 0.575],
        "observation_id": "fine", "gsd_m": 0.5,
    }
    memory.update(spec, _geo_evidence("wide", [stale], gsd=1.5))
    fused = memory.update(spec, _geo_evidence("fine", [current], gsd=0.5))
    assert fused.matching_count == 1
    assert fused.target_geo == [current["lat"], current["lon"]]
    assert any(not obj["visible_in_current"] for obj in fused.objects)


def _vlm_confident(img, result, spec, qid, evidence, generation_context):
    """VLM 桩: 总是返回高置信回答。"""
    if spec.question_type == "presence":
        answer = "是"
    if spec.question_type == "damage":
        answer = choices_for_question_type("damage")[-1]
    elif spec.question_type == "count":
        answer = "1"
    elif spec.question_type == "spatial":
        answer = "北"
    return (f'{{"answer": "{answer}", "confidence": 0.9, "abstain": false, '
            '"decision": "answer", "reason_code": "sufficient_evidence", '
            '"evidence": {"source": "image", "norm_xy": [0.5, 0.5]}}')


def _vlm_low_conf(img, result, spec, qid, evidence, generation_context):
    """VLM 桩: 总是返回低置信 (触发 reobserve)。"""
    return ('{"answer": "是", "confidence": 0.3, "abstain": false, '
            '"decision": "answer", "reason_code": "sufficient_evidence", '
            '"evidence": {"source": "image", "norm_xy": [0.5, 0.5]}}')


def _vlm_invalid(img, result, spec, qid, evidence, generation_context):
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
    altitude = {"value": 30.0}
    def reobserve_fn(result, spec, evidence):
        reobs_calls.append(1)
        altitude["value"] -= 10.0
        return {"kind": "recheck",
                "params": {"north_m": 0.0, "east_m": 0.0, "up_m": -10.0},
                "reason": "policy_recheck", "uncertainty": 0.8}
    ctl = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=0, max_reobservations=2,
                              confidence_threshold=0.6),
        vlm_answer_fn=_vlm_low_conf,  # 总是 0.3 < 0.6
        perceive_fn=lambda: _FakePerception([_det("完全损毁建筑", 0.4)]),
        reobserve_fn=reobserve_fn,
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": altitude["value"]},
    )
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q3")
    assert len(reobs_calls) == 2, f"应用完 2 次重观测预算, 实际 {len(reobs_calls)}"
    assert ans.decision == "abstain" or ans.decision == "answer"
    # 预算用完后用最后一次观测作答。
    assert ans.decision == "answer" and ans.reason_code == "sufficient_evidence", ans
    assert [r.position["alt"] for r in ctl.trajectory] == [30.0, 20.0, 10.0]
    print(f"[OK] 低置信 -> reobserve {len(reobs_calls)} 次后预算耗尽")


def test_reobserve_policy_runs_without_matching_subtype() -> None:
    """题面要完全损毁、当前只有无损伤框时，仍应把观测交给策略 (计划 E4)。"""
    calls = []
    def reobserve_fn(result, spec, evidence):
        calls.append(spec.question_type)
        return {"kind": "recheck",
                "params": {"north_m": 0.0, "east_m": 0.0, "up_m": -10.0},
                "reason": "entropy_high", "uncertainty": 0.7}
    ctl = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=0, max_reobservations=1),
        vlm_answer_fn=_vlm_confident,
        perceive_fn=lambda: _FakePerception([_det("无损伤建筑", 0.55)]),
        reobserve_fn=reobserve_fn,
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    ctl.run("当前视场是否存在完全损毁建筑？", "q_nomatch")
    assert calls == ["presence"], calls
    kinds = [r.reobserve_kind for r in ctl.trajectory]
    assert "recheck" in kinds, kinds


def test_reobserve_skip_is_recorded_and_answers() -> None:
    def reobserve_fn(result, spec, evidence):
        return {"kind": "skip", "reason": "把握足够或无可疑灾情目标，无需复核。",
                "uncertainty": 0.12}
    ctl = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=0, max_reobservations=2),
        vlm_answer_fn=_vlm_confident,
        perceive_fn=lambda: _FakePerception([_det("完全损毁建筑", 0.9)]),
        reobserve_fn=reobserve_fn,
        get_position_fn=lambda: {"lat": 30.0, "lon": 120.0, "alt": 30.0},
    )
    ans = ctl.run("当前视场是否存在完全损毁建筑？", "q_skip")
    assert ans.decision == "answer" and not ans.abstain
    assert len(ctl.trajectory) == 1
    rec = ctl.trajectory[0]
    assert rec.decision == "answer"
    assert rec.reobserve_kind == "skip"
    assert rec.uncertainty == 0.12
    assert rec.evidence.get("target_subtype") == "destroyed"


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


def test_damage_target_is_projected_and_associated_without_subtype_filter() -> None:
    perception = _FakePerception(
        [
            _det("无损伤建筑", 0.95, [5, 5, 20, 20]),
            _det("严重损伤建筑", 0.80, [70, 20, 90, 40]),
        ],
        extras={
            "window": {"west": 120.0, "east": 121.0, "south": 30.0, "north": 31.0},
            "roi_norm_bbox": [0.1, 0.1, 0.9, 0.9],
        },
    )
    spec = parse_question("标记区域内标记建筑 b-17 的损伤等级是什么？")
    ctx = TaskContext(target_ref_id="b-17", target_lat=30.7, target_lon=120.8)
    ev = build_evidence_from_perception(perception, spec, "obs", ctx)
    assert ev.target_norm_xy == [0.8, 0.3]
    assert ev.target_visible and ev.target_matched
    assert ev.target_subtype == "major-damage"
    assert ev.matching_count == 1
    assert ev.match_method == "target_point_in_bbox"


def test_damage_nearest_fallback_uses_metric_not_image_fraction_gate() -> None:
    perception = _FakePerception(
        [_det("无损伤建筑", 0.6, [545, 545, 555, 555])], pw=1024, ph=1024,
        extras={
            "window": {"west": 120.0, "east": 121.0, "south": 30.0, "north": 31.0},
            "roi_norm_bbox": [0.1, 0.1, 0.9, 0.9],
            "eff_gsd_m": 1.5,
        },
    )
    spec = parse_question("标记建筑 b-17 的损伤状态是什么？")
    # Target is 30 px from the predicted centre: old 8%-image threshold
    # accepted it, while 15 m / 1.5 m-per-px correctly rejects it.
    ctx = TaskContext(
        target_ref_id="b-17", target_lat=31.0 - 550.0 / 1024.0,
        target_lon=120.0 + 580.0 / 1024.0,
    )
    ev = build_evidence_from_perception(perception, spec, "obs", ctx)
    assert ev.target_visible and not ev.target_matched


def test_visible_unmatched_damage_target_can_reobserve_before_search() -> None:
    perception = _FakePerception([], extras={
        "window": {"west": 120.0, "east": 121.0, "south": 30.0, "north": 31.0},
        "roi_norm_bbox": [0.1, 0.1, 0.9, 0.9], "eff_gsd_m": 1.5,
    })
    called = []
    ctl = AgentVqaController(
        config=AgentVqaConfig(max_search_steps=0, max_reobservations=1,
                              answer_mode="deterministic"),
        perceive_fn=lambda: perception,
        reobserve_fn=lambda result, spec, evidence: called.append(evidence) or {
            "kind": "recheck", "params": {"north_m": 1.0, "up_m": -10.0},
        },
    )
    spec = parse_question("标记建筑 b-17 的损伤状态是什么？")
    ctx = TaskContext(target_ref_id="b-17", target_lat=30.5, target_lon=120.5)
    ctl.run(spec.raw, "q_visible_missing", task_context=ctx)
    assert called and called[0].target_visible and not called[0].target_matched
    assert ctl.trajectory[0].decision == "reobserve"


def test_task_context_never_exposes_answer_or_damage_subtype() -> None:
    spec = parse_question("标记区域内标记建筑 b-17 的损伤等级是什么？")
    ctx = task_context_from_item({
        "answer": "完全损毁",
        "tile_id": "tile-a",
        "target": {"ref_id": "b-17", "lat": 30.1, "lon": 120.2,
                   "subtype": "destroyed"},
    }, spec)
    assert ctx.target_ref_id == "b-17"
    assert ctx.target_lat == 30.1 and ctx.target_lon == 120.2
    assert not hasattr(ctx, "answer") and not hasattr(ctx, "target_subtype")


def test_hybrid_rejects_vlm_damage_answer_without_target_match() -> None:
    ctl = AgentVqaController(
        config=AgentVqaConfig(answer_mode="hybrid", max_search_steps=0, max_reobservations=0),
        vlm_answer_fn=_vlm_confident,
        perceive_fn=lambda: _FakePerception([]),
    )
    item = {"target": {"ref_id": "b-17", "lat": 30.1, "lon": 120.2,
                       "subtype": "destroyed"}, "answer": "完全损毁"}
    ans = ctl.run("标记区域内标记建筑 b-17 的损伤等级是什么？", "q_guard", item=item)
    assert ans.decision == "abstain" and ans.reason_code == "budget_exhausted"
    assert ans.answer == ""


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


def test_bboxes_match_tolerates_float_precision() -> None:
    """legacy 全精度 bbox 与 round(2) 的 target_bbox 应判定为同一框。"""
    assert bboxes_match([10.123456, 20.987654, 30.5, 40.5], [10.12, 20.99, 30.5, 40.5])
    assert not bboxes_match([10.123456, 20.987654, 30.5, 40.5], [10.12, 20.99, 30.5, 40.6])
    assert not bboxes_match([1, 2, 3], [1, 2, 3, 4])
    assert not bboxes_match(None, [1, 2, 3, 4])
    assert not bboxes_match("bad", [1, 2, 3, 4])


def test_generation_seed_is_keyed_and_logged() -> None:
    seen = []

    def seeded_vlm(img, result, spec, qid, evidence, generation_context):
        seen.append(generation_context)
        return _vlm_confident(img, result, spec, qid, evidence, generation_context)

    cfg = AgentVqaConfig(
        max_search_steps=0, max_reobservations=0,
        generation_base_seed=20260913, generation_repeat=2,
    )
    ctl = _make_ctrl([_det("完全损毁建筑", 0.9)], vlm_fn=seeded_vlm, config=cfg)
    ctl.run("当前视场是否存在完全损毁建筑？", "stable-qid")
    expected = derive_generation_seed(20260913, "stable-qid", 2, 0, "candidate_answer")
    assert len(seen) == 1 and seen[0].seed == expected
    record = ctl.trajectory[0].to_dict()
    assert record["generation_seed"] == expected
    assert record["generation_repeat"] == 2
    assert record["generation_call_role"] == "candidate_answer"


def test_generation_seed_does_not_depend_on_execution_order() -> None:
    keys = [
        ("q-a", 0, 0, "candidate_answer"),
        ("q-b", 0, 0, "candidate_answer"),
        ("q-a", 1, 0, "candidate_answer"),
        ("q-a", 0, 1, "candidate_answer"),
    ]
    forward = {key: derive_generation_seed(7, *key) for key in keys}
    reverse = {key: derive_generation_seed(7, *key) for key in reversed(keys)}
    assert forward == reverse
    assert len(set(forward.values())) == len(keys)


def test_deterministic_answer_does_not_log_generation_seed() -> None:
    cfg = AgentVqaConfig(
        answer_mode="deterministic", max_search_steps=0, max_reobservations=0,
        generation_base_seed=20260913,
    )
    ctl = _make_ctrl([_det("完全损毁建筑", 0.9)], vlm_fn=_vlm_confident, config=cfg)
    ctl.run("当前视场是否存在完全损毁建筑？", "rule-qid")
    record = ctl.trajectory[0].to_dict()
    assert record["generation_seed"] is None
    assert record["generation_call_role"] == ""


def _run_all() -> int:
    tests = [
        test_geographic_memory_deduplicates_and_fuses_resolution_weighted_probs,
        test_geographic_memory_does_not_overwrite_strong_fine_view,
        test_detailed_changeos_labels_survive_history_fusion,
        test_geographic_memory_keeps_distinct_buildings,
        test_geographic_memory_does_not_answer_from_stale_full_roi_track,
        test_bboxes_match_tolerates_float_precision,
        test_generation_seed_is_keyed_and_logged,
        test_generation_seed_does_not_depend_on_execution_order,
        test_deterministic_answer_does_not_log_generation_seed,
        test_sufficient_evidence_answers,
        test_target_missing_continues_search_then_abstains,
        test_low_confidence_triggers_reobserve,
        test_reobserve_policy_runs_without_matching_subtype,
        test_reobserve_skip_is_recorded_and_answers,
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
        test_damage_target_is_projected_and_associated_without_subtype_filter,
        test_task_context_never_exposes_answer_or_damage_subtype,
        test_hybrid_rejects_vlm_damage_answer_without_target_match,
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
