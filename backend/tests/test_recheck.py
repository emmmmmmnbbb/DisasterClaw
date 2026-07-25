"""
backend/tests/test_recheck.py — P2 灾情不确定性驱动主动复核 单测

验收点（对应 docs/vln_rescue_agent_实施计划.md 的 P2 + P5）：
    1. 不确定性评分：低 risk('low'/可疑) + 低 conf → 高；high + 高 conf → 低。
    2. best_evidence 取受灾相关检测里 conf 最高者，忽略完好建筑；附带回传 class_probs。
    3. 低置信/可疑场景 → 触发复核（降高 up_m<0 + 朝目标居中）。
    4. 高置信场景 → 不触发复核（不浪费步数）。
    5. 复核预算耗尽 / 到高度下限 → 收尾定论；降高不越过 alt_min。
    6. 改善路径：先复核 → 再观测变笃定 → resolve confirmed 且不确定性下降 >0。
    7. degraded 视场 → 不居中（仅降高），不崩。
    P5 升级接口：
    8. entropy_uncertainty：均匀分布熵最大(≈1)，one-hot 分布熵最小(≈0)。
    9. uncertainty_score(mode="entropy")：给了 class_probs 用校准熵；没给自动退化 heuristic。
    10. info_gain_descend：离 alt_min 越远，降高复核的期望信息增益越大；已在 alt_min 时增益为 0。
    11. trigger_mode="info_gain" 与 uncertainty_mode="entropy" 组合在 assess() 里能跑通，
        且与默认 heuristic/threshold 模式互不干扰（同一输入下两种模式都不崩）。

运行：`python backend/tests/test_recheck.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recheck import (  # noqa: E402
    RecheckConfig,
    RecheckController,
    best_evidence,
    entropy_uncertainty,
    info_gain_descend,
    uncertainty_score,
)

LAT, LON = 31.2304, 121.4737


def _det(cls: str, conf: float, bbox=None) -> dict:
    return {"class_name": cls, "conf": conf, "bbox": bbox or [10, 10, 30, 30]}


def test_uncertainty_score() -> None:
    u_low = uncertainty_score("low", 0.3, True)
    u_high = uncertainty_score("high", 0.9, True)
    assert u_low > u_high, (u_low, u_high)
    assert uncertainty_score("any", 0.0, False) == 0.0
    print(f"[OK] 不确定性：low/低conf={u_low} > high/高conf={u_high}")


def test_best_evidence() -> None:
    dets = [
        _det("无损伤建筑", 0.95),       # 非证据，忽略
        _det("严重损伤建筑", 0.4),
        _det("完全损毁建筑", 0.6),
    ]
    conf, cls, _, probs = best_evidence(dets)
    assert cls == "完全损毁建筑" and abs(conf - 0.6) < 1e-6, (cls, conf)
    assert probs is None, probs  # 未附加 class_probs（VLN_CHANGE_PERCEPTION 关闭）
    assert best_evidence([_det("无损伤建筑", 0.9)]) == (0.0, "", None, None)
    print(f"[OK] best_evidence: {cls}@{conf}")


def test_best_evidence_class_probs() -> None:
    """P5：class_probs 字段（perception.py 在 VLN_CHANGE_PERCEPTION=1 时才会附加）随最佳证据回传。"""
    det = _det("完全损毁建筑", 0.6)
    det["class_probs"] = {"no-damage": 0.05, "minor-damage": 0.05, "major-damage": 0.1, "destroyed": 0.8}
    conf, cls, _, probs = best_evidence([det])
    assert cls == "完全损毁建筑" and probs == det["class_probs"], probs
    print(f"[OK] best_evidence 回传 class_probs: {probs}")


def test_entropy_uncertainty() -> None:
    uniform = {"no-damage": 0.25, "minor-damage": 0.25, "major-damage": 0.25, "destroyed": 0.25}
    one_hot = {"no-damage": 0.0, "minor-damage": 0.0, "major-damage": 0.0, "destroyed": 1.0}
    h_uniform = entropy_uncertainty(uniform)
    h_onehot = entropy_uncertainty(one_hot)
    assert h_uniform > 0.95, h_uniform  # 均匀分布 → 归一化熵≈1（最不确定）
    assert h_onehot < 0.05, h_onehot    # one-hot → 归一化熵≈0（最确定）
    assert h_uniform > h_onehot
    print(f"[OK] entropy_uncertainty: uniform={h_uniform} one_hot={h_onehot}")


def test_uncertainty_score_entropy_mode() -> None:
    probs_uncertain = {"no-damage": 0.3, "minor-damage": 0.3, "major-damage": 0.2, "destroyed": 0.2}
    probs_confident = {"no-damage": 0.02, "minor-damage": 0.02, "major-damage": 0.02, "destroyed": 0.94}
    u1 = uncertainty_score("low", 0.3, True, class_probs=probs_uncertain, mode="entropy")
    u2 = uncertainty_score("low", 0.3, True, class_probs=probs_confident, mode="entropy")
    assert u1 > u2, (u1, u2)
    # 没给 class_probs → 自动退化 heuristic（数值应等于纯 heuristic 调用）
    u_fallback = uncertainty_score("low", 0.3, True, class_probs=None, mode="entropy")
    u_heuristic = uncertainty_score("low", 0.3, True)
    assert u_fallback == u_heuristic, (u_fallback, u_heuristic)
    # has_evidence=False 时恒 0，与 mode 无关
    assert uncertainty_score("any", 0.0, False, class_probs=probs_uncertain, mode="entropy") == 0.0
    print(f"[OK] entropy 模式：不确定分布={u1} > 确定分布={u2}；无 class_probs 退化={u_fallback}")


def test_info_gain_descend() -> None:
    g_far = info_gain_descend(entropy_now=0.8, alt=120.0, descend_step_m=20.0, alt_min_m=30.0)
    g_near_floor = info_gain_descend(entropy_now=0.8, alt=35.0, descend_step_m=20.0, alt_min_m=30.0)
    g_at_floor = info_gain_descend(entropy_now=0.8, alt=30.0, descend_step_m=20.0, alt_min_m=30.0)
    assert g_far > g_near_floor >= 0, (g_far, g_near_floor)
    assert g_at_floor == 0.0, g_at_floor  # 已到高度下限，降不动 → 期望增益为 0
    print(f"[OK] info_gain_descend：远离下限={g_far} > 靠近下限={g_near_floor} > 已到下限={g_at_floor}")


def test_trigger_recheck() -> None:
    ctl = RecheckController()
    out = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="low",
        detections=[_det("严重损伤建筑", 0.35, bbox=[70, 10, 90, 30])],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert out.kind == "recheck", out.kind
    assert out.params["up_m"] < 0, "应降高度"
    assert out.params["east_m"] > 0, "目标在右(东) → 应朝东居中"
    assert out.count == 1
    print(f"[OK] 触发复核：{out.params}（{out.reason}）")


def test_skip_when_confident() -> None:
    ctl = RecheckController()
    out = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="high",
        detections=[_det("完全损毁建筑", 0.9)],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert out.kind == "skip", out.kind
    print("[OK] 高置信 → 跳过复核（不浪费步数）")


def test_budget_exhausted_resolves() -> None:
    ctl = RecheckController(RecheckConfig(max_rechecks=2))
    kw = dict(
        lat=LAT, lon=LON, risk_level="low",
        detections=[_det("严重损伤建筑", 0.3)],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    o1 = ctl.assess(alt=120.0, **kw)
    o2 = ctl.assess(alt=100.0, **kw)
    o3 = ctl.assess(alt=80.0, **kw)
    assert o1.kind == "recheck" and o2.kind == "recheck", (o1.kind, o2.kind)
    assert o3.kind == "resolve", o3.kind
    assert o3.status in ("inconclusive", "confirmed")
    assert o3.reduction is not None
    print(f"[OK] 预算耗尽 → 定论 {o3.status}，不确定性下降 {o3.reduction}")


def test_altitude_floor() -> None:
    ctl = RecheckController(RecheckConfig(alt_min_m=30.0, descend_step_m=20.0))
    # alt 已在下限 → 直接定论，不再降
    out = ctl.assess(
        lat=LAT, lon=LON, alt=30.0, risk_level="low",
        detections=[_det("严重损伤建筑", 0.3)],
        patch_radius_m=40.0, patch_width=100, patch_height=100,
    )
    assert out.kind == "resolve", out.kind
    assert "高度下限" in out.reason
    print("[OK] 到高度下限 → 不再降高，直接定论")


def test_improvement_resolves_confirmed() -> None:
    ctl = RecheckController()
    # 第一次：可疑低置信 → 复核
    o1 = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="low",
        detections=[_det("严重损伤建筑", 0.3)],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert o1.kind == "recheck"
    # 第二次（降高后看清）：risk 升级 high + 高 conf → 把握足够 → resolve confirmed
    o2 = ctl.assess(
        lat=LAT, lon=LON, alt=100.0, risk_level="high",
        detections=[_det("完全损毁建筑", 0.85)],
        patch_radius_m=50.0, patch_width=100, patch_height=100,
    )
    assert o2.kind == "resolve" and o2.status == "confirmed", (o2.kind, o2.status)
    assert o2.reduction is not None and o2.reduction > 0, o2.reduction
    print(f"[OK] 改善路径：复核 → confirmed，不确定性下降 {o2.reduction}")
    assert ctl.stats()["confirmed"] == 1


def test_assess_entropy_info_gain_modes() -> None:
    """P5：uncertainty_mode=entropy + trigger_mode=info_gain 组合能跑通，且与默认
    heuristic/threshold 模式行为方向一致（低置信触发复核，高置信跳过）。"""
    # 显式指定 alt_min_m/descend_step_m（而不是依赖默认值），因为本测试要验证的是
    # "越接近高度下限，继续降高的期望信息增益越小"这一相对关系，与默认值具体是多少无关。
    cfg = RecheckConfig(
        uncertainty_mode="entropy", trigger_mode="info_gain", min_info_gain=0.02,
        alt_min_m=30.0, descend_step_m=20.0,
    )
    ctl = RecheckController(cfg)

    uncertain_det = _det("严重损伤建筑", 0.35, bbox=[70, 10, 90, 30])
    uncertain_det["class_probs"] = {
        "no-damage": 0.3, "minor-damage": 0.3, "major-damage": 0.25, "destroyed": 0.15,
    }
    out = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="low",
        detections=[uncertain_det],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert out.kind == "recheck", (out.kind, out.reason)

    # 已经很接近高度下限（32m，alt_min=30m）→ 即使还有一点不确定性，
    # 继续降高能带来的期望信息增益也很小 → info_gain 模式应判定不值得复核。
    ctl2 = RecheckController(cfg)
    confident_det = _det("完全损毁建筑", 0.9)
    confident_det["class_probs"] = {
        "no-damage": 0.01, "minor-damage": 0.01, "major-damage": 0.03, "destroyed": 0.95,
    }
    out2 = ctl2.assess(
        lat=LAT, lon=LON, alt=32.0, risk_level="high",
        detections=[confident_det],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert out2.kind == "skip", (out2.kind, out2.reason)
    print(f"[OK] entropy+info_gain 组合：不确定→{out.kind}，确定→{out2.kind}")


def test_degraded_no_recenter() -> None:
    ctl = RecheckController()
    out = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="low",
        detections=[_det("严重损伤建筑", 0.3, bbox=[70, 10, 90, 30])],
        patch_radius_m=60.0, patch_width=100, patch_height=100, degraded=True,
    )
    assert out.kind == "recheck"
    assert out.params["north_m"] == 0.0 and out.params["east_m"] == 0.0, out.params
    assert out.params["up_m"] < 0, "degraded 仍应降高"
    print("[OK] degraded → 仅降高、不居中")


def test_trigger_mode_fixed_always_rechecks() -> None:
    """E11 基线 trigger_mode='fixed'：只要有可疑证据就必复核，不看不确定性数值——
    即使 risk='high' 且 conf 很高（heuristic 模式下 unc 会很低、threshold 模式会 skip），
    fixed 模式仍应触发复核。"""
    cfg = RecheckConfig(trigger_mode="fixed")
    ctl = RecheckController(cfg)
    out = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="high",
        detections=[_det("完全损毁建筑", 0.95, bbox=[70, 10, 90, 30])],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert out.kind == "recheck", (out.kind, out.reason)
    print("[OK] trigger_mode=fixed：高置信证据仍强制复核")


def test_trigger_mode_random_reproducible() -> None:
    """E11 基线 trigger_mode='random'：同一 random_seed 下行为可复现（两次独立
    controller 用相同 seed，对同一输入序列应产生完全相同的 skip/recheck 序列）。"""
    def _make_ctl() -> RecheckController:
        return RecheckController(RecheckConfig(trigger_mode="random", random_prob=0.5, random_seed=7))

    det = _det("严重损伤建筑", 0.5, bbox=[70, 10, 90, 30])
    kinds_a = []
    kinds_b = []
    for ctl, out_list in ((_make_ctl(), kinds_a), (_make_ctl(), kinds_b)):
        for _ in range(5):
            out = ctl.assess(
                lat=LAT, lon=LON, alt=120.0, risk_level="low",
                detections=[det], patch_radius_m=60.0, patch_width=100, patch_height=100,
            )
            out_list.append(out.kind)
    assert kinds_a == kinds_b, (kinds_a, kinds_b)
    print(f"[OK] trigger_mode=random：seed=7 两次运行序列一致 {kinds_a}")


def test_finalize_flushes_pending_reduction() -> None:
    """回归测试（对应 E11 实测发现的 ΔU≈0 统计漏洞）：episode 在复核循环走到正式
    定论之前就结束（到达终点/步数耗尽）时，finalize() 应该把这个"未收尾"的位置
    也按最新一次观测补记进 avg_uncertainty_reduction，而不是让它从统计里消失。"""
    ctl = RecheckController()
    # 第一次：可疑低置信 → 触发复核（此时还没到 resolve，episode 假设到这里就结束了）。
    out1 = ctl.assess(
        lat=LAT, lon=LON, alt=120.0, risk_level="low",
        detections=[_det("严重损伤建筑", 0.3)],
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    assert out1.kind == "recheck", out1.kind
    # 复核修复前：episode 结束时直接调 stats()，pending 的位置从未进 resolved_log。
    stats_before_finalize = ctl.stats()
    assert stats_before_finalize["resolved"] == 0
    assert stats_before_finalize["avg_uncertainty_reduction"] == 0.0
    assert stats_before_finalize["pending"] == 1
    # 修复后：episode 结束时先 finalize()，pending 位置补记一笔账（status=episode_end），
    # 不计入 confirmed/dismissed/inconclusive，但计入 avg_uncertainty_reduction。
    ctl.finalize()
    stats_after = ctl.stats()
    assert stats_after["resolved"] == 1, stats_after
    assert stats_after["episode_end_pending"] == 1, stats_after
    assert stats_after["confirmed"] == 0 and stats_after["dismissed"] == 0 and stats_after["inconclusive"] == 0
    assert stats_after["pending"] == 0
    # 只观测了一次，before==after（还没有降高复核带来的新观测），下降量应为 0，
    # 但关键是这次已经被记进了统计分母里，不再是"凭空消失"。
    assert ctl.resolved_log[-1]["status"] == "episode_end"
    print(f"[OK] finalize() 补记未收尾的复核：{stats_before_finalize} → {stats_after}")


def test_finalize_uses_latest_observation() -> None:
    """finalize() 应该用"最新一次观测"而不是首次触发时的不确定性来算下降量——
    模拟"复核了一次、把握有所提升但还没触发 resolve 判定"的场景。"""
    ctl = RecheckController(RecheckConfig(max_rechecks=3))
    kw = dict(
        lat=LAT, lon=LON,
        patch_radius_m=60.0, patch_width=100, patch_height=100,
    )
    o1 = ctl.assess(alt=120.0, risk_level="low",
                     detections=[_det("严重损伤建筑", 0.3)], **kw)
    assert o1.kind == "recheck"
    # 降高后看得更清楚一点，但还没到"把握足够"或"预算耗尽"的定论条件。
    o2 = ctl.assess(alt=100.0, risk_level="low",
                     detections=[_det("严重损伤建筑", 0.5)], **kw)
    assert o2.kind == "recheck", o2.kind
    unc_after_o2 = o2.uncertainty
    ctl.finalize()
    entry = ctl.resolved_log[-1]
    assert entry["status"] == "episode_end"
    assert abs(entry["uncertainty_after"] - unc_after_o2) < 1e-9, (entry, unc_after_o2)
    assert entry["uncertainty_before"] == o1.uncertainty
    print(f"[OK] finalize() 用最新观测算下降量：{entry['uncertainty_before']} → {entry['uncertainty_after']}")


def _run_all() -> int:
    tests = [
        test_uncertainty_score,
        test_best_evidence,
        test_best_evidence_class_probs,
        test_entropy_uncertainty,
        test_uncertainty_score_entropy_mode,
        test_info_gain_descend,
        test_trigger_recheck,
        test_skip_when_confident,
        test_budget_exhausted_resolves,
        test_altitude_floor,
        test_improvement_resolves_confirmed,
        test_assess_entropy_info_gain_modes,
        test_degraded_no_recenter,
        test_trigger_mode_fixed_always_rechecks,
        test_trigger_mode_random_reproducible,
        test_finalize_flushes_pending_reduction,
        test_finalize_uses_latest_observation,
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
    print(f"\n{'='*48}\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
