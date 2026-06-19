"""
backend/tests/test_recheck.py — P2 灾情不确定性驱动主动复核 单测

验收点（对应 docs/vln_rescue_agent_实施计划.md 的 P2）：
    1. 不确定性评分：低 risk('low'/可疑) + 低 conf → 高；high + 高 conf → 低。
    2. best_evidence 取受灾相关检测里 conf 最高者，忽略完好建筑。
    3. 低置信/可疑场景 → 触发复核（降高 up_m<0 + 朝目标居中）。
    4. 高置信场景 → 不触发复核（不浪费步数）。
    5. 复核预算耗尽 / 到高度下限 → 收尾定论；降高不越过 alt_min。
    6. 改善路径：先复核 → 再观测变笃定 → resolve confirmed 且不确定性下降 >0。
    7. degraded 视场 → 不居中（仅降高），不崩。

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
    conf, cls, _ = best_evidence(dets)
    assert cls == "完全损毁建筑" and abs(conf - 0.6) < 1e-6, (cls, conf)
    assert best_evidence([_det("无损伤建筑", 0.9)]) == (0.0, "", None)
    print(f"[OK] best_evidence: {cls}@{conf}")


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


def _run_all() -> int:
    tests = [
        test_uncertainty_score,
        test_best_evidence,
        test_trigger_recheck,
        test_skip_when_confident,
        test_budget_exhausted_resolves,
        test_altitude_floor,
        test_improvement_resolves_confirmed,
        test_degraded_no_recenter,
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
