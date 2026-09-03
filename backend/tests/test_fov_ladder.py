"""backend/tests/test_fov_ladder.py — 视场收缩阶梯的几何不变量 (计划 §2.5)

这些断言锁死改动一的核心科学声明：
  - 下限视场 = 恰好一整张 xBD 瓦片
  - 下限有效 GSD = 原生 0.5 m/px（信息天花板是**推导出来的**，不是设定的）
  - 降高严格提升 ROI 单位目标像素数
  - 旧的合成模糊阶梯不得复活
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fov_ladder as FL  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


def test_floor_fov_is_exactly_one_tile():
    """硬不变量：最小视场 = 一整张瓦片 (512 m)。"""
    alt = FL.alt_min_m()
    assert FL.span_m_for_alt(alt) == pytest.approx(FL.TILE_SPAN_M, rel=1e-9)
    assert FL.span_tiles_for_alt(alt) == pytest.approx(1.0, rel=1e-9)


def test_floor_gsd_equals_native():
    """信息天花板 = 源影像原生 GSD，且是被几何推导出来的。"""
    assert FL.eff_gsd_for_alt(FL.alt_min_m()) == pytest.approx(FL.NATIVE_GSD_M, abs=1e-12)
    assert FL.resample_ratio(FL.alt_min_m()) == pytest.approx(1.0, abs=1e-12)


def test_cruise_is_three_tiles_and_1p5m_gsd():
    alt = FL.alt_cruise_m()
    assert FL.span_tiles_for_alt(alt) == pytest.approx(3.0, rel=1e-9)
    assert FL.eff_gsd_for_alt(alt) == pytest.approx(1.5, rel=1e-9)


def test_altitudes_are_physically_plausible():
    """review2 B2 的后半句：真实低空飞行给的是厘米级 GSD。

    重标定后 0.5 m/px 对应 443 m，属固定翼应急侦察的真实作业高度，
    不再出现「10 m 高度 / 0.5 m 分辨率」这种差两个数量级的设定。
    """
    assert 300.0 < FL.alt_min_m() < 600.0
    assert 1000.0 < FL.alt_cruise_m() < 1800.0


def test_gsd_strictly_increases_with_altitude():
    alts = [FL.alt_min_m() + i * 50.0 for i in range(18)]
    gsds = [FL.eff_gsd_for_alt(a) for a in alts]
    assert all(b > a for a, b in zip(gsds, gsds[1:]))


def test_descending_strictly_increases_roi_pixels():
    """降高必须严格提升 ROI 的单位目标像素数 —— 这就是「真实信息增益」。"""
    fr_cruise = FL.roi_pixel_fraction(FL.alt_cruise_m())
    fr_mid = FL.roi_pixel_fraction(FL.alt_for_span_tiles(2.0))
    fr_floor = FL.roi_pixel_fraction(FL.alt_min_m())
    assert fr_cruise == pytest.approx(1 / 3, rel=1e-9)
    assert fr_mid == pytest.approx(0.5, rel=1e-9)
    assert fr_floor == pytest.approx(1.0, rel=1e-9)
    assert fr_cruise < fr_mid < fr_floor


def test_ladder_points_span_full_range():
    pts = FL.ladder_points(3)
    assert [p["span_tiles"] for p in pts] == [3.0, 2.0, 1.0]
    assert [p["gsd_m"] for p in pts] == [1.5, 1.0, 0.5]
    assert pts[-1]["resample_ratio"] == pytest.approx(1.0)


def test_alt_span_roundtrip():
    for n in (1.0, 1.5, 2.0, 2.5, 3.0):
        assert FL.span_tiles_for_alt(FL.alt_for_span_tiles(n)) == pytest.approx(n, rel=1e-9)


def test_clamp_alt_bounds():
    assert FL.clamp_alt(10.0) == pytest.approx(FL.alt_min_m())
    assert FL.clamp_alt(99999.0) == pytest.approx(FL.alt_cruise_m())


def test_descend_step_divides_range():
    step = FL.descend_step_m(2)
    assert FL.alt_cruise_m() - 2 * step == pytest.approx(FL.alt_min_m(), rel=1e-9)


def test_recheck_altitude_invariant_holds():
    """app.py:907-914 记录过的历史 bug：巡航高度 == alt_min 会让 recheck 恒真。

    重标定后必须仍然容得下 max_rechecks 步下降。
    """
    assert FL.alt_cruise_m() > FL.alt_min_m()
    step = FL.descend_step_m(2)
    assert step > 1.0
    assert FL.alt_cruise_m() - step > FL.alt_min_m()


def test_synthetic_blur_ladder_is_confined_to_legacy_path():
    """`degrade_to_scale` 只允许出现在 legacy_crop 回退路径里 (计划 §2.5 第 6 条)。

    P2 已完成：`perceive_at` 默认走 `_render_uav_view`（mosaic_fov 观测模型），
    合成模糊仅在 `MOSAIC_VIEW=0` 的旧裁块路径下可达，用于复现旧产物。
    这里断言它没有泄漏到其他模块。
    """
    allowed = {"perception.py"}  # 仅 _crop_uav_view（legacy_crop）内部
    offenders = set()
    for py in BACKEND.glob("*.py"):
        if py.name == "gsd_ladder.py":
            continue  # 旧模块本体保留只读，供历史产物复现
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "degrade_to_scale(" in stripped:
                offenders.add(py.name)
                break
    assert offenders <= allowed, f"合成模糊阶梯泄漏到了非 legacy 路径: {sorted(offenders - allowed)}"


def test_default_observation_model_is_mosaic_fov():
    """默认观测模型必须是视场收缩，不能是合成模糊阶梯。"""
    import perception

    assert perception.MOSAIC_VIEW is True
    assert hasattr(perception, "_render_uav_view") or hasattr(
        perception.DisasterPerception, "_render_uav_view"
    )


def test_entropy_table_rejects_stale_synthetic_schema(tmp_path):
    """旧熵表是在合成模糊上拟合的，必须拒绝，不得静默复用。"""
    import json

    p = tmp_path / "stale.json"
    p.write_text(json.dumps({"schema": "gsd-ladder/1.0", "bins": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        FL.ExpectedEntropyTable.load(p)


def test_entropy_table_fails_closed_when_missing_or_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        FL.ExpectedEntropyTable.load(tmp_path / "missing.json")
    empty = tmp_path / "empty.json"
    empty.write_text(
        '{"schema":"fov-ladder-entropy/1.0","bins":[]}', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no fitted bins"):
        FL.ExpectedEntropyTable.load(empty)


def test_fov_entropy_table_schema_uses_cruise_floor_not_legacy_scales():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"))
    from eval_fov_ladder import build_entropy_table

    items = [{
        "views": {
            "cruise": {"entropy": 0.9, "pred": "no-damage"},
            "mid": {"entropy": 0.6, "pred": "minor-damage"},
            "floor": {"entropy": 0.2, "pred": "destroyed"},
        }
    }]
    table = build_entropy_table(items, ["hurricane-harvey"])
    assert table["schema"] == "fov-ladder-entropy/1.0"
    views = {row["view"] for row in table["bins"]}
    assert views == {"cruise", "mid", "floor"}
    assert "1.0" not in views and "4.0" not in views
