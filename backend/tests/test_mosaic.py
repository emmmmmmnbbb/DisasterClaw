"""backend/tests/test_mosaic.py — 马赛克合成的几何与信息边界断言 (计划 §2.5)

不联网：用桩 basemap 与人造 manifest，验证仿射、覆盖掩码、nodata 处理、
ROI 约束与双时相配准。真实底图抓取由 `test_basemap.py` 单独标记。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fov_ladder as FL  # noqa: E402
import mosaic as M  # noqa: E402
from basemap import WindowGeo, window_from_center_span  # noqa: E402


class StubBasemap:
    """返回纯品红背景，便于一眼分辨「哪些像素来自底图」。"""

    provider = {"provider": "stub", "max_zoom": 19}

    def __init__(self):
        self.calls = 0

    def render_window(self, window, out_px, zoom=None):
        self.calls += 1
        return Image.new("RGB", (out_px, out_px), (255, 0, 255))


TILE_PX = 1024
GSD = 0.5
# 1024 px @ 0.5 m/px，放在赤道附近简化经纬换算
LAT0, LON0 = 0.0, 0.0
DLAT = (TILE_PX * GSD) / 110_540.0
DLON = (TILE_PX * GSD) / 111_320.0


def _make_entry(tile_id, i, j, stage="post"):
    """构造一个理想网格瓦片：第 (i,j) 格。"""
    west = LON0 + j * DLON
    east = west + DLON
    north = LAT0 - i * DLAT
    south = north - DLAT
    # pixel->geo: lon = west + x/TILE*DLON ; lat = north - y/TILE*DLAT
    a = DLON / TILE_PX
    e = -DLAT / TILE_PX
    return {
        "tile_id": tile_id, "stage": stage, "disaster": "unit-test",
        "has_georef": True, "gsd": GSD, "width": TILE_PX, "height": TILE_PX,
        "image_relpath": f"{tile_id}.png",
        "bounds": {"west": west, "east": east, "south": south, "north": north},
        "pixel_to_geo": {"lon": [a, 0.0, west], "lat": [0.0, e, north]},
        "geo_to_pixel": {"x": [1.0 / a, 0.0, -west / a], "y": [0.0, 1.0 / e, -north / e]},
    }


@pytest.fixture()
def grid_mosaic(tmp_path):
    """3×3 理想网格，中心格 (1,1) 作 ROI。"""
    entries = []
    for i in range(3):
        for j in range(3):
            tid = f"t_{i}{j}"
            entries.append(_make_entry(tid, i, j))
            # 每格填不同灰度，便于定位
            Image.new("RGB", (TILE_PX, TILE_PX), (10 + 20 * (i * 3 + j), 128, 64)).save(
                tmp_path / f"{tid}.png"
            )
    mo = M.TileMosaic(entries, dataset_root=str(tmp_path), basemap=StubBasemap(),
                      harmonize=False)
    return mo, entries


def _center_of(entry):
    b = entry["bounds"]
    return (b["north"] + b["south"]) / 2, (b["east"] + b["west"]) / 2


def test_floor_window_matches_roi_tile_bounds(grid_mosaic):
    """§2.5-1：下限高度渲染的窗口与 ROI 瓦片地理范围重合，IoU > 0.99。"""
    mo, entries = grid_mosaic
    roi = entries[4]  # (1,1)
    clat, clon = _center_of(roi)
    _, meta = mo.render_for_alt(clat, clon, FL.alt_min_m(), roi_tile_id=roi["tile_id"])
    w, b = meta.window, roi["bounds"]
    inter = (
        max(0.0, min(w["east"], b["east"]) - max(w["west"], b["west"]))
        * max(0.0, min(w["north"], b["north"]) - max(w["south"], b["south"]))
    )
    union = (
        (w["east"] - w["west"]) * (w["north"] - w["south"])
        + (b["east"] - b["west"]) * (b["north"] - b["south"]) - inter
    )
    assert inter / union > 0.99


def test_floor_gsd_is_native(grid_mosaic):
    """§2.5-2：下限有效 GSD == 原生。"""
    mo, entries = grid_mosaic
    clat, clon = _center_of(entries[4])
    _, meta = mo.render_for_alt(clat, clon, FL.alt_min_m())
    assert meta.eff_gsd_m == pytest.approx(FL.NATIVE_GSD_M, rel=1e-6)
    assert meta.resample_ratio == pytest.approx(1.0, rel=1e-6)


def test_full_grid_has_no_basemap_bleed(grid_mosaic):
    """3×3 全覆盖时 xbd_fraction==1，且输出中不得出现底图品红。"""
    mo, entries = grid_mosaic
    clat, clon = _center_of(entries[4])
    img, meta = mo.render_for_alt(clat, clon, FL.alt_cruise_m(),
                                  roi_tile_id=entries[4]["tile_id"])
    assert meta.xbd_fraction == pytest.approx(1.0, abs=1e-3)
    a = np.asarray(img)
    magenta = (a[..., 0] > 250) & (a[..., 1] < 5) & (a[..., 2] > 250)
    assert magenta.mean() < 1e-4


def test_basemap_fills_holes(tmp_path):
    """§2.5-3：只有中心一格时，四周必须由底图填充且被如实计入 xbd_fraction。"""
    e = _make_entry("solo", 1, 1)
    Image.new("RGB", (TILE_PX, TILE_PX), (20, 120, 60)).save(tmp_path / "solo.png")
    mo = M.TileMosaic([e], dataset_root=str(tmp_path), basemap=StubBasemap(),
                      harmonize=False)
    clat, clon = _center_of(e)
    img, meta = mo.render_for_alt(clat, clon, FL.alt_cruise_m(), roi_tile_id="solo")
    # 视场 3×3 瓦片，只有中心 1 格是真实影像 → 约 1/9
    assert meta.xbd_fraction == pytest.approx(1 / 9, abs=0.02)
    a = np.asarray(img)
    magenta = (a[..., 0] > 250) & (a[..., 1] < 5) & (a[..., 2] > 250)
    assert magenta.mean() == pytest.approx(8 / 9, abs=0.02)


def test_nodata_pixels_let_basemap_through(tmp_path):
    """xBD 瓦片自带黑色 nodata（实测各事件均值 1.5%–11%），
    必须让底图透出来，且不得被记为「有 xBD 覆盖」。"""
    e = _make_entry("nd", 1, 1)
    arr = np.full((TILE_PX, TILE_PX, 3), 90, dtype=np.uint8)
    arr[:, : TILE_PX // 2] = 0  # 左半 nodata
    Image.fromarray(arr).save(tmp_path / "nd.png")
    mo = M.TileMosaic([e], dataset_root=str(tmp_path), basemap=StubBasemap(),
                      harmonize=False)
    clat, clon = _center_of(e)
    img, meta = mo.render_for_alt(clat, clon, FL.alt_min_m(), enforce_roi=False)
    assert meta.xbd_fraction == pytest.approx(0.5, abs=0.02)
    a = np.asarray(img)
    assert (a.sum(axis=2) == 0).mean() < 1e-3  # 输出里不应残留纯黑


def test_roi_coverage_enforced(tmp_path):
    """ROI 必须 100% 真实 xBD；被底图渗入时显式报错，不静默通过。"""
    e = _make_entry("nd", 1, 1)
    arr = np.full((TILE_PX, TILE_PX, 3), 90, dtype=np.uint8)
    arr[:, : TILE_PX // 2] = 0
    Image.fromarray(arr).save(tmp_path / "nd.png")
    mo = M.TileMosaic([e], dataset_root=str(tmp_path), basemap=StubBasemap(),
                      harmonize=False)
    clat, clon = _center_of(e)
    with pytest.raises(M.RoiCoverageError):
        mo.render_for_alt(clat, clon, FL.alt_min_m(), roi_tile_id="nd", enforce_roi=True)


def test_roi_norm_bbox_shrinks_to_full_frame_on_descent(grid_mosaic):
    """ROI 在画面中的归一化 bbox：巡航约占 1/3 边长，下限铺满。"""
    mo, entries = grid_mosaic
    roi = entries[4]
    clat, clon = _center_of(roi)
    _, m_cruise = mo.render_for_alt(clat, clon, FL.alt_cruise_m(), roi_tile_id=roi["tile_id"])
    _, m_floor = mo.render_for_alt(clat, clon, FL.alt_min_m(), roi_tile_id=roi["tile_id"])
    wc = m_cruise.roi_norm_bbox[2] - m_cruise.roi_norm_bbox[0]
    wf = m_floor.roi_norm_bbox[2] - m_floor.roi_norm_bbox[0]
    assert wc == pytest.approx(1 / 3, abs=0.02)
    assert wf == pytest.approx(1.0, abs=0.02)
    assert wc < wf


def test_pre_post_windows_are_identical(tmp_path):
    """§2.5-4：同一 (center, span) 的 pre/post 窗口地理边界严格相等 —— 双时相配准不破。"""
    entries = []
    for stage in ("pre", "post"):
        e = _make_entry(f"x_{stage}", 1, 1, stage=stage)
        entries.append(e)
        Image.new("RGB", (TILE_PX, TILE_PX), (30, 90, 30)).save(tmp_path / f"x_{stage}.png")
    mo = M.TileMosaic(entries, dataset_root=str(tmp_path), basemap=StubBasemap(),
                      harmonize=False)
    clat, clon = _center_of(entries[0])
    _, mp = mo.render_for_alt(clat, clon, FL.alt_min_m(), stage="pre", enforce_roi=False)
    _, mq = mo.render_for_alt(clat, clon, FL.alt_min_m(), stage="post", enforce_roi=False)
    assert mp.window == mq.window
    assert mp.eff_gsd_m == pytest.approx(mq.eff_gsd_m)


def test_geometric_coverage_matches_grid(grid_mosaic, tmp_path):
    """覆盖率索引：3×3 满格 → 1.0；孤立格 → 约 1/9。"""
    mo, entries = grid_mosaic
    assert mo.geometric_coverage(entries[4]["tile_id"]) == pytest.approx(1.0, abs=0.02)

    solo = _make_entry("solo", 1, 1)
    Image.new("RGB", (TILE_PX, TILE_PX), (20, 120, 60)).save(tmp_path / "solo.png")
    mo2 = M.TileMosaic([solo], dataset_root=str(tmp_path), basemap=StubBasemap())
    assert mo2.geometric_coverage("solo") == pytest.approx(1 / 9, abs=0.03)


def test_roi_candidates_respects_min_coverage(grid_mosaic, tmp_path):
    """作者决策：巡航真实覆盖 < 0.80 的场景不得进入 ROI 池。"""
    mo, entries = grid_mosaic
    assert entries[4]["tile_id"] in mo.roi_candidates(min_coverage=0.8)

    solo = _make_entry("solo", 1, 1)
    Image.new("RGB", (TILE_PX, TILE_PX), (20, 120, 60)).save(tmp_path / "solo.png")
    mo2 = M.TileMosaic([solo], dataset_root=str(tmp_path), basemap=StubBasemap())
    assert mo2.roi_candidates(min_coverage=0.8) == []
    assert mo2.roi_candidates(min_coverage=0.0) == ["solo"]


def test_descend_increases_roi_pixel_detail(grid_mosaic):
    """降高后 ROI 在输出图上占的像素数必须真实增加（不是重复放大同一批源像素）。"""
    mo, entries = grid_mosaic
    roi = entries[4]
    clat, clon = _center_of(roi)
    _, mc = mo.render_for_alt(clat, clon, FL.alt_cruise_m(), roi_tile_id=roi["tile_id"])
    _, mf = mo.render_for_alt(clat, clon, FL.alt_min_m(), roi_tile_id=roi["tile_id"])
    # 巡航时 ROI 被 3× 下采样，下限时 1:1
    assert mc.resample_ratio == pytest.approx(3.0, rel=1e-6)
    assert mf.resample_ratio == pytest.approx(1.0, rel=1e-6)
