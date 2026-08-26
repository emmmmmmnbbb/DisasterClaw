"""backend/mosaic.py — [Esri 背景 + xBD 瓦片] 地理窗口合成 (改动一核心)

把前端 `SituationMap.jsx` 已经在做的事（xBD 瓦片地理配准后压在 Esri 卫星底图上）
挪到服务端，并从「一次叠一张」推广到「叠窗口内全部瓦片」。

与旧 `perception._crop_uav_view()` 的关键区别::

    旧: 从单张瓦片裁一个像素方块 → 用合成高斯模糊阶梯降质
        视场几乎不随高度变化（MIN_PATCH_PX 下限恒定生效），
        唯一真实变化量是人工模糊强度。

    新: 按 fov_ladder 算出地面足迹 → 从合成底图上取该足迹 → 重采样到固定 1024 px
        降高 = 足迹收缩 = 重采样比下降 = 单位目标像素数真实提升。
        无任何人工模糊。

ROI-scoped 约束（计划 §2.3）::
    - 题目锚定在中心那张有 xBD 标注的 post 瓦片（ROI）。
    - **ROI 区域必须 100% 由真实 xBD 像素覆盖**，绝不允许 Esri 背景渗入 ——
      `render()` 会校验并在违反时抛错。
    - ROI 以地理 bbox 告知智能体；**不得**在图像上画高亮框。
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

import fov_ladder as FL
from basemap import BasemapTiles, WindowGeo, window_from_center_span, zoom_for_gsd

logger = logging.getLogger(__name__)

MOSAIC_HARMONIZE = os.getenv("MOSAIC_HARMONIZE", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
MOSAIC_LRU_SIZE = int(os.getenv("MOSAIC_LRU_SIZE", "32"))
# ROI 覆盖率低于此值即报错。ROI 必须是真实 xBD 影像。
ROI_MIN_XBD_FRACTION = float(os.getenv("ROI_MIN_XBD_FRACTION", "0.999"))


class RoiCoverageError(RuntimeError):
    """ROI 区域未被真实 xBD 像素完全覆盖。"""


@dataclass
class MosaicMeta:
    """一次渲染的完整溯源信息，进 PerceptionResult 与 benchmark 产物。"""
    window: dict = field(default_factory=dict)
    span_m: float = 0.0
    span_tiles: float = 0.0
    alt_m: float = 0.0
    eff_gsd_m: float = 0.0
    resample_ratio: float = 1.0
    out_px: int = FL.SENSOR_PX
    stage: str = "post"
    xbd_fraction: float = 0.0
    contributing_tile_ids: list[str] = field(default_factory=list)
    roi_tile_id: str = ""
    roi_norm_bbox: Optional[list[float]] = None
    roi_xbd_fraction: Optional[float] = None
    basemap_provider: str = ""
    basemap_zoom: int = 0
    harmonized: bool = False

    def to_dict(self) -> dict:
        return {
            "window": self.window, "span_m": round(self.span_m, 3),
            "span_tiles": round(self.span_tiles, 4), "alt_m": round(self.alt_m, 3),
            "eff_gsd_m": round(self.eff_gsd_m, 6),
            "resample_ratio": round(self.resample_ratio, 4),
            "out_px": self.out_px, "stage": self.stage,
            "xbd_fraction": round(self.xbd_fraction, 6),
            "contributing_tile_ids": list(self.contributing_tile_ids),
            "roi_tile_id": self.roi_tile_id,
            "roi_norm_bbox": self.roi_norm_bbox,
            "roi_xbd_fraction": (
                None if self.roi_xbd_fraction is None else round(self.roi_xbd_fraction, 6)
            ),
            "basemap_provider": self.basemap_provider,
            "basemap_zoom": self.basemap_zoom,
            "harmonized": self.harmonized,
        }


def _bounds_intersect(a: dict, b: WindowGeo) -> bool:
    return not (
        float(a["east"]) < b.west or float(a["west"]) > b.east
        or float(a["north"]) < b.south or float(a["south"]) > b.north
    )


def _window_affine_to_tile(
    window: WindowGeo, out_px: int, geo_to_pixel: dict, reduce_k: int = 1
) -> tuple[float, float, float, float, float, float]:
    """输出像素 (u,v) → 瓦片像素 (x,y) 的仿射系数。

    输出网格与瓦片仿射都是线性的，所以复合仍是仿射，可直接交给
    ``Image.transform(..., Image.AFFINE, ...)``（它期望的正是
    输出→输入 的映射）。
    """
    sw = (window.east - window.west) / out_px          # 每输出像素的经度增量
    sh = (window.north - window.south) / out_px        # 每输出像素的纬度减量
    gx = geo_to_pixel["x"]
    gy = geo_to_pixel["y"]
    lon0 = window.west + 0.5 * sw
    lat0 = window.north - 0.5 * sh
    k = float(reduce_k)
    return (
        gx[0] * sw / k,
        -gx[1] * sh / k,
        (gx[0] * lon0 + gx[1] * lat0 + gx[2]) / k,
        gy[0] * sw / k,
        -gy[1] * sh / k,
        (gy[0] * lon0 + gy[1] * lat0 + gy[2]) / k,
    )


def _harmonize(bg: Image.Image, xbd: Image.Image, mask: Image.Image) -> Image.Image:
    """把 Esri 背景的逐通道均值/方差匹配到窗口内 xBD 像素的统计量。

    只做线性匹配，不做直方图规定化 —— 后者会破坏损伤纹理，
    而损伤纹理正是任务信号本身。
    """
    m = np.asarray(mask, dtype=np.uint8) > 127
    if m.sum() < 64 or (~m).sum() < 64:
        return bg
    a_bg = np.asarray(bg, dtype=np.float32)
    a_xb = np.asarray(xbd, dtype=np.float32)
    out = a_bg.copy()
    for c in range(3):
        src = a_bg[..., c][m]      # 背景在「被 xBD 覆盖处」的统计
        dst = a_xb[..., c][m]      # xBD 自身的统计
        s_src, s_dst = src.std(), dst.std()
        if s_src < 1e-3:
            out[..., c] = a_bg[..., c] - src.mean() + dst.mean()
        else:
            out[..., c] = (a_bg[..., c] - src.mean()) * (s_dst / s_src) + dst.mean()
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


class TileMosaic:
    """按地理窗口合成 [Esri 背景 + xBD 瓦片] 的虚拟大图。"""

    def __init__(
        self,
        entries: Iterable[dict],
        dataset_root: str | os.PathLike[str],
        basemap: Optional[BasemapTiles] = None,
        harmonize: bool = MOSAIC_HARMONIZE,
    ):
        self.dataset_root = str(dataset_root)
        self.basemap = basemap or BasemapTiles()
        self.harmonize = bool(harmonize)
        self._by_stage: dict[str, list[dict]] = {"pre": [], "post": []}
        self._by_id: dict[str, dict] = {}
        for e in entries:
            if not e.get("has_georef") or not e.get("bounds"):
                continue
            stage = e.get("stage")
            if stage in self._by_stage:
                self._by_stage[stage].append(e)
            self._by_id[e.get("tile_id", "")] = e
        self._lock = threading.Lock()
        self._lru: OrderedDict[tuple, tuple[Image.Image, MosaicMeta]] = OrderedDict()

    # ── 场景池 ──────────────────────────────────────────────────────────────

    def geometric_coverage(
        self, tile_id: str, span_m: Optional[float] = None,
        stage: str = "post", grid: int = 64,
    ) -> float:
        """瓦片为中心的窗口被**真实 xBD 影像足迹**覆盖的面积比（几何近似）。

        只用 manifest 的 bounds 做栅格化，不读图，因此很快；不含 nodata 效应
        （nodata 由 `render()` 的掩码精确处理，且实测各事件均值仅 1.5%–11%）。
        """
        entry = self._by_id.get(tile_id)
        if not entry or not entry.get("bounds"):
            return 0.0
        span = float(span_m if span_m is not None else FL.span_m_for_alt(FL.alt_cruise_m()))
        b = entry["bounds"]
        clat = (float(b["north"]) + float(b["south"])) * 0.5
        clon = (float(b["east"]) + float(b["west"])) * 0.5
        w = window_from_center_span(clat, clon, span)
        # 用栅格**单元中心**采样估面积；linspace 含两端会把窗口外沿也算进去，
        # 导致满格网格算出 62/64 这类偏低值。
        gx = w.west + (w.east - w.west) * (np.arange(grid) + 0.5) / grid
        gy = w.north - (w.north - w.south) * (np.arange(grid) + 0.5) / grid
        occ = np.zeros((grid, grid), dtype=bool)
        for o in self._by_stage.get(stage, []):
            ob = o["bounds"]
            if (float(ob["east"]) < w.west or float(ob["west"]) > w.east
                    or float(ob["north"]) < w.south or float(ob["south"]) > w.north):
                continue
            occ |= (
                (gx[None, :] >= float(ob["west"])) & (gx[None, :] <= float(ob["east"]))
                & (gy[:, None] <= float(ob["north"])) & (gy[:, None] >= float(ob["south"]))
            )
        return float(occ.mean())

    def roi_candidates(
        self,
        disaster: Optional[str] = None,
        min_coverage: float = 0.0,
        span_m: Optional[float] = None,
        coverage_index: Optional[dict] = None,
    ) -> list[str]:
        """可作 ROI 的瓦片：有 georef 的 post 瓦片，且巡航视场真实覆盖达标。

        ROI-scoped 设定下**不要求**邻接瓦片有标注 —— 这是它相对 FOV-scoped 的
        关键优势（计划 §2.3）。但作者决策要求巡航视场的真实 xBD 覆盖 >= 80%，
        以免半个画面是异日期的灾前底图（实测全量均值仅 0.50）。

        `coverage_index` 传入 `build_roi_index()` 的产物可避免重算。
        """
        out = []
        for e in self._by_stage["post"]:
            if disaster and e.get("disaster") != disaster:
                continue
            tid = e["tile_id"]
            if min_coverage > 0.0:
                cov = (
                    coverage_index.get(tid) if coverage_index is not None
                    else self.geometric_coverage(tid, span_m=span_m)
                )
                if cov is None or float(cov) < min_coverage:
                    continue
            out.append(tid)
        return sorted(out)

    def build_roi_index(
        self, span_m: Optional[float] = None, stage: str = "post", grid: int = 64,
    ) -> dict[str, float]:
        """为全部 post 瓦片预计算巡航视场覆盖率。落盘后供题库生成复用。"""
        return {
            e["tile_id"]: round(
                self.geometric_coverage(e["tile_id"], span_m=span_m, stage=stage, grid=grid), 6
            )
            for e in self._by_stage.get(stage, [])
        }


    def get_entry(self, tile_id: str) -> Optional[dict]:
        return self._by_id.get(tile_id)

    # ── 渲染 ────────────────────────────────────────────────────────────────

    def render_for_alt(
        self,
        center_lat: float,
        center_lon: float,
        alt_m: float,
        stage: str = "post",
        out_px: int = FL.SENSOR_PX,
        roi_tile_id: str = "",
        enforce_roi: bool = True,
    ) -> tuple[Image.Image, MosaicMeta]:
        """按高度渲染观测图。足迹由 `fov_ladder` 决定。"""
        alt = FL.clamp_alt(alt_m)
        span_m = FL.span_m_for_alt(alt)
        img, meta = self.render(
            center_lat, center_lon, span_m, stage=stage, out_px=out_px,
            roi_tile_id=roi_tile_id, enforce_roi=enforce_roi,
        )
        meta.alt_m = alt
        meta.span_tiles = FL.span_tiles_for_alt(alt)
        meta.resample_ratio = FL.resample_ratio(alt, out_px)
        return img, meta

    def render(
        self,
        center_lat: float,
        center_lon: float,
        span_m: float,
        stage: str = "post",
        out_px: int = FL.SENSOR_PX,
        roi_tile_id: str = "",
        enforce_roi: bool = True,
    ) -> tuple[Image.Image, MosaicMeta]:
        """渲染一个地理窗口。

        顺序：Esri 背景 → 叠 xBD 瓦片 → 色调协调 → 合成。
        """
        key = (
            round(center_lat, 7), round(center_lon, 7), round(float(span_m), 3),
            stage, int(out_px), roi_tile_id,
        )
        with self._lock:
            hit = self._lru.get(key)
            if hit is not None:
                self._lru.move_to_end(key)
                return hit[0].copy(), hit[1]

        window = window_from_center_span(center_lat, center_lon, span_m)
        eff_gsd = float(span_m) / float(out_px)

        # 背景 zoom 选到不粗于目标 GSD，避免背景比 xBD 明显糊。
        z = min(
            zoom_for_gsd(eff_gsd, center_lat, self.basemap.provider.get("max_zoom", 19)),
            int(self.basemap.provider.get("max_zoom", 19)),
        )
        bg = self.basemap.render_window(window, out_px, zoom=z)

        # xBD 图层 + 覆盖掩码
        xbd_layer = Image.new("RGB", (out_px, out_px))
        cover = Image.new("L", (out_px, out_px), 0)
        contributing: list[str] = []

        for entry in self._by_stage.get(stage, []):
            if not _bounds_intersect(entry["bounds"], window):
                continue
            path = os.path.join(self.dataset_root, entry.get("image_relpath") or "")
            if not os.path.isfile(path):
                continue
            try:
                with Image.open(path) as im:
                    im.load()
                    tile = im.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Mosaic] 瓦片读取失败 %s: %s", path, exc)
                continue

            # 下采样超过 1.5× 时先做整数 reduce，避免 AFFINE 双线性混叠。
            tile_gsd = float(entry.get("gsd") or FL.NATIVE_GSD_M)
            k = 1
            if eff_gsd > tile_gsd * 1.5:
                k = max(1, int(eff_gsd / tile_gsd))

            # xBD 瓦片自带黑色 nodata 区（实测各事件均值 1.5%–11%，
            # 个别瓦片可达 30%+）。这些像素不是观测，必须让底图透出来，
            # 否则会在观测图上留下大片纯黑，且被错记为「有 xBD 覆盖」。
            valid = Image.fromarray(
                ((np.asarray(tile, dtype=np.uint16).sum(axis=2) > 0) * 255).astype(np.uint8)
            ).convert("L")
            if k > 1:
                tile = tile.reduce(k)
                # reduce 取均值；要求整个 k×k 块全有效才算有效（保守，避免边缘渗黑）
                valid = valid.reduce(k).point(lambda p: 255 if p >= 255 else 0)

            coeffs = _window_affine_to_tile(window, out_px, entry["geo_to_pixel"], reduce_k=k)
            warped = tile.transform(
                (out_px, out_px), Image.AFFINE, coeffs, resample=Image.BILINEAR
            )
            tile_mask = valid.transform(
                (out_px, out_px), Image.AFFINE, coeffs, resample=Image.NEAREST, fillcolor=0
            )
            xbd_layer.paste(warped, (0, 0), tile_mask)
            cover.paste(255, (0, 0), tile_mask)
            contributing.append(entry["tile_id"])

        cover_arr = np.asarray(cover, dtype=np.uint8) > 127
        xbd_fraction = float(cover_arr.mean())

        if self.harmonize and xbd_fraction > 0.0:
            bg = _harmonize(bg, xbd_layer, cover)

        out = bg.copy()
        out.paste(xbd_layer, (0, 0), cover)

        meta = MosaicMeta(
            window={
                "west": window.west, "south": window.south,
                "east": window.east, "north": window.north,
            },
            span_m=float(span_m),
            span_tiles=float(span_m) / FL.TILE_SPAN_M,
            eff_gsd_m=eff_gsd,
            resample_ratio=eff_gsd / FL.NATIVE_GSD_M,
            out_px=int(out_px),
            stage=stage,
            xbd_fraction=xbd_fraction,
            contributing_tile_ids=sorted(contributing),
            basemap_provider=self.basemap.provider.get("provider", ""),
            basemap_zoom=z,
            harmonized=bool(self.harmonize and xbd_fraction > 0.0),
        )

        if roi_tile_id:
            self._attach_roi(meta, roi_tile_id, window, out_px, cover_arr, enforce_roi)

        with self._lock:
            self._lru[key] = (out.copy(), meta)
            while len(self._lru) > MOSAIC_LRU_SIZE:
                self._lru.popitem(last=False)
        return out, meta

    def _attach_roi(
        self, meta: MosaicMeta, roi_tile_id: str, window: WindowGeo,
        out_px: int, cover_arr: np.ndarray, enforce: bool,
    ) -> None:
        entry = self._by_id.get(roi_tile_id)
        if not entry or not entry.get("bounds"):
            raise ValueError(f"ROI tile not in manifest or lacks bounds: {roi_tile_id}")
        b = entry["bounds"]
        sw = (window.east - window.west) / out_px
        sh = (window.north - window.south) / out_px
        u0 = (float(b["west"]) - window.west) / sw
        u1 = (float(b["east"]) - window.west) / sw
        v0 = (window.north - float(b["north"])) / sh
        v1 = (window.north - float(b["south"])) / sh
        meta.roi_tile_id = roi_tile_id
        meta.roi_norm_bbox = [
            round(u0 / out_px, 6), round(v0 / out_px, 6),
            round(u1 / out_px, 6), round(v1 / out_px, 6),
        ]

        iu0, iv0 = max(0, int(math.floor(u0))), max(0, int(math.floor(v0)))
        iu1, iv1 = min(out_px, int(math.ceil(u1))), min(out_px, int(math.ceil(v1)))
        if iu1 <= iu0 or iv1 <= iv0:
            meta.roi_xbd_fraction = 0.0
        else:
            meta.roi_xbd_fraction = float(cover_arr[iv0:iv1, iu0:iu1].mean())

        if enforce and (meta.roi_xbd_fraction or 0.0) < ROI_MIN_XBD_FRACTION:
            raise RoiCoverageError(
                f"ROI {roi_tile_id} 只有 {meta.roi_xbd_fraction:.4f} 的像素来自真实 xBD 影像 "
                f"(要求 >= {ROI_MIN_XBD_FRACTION})。ROI 不得被 Esri 背景渗入 —— "
                "所有 GT 与问题作用域都锚定在 ROI 上。"
            )


def from_manifest(
    manifest: dict, basemap: Optional[BasemapTiles] = None, **kwargs
) -> TileMosaic:
    """从 xbd_map manifest 构造 TileMosaic。"""
    return TileMosaic(
        entries=manifest.get("items") or [],
        dataset_root=manifest.get("dataset_root") or "",
        basemap=basemap,
        **kwargs,
    )
