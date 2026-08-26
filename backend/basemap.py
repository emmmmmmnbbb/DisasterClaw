"""backend/basemap.py — Esri World Imagery 瓦片抓取 / 缓存 / 按地理窗口拼接

与前端 `SituationMap.jsx` 的 `TileLayer` 同源（`world.DEFAULT_BASEMAP`），
把「xBD 瓦片叠在连续卫星底图上」这件前端已经在做的事挪到服务端，
用于渲染智能体的观测图像。

实测（2026-08-24，lat 30.7 / hurricane-michael 区域）：
    Esri World Imagery z=18 → 0.513 m/px，256×256 RGB，约 20 KB/瓦片
    xBD 原生                → 0.500 m/px
两者 GSD 差 2.6%，锐度相当，因此「智能体靠源锐度差异认出出题瓦片」这条
泄漏路径基本不成立。剩余的日期/色调差异由 `mosaic.harmonize` 压制。

信息边界（重要）:
    - 底图是**非事件当天**影像（多为灾前），在观测中仅作**上下文背景**。
    - 所有 GT 与问题作用域严格限制在 ROI 内的 xBD post 瓦片上。
    - **离线且缓存未命中时抛显式错误，绝不静默返回灰图** —— 否则会在评测中
      悄悄改变输入分布，属于计划 §12.3 禁止的静默降级。
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from geo import TILE_SIZE, latlon_to_world_pixels, normalize_tile_x, world_pixels_to_latlon

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent

DEFAULT_ZOOM = int(os.getenv("BASEMAP_ZOOM", "18"))
DEFAULT_CACHE_DIR = Path(
    os.getenv("BASEMAP_CACHE_DIR", str(BACKEND_DIR / "data" / "basemap_cache"))
).expanduser()
# 允许联网抓取未命中的瓦片。评测复现时应设为 0，强制只用归档缓存。
BASEMAP_ALLOW_FETCH = os.getenv("BASEMAP_ALLOW_FETCH", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
BASEMAP_TIMEOUT_S = float(os.getenv("BASEMAP_TIMEOUT_S", "20"))
BASEMAP_MAX_RETRIES = int(os.getenv("BASEMAP_MAX_RETRIES", "3"))
BASEMAP_USER_AGENT = os.getenv(
    "BASEMAP_USER_AGENT",
    "disasterclaw-research/1.0 (academic use; xBD damage assessment)",
)

ESRI_WORLD_IMAGERY = {
    "provider": "esri-world-imagery",
    # 注意 Esri 的 URL 是 {z}/{y}/{x}（行在列之前），与 OSM 的 {z}/{x}/{y} 不同。
    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "attribution": (
        "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
        "and the GIS User Community"
    ),
    "max_zoom": 19,
    "ext": "jpg",
}


class BasemapUnavailable(RuntimeError):
    """缓存未命中且不允许/无法联网抓取。显式失败，不静默降级。"""


@dataclass(frozen=True)
class WindowGeo:
    """一个地理窗口（Web Mercator 轴对齐）。"""
    west: float
    south: float
    east: float
    north: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) * 0.5, (self.west + self.east) * 0.5)


def window_from_center_span(center_lat: float, center_lon: float, span_m: float) -> WindowGeo:
    """以 (lat, lon) 为中心、边长 span_m 的正方形窗口（局部线性化）。

    在 512–1536 m 尺度、xBD 覆盖的纬度范围内，局部线性化误差远小于一个像素。
    """
    half = float(span_m) * 0.5
    dlat = half / 110_540.0
    dlon = half / (111_320.0 * max(math.cos(math.radians(center_lat)), 1e-6))
    return WindowGeo(
        west=center_lon - dlon,
        south=center_lat - dlat,
        east=center_lon + dlon,
        north=center_lat + dlat,
    )


def zoom_for_gsd(target_gsd_m: float, lat: float, max_zoom: int = 19) -> int:
    """选一个地面分辨率不粗于 target_gsd_m 的最小 zoom。"""
    for z in range(0, max_zoom + 1):
        gsd = 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)
        if gsd <= target_gsd_m:
            return z
    return max_zoom


def tile_ground_gsd_m(zoom: int, lat: float) -> float:
    """某 zoom 在给定纬度的地面分辨率（m/px）。"""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


class BasemapTiles:
    """XYZ 卫星底图瓦片的抓取、磁盘缓存与窗口拼接。"""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        provider: Optional[dict] = None,
        zoom: int = DEFAULT_ZOOM,
        allow_fetch: bool = BASEMAP_ALLOW_FETCH,
        fetch_workers: int = int(os.getenv("BASEMAP_FETCH_WORKERS", "16")),
    ):
        self.provider = dict(provider or ESRI_WORLD_IMAGERY)
        self.zoom = int(zoom)
        self.allow_fetch = bool(allow_fetch)
        self.fetch_workers = max(1, int(fetch_workers))
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[tuple[int, int, int], Image.Image] = {}
        self.stats = {"hit": 0, "miss": 0, "fetch": 0, "fail": 0}

    # ── 单瓦片 ──────────────────────────────────────────────────────────────

    def _cache_path(self, z: int, x: int, y: int) -> Path:
        ext = self.provider.get("ext", "jpg")
        return self.cache_dir / self.provider["provider"] / str(z) / str(x) / f"{y}.{ext}"

    def _tile_url(self, z: int, x: int, y: int) -> str:
        return (
            self.provider["url"]
            .replace("{z}", str(z))
            .replace("{x}", str(x))
            .replace("{y}", str(y))
        )

    def fetch_tile(self, z: int, x: int, y: int) -> Image.Image:
        """取一张瓦片。优先内存 → 磁盘 → 网络。"""
        key = (z, x, y)
        with self._lock:
            cached = self._mem.get(key)
        if cached is not None:
            self.stats["hit"] += 1
            return cached

        path = self._cache_path(z, x, y)
        if path.is_file():
            try:
                with Image.open(path) as im:
                    im.load()
                    img = im.convert("RGB")
                self.stats["hit"] += 1
                with self._lock:
                    self._mem[key] = img
                return img
            except Exception as exc:  # noqa: BLE001 — 损坏缓存则重抓
                logger.warning("[Basemap] 缓存瓦片损坏，将重抓 %s: %s", path, exc)
                path.unlink(missing_ok=True)

        self.stats["miss"] += 1
        if not self.allow_fetch:
            raise BasemapUnavailable(
                f"basemap tile {z}/{x}/{y} not in cache and BASEMAP_ALLOW_FETCH=0. "
                "评测复现模式下必须使用归档缓存；不得静默替换为灰图。"
            )

        data = self._http_get(self._tile_url(z, x, y))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            im.load()
            img = im.convert("RGB")
        self.stats["fetch"] += 1
        with self._lock:
            self._mem[key] = img
        return img

    def _http_get(self, url: str) -> bytes:
        last: Optional[Exception] = None
        for attempt in range(BASEMAP_MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": BASEMAP_USER_AGENT})
                with urllib.request.urlopen(req, timeout=BASEMAP_TIMEOUT_S) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt < BASEMAP_MAX_RETRIES - 1:
                    time.sleep(0.5 * (2 ** attempt))
        self.stats["fail"] += 1
        raise BasemapUnavailable(f"basemap fetch failed after {BASEMAP_MAX_RETRIES} tries: {url}") from last

    # ── 窗口拼接 ────────────────────────────────────────────────────────────

    def render_window(
        self,
        window: WindowGeo,
        out_px: int,
        zoom: Optional[int] = None,
    ) -> Image.Image:
        """把地理窗口渲染成 out_px × out_px 的 RGB 图。

        做法：在 Web Mercator 世界像素坐标里定位窗口 → 取覆盖它的所有瓦片 →
        拼成一张大图 → 按窗口精确裁剪 → 重采样到 out_px。
        """
        z = int(zoom if zoom is not None else self.zoom)

        # 窗口四角 → 世界像素。Mercator 下 y 随纬度单调，所以 north→top。
        x0, y0 = latlon_to_world_pixels(window.north, window.west, z)
        x1, y1 = latlon_to_world_pixels(window.south, window.east, z)
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)

        tx0 = int(math.floor(left / TILE_SIZE))
        tx1 = int(math.floor((right - 1e-9) / TILE_SIZE))
        ty0 = int(math.floor(top / TILE_SIZE))
        ty1 = int(math.floor((bottom - 1e-9) / TILE_SIZE))

        n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        if n_tiles > 4096:
            raise ValueError(
                f"basemap window needs {n_tiles} tiles at z={z}; refuse to fetch. "
                "降低 zoom 或缩小窗口。"
            )

        canvas_w = (tx1 - tx0 + 1) * TILE_SIZE
        canvas_h = (ty1 - ty0 + 1) * TILE_SIZE
        canvas = Image.new("RGB", (canvas_w, canvas_h))

        max_ty = (1 << z) - 1
        wanted = [
            (tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)
            if 0 <= ty <= max_ty
        ]

        # 冷缓存窗口要抓上百张瓦片（1536 m @ z18 约 12×12=144 张）；串行抓取
        # 实测单窗口要 ~50 s，会主导整个评测的墙钟时间。并发抓取只影响未命中的部分，
        # 命中缓存时线程池几乎零开销。
        missing = [
            (tx, ty) for (tx, ty) in wanted
            if not self._cache_path(z, normalize_tile_x(z, tx), ty).is_file()
        ]
        if len(missing) > 1 and self.allow_fetch:
            from concurrent.futures import ThreadPoolExecutor

            workers = min(self.fetch_workers, len(missing))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(
                    lambda t: self.fetch_tile(z, normalize_tile_x(z, t[0]), t[1]),
                    missing,
                ))

        for (tx, ty) in wanted:
            px = (tx - tx0) * TILE_SIZE
            py = (ty - ty0) * TILE_SIZE
            tile = self.fetch_tile(z, normalize_tile_x(z, tx), ty)
            if tile.size != (TILE_SIZE, TILE_SIZE):
                tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
            canvas.paste(tile, (px, py))

        crop = (
            left - tx0 * TILE_SIZE,
            top - ty0 * TILE_SIZE,
            right - tx0 * TILE_SIZE,
            bottom - ty0 * TILE_SIZE,
        )
        chip = canvas.resize((int(out_px), int(out_px)), Image.BILINEAR, box=crop)
        return chip

    # ── 溯源 ────────────────────────────────────────────────────────────────

    def provenance(self) -> dict:
        return {
            "provider": self.provider.get("provider"),
            "url_template": self.provider.get("url"),
            "attribution": self.provider.get("attribution"),
            "zoom": self.zoom,
            "cache_dir": str(self.cache_dir),
            "allow_fetch": self.allow_fetch,
            "stats": dict(self.stats),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

    def write_provenance(self, path: str | os.PathLike[str] | None = None) -> Path:
        out = Path(path) if path else (self.cache_dir / "provenance.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.provenance(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out
