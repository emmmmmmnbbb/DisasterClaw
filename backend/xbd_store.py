"""
backend/xbd_store.py — 封装 xBD manifest 加载、缓存、过滤与查找。

上层路由（app.py 中的 /api/xbd/*）只需调用这里的 load_cached / filter_catalog /
get_entry / summary。根据 mtime 自动刷新，避免 manifest.json 更新后还在读旧内容。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from xbd_map import load_manifest, resolve_output_dir  # noqa: F401  (re-exported for app.py)

logger = logging.getLogger(__name__)


# 全局开关：只保留灾后 (post_disaster) 瓦片，过滤掉所有 pre_disaster。
# 可通过环境变量 XBD_POST_ONLY=0 临时关闭（默认 True）。
POST_ONLY_MODE: bool = os.environ.get("XBD_POST_ONLY", "1").strip().lower() not in {"0", "false", "no"}


def _is_post(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    return str(item.get("stage") or "").lower() in {"post_disaster", "post"}


def _is_pre(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    return str(item.get("stage") or "").lower() in {"pre_disaster", "pre"}


def _stage_matches(item: dict[str, Any] | None, stage: str | None) -> bool:
    """Alias-aware stage equality.

    manifest 里的 stage 是 ``post`` / ``pre``；上层调用方经常传
    ``post_disaster`` / ``pre_disaster``。这里统一吸收别名，避免 ``!=`` 比较
    永远为真的静默 bug。
    """
    if stage is None:
        return True
    s = str(stage).lower()
    if s in {"post", "post_disaster"}:
        return _is_post(item)
    if s in {"pre", "pre_disaster"}:
        return _is_pre(item)
    return str(item.get("stage") or "").lower() == s


_cache_lock = threading.Lock()
_cache: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "manifest": None,
    "index": {},
}


def get_manifest_path() -> str:
    """解析当前 manifest.json 路径，允许通过 XBD_MANIFEST_PATH 覆盖。"""
    override = os.environ.get("XBD_MANIFEST_PATH")
    if override:
        return str(Path(override).expanduser().resolve())
    return str(resolve_output_dir() / "manifest.json")


def get_footprints_path() -> str:
    """与 manifest 并置的 footprints.geojson。"""
    override = os.environ.get("XBD_FOOTPRINTS_PATH")
    if override:
        return str(Path(override).expanduser().resolve())
    return str(resolve_output_dir() / "footprints.geojson")


def load_cached() -> tuple[dict[str, Any] | None, str]:
    """
    返回 (manifest_dict, manifest_path)。
    manifest 文件不存在时返回 (None, path)，由上层决定报错。
    """
    manifest_path = get_manifest_path()
    with _cache_lock:
        try:
            stat = os.stat(manifest_path)
        except FileNotFoundError:
            _cache["manifest"] = None
            _cache["index"] = {}
            _cache["path"] = manifest_path
            _cache["mtime"] = None
            return None, manifest_path

        mtime = stat.st_mtime
        if (
            _cache["manifest"] is not None
            and _cache["path"] == manifest_path
            and _cache["mtime"] == mtime
        ):
            return _cache["manifest"], manifest_path

        logger.info("reloading xBD manifest: %s", manifest_path)
        manifest = load_manifest(manifest_path)
        index: dict[str, dict[str, Any]] = {}
        for item in manifest.get("items", []):
            tile_id = item.get("tile_id")
            if tile_id:
                index[tile_id] = item

        _cache["manifest"] = manifest
        _cache["index"] = index
        _cache["path"] = manifest_path
        _cache["mtime"] = mtime
        return manifest, manifest_path


def get_entry(tile_id: str) -> dict[str, Any] | None:
    load_cached()
    with _cache_lock:
        return _cache["index"].get(tile_id)


def summary() -> dict[str, Any]:
    manifest, _ = load_cached()
    if not manifest:
        return {}
    return manifest.get("summary", {})


def dataset_root() -> str | None:
    manifest, _ = load_cached()
    if not manifest:
        return None
    return manifest.get("dataset_root")


def filter_catalog(
    split: str | None = None,
    disaster: str | None = None,
    disaster_type: str | None = None,
    stage: str | None = None,
    georef: bool | None = None,
    offset: int = 0,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], int]:
    """按条件过滤 items，返回 (分页后的items, 过滤后总数)。"""
    manifest, _ = load_cached()
    if not manifest:
        return [], 0

    items = manifest.get("items", [])
    filtered: list[dict[str, Any]] = []
    for item in items:
        if POST_ONLY_MODE and not _is_post(item):
            continue
        if split and item.get("split") != split:
            continue
        if disaster and item.get("disaster") != disaster:
            continue
        if disaster_type and item.get("disaster_type") != disaster_type:
            continue
        if stage and not _stage_matches(item, stage):
            continue
        if georef is not None and bool(item.get("has_georef")) != georef:
            continue
        filtered.append(item)

    total = len(filtered)
    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = 1
    page = filtered[offset:offset + limit]
    return page, total


def first_georef_entry(
    disaster: str | None = None,
    stage: str | None = None,
) -> dict[str, Any] | None:
    """
    用于启动时挑选默认激活瓦片。
    若未指定 disaster/stage，则取 manifest 里第一个 has_georef 的条目。
    """
    manifest, _ = load_cached()
    if not manifest:
        return None

    for item in manifest.get("items", []):
        if not item.get("has_georef"):
            continue
        if POST_ONLY_MODE and not _is_post(item):
            continue
        if disaster and item.get("disaster") != disaster:
            continue
        if stage and not _stage_matches(item, stage):
            continue
        return item

    # 回退：忽略 disaster/stage 再试一次（仍然受 POST_ONLY_MODE 约束）
    if disaster or stage:
        for item in manifest.get("items", []):
            if not item.get("has_georef"):
                continue
            if POST_ONLY_MODE and not _is_post(item):
                continue
            return item
    return None


def tile_contains(entry: dict[str, Any] | None, lat: float, lon: float) -> bool:
    """
    判断给定 (lat, lon) 是否落在瓦片外接 bbox 内。

    优先读 entry["bounds"] 的 north/south/east/west；
    若没有 bounds，再尝试用 entry["corner_coordinates"] 四点求包围盒。
    未经纬度标定 (无 has_georef) 的瓦片直接返回 False。
    """
    if not entry:
        return False
    if not entry.get("has_georef"):
        return False

    bounds = entry.get("bounds") or {}
    north = bounds.get("north")
    south = bounds.get("south")
    east = bounds.get("east")
    west = bounds.get("west")

    if north is None or south is None or east is None or west is None:
        corners = entry.get("corner_coordinates") or []
        if not corners:
            return False
        lats = [float(c["lat"]) for c in corners if "lat" in c]
        lons = [float(c["lon"]) for c in corners if "lon" in c]
        if not lats or not lons:
            return False
        north, south = max(lats), min(lats)
        east, west = max(lons), min(lons)

    try:
        nf = float(north); sf = float(south); ef = float(east); wf = float(west)
    except (TypeError, ValueError):
        return False

    return sf <= lat <= nf and wf <= lon <= ef


def find_tile_containing(
    lat: float,
    lon: float,
    *,
    disaster: str | None = None,
    stage_priority: tuple[str, ...] = ("post_disaster", "pre_disaster"),
    require_georef: bool = True,
) -> dict[str, Any] | None:
    """
    在 manifest.items 中搜索第一个覆盖 (lat, lon) 的瓦片。

    - 按 stage_priority 顺序优先匹配（默认 post 优先 pre 兜底）。
    - disaster 非空时只在该灾情内搜索；否则整个 manifest 搜。
    - 命中不到返回 None。
    """
    manifest, _ = load_cached()
    if not manifest:
        return None
    items = manifest.get("items", [])

    def _iter(stage: str | None):
        for item in items:
            if require_georef and not item.get("has_georef"):
                continue
            if POST_ONLY_MODE and not _is_post(item):
                continue
            if disaster and item.get("disaster") != disaster:
                continue
            if not _stage_matches(item, stage):
                continue
            yield item

    for stage in stage_priority:
        for item in _iter(stage):
            if tile_contains(item, lat, lon):
                return item

    if POST_ONLY_MODE:
        # POST-only 模式下不再 fallback 到其它 stage
        return None

    # stage_priority 之外的 stage 也再扫一遍（防御性 fallback）
    for item in items:
        if require_georef and not item.get("has_georef"):
            continue
        if disaster and item.get("disaster") != disaster:
            continue
        if any(_stage_matches(item, s) for s in stage_priority):
            continue
        if tile_contains(item, lat, lon):
            return item
    return None


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None
