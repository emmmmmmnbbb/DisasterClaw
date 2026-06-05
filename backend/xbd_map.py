"""
backend/xbd_map.py — xBD/xView2 灾害地图数据工具

职责：
    - 解析 xBD label JSON
    - 从 xy ↔ lng_lat 控制点拟合像素到地理坐标的仿射变换
    - 生成可供后端 / 前端复用的 manifest
    - 按需输出 GeoJSON footprint / annotation

源自 AerialClaw/core/xbd_map.py，仅替换日志导入并把默认输出目录指到
disasterclaw/backend/data/xbd。
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PAIR_RE = re.compile(r"_(pre|post)_disaster$")


def resolve_dataset_root(dataset_root: str | os.PathLike[str] | None = None) -> Path:
    """
    解析 xBD 数据集根目录。

    优先级：
        1. 显式传入参数
        2. 环境变量 XBD_DATASET_ROOT
        3. ~/datasets/xbd
    """
    candidates: list[Path] = []
    if dataset_root:
        candidates.append(Path(dataset_root).expanduser())

    env_root = os.environ.get("XBD_DATASET_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(Path.home() / "datasets" / "xbd")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "xBD dataset root not found. Set XBD_DATASET_ROOT or pass --dataset-root."
    )


def resolve_output_dir(output_dir: str | os.PathLike[str] | None = None) -> Path:
    """解析 manifest 输出目录。

    默认指到 ``disasterclaw/backend/data/xbd``；若环境变量 ``DATASET_MODE`` 被
    设为 ``rescuenet``（或其他子目录名），则指到 ``data/<mode>``。整个 xbd_store
    / perception 管线 schema 不变，只是数据源替换成 tune7b 的 4000×3000 航拍。
    """
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    mode = (os.environ.get("DATASET_MODE") or "xbd").strip().lower() or "xbd"
    return (Path(__file__).resolve().parent / "data" / mode).resolve()


def _parse_polygon_wkt(wkt: str) -> list[tuple[float, float]]:
    """解析单环 POLYGON WKT。"""
    text = (wkt or "").strip()
    if not text.startswith("POLYGON"):
        return []

    start = text.find("((")
    end = text.rfind("))")
    if start < 0 or end < 0 or end <= start + 2:
        return []

    rings_text = text[start + 2:end]
    outer_ring = re.split(r"\)\s*,\s*\(", rings_text, maxsplit=1)[0]
    coords: list[tuple[float, float]] = []
    for pair in outer_ring.split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords.pop()
    return coords


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _feature_map(features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for feature in features:
        uid = (((feature or {}).get("properties") or {}).get("uid") or "").strip()
        if uid:
            out[uid] = feature
    return out


def _stage_from_tile_id(tile_id: str) -> str:
    if tile_id.endswith("_pre_disaster"):
        return "pre"
    if tile_id.endswith("_post_disaster"):
        return "post"
    return "unknown"


def pair_group_id(tile_id: str) -> str:
    """将 pre/post 成对图像归到同一个 group。"""
    return _PAIR_RE.sub("", tile_id)


def _local_error_m(
    pred_lon: np.ndarray,
    pred_lat: np.ndarray,
    ref_lon: np.ndarray,
    ref_lat: np.ndarray,
) -> np.ndarray:
    """局部近似将经纬度误差换算为米。"""
    lat0 = np.deg2rad((pred_lat + ref_lat) * 0.5)
    dx = (pred_lon - ref_lon) * 111_320.0 * np.cos(lat0)
    dy = (pred_lat - ref_lat) * 110_540.0
    return np.hypot(dx, dy)


def _fit_affine(
    xy_points: list[tuple[float, float]],
    geo_points: list[tuple[float, float]],
) -> dict[str, Any] | None:
    """
    拟合像素到经纬度的仿射变换：
        lon = a*x + b*y + c
        lat = d*x + e*y + f
    """
    if len(xy_points) < 3 or len(geo_points) < 3 or len(xy_points) != len(geo_points):
        return None

    xy = np.asarray(xy_points, dtype=float)
    geo = np.asarray(geo_points, dtype=float)

    if np.linalg.matrix_rank(np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])) < 3:
        return None

    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    lon_coeffs, _, _, _ = np.linalg.lstsq(design, geo[:, 0], rcond=None)
    lat_coeffs, _, _, _ = np.linalg.lstsq(design, geo[:, 1], rcond=None)

    linear = np.array([
        [float(lon_coeffs[0]), float(lon_coeffs[1])],
        [float(lat_coeffs[0]), float(lat_coeffs[1])],
    ])
    det = float(np.linalg.det(linear))
    if abs(det) < 1e-12:
        return None

    pred_lon = design @ lon_coeffs
    pred_lat = design @ lat_coeffs
    residuals = _local_error_m(pred_lon, pred_lat, geo[:, 0], geo[:, 1])

    inverse = np.linalg.inv(linear)
    offset = np.array([float(lon_coeffs[2]), float(lat_coeffs[2])], dtype=float)
    geo_to_pixel_bias = -inverse @ offset

    return {
        "pixel_to_geo": {
            "lon": [float(v) for v in lon_coeffs.tolist()],
            "lat": [float(v) for v in lat_coeffs.tolist()],
        },
        "geo_to_pixel": {
            "x": [float(inverse[0, 0]), float(inverse[0, 1]), float(geo_to_pixel_bias[0])],
            "y": [float(inverse[1, 0]), float(inverse[1, 1]), float(geo_to_pixel_bias[1])],
        },
        "fit": {
            "point_count": int(len(xy_points)),
            "mean_error_m": float(np.mean(residuals)),
            "rms_error_m": float(np.sqrt(np.mean(np.square(residuals)))),
            "max_error_m": float(np.max(residuals)),
        },
    }


def pixel_to_geo(
    transform: dict[str, Any],
    x: float,
    y: float,
) -> tuple[float, float]:
    lon = transform["pixel_to_geo"]["lon"]
    lat = transform["pixel_to_geo"]["lat"]
    return (
        lon[0] * x + lon[1] * y + lon[2],
        lat[0] * x + lat[1] * y + lat[2],
    )


def geo_to_pixel(
    transform: dict[str, Any],
    lon: float,
    lat: float,
) -> tuple[float, float]:
    x_coeffs = transform["geo_to_pixel"]["x"]
    y_coeffs = transform["geo_to_pixel"]["y"]
    return (
        x_coeffs[0] * lon + x_coeffs[1] * lat + x_coeffs[2],
        y_coeffs[0] * lon + y_coeffs[1] * lat + y_coeffs[2],
    )


def _entry_corners(width: int, height: int, transform: dict[str, Any]) -> list[dict[str, float]]:
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    corners = [
        ("top_left", 0.0, 0.0),
        ("top_right", max_x, 0.0),
        ("bottom_right", max_x, max_y),
        ("bottom_left", 0.0, max_y),
    ]
    out: list[dict[str, float]] = []
    for name, x, y in corners:
        lon, lat = pixel_to_geo(transform, x, y)
        out.append({
            "name": name,
            "x": x,
            "y": y,
            "lon": float(lon),
            "lat": float(lat),
        })
    return out


def _entry_bounds(corners: list[dict[str, float]]) -> dict[str, float]:
    lons = [c["lon"] for c in corners]
    lats = [c["lat"] for c in corners]
    return {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats),
    }


def _matched_control_points(label_data: dict[str, Any]) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int]:
    features = label_data.get("features") or {}
    xy_map = _feature_map(features.get("xy") or [])
    geo_map = _feature_map(features.get("lng_lat") or [])
    shared_uids = sorted(set(xy_map) & set(geo_map))

    xy_points: list[tuple[float, float]] = []
    geo_points: list[tuple[float, float]] = []
    matched_feature_count = 0

    for uid in shared_uids:
        xy_poly = _parse_polygon_wkt(xy_map[uid].get("wkt", ""))
        geo_poly = _parse_polygon_wkt(geo_map[uid].get("wkt", ""))
        if not xy_poly or not geo_poly:
            continue

        matched_feature_count += 1
        if len(xy_poly) == len(geo_poly):
            pairs = zip(xy_poly, geo_poly)
        else:
            pairs = [(_polygon_centroid(xy_poly), _polygon_centroid(geo_poly))]

        for (x, y), (lon, lat) in pairs:
            xy_points.append((float(x), float(y)))
            geo_points.append((float(lon), float(lat)))

    return xy_points, geo_points, matched_feature_count


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _entry_from_label(dataset_root: Path, split: str, label_path: Path) -> dict[str, Any]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}

    tile_id = label_path.stem
    width = int(metadata.get("width") or metadata.get("original_width") or 1024)
    height = int(metadata.get("height") or metadata.get("original_height") or 1024)

    image_name = metadata.get("img_name") or f"{tile_id}.png"
    image_path = dataset_root / split / "images" / image_name

    xy_points, geo_points, matched_feature_count = _matched_control_points(data)
    transform = _fit_affine(xy_points, geo_points)

    entry: dict[str, Any] = {
        "tile_id": tile_id,
        "group_id": pair_group_id(tile_id),
        "split": split,
        "stage": _stage_from_tile_id(tile_id),
        "image_name": image_name,
        "image_relpath": _relative_path(image_path, dataset_root),
        "label_relpath": _relative_path(label_path, dataset_root),
        "disaster": metadata.get("disaster"),
        "disaster_type": metadata.get("disaster_type"),
        "sensor": metadata.get("sensor"),
        "capture_date": metadata.get("capture_date"),
        "catalog_id": metadata.get("catalog_id"),
        "gsd": metadata.get("gsd"),
        "width": width,
        "height": height,
        "has_label_geo": bool(transform),
        "has_georef": bool(transform),
        "transform_source": "label_points" if transform else None,
        "matched_feature_count": matched_feature_count,
        "matched_point_count": len(xy_points),
        "pixel_to_geo": transform["pixel_to_geo"] if transform else None,
        "geo_to_pixel": transform["geo_to_pixel"] if transform else None,
        "fit": transform["fit"] if transform else None,
        "corner_coordinates": None,
        "bounds": None,
    }

    if transform:
        corners = _entry_corners(width, height, transform)
        entry["corner_coordinates"] = corners
        entry["bounds"] = _entry_bounds(corners)

    return entry


def _apply_pair_fallback(entries: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry["group_id"], []).append(entry)

    for group_entries in groups.values():
        tile_ids = [item["tile_id"] for item in group_entries]
        for entry in group_entries:
            peers = [tile_id for tile_id in tile_ids if tile_id != entry["tile_id"]]
            entry["paired_tile_ids"] = peers
            entry["paired_tile_id"] = peers[0] if len(peers) == 1 else None

    for group_entries in groups.values():
        source = next(
            (item for item in group_entries if item.get("transform_source") == "label_points"),
            None,
        )
        if not source:
            continue
        for target in group_entries:
            if target.get("has_georef"):
                continue
            if source.get("width") != target.get("width") or source.get("height") != target.get("height"):
                continue
            target["has_georef"] = True
            target["transform_source"] = "paired_tile"
            target["pixel_to_geo"] = source.get("pixel_to_geo")
            target["geo_to_pixel"] = source.get("geo_to_pixel")
            target["corner_coordinates"] = source.get("corner_coordinates")
            target["bounds"] = source.get("bounds")
            target["fit"] = {
                **(source.get("fit") or {}),
                "copied_from": source["tile_id"],
            }


def build_manifest(
    dataset_root: str | os.PathLike[str] | None = None,
    splits: list[str] | None = None,
    disasters: set[str] | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """扫描 xBD 数据集并生成 manifest 字典。"""
    root = resolve_dataset_root(dataset_root)
    selected_splits = splits or [name for name in ("train", "test", "tier3") if (root / name).exists()]

    entries: list[dict[str, Any]] = []
    processed = 0
    for split in selected_splits:
        labels_dir = root / split / "labels"
        if not labels_dir.exists():
            logger.warning("xBD split missing labels dir: %s", labels_dir)
            continue

        for label_path in sorted(labels_dir.glob("*.json")):
            if max_items is not None and processed >= max_items:
                break

            entry = _entry_from_label(root, split, label_path)
            if disasters and (entry.get("disaster") not in disasters):
                continue

            entries.append(entry)
            processed += 1

        if max_items is not None and processed >= max_items:
            break

    _apply_pair_fallback(entries)

    by_split: dict[str, dict[str, int]] = {}
    by_disaster: dict[str, dict[str, Any]] = {}
    by_disaster_type: dict[str, dict[str, Any]] = {}
    for entry in entries:
        bucket = by_split.setdefault(entry["split"], {
            "tiles": 0,
            "has_label_geo": 0,
            "has_georef": 0,
            "paired_fallback": 0,
        })
        bucket["tiles"] += 1
        bucket["has_label_geo"] += int(bool(entry.get("has_label_geo")))
        bucket["has_georef"] += int(bool(entry.get("has_georef")))
        bucket["paired_fallback"] += int(entry.get("transform_source") == "paired_tile")

        disaster = entry.get("disaster") or "unknown"
        disaster_bucket = by_disaster.setdefault(disaster, {
            "tiles": 0,
            "has_georef": 0,
            "sample_tile_id": None,
            "splits": {},
            "stages": {},
            "disaster_type": entry.get("disaster_type") or "unknown",
        })
        disaster_bucket["tiles"] += 1
        disaster_bucket["has_georef"] += int(bool(entry.get("has_georef")))
        if disaster_bucket["sample_tile_id"] is None and entry.get("has_georef"):
            disaster_bucket["sample_tile_id"] = entry.get("tile_id")
        disaster_bucket["splits"][entry["split"]] = disaster_bucket["splits"].get(entry["split"], 0) + 1
        disaster_bucket["stages"][entry["stage"]] = disaster_bucket["stages"].get(entry["stage"], 0) + 1

        disaster_type = entry.get("disaster_type") or "unknown"
        disaster_type_bucket = by_disaster_type.setdefault(disaster_type, {
            "tiles": 0,
            "has_georef": 0,
            "disasters": {},
        })
        disaster_type_bucket["tiles"] += 1
        disaster_type_bucket["has_georef"] += int(bool(entry.get("has_georef")))
        disaster_type_bucket["disasters"][disaster] = disaster_type_bucket["disasters"].get(disaster, 0) + 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "splits": selected_splits,
        "summary": {
            "tiles": len(entries),
            "has_label_geo": sum(1 for item in entries if item.get("has_label_geo")),
            "has_georef": sum(1 for item in entries if item.get("has_georef")),
            "paired_fallback": sum(1 for item in entries if item.get("transform_source") == "paired_tile"),
            "by_split": by_split,
            "by_disaster": by_disaster,
            "by_disaster_type": by_disaster_type,
        },
        "items": entries,
    }
    return manifest


def manifest_to_footprints_geojson(manifest: dict[str, Any]) -> dict[str, Any]:
    """将 manifest 转成 footprint GeoJSON。"""
    features: list[dict[str, Any]] = []
    for entry in manifest.get("items", []):
        corners = entry.get("corner_coordinates") or []
        if not corners:
            continue
        ring = [[corner["lon"], corner["lat"]] for corner in corners]
        if ring:
            ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {
                "tile_id": entry.get("tile_id"),
                "split": entry.get("split"),
                "stage": entry.get("stage"),
                "disaster": entry.get("disaster"),
                "disaster_type": entry.get("disaster_type"),
                "transform_source": entry.get("transform_source"),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
        })
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def build_annotation_geojson(label_path: str | os.PathLike[str], entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    输出单张图的 GeoJSON 标注。

    优先直接使用 `features.lng_lat`；
    如果没有经纬度标注但 entry 有 georef，则把 `features.xy` 投影到地理坐标。
    """
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    features = data.get("features") or {}
    geo_features = features.get("lng_lat") or []
    xy_features = features.get("xy") or []

    out_features: list[dict[str, Any]] = []

    if geo_features:
        for feature in geo_features:
            ring = _parse_polygon_wkt(feature.get("wkt", ""))
            if not ring:
                continue
            coords = [[lon, lat] for lon, lat in ring]
            coords.append(coords[0])
            out_features.append({
                "type": "Feature",
                "properties": {
                    **(feature.get("properties") or {}),
                    "geometry_source": "lng_lat",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })
    elif entry and entry.get("has_georef") and xy_features:
        transform = {
            "pixel_to_geo": entry.get("pixel_to_geo"),
            "geo_to_pixel": entry.get("geo_to_pixel"),
        }
        for feature in xy_features:
            ring = _parse_polygon_wkt(feature.get("wkt", ""))
            if not ring:
                continue
            coords = []
            for x, y in ring:
                lon, lat = pixel_to_geo(transform, x, y)
                coords.append([lon, lat])
            coords.append(coords[0])
            out_features.append({
                "type": "Feature",
                "properties": {
                    **(feature.get("properties") or {}),
                    "geometry_source": "xy_projected",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })

    return {
        "type": "FeatureCollection",
        "features": out_features,
    }


def write_manifest_bundle(
    manifest: dict[str, Any],
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """将 manifest 和 footprint GeoJSON 写入输出目录。"""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    footprints_path = out_dir / "footprints.geojson"

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    footprints_path.write_text(
        json.dumps(manifest_to_footprints_geojson(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "manifest": manifest_path,
        "footprints": footprints_path,
    }


def load_manifest(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """读取 manifest.json。"""
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
