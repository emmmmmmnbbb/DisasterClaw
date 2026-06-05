#!/usr/bin/env python
"""
build_rescuenet_dataset.py — 把 /home/lc/tune7b (RescueNet / 2018 Hurricane Michael
DJI 无人机航拍, 4000x3000) 转成 disasterclaw xBD 管线能吃的资源：

    backend/data/rescuenet/manifest.json          # xBD schema
    backend/data/rescuenet/footprints.geojson     # 每张图的 bbox 作 footprint
    backend/data/rescuenet/damage_ranking.json    # 重灾排名
    backend/data/rescuenet/labels/<id>_post.json  # xBD-style building polygons

之后只要设 DATASET_MODE=rescuenet 启动后端，xbd_store / perception / footprints
端到端就会走 tune7b 的 4000x3000 原图，而不是 xBD 的 1024x1024 卫星瓦片。

用法:
    python scripts/build_rescuenet_dataset.py                 # 默认全量
    python scripts/build_rescuenet_dataset.py --limit 200     # 调试
    python scripts/build_rescuenet_dataset.py --force         # 重算
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import ExifTags, Image


# --------------------------- 常量 / 路径 --------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNE7B_ROOT = Path("/home/lc/tune7b")
OUTPUT_DIR = REPO_ROOT / "backend" / "data" / "rescuenet"
LABELS_DIR = OUTPUT_DIR / "labels"

DISASTER_NAME = "hurricane-michael"
DISASTER_TYPE = "hurricane"
SENSOR = "DJI-DRONE-RGB"
CAPTURE_DATE = "2018-10-13T00:00:00.000Z"

# RescueNet detection classes (per paper + YOLO file convention observed in tune7b):
#   0: Building_No_Damage
#   1: Building_Minor_Damage
#   2: Building_Major_Damage
#   3: Building_Total_Destruction
#   4: Vehicle
#   5: Debris / Road-Blocked
DET_CLASS_TO_SUBTYPE: dict[int, tuple[str, str]] = {
    0: ("building", "no-damage"),
    1: ("building", "minor-damage"),
    2: ("building", "major-damage"),
    3: ("building", "destroyed"),
    4: ("vehicle", ""),
    5: ("debris", ""),
}

SEVERITY_WEIGHTS = {
    "destroyed": 5,
    "major-damage": 3,
    "minor-damage": 1,
    "no-damage": 0,
    "": 0,
}

# 假设的 DJI 无人机等效水平 FOV（Phantom 4 / Mavic 2 Pro ≈ 73°）；若 EXIF
# 能给 FocalLengthIn35mmFormat 会在 per-image 里换算，否则走这个默认值。
DEFAULT_HFOV_DEG = 73.0
# 若 EXIF 无 altitude，就假设 50 m（~tune7b 中位值）
DEFAULT_ALTITUDE_M = 50.0
# 最小 / 最大合成 footprint（米），防止某些 altitude=0 的异常值
MIN_FOOTPRINT_M = 30.0
MAX_FOOTPRINT_M = 400.0


# --------------------------- EXIF --------------------------- #

_TAG_NAME_TO_ID = {v: k for k, v in ExifTags.TAGS.items()}
_GPS_NAME_TO_ID = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _dms_to_deg(dms) -> float:
    d, m, s = (float(x) for x in dms)
    return d + m / 60.0 + s / 3600.0


def read_drone_exif(path: Path) -> dict[str, Any] | None:
    """从 DJI 无人机 JPG 里抽 GPS / altitude / focal。缺 GPS 则返回 None。"""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            gps_idx = _TAG_NAME_TO_ID.get("GPSInfo")
            gps = exif.get_ifd(gps_idx) if gps_idx else None
            if not gps or 2 not in gps or 4 not in gps:
                return None
            lat = _dms_to_deg(gps[2])
            if str(gps.get(1, "N")).upper() == "S":
                lat = -lat
            lon = _dms_to_deg(gps[4])
            if str(gps.get(3, "E")).upper() == "W":
                lon = -lon
            alt_val = gps.get(6)
            alt = float(alt_val) if alt_val is not None else None
            # 参考高度（AboveSeaLevel 还是 BelowSeaLevel）
            if alt is not None and gps.get(5) == 1:
                alt = -alt

            fl35 = exif.get(_TAG_NAME_TO_ID.get("FocalLengthIn35mmFilm"))
            datetime_orig = exif.get(_TAG_NAME_TO_ID.get("DateTimeOriginal")) or exif.get(
                _TAG_NAME_TO_ID.get("DateTime")
            )
            return {
                "lat": lat,
                "lon": lon,
                "altitude_m": alt,
                "focal_35mm": float(fl35) if fl35 else None,
                "datetime": str(datetime_orig) if datetime_orig else None,
                "size": im.size,
            }
    except Exception:
        return None


# --------------------------- footprint 合成 --------------------------- #


def _image_hfov_deg(focal_35mm: float | None) -> float:
    """35mm 等效焦距 → 水平 FOV (36mm sensor width)。"""
    if not focal_35mm or focal_35mm <= 0:
        return DEFAULT_HFOV_DEG
    return 2.0 * math.degrees(math.atan((36.0 / 2.0) / focal_35mm))


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """返回 (dm_per_dlat_deg, dm_per_dlon_deg)。"""
    dlat_m = 110540.0  # 约定值
    dlon_m = 111320.0 * math.cos(math.radians(lat_deg))
    return dlat_m, max(dlon_m, 1.0)


def synth_footprint(
    lat: float, lon: float, altitude_m: float | None, focal_35mm: float | None,
    width: int, height: int,
) -> tuple[dict, float]:
    """返回 (bounds dict, gsd_m_per_px)."""
    alt = altitude_m if altitude_m and altitude_m > 5 else DEFAULT_ALTITUDE_M
    hfov = _image_hfov_deg(focal_35mm)
    footprint_w_m = 2.0 * alt * math.tan(math.radians(hfov / 2.0))
    footprint_w_m = max(MIN_FOOTPRINT_M, min(MAX_FOOTPRINT_M, footprint_w_m))
    gsd = footprint_w_m / max(width, 1)
    footprint_h_m = gsd * height  # 保持等比

    dlat_m, dlon_m = _meters_per_degree(lat)
    half_dlat = (footprint_h_m / 2.0) / dlat_m
    half_dlon = (footprint_w_m / 2.0) / dlon_m

    bounds = {
        "north": lat + half_dlat,
        "south": lat - half_dlat,
        "east": lon + half_dlon,
        "west": lon - half_dlon,
    }
    return bounds, gsd


def bounds_to_affine(bounds: dict, width: int, height: int) -> tuple[dict, dict]:
    """以 bounds 构造轴对齐的 pixel<->geo 仿射系数（配合 xbd_map.{pixel_to_geo, geo_to_pixel}）。"""
    west = float(bounds["west"])
    east = float(bounds["east"])
    north = float(bounds["north"])
    south = float(bounds["south"])
    dx = (east - west) / max(width, 1)
    dy = (south - north) / max(height, 1)

    pixel_to_geo = {
        "lon": [dx, 0.0, west],
        "lat": [0.0, dy, north],
    }
    # inverse: x = (lon - west) / dx; y = (lat - north) / dy
    geo_to_pixel = {
        "x": [1.0 / dx, 0.0, -west / dx],
        "y": [0.0, 1.0 / dy, -north / dy],
    }
    return pixel_to_geo, geo_to_pixel


# --------------------------- 检测 JSON → xBD label polygons --------------------------- #


def bbox_to_wkt(x: int, y: int, w: int, h: int) -> str:
    x2 = x + w
    y2 = y + h
    return (
        f"POLYGON (({x} {y}, {x2} {y}, {x2} {y2}, {x} {y2}, {x} {y}))"
    )


def load_detections(det_json: Path) -> list[dict]:
    try:
        data = json.load(open(det_json, "r", encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        bbox = item.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x, y, w, h = (int(v) for v in bbox)
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        cls = int(item.get("type", -1))
        feature_type, subtype = DET_CLASS_TO_SUBTYPE.get(cls, ("building", ""))
        out.append(
            {
                "cls": cls,
                "feature_type": feature_type,
                "subtype": subtype,
                "bbox": [x, y, w, h],
            }
        )
    return out


def build_label_json(detections: list[dict], tile_id: str) -> dict:
    features_xy: list[dict] = []
    for idx, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        features_xy.append(
            {
                "properties": {
                    "uid": f"{tile_id}#{idx}",
                    "feature_type": det["feature_type"],
                    "subtype": det["subtype"] or None,
                    "source_class": det["cls"],
                },
                "wkt": bbox_to_wkt(x, y, w, h),
            }
        )
    return {
        "metadata": {
            "sensor": SENSOR,
            "disaster_type": DISASTER_TYPE,
            "img_name": tile_id + ".jpg",
            "catalog_id": tile_id,
        },
        "features": {"xy": features_xy, "lng_lat": []},
    }


# --------------------------- footprints geojson --------------------------- #


def bounds_to_geojson_polygon(bounds: dict) -> dict:
    w, e = float(bounds["west"]), float(bounds["east"])
    s, n = float(bounds["south"]), float(bounds["north"])
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


# --------------------------- 主流程 --------------------------- #


def iter_images(root: Path) -> Iterable[tuple[str, Path, Path | None]]:
    """yield (split, image_path, det_json_path_or_None)."""
    mapping = [
        ("train", root / "train" / "train-org-img", root / "train" / "train-label-det"),
        ("val", root / "val" / "val-org-img", root / "val" / "val-label-det"),
    ]
    for split, img_dir, det_dir in mapping:
        if not img_dir.is_dir():
            continue
        for jpg in sorted(img_dir.glob("*.jpg")):
            stem = jpg.stem  # e.g. "10778"
            det_json = det_dir / f"{stem}_lab.json"
            yield split, jpg, det_json if det_json.is_file() else None


def build(limit: int | None, force: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "manifest.json"
    footprints_path = OUTPUT_DIR / "footprints.geojson"
    ranking_path = OUTPUT_DIR / "damage_ranking.json"

    items: list[dict] = []
    footprint_features: list[dict] = []
    disaster_counter: collections.Counter[str] = collections.Counter()
    split_counter: collections.Counter[str] = collections.Counter()
    stage_counter: collections.Counter[str] = collections.Counter()

    processed = 0
    skipped_no_gps = 0
    t0 = time.time()

    for split, jpg, det_json in iter_images(TUNE7B_ROOT):
        if limit and processed >= limit:
            break
        stem = jpg.stem
        tile_id = f"rescuenet_{stem}_post_disaster"

        exif = read_drone_exif(jpg)
        if not exif:
            skipped_no_gps += 1
            continue

        width, height = exif["size"]
        bounds, gsd = synth_footprint(
            exif["lat"], exif["lon"], exif.get("altitude_m"), exif.get("focal_35mm"),
            width, height,
        )
        pixel_to_geo, geo_to_pixel = bounds_to_affine(bounds, width, height)

        # ----- 检测 → xBD label polygon -----
        detections = load_detections(det_json) if det_json else []
        cnt_sub: collections.Counter[str] = collections.Counter()
        for det in detections:
            if det["feature_type"] == "building" and det["subtype"]:
                cnt_sub[det["subtype"]] += 1
        label_rel = f"labels/{stem}_post_label.json"
        label_abs = OUTPUT_DIR / label_rel
        label_abs.parent.mkdir(parents=True, exist_ok=True)
        with open(label_abs, "w", encoding="utf-8") as fp:
            json.dump(build_label_json(detections, tile_id), fp, ensure_ascii=False)

        image_relpath = f"{split}/{split}-org-img/{stem}.jpg"
        abs_img = TUNE7B_ROOT / image_relpath
        if not abs_img.is_file():
            continue

        item = {
            "tile_id": tile_id,
            "group_id": f"rescuenet_{stem}",
            "split": split,
            "stage": "post",  # 所有 tune7b 图像都是灾后
            "image_name": abs_img.name,
            "image_relpath": image_relpath,
            # label 文件放在 backend/data/rescuenet/labels 下；用绝对路径让
            # `Path(dataset_root) / relpath` 直接跳到绝对路径（xbd_store 逻辑不改）。
            "label_relpath": str(label_abs.resolve()),
            "disaster": DISASTER_NAME,
            "disaster_type": DISASTER_TYPE,
            "sensor": SENSOR,
            "capture_date": exif.get("datetime") or CAPTURE_DATE,
            "catalog_id": f"rescuenet-{stem}",
            "gsd": gsd,
            "width": width,
            "height": height,
            "has_label_geo": False,  # 我们不做真多边形地理配准，只用轴对齐仿射
            "has_georef": True,
            "transform_source": "synthesized_from_exif_gps_altitude",
            "matched_feature_count": len(detections),
            "matched_point_count": 0,
            "pixel_to_geo": pixel_to_geo,
            "geo_to_pixel": geo_to_pixel,
            "bounds": bounds,
            "center": {
                "lat": (bounds["north"] + bounds["south"]) / 2,
                "lon": (bounds["east"] + bounds["west"]) / 2,
            },
            "exif": {
                "altitude_m": exif.get("altitude_m"),
                "focal_35mm": exif.get("focal_35mm"),
            },
            "counts": {
                "destroyed": int(cnt_sub.get("destroyed", 0)),
                "major_damage": int(cnt_sub.get("major-damage", 0)),
                "minor_damage": int(cnt_sub.get("minor-damage", 0)),
                "no_damage": int(cnt_sub.get("no-damage", 0)),
                "total_buildings": int(sum(cnt_sub.values())),
            },
        }
        items.append(item)

        # ---- footprint GeoJSON feature ----
        severity = (
            item["counts"]["destroyed"] * SEVERITY_WEIGHTS["destroyed"]
            + item["counts"]["major_damage"] * SEVERITY_WEIGHTS["major-damage"]
            + item["counts"]["minor_damage"] * SEVERITY_WEIGHTS["minor-damage"]
        )
        footprint_features.append(
            {
                "type": "Feature",
                "geometry": bounds_to_geojson_polygon(bounds),
                "properties": {
                    "tile_id": tile_id,
                    "disaster": DISASTER_NAME,
                    "disaster_type": DISASTER_TYPE,
                    "stage": "post_disaster",
                    "split": split,
                    "has_georef": True,
                    "damage": {
                        "destroyed": item["counts"]["destroyed"],
                        "major": item["counts"]["major_damage"],
                        "minor": item["counts"]["minor_damage"],
                        "no": item["counts"]["no_damage"],
                        "severity": severity,
                    },
                },
            }
        )

        disaster_counter[DISASTER_NAME] += 1
        split_counter[split] += 1
        stage_counter["post"] += 1
        processed += 1
        if processed % 500 == 0:
            print(f"[rescuenet] processed {processed} images ({time.time()-t0:.1f}s)")

    # --------- manifest ---------
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "scripts/build_rescuenet_dataset.py",
        "dataset_root": str(TUNE7B_ROOT),
        "dataset_mode": "rescuenet",
        "splits": dict(split_counter),
        "summary": {
            "tiles": len(items),
            "has_georef": len(items),
            "disasters": dict(disaster_counter),
            "stages": dict(stage_counter),
            "skipped_no_gps": skipped_no_gps,
        },
        "items": items,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False)

    # --------- footprints.geojson ---------
    with open(footprints_path, "w", encoding="utf-8") as fp:
        json.dump({"type": "FeatureCollection", "features": footprint_features}, fp, ensure_ascii=False)

    # --------- damage_ranking.json ---------
    ranking_items = []
    for it in items:
        c = it["counts"]
        severity = (
            c["destroyed"] * SEVERITY_WEIGHTS["destroyed"]
            + c["major_damage"] * SEVERITY_WEIGHTS["major-damage"]
            + c["minor_damage"] * SEVERITY_WEIGHTS["minor-damage"]
        )
        total = c["destroyed"] + c["major_damage"] + c["minor_damage"] + c["no_damage"]
        damaged = c["destroyed"] + c["major_damage"] + c["minor_damage"]
        ranking_items.append(
            {
                "tile_id": it["tile_id"],
                "disaster": it["disaster"],
                "disaster_type": it["disaster_type"],
                "split": it["split"],
                "stage": it["stage"],
                "center": it["center"],
                "bounds": it["bounds"],
                "gsd": it["gsd"],
                "capture_date": it.get("capture_date"),
                "counts": {
                    "destroyed": c["destroyed"],
                    "major_damage": c["major_damage"],
                    "minor_damage": c["minor_damage"],
                    "no_damage": c["no_damage"],
                    "un_classified": 0,
                    "total_buildings": total,
                },
                "severity": severity,
                "destroyed_ratio": round(c["destroyed"] / total, 4) if total else 0.0,
                "damaged_ratio": round(damaged / total, 4) if total else 0.0,
            }
        )
    ranking_items.sort(key=lambda r: (-r["severity"], -r["counts"]["destroyed"]))
    for rank, row in enumerate(ranking_items, start=1):
        row["rank"] = rank
    with open(ranking_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "generated_from": str(manifest_path),
                "dataset_root": str(TUNE7B_ROOT),
                "weights": SEVERITY_WEIGHTS,
                "total_post_tiles": len(items),
                "total_scored": len(ranking_items),
                "items": ranking_items,
            },
            fp,
            ensure_ascii=False,
        )

    print(
        f"[ok] wrote manifest/footprints/damage-ranking to {OUTPUT_DIR} — "
        f"{len(items)} tiles, skipped_no_gps={skipped_no_gps}, elapsed={time.time()-t0:.1f}s"
    )
    print("Top 10 by severity:")
    print(f"{'rank':>4}  {'sev':>6}  {'dest':>4}  {'maj':>4}  {'min':>4}  {'tot':>4}  tile")
    for row in ranking_items[:10]:
        c = row["counts"]
        print(
            f"{row['rank']:>4}  {row['severity']:>6}  {c['destroyed']:>4}  "
            f"{c['major_damage']:>4}  {c['minor_damage']:>4}  {c['total_buildings']:>4}  {row['tile_id']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists() and not args.force and not args.limit:
        print(f"[skip] {manifest_path} exists; use --force to rebuild.")
        return 0

    if not TUNE7B_ROOT.is_dir():
        print(f"ERROR: TUNE7B_ROOT not found: {TUNE7B_ROOT}", file=sys.stderr)
        return 1

    build(limit=args.limit, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
