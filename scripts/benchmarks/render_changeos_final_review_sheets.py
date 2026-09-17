#!/usr/bin/env python3
"""Render ROI-scoped, GT-annotated sheets for author review of final candidates.

These sheets are review aids only. They never approve a question or change its
human_review record, and they contain no benchmark model predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import xbd_map  # noqa: E402

COLORS = {
    "no-damage": "#2f6bdf", "minor-damage": "#e6ad26",
    "major-damage": "#ee5b3d", "destroyed": "#ad2594",
}
PANEL_PX = 512
TEXT_PX = 175


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def geo_xy(entry: dict, lon: float, lat: float, image_size: tuple[int, int]) -> tuple[float, float]:
    x, y = xbd_map.geo_to_pixel({
        "pixel_to_geo": entry["pixel_to_geo"],
        "geo_to_pixel": entry["geo_to_pixel"],
    }, lon, lat)
    return x * PANEL_PX / image_size[0], y * PANEL_PX / image_size[1]


def gt_points(entry: dict, dataset_root: Path) -> list[dict]:
    label_path = dataset_root / str(entry.get("label_relpath") or "")
    if not entry.get("label_relpath") or not label_path.is_file():
        raise ValueError(f"missing label for {entry.get('tile_id')}")
    label = json.loads(label_path.read_text(encoding="utf-8"))
    points = []
    for feature in ((label.get("features") or {}).get("lng_lat") or []):
        props = feature.get("properties") or {}
        subtype = str(props.get("subtype") or "")
        if subtype not in COLORS:
            continue
        ring = xbd_map._parse_polygon_wkt(feature.get("wkt") or "")
        if not ring:
            continue
        lon, lat = xbd_map._polygon_centroid(ring)
        points.append({"lon": lon, "lat": lat, "subtype": subtype,
                       "uid": str(props.get("uid") or "")})
    return points


def panel(item: dict, entry: dict, dataset_root: Path, points: list[dict]) -> Image.Image:
    image_path = dataset_root / str(entry["image_relpath"])
    with Image.open(image_path) as original:
        original_size = original.size
        post = original.convert("RGB").resize((PANEL_PX, PANEL_PX), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(post)
    roi = item["roi"]
    bounds = roi["bounds"]
    corners = [
        (bounds["west"], bounds["north"]),
        (bounds["east"], bounds["north"]),
        (bounds["east"], bounds["south"]),
        (bounds["west"], bounds["south"]),
    ]
    roi_xy = [geo_xy(entry, lon, lat, original_size) for lon, lat in corners]
    draw.line(roi_xy + roi_xy[:1], fill="white", width=5)
    draw.line(roi_xy + roi_xy[:1], fill="#12b886", width=2)
    for building in points:
        if not (bounds["south"] <= building["lat"] <= bounds["north"]
                and bounds["west"] <= building["lon"] <= bounds["east"]):
            continue
        x, y = geo_xy(entry, building["lon"], building["lat"], original_size)
        if 0 <= x < PANEL_PX and 0 <= y < PANEL_PX:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=COLORS[building["subtype"]], outline="white", width=1)
    center = roi["center"]
    cx, cy = geo_xy(entry, center["lon"], center["lat"], original_size)
    draw.line((cx - 10, cy, cx + 10, cy), fill="#21e6a1", width=3)
    draw.line((cx, cy - 10, cx, cy + 10), fill="#21e6a1", width=3)
    target = item.get("target") or {}
    if "lat" in target and "lon" in target:
        tx, ty = geo_xy(entry, target["lon"], target["lat"], original_size)
        draw.ellipse((tx - 12, ty - 12, tx + 12, ty + 12), outline="white", width=5)
        draw.ellipse((tx - 12, ty - 12, tx + 12, ty + 12), outline="#ff2d55", width=3)
        draw.line((tx - 17, ty, tx + 17, ty), fill="#ff2d55", width=3)
        draw.line((tx, ty - 17, tx, ty + 17), fill="#ff2d55", width=3)
    start = item["start"]
    sx, sy = geo_xy(entry, start["lon"], start["lat"], original_size)
    start_inside = 0 <= sx < PANEL_PX and 0 <= sy < PANEL_PX
    if start_inside:
        draw.rectangle((sx - 6, sy - 6, sx + 6, sy + 6), outline="#ffe066", width=3)

    result = Image.new("RGB", (PANEL_PX, PANEL_PX + TEXT_PX), "white")
    result.paste(post, (0, 0))
    text_draw = ImageDraw.Draw(result)
    flags = ",".join((item.get("review") or {}).get("ambiguity_flags") or []) or "-"
    short_id = str(item.get("id") or "")[-38:]
    lines = [
        f"{short_id}",
        f"{item['disaster']} | {item['question_type']} | 答案 {item['answer']}",
        str(item["question"])[:42],
        f"歧义: {flags[:45]}",
        f"起点: {'ROI 内' if start_inside else 'ROI 外'} | 圆=目标, 方=起点",
        "点色: 蓝=无损伤 黄=轻微 橙=严重 紫=完全损毁",
    ]
    for line_no, line in enumerate(lines):
        text_draw.text((8, PANEL_PX + 5 + line_no * 26), line,
                       font=font(16), fill="#111111")
    return result


def render(taskset: dict, manifest: dict, dataset_root: Path, out_dir: Path,
           per_sheet: int = 8) -> dict:
    if per_sheet <= 0:
        raise ValueError("per_sheet must be positive")
    entries = {str(entry.get("tile_id")): entry for entry in manifest.get("items", [])
               if entry.get("stage") == "post"}
    items = taskset.get("items") or []
    if not items:
        raise ValueError("taskset has no items")
    out_dir.mkdir(parents=True, exist_ok=False)
    point_cache: dict[str, list[dict]] = {}
    index = []
    cols = 4
    rows_per_sheet = math.ceil(per_sheet / cols)
    panel_h = PANEL_PX + TEXT_PX
    for sheet_no, offset in enumerate(range(0, len(items), per_sheet), start=1):
        page = Image.new("RGB", (cols * PANEL_PX, rows_per_sheet * panel_h), "#dddddd")
        for local_index, item in enumerate(items[offset:offset + per_sheet]):
            tile_id = str(item.get("tile_id") or "")
            entry = entries.get(tile_id)
            if entry is None:
                raise ValueError(f"tile missing from manifest: {tile_id}")
            if tile_id not in point_cache:
                point_cache[tile_id] = gt_points(entry, dataset_root)
            picture = panel(item, entry, dataset_root, point_cache[tile_id])
            page.paste(picture, ((local_index % cols) * PANEL_PX,
                                 (local_index // cols) * panel_h))
            index.append({
                "qid": item.get("id"), "sheet": sheet_no, "panel": local_index + 1,
                "question_type": item.get("question_type"),
                "event": item.get("disaster"), "answer": item.get("answer"),
                "ambiguity_flags": list((item.get("review") or {}).get("ambiguity_flags") or []),
                "needs_author_check": bool((item.get("review") or {}).get("ambiguity_flags")),
            })
        page.save(out_dir / f"review_{sheet_no:02d}.jpg", quality=90)
    report = {
        "schema": "changeos-roi-final-review-sheets/1.0",
        "role": "author review aid; not an approval or freeze receipt",
        "n_questions": len(items), "n_sheets": sheet_no,
        "n_unique_roi": len(point_cache), "index": index,
    }
    (out_dir / "index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "author_decisions_template.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fp:
        fields = ("qid", "sheet", "panel", "question_type", "event",
                  "answer", "ambiguity_flags", "decision", "reviewer", "note")
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for record in index:
            writer.writerow({
                **{field: record[field] for field in fields[:6]},
                "ambiguity_flags": ",".join(record["ambiguity_flags"]),
                "decision": "", "reviewer": "", "note": "",
            })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testset", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-sheet", type=int, default=8)
    args = parser.parse_args()
    taskset = json.loads(args.testset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = render(taskset, manifest, Path(str(manifest["dataset_root"])),
                    args.out_dir, args.per_sheet)
    print(json.dumps({key: report[key] for key in (
        "n_questions", "n_sheets", "n_unique_roi",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
