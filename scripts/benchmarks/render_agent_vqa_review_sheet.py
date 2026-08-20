#!/usr/bin/env python3
"""scripts/benchmarks/render_agent_vqa_review_sheet.py — Agent-VQA 审核表渲染 (D2).

为人工抽查渲染 contact sheet: 每格显示灾后瓦片、UAV 起点(蓝)、目标质心(红)、
巡航视场圆(蓝虚线)、问题/答案/题型/事件/歧义标志。脚本不替作者批准题——
作者检视后须更新各题 review 记录方可进入论文主实验。

用法:
    python scripts/benchmarks/render_agent_vqa_review_sheet.py \
        backend/data/benchmarks/agent_vqa_testset.json \
        --out-dir runs/benchmarks/cja_agent_vqa/review_sheets
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from xbd_map import geo_to_pixel  # noqa: E402

CRUISE_ALT_M = 30.0
CRUISE_RADIUS_M = max(20.0, min(300.0, CRUISE_ALT_M * 2.0))


def _font(size: int) -> ImageFont.ImageFont:
    # 优先 CJK 字体 (题面为中文); DejaVu/Liberation 不含 CJK 字形会显示方块
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _marker(draw, xy, color, label):
    x, y = xy
    r = 12
    draw.ellipse((x - r, y - r, x + r, y + r), outline="white", width=6)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    draw.text((x + 15, y - 12), label, fill=color, font=_font(22), stroke_width=3, stroke_fill="white")


def _fov_circle(draw, transform, start, radius_m):
    """在起点周围画巡航视场圆 (用经纬度近似, 跨度小时足够可视化)。"""
    cx, cy = geo_to_pixel(transform, start["lon"], start["lat"])
    # 用 1 度纬度≈111km 把半径换算成像素, 再按 transform 缩放近似
    lat_per_m = 1.0 / 111_000.0
    edge_lat = start["lat"] + lat_per_m * radius_m
    edge_lon = start["lon"]
    ex, ey = geo_to_pixel(transform, edge_lon, edge_lat)
    r_px = max(8, math.hypot(ex - cx, ey - cy))
    draw.ellipse((cx - r_px, cy - r_px, cx + r_px, cy + r_px),
                 outline="#1976d2", width=2)


def _panel(item, entry, dataset_root, size):
    image_path = dataset_root / entry["image_relpath"]
    image = Image.open(image_path).convert("RGB")
    transform = {"pixel_to_geo": entry["pixel_to_geo"], "geo_to_pixel": entry["geo_to_pixel"]}
    start = item["start"]
    cx, cy = geo_to_pixel(transform, start["lon"], start["lat"])
    gsd = float(entry.get("gsd") or 0.5)
    radius_px = int(max(128, CRUISE_RADIUS_M / max(gsd, 1e-3)))
    left = max(0, int(round(cx - radius_px)))
    top = max(0, int(round(cy - radius_px)))
    right = min(image.width, int(round(cx + radius_px)))
    bottom = min(image.height, int(round(cy + radius_px)))
    image = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(image)
    start_xy = (cx - left, cy - top)
    _marker(draw, start_xy, "#1976d2", "START")
    target = item.get("target")
    if target:
        tx, ty = geo_to_pixel(transform, target["lon"], target["lat"])
        _marker(draw, (tx - left, ty - top), "#d32f2f", "TARGET")
    if item.get("question_type") == "damage":
        # 与实际 VQA 输入一致：视场中心具有显式十字标记。
        mx, my = image.width // 2, image.height // 2
        arm = max(12, min(image.size) // 16)
        draw.line((mx - arm, my, mx + arm, my), fill="white", width=8)
        draw.line((mx, my - arm, mx, my + arm), fill="white", width=8)
        draw.line((mx - arm, my, mx + arm, my), fill="#ff2d55", width=4)
        draw.line((mx, my - arm, mx, my + arm), fill="#ff2d55", width=4)

    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size + 110), "white")
    panel.paste(image, ((size - image.width) // 2, 0))
    flags = ",".join(item.get("review", {}).get("ambiguity_flags", [])) or "-"
    text = (
        f"[{item['question_type']}] {item['question']}\n"
        f"答案: {item['answer']} | {item['disaster']} | flags={flags}"
    )
    ImageDraw.Draw(panel).multiline_text(
        (8, size + 6), text, fill="black", font=_font(16), spacing=3)
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "benchmarks" / "cja_agent_vqa" / "review_sheets"))
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--panel-size", type=int, default=420)
    ap.add_argument("--only-needs-author", action="store_true",
                    help="只渲染需作者检查的题 (带歧义标志)")
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = {e["tile_id"]: e for e in manifest.get("items", [])}
    items = testset.get("items", [])
    if args.only_needs_author:
        items = [it for it in items if it.get("review", {}).get("ambiguity_flags")]

    import xbd_map as _xbd
    dataset_root = _xbd.resolve_dataset_root(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = 3
    rows = math.ceil(args.per_sheet / cols)
    panel_h = args.panel_size + 110
    for sheet_idx, offset in enumerate(range(0, len(items), args.per_sheet), start=1):
        batch = items[offset: offset + args.per_sheet]
        sheet = Image.new("RGB", (cols * args.panel_size, rows * panel_h), "#dddddd")
        for idx, item in enumerate(batch):
            entry = entries.get(item["tile_id"])
            if entry is None:
                print(f"[WARN] manifest 缺瓦片 {item['tile_id']}, 跳过", file=sys.stderr)
                continue
            panel = _panel(item, entry, dataset_root, args.panel_size)
            sheet.paste(panel, ((idx % cols) * args.panel_size, (idx // cols) * panel_h))
        path = out_dir / f"review_{sheet_idx:02d}.jpg"
        sheet.save(path, quality=92)
        print(path)
    print(f"[OK] 渲染 {len(items)} 条题 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
