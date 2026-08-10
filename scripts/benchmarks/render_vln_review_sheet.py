#!/usr/bin/env python3
"""Render contact sheets for manual VLN benchmark review.

Each panel shows the full post-disaster tile, the sampled UAV start (blue),
the target building centroid (red), and the instruction.  The script does not
approve examples: reviewers must inspect the sheets and update each item's
``review`` record before it can enter a paper benchmark.
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


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _marker(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: str, label: str) -> None:
    x, y = xy
    r = 12
    draw.ellipse((x - r, y - r, x + r, y + r), outline="white", width=6)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    draw.text((x + 15, y - 12), label, fill=color, font=_font(22), stroke_width=3, stroke_fill="white")


def _panel(item: dict, entry: dict, dataset_root: Path, size: int) -> Image.Image:
    image_path = dataset_root / entry["image_relpath"]
    image = Image.open(image_path).convert("RGB")
    transform = {
        "pixel_to_geo": entry["pixel_to_geo"],
        "geo_to_pixel": entry["geo_to_pixel"],
    }
    draw = ImageDraw.Draw(image)
    start = item["start"]
    goal = item["goals"][-1]
    _marker(draw, geo_to_pixel(transform, start["lon"], start["lat"]), "#1976d2", "START")
    _marker(draw, geo_to_pixel(transform, goal["lon"], goal["lat"]), "#d32f2f", "GOAL")

    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size + 92), "white")
    panel.paste(image, ((size - image.width) // 2, 0))
    text = (
        f"{item['id']}\n"
        f"{item['instruction']} | {item['disaster']} | {item['shortest_path_m']}m"
    )
    ImageDraw.Draw(panel).multiline_text(
        (8, size + 6), text, fill="black", font=_font(16), spacing=3
    )
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--dataset-root", default="/home/lc/datasets/xbd")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "benchmarks" / "vln_recheck_review"))
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--panel-size", type=int, default=420)
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = {entry["tile_id"]: entry for entry in manifest.get("items", [])}
    items = testset.get("items", [])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = 3
    rows = math.ceil(args.per_sheet / cols)
    panel_h = args.panel_size + 92
    for sheet_idx, offset in enumerate(range(0, len(items), args.per_sheet), start=1):
        batch = items[offset: offset + args.per_sheet]
        sheet = Image.new("RGB", (cols * args.panel_size, rows * panel_h), "#dddddd")
        for idx, item in enumerate(batch):
            entry = entries.get(item["tile_id"])
            if entry is None:
                raise KeyError(f"manifest missing tile {item['tile_id']}")
            panel = _panel(item, entry, Path(args.dataset_root), args.panel_size)
            sheet.paste(panel, ((idx % cols) * args.panel_size, (idx // cols) * panel_h))
        path = out_dir / f"review_{sheet_idx:02d}.jpg"
        sheet.save(path, quality=92)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
