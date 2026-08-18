#!/usr/bin/env python3
"""
scripts/training/gen_xbd_loc_dataset.py — xBD → 建筑定位（二值 mask）数据集

只做定位脚手架：用 post 标签里的建筑多边形 rasterize 成二值 mask，图像侧软链
对应的 pre_disaster PNG（对齐 xView2 冠军定位阶段：在 pre 上训定位，避开 post
噪声）。损伤分级不在此数据集里——推理时由 change_perception 完成。

严格事件切分参数与 gen_xbd_yolo_dataset.py / paper_reproduction.md 一致，
manifest 含 split_audit.event_disjoint，供 train/eval 的 --require-event-disjoint 校验。

用法：
    python scripts/training/gen_xbd_loc_dataset.py \
        --xbd-root /home/lc/datasets/xbd \
        --out /home/lc/datasets/xbd_loc_strict_v1 \
        --splits train,tier3 --test-split test --strict-event-split \
        --val-disasters hurricane-harvey,mexico-earthquake \
        --test-disasters hurricane-michael,palu-tsunami \
        --holdout-disasters nepal-flooding,moore-tornado,pinery-bushfire \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from xbd_map import _parse_polygon_wkt  # noqa: E402

# un-classified 按冠军惯例并入 building；其余有 subtype 的建筑也一律算前景。
_BUILDING_SUBTYPES = {
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
    "un-classified",
}

_DISASTER_RE = re.compile(r"_\d+_(?:pre|post)_disaster$")


def disaster_of(label_path: Path) -> str:
    return _DISASTER_RE.sub("", label_path.stem)


def parse_disasters(raw: str) -> set[str]:
    return {d.strip() for d in raw.split(",") if d.strip()}


def validate_event_subsets(subsets: dict[str, list[tuple[Path, str]]]) -> dict:
    events = {
        name: sorted({disaster_of(path) for path, _ in paths})
        for name, paths in subsets.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = list(events)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = sorted(set(events[left]) & set(events[right]))
            if shared:
                overlaps[f"{left}__{right}"] = shared
    if overlaps:
        raise ValueError(f"事件级切分泄漏：{overlaps}")
    return {"event_disjoint": True, "events": events, "overlaps": overlaps}


def _post_label_paths(xbd_root: Path, split: str) -> list[Path]:
    labels_dir = xbd_root / split / "labels"
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.glob("*post_disaster.json"))


def _pre_image_for_post_label(xbd_root: Path, src_split: str, post_label: Path) -> Path | None:
    """post 标签 stem → 同 tile 的 pre PNG。"""
    pre_stem = post_label.stem.replace("_post_disaster", "_pre_disaster")
    pre_path = xbd_root / src_split / "images" / f"{pre_stem}.png"
    return pre_path if pre_path.exists() else None


def _rasterize_building_mask(label_path: Path) -> tuple[Image.Image, int, dict]:
    """post JSON → 二值 mask (L, 0/255) + 建筑多边形数 + 元信息。"""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    w = int(meta.get("width") or meta.get("original_width") or 1024)
    h = int(meta.get("height") or meta.get("original_height") or 1024)
    feats = (data.get("features") or {}).get("xy") or []

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    n_poly = 0
    subtype_counts: dict[str, int] = {}
    for feat in feats:
        props = feat.get("properties") or {}
        sub = (props.get("subtype") or "").strip() or "un-classified"
        if sub not in _BUILDING_SUBTYPES:
            continue
        poly = _parse_polygon_wkt(feat.get("wkt", ""))
        if len(poly) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in poly], outline=255, fill=255)
        n_poly += 1
        subtype_counts[sub] = subtype_counts.get(sub, 0) + 1
    return mask, n_poly, {"width": w, "height": h, "subtype_counts": subtype_counts}


def _emit(
    label_paths: list[tuple[Path, str]],
    xbd_root: Path,
    out_root: Path,
    subset: str,
    limit: int,
) -> dict:
    img_dir = out_root / subset / "images"
    mask_dir = out_root / subset / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_empty = 0
    n_missing_pre = 0
    n_buildings = 0
    disaster_counts: dict[str, int] = {}
    subtype_total: dict[str, int] = {}

    for label_path, src_split in label_paths:
        if limit and n_img >= limit:
            break
        pre_src = _pre_image_for_post_label(xbd_root, src_split, label_path)
        if pre_src is None:
            n_missing_pre += 1
            continue
        mask, n_poly, info = _rasterize_building_mask(label_path)
        if n_poly == 0:
            n_empty += 1
            # 仍保留空 mask 样本，让模型学会「无建筑」场景
        tile_id = label_path.stem.replace("_post_disaster", "_pre_disaster")
        link = img_dir / f"{tile_id}.png"
        if not link.exists():
            try:
                link.symlink_to(pre_src.resolve())
            except FileExistsError:
                pass
        mask.save(mask_dir / f"{tile_id}.png")
        n_img += 1
        n_buildings += n_poly
        disaster = disaster_of(label_path)
        disaster_counts[disaster] = disaster_counts.get(disaster, 0) + 1
        for k, v in info["subtype_counts"].items():
            subtype_total[k] = subtype_total.get(k, 0) + v

    return {
        "subset": subset,
        "images": n_img,
        "buildings": n_buildings,
        "empty_masks": n_empty,
        "missing_pre": n_missing_pre,
        "disaster_counts": disaster_counts,
        "subtype_counts": subtype_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", default="/home/lc/datasets/xbd")
    ap.add_argument("--out", default="/home/lc/datasets/xbd_loc")
    ap.add_argument("--splits", default="train,tier3")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--holdout-disasters",
        default="nepal-flooding,moore-tornado,pinery-bushfire",
    )
    ap.add_argument("--strict-event-split", action="store_true")
    ap.add_argument("--val-disasters", default="")
    ap.add_argument("--test-disasters", default="")
    args = ap.parse_args()

    xbd_root = Path(args.xbd_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    if not xbd_root.is_dir():
        raise FileNotFoundError(f"xBD 根目录不存在: {xbd_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    holdout_disasters = parse_disasters(args.holdout_disasters)
    val_disasters = parse_disasters(args.val_disasters)
    test_disasters = parse_disasters(args.test_disasters)
    requested_groups = {
        "val": val_disasters,
        "test": test_disasters,
        "holdout": holdout_disasters,
    }
    requested_overlaps = {}
    requested_names = list(requested_groups)
    for i, left in enumerate(requested_names):
        for right in requested_names[i + 1 :]:
            shared = sorted(requested_groups[left] & requested_groups[right])
            if shared:
                requested_overlaps[f"{left}__{right}"] = shared
    if requested_overlaps:
        raise ValueError(f"--val/--test/--holdout-disasters 互相重叠: {requested_overlaps}")

    source_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.strict_event_split:
        if not val_disasters or not test_disasters or not holdout_disasters:
            raise ValueError(
                "--strict-event-split 要求显式提供非空的 --val-disasters、"
                "--test-disasters 和 --holdout-disasters"
            )
        if args.test_split and args.test_split not in source_splits:
            source_splits.append(args.test_split)

    candidates: list[tuple[Path, str]] = []
    for sp in source_splits:
        paths = _post_label_paths(xbd_root, sp)
        if not paths:
            raise FileNotFoundError(f"split 缺少 post 标签: {xbd_root / sp / 'labels'}")
        candidates.extend((p, sp) for p in paths)
        print(f"[gen-loc] {sp}: {len(paths)} post labels")

    if args.strict_event_split:
        known_disasters = {disaster_of(p) for p, _ in candidates}
        missing = {
            name: sorted(values - known_disasters)
            for name, values in requested_groups.items()
            if values - known_disasters
        }
        if missing:
            raise ValueError(f"指定了不存在的灾害事件: {missing}")
        subsets: dict[str, list[tuple[Path, str]]] = {
            "train": [],
            "val": [],
            "test": [],
            "holdout": [],
        }
        for item in candidates:
            disaster = disaster_of(item[0])
            if disaster in holdout_disasters:
                subset = "holdout"
            elif disaster in test_disasters:
                subset = "test"
            elif disaster in val_disasters:
                subset = "val"
            else:
                subset = "train"
            subsets[subset].append(item)
        split_audit = validate_event_subsets(subsets)
        split_strategy = "strict_event"
    else:
        pool = [item for item in candidates if disaster_of(item[0]) not in holdout_disasters]
        holdout_pool = [item for item in candidates if disaster_of(item[0]) in holdout_disasters]
        rng.shuffle(pool)
        n_val = int(len(pool) * args.val_frac)
        subsets = {"train": pool[n_val:], "val": pool[:n_val]}
        if args.test_split:
            subsets["test"] = [
                (p, args.test_split) for p in _post_label_paths(xbd_root, args.test_split)
            ]
        if holdout_pool:
            subsets["holdout"] = holdout_pool
        event_sets = {
            name: sorted({disaster_of(path) for path, _ in paths})
            for name, paths in subsets.items()
        }
        overlaps = {}
        names = list(event_sets)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = sorted(set(event_sets[left]) & set(event_sets[right]))
                if shared:
                    overlaps[f"{left}__{right}"] = shared
        split_audit = {
            "event_disjoint": not overlaps,
            "events": event_sets,
            "overlaps": overlaps,
        }
        split_strategy = "legacy_tile_random"

    if args.strict_event_split:
        occupied = []
        for name in subsets:
            for leaf in ("images", "masks"):
                directory = out_root / name / leaf
                if directory.exists() and any(directory.iterdir()):
                    occupied.append(str(directory))
        if occupied:
            raise FileExistsError(
                "严格事件模式拒绝复用非空输出目录；请换一个新的 --out。"
                f"非空目录: {occupied}"
            )

    stats = [
        _emit(paths, xbd_root, out_root, name, args.limit)
        for name, paths in subsets.items()
    ]

    data_cfg = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images",
        "masks": {
            "train": "train/masks",
            "val": "val/masks",
        },
        "task": "building_localization",
        "image_phase": "pre",
        "mask_source": "post_polygons",
    }
    if "test" in subsets:
        data_cfg["test"] = "test/images"
        data_cfg["masks"]["test"] = "test/masks"
    if "holdout" in subsets:
        data_cfg["holdout"] = str(out_root / "holdout" / "images")
        data_cfg["masks"]["holdout"] = str(out_root / "holdout" / "masks")
    (out_root / "data.json").write_text(
        json.dumps(data_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "generator": str(Path(__file__).resolve()),
        "xbd_root": str(xbd_root),
        "seed": args.seed,
        "split_strategy": split_strategy,
        "strict_event_split": bool(args.strict_event_split),
        "source_splits": source_splits,
        "requested_disasters": {
            name: sorted(values) for name, values in requested_groups.items()
        },
        "split_audit": split_audit,
        "subsets": {s["subset"]: {k: v for k, v in s.items() if k != "subset"} for s in stats},
        "task": "building_localization",
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[gen-loc] 完成：")
    for s in stats:
        print(
            f"  {s['subset']:7s}: imgs={s['images']:6d} buildings={s['buildings']:7d} "
            f"empty={s['empty_masks']:5d} missing_pre={s['missing_pre']:4d}"
        )
    print(f"[gen-loc] data.json → {out_root / 'data.json'}")
    print(f"[gen-loc] manifest → {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
