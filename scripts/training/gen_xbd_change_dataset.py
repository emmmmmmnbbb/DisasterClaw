#!/usr/bin/env python3
"""
scripts/training/gen_xbd_change_dataset.py — xBD pre/post 配对 → 双时相变化数据集清单

为什么：P5（docs/vln_rescue_agent_实施计划.md 第六节"升级接口"）需要一个能输出**校准过的
4 类损伤 softmax 概率**的感知后端，而不是当前 YOLO 的单一 top-1 conf。xBD 本地已有完整的
pre/post 配对影像（train 2799 对 + tier3 6369 对 + test 933 对），且同一栋建筑在 pre/post
两份标注里共享同一个 `uid`，天然对齐——不需要额外配准或采购新数据。

本脚本不裁剪、不复制图片（避免生成成百上千的小文件），只产出一份 **JSONL 清单**：
每行是一栋建筑的 pre/post 配对记录（tile 路径 + 该建筑在 pre/post 图上各自的 bbox +
subtype + 二值变化标签），供 `backend/change_perception.py` 的 Dataset 在训练时按需
从原始 tile PNG 里现场裁剪配对 patch（复用 perception.py"裁剪不落盘复制"的思路）。

变化标签构造（零成本，标注里已经有）：
    pre 侧 features 里同一 uid 的建筑没有 subtype（灾前，默认视为 no-damage 基线）；
    post 侧 subtype 就是灾后结果。changed = (post.subtype != 'no-damage')。

跨灾害留出：复用 gen_xbd_yolo_dataset.py 的 disaster_of() / holdout 逻辑，保证两个数据集
（检测器训练 + 变化感知训练）用同一套 train/val/test/holdout 灾害事件划分，不会出现
"YOLO 用了修复后的划分、change_perception 却用了旧划分"的不一致。

用法：
    python scripts/training/gen_xbd_change_dataset.py \
        --xbd-root /home/lc/datasets/xbd --out /home/lc/datasets/xbd_change_event_v1 \
        --splits train,tier3 --test-split test --strict-event-split \
        --val-disasters hurricane-florence,hurricane-matthew \
        --test-disasters hurricane-michael,joplin-tornado \
        --holdout-disasters nepal-flooding,moore-tornado,pinery-bushfire
    # 调试： --limit 200（按 tile 数限制，不是按建筑数）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xbd_map import _parse_polygon_wkt  # noqa: E402
from gen_xbd_yolo_dataset import (  # noqa: E402
    CLASS_NAMES,
    SUBTYPE_TO_CLASS,
    disaster_of,
    parse_disasters,
    validate_event_subsets,
)

MIN_BOX_PX = 4.0  # 退化框（宽或高小于此值）跳过


def _bbox_from_poly(
    poly: list[tuple[float, float]], w: int, h: int
) -> tuple[float, float, float, float] | None:
    if len(poly) < 3:
        return None
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(w), max(xs))
    y2 = min(float(h), max(ys))
    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        return None
    return (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))


def _load_features(label_path: Path) -> tuple[dict, int, int]:
    """返回 {uid: bbox} 与图像宽高。"""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    w = int(meta.get("width") or meta.get("original_width") or 1024)
    h = int(meta.get("height") or meta.get("original_height") or 1024)
    feats = (data.get("features") or {}).get("xy") or []
    out: dict[str, dict] = {}
    for f in feats:
        props = f.get("properties") or {}
        uid = props.get("uid")
        if not uid:
            continue
        poly = _parse_polygon_wkt(f.get("wkt", ""))
        bbox = _bbox_from_poly(poly, w, h)
        if bbox is None:
            continue
        out[uid] = {"bbox": bbox, "subtype": (props.get("subtype") or "").strip()}
    return out, w, h


def _records_for_tile(
    post_label_path: Path, xbd_root: Path, src_split: str
) -> list[dict]:
    tile_id = post_label_path.stem  # "<disaster>_<idx>_post_disaster"
    base_id = tile_id[: -len("_post_disaster")]
    pre_label_path = post_label_path.with_name(f"{base_id}_pre_disaster.json")
    if not pre_label_path.exists():
        return []

    pre_img = xbd_root / src_split / "images" / f"{base_id}_pre_disaster.png"
    post_img = xbd_root / src_split / "images" / f"{base_id}_post_disaster.png"
    if not pre_img.exists() or not post_img.exists():
        return []

    post_feats, pw, ph = _load_features(post_label_path)
    pre_feats, _, _ = _load_features(pre_label_path)

    records: list[dict] = []
    for uid, post_rec in post_feats.items():
        sub = post_rec["subtype"]
        cid = SUBTYPE_TO_CLASS.get(sub)
        if cid is None:  # un-classified / 空 → 跳过
            continue
        pre_rec = pre_feats.get(uid)
        if pre_rec is None:  # 该建筑在 pre 图里没有对应标注（极少见）
            continue
        records.append({
            "tile_id": base_id,
            "disaster": disaster_of(post_label_path),
            "uid": uid,
            "pre_image": str(pre_img),
            "post_image": str(post_img),
            "image_width": pw,
            "image_height": ph,
            "bbox_pre": list(pre_rec["bbox"]),
            "bbox_post": list(post_rec["bbox"]),
            "subtype": sub,
            "class_id": cid,
            "changed": int(sub != "no-damage"),
        })
    return records


def _write_jsonl(records: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    cls_counts = {name: 0 for name in CLASS_NAMES}
    n_changed = 0
    disaster_counts: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cls_counts[rec["subtype"]] = cls_counts.get(rec["subtype"], 0) + 1
            n_changed += rec["changed"]
            disaster = rec["disaster"]
            disaster_counts[disaster] = disaster_counts.get(disaster, 0) + 1
    return {
        "path": str(path),
        "buildings": len(records),
        "changed": n_changed,
        "class_counts": cls_counts,
        "disaster_counts": disaster_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", default="/home/lc/datasets/xbd")
    ap.add_argument("--out", default="/home/lc/datasets/xbd_change")
    ap.add_argument("--splits", default="train,tier3")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="每个 split 最多扫描多少张 tile（调试用）")
    ap.add_argument(
        "--holdout-disasters",
        default="nepal-flooding,moore-tornado,pinery-bushfire",
        help="同 gen_xbd_yolo_dataset.py，保持两套数据集切分一致；传空字符串关闭。",
    )
    ap.add_argument(
        "--strict-event-split",
        action="store_true",
        help=(
            "按灾害事件严格切 train/val/test/holdout；会把 --test-split 并入候选池，"
            "确保官方 train/test 中同名事件不会跨 subset。"
        ),
    )
    ap.add_argument("--val-disasters", default="", help="严格事件模式的 val 事件（逗号分隔）。")
    ap.add_argument("--test-disasters", default="", help="严格事件模式的 test 事件（逗号分隔）。")
    args = ap.parse_args()

    xbd_root = Path(args.xbd_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    if not xbd_root.is_dir():
        raise FileNotFoundError(f"xBD 根目录不存在: {xbd_root}")
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("--val-frac 必须在 (0, 1) 内")
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
        for right in requested_names[i + 1:]:
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
        labels_dir = xbd_root / sp / "labels"
        post_paths = sorted(labels_dir.glob("*post_disaster.json")) if labels_dir.exists() else []
        if not post_paths:
            raise FileNotFoundError(f"split 缺少 post 标签: {labels_dir}")
        if args.limit:
            post_paths = post_paths[: args.limit]
        candidates.extend((p, sp) for p in post_paths)
        print(f"[gen] {sp}: {len(post_paths)} tiles 扫描")

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
            "train": [], "val": [], "test": [], "holdout": [],
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
        pool_tiles = [
            item for item in candidates if disaster_of(item[0]) not in holdout_disasters
        ]
        holdout_tiles = [
            item for item in candidates if disaster_of(item[0]) in holdout_disasters
        ]
        rng.shuffle(pool_tiles)
        n_val = int(len(pool_tiles) * args.val_frac)
        subsets = {"train": pool_tiles[n_val:], "val": pool_tiles[:n_val]}
        if args.test_split:
            labels_dir = xbd_root / args.test_split / "labels"
            test_paths = sorted(labels_dir.glob("*post_disaster.json")) if labels_dir.exists() else []
            if args.limit:
                test_paths = test_paths[: args.limit]
            if test_paths:
                subsets["test"] = [(p, args.test_split) for p in test_paths]
        if holdout_tiles:
            subsets["holdout"] = holdout_tiles
        event_sets = {
            name: sorted({disaster_of(path) for path, _ in paths})
            for name, paths in subsets.items()
        }
        overlaps = {}
        names = list(event_sets)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                shared = sorted(set(event_sets[left]) & set(event_sets[right]))
                if shared:
                    overlaps[f"{left}__{right}"] = shared
        split_audit = {
            "event_disjoint": not overlaps, "events": event_sets, "overlaps": overlaps,
        }
        split_strategy = "legacy_tile_random"

    if args.strict_event_split:
        occupied = [
            str(out_root / name)
            for name in ("train.jsonl", "val.jsonl", "test.jsonl", "holdout.jsonl", "manifest.json")
            if (out_root / name).exists()
        ]
        if occupied:
            raise FileExistsError(
                "严格事件模式拒绝覆盖已有清单；请换一个新的 --out。"
                f"已有文件: {occupied}"
            )

    summary = {}
    for name, tiles in subsets.items():
        records: list[dict] = []
        for label_path, src_split in tiles:
            records.extend(_records_for_tile(label_path, xbd_root, src_split))
        summary[name] = _write_jsonl(records, out_root / f"{name}.jsonl")

    manifest = {
        "generator": str(Path(__file__).resolve()),
        "xbd_root": str(xbd_root),
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "split_strategy": split_strategy,
        "strict_event_split": bool(args.strict_event_split),
        "source_splits": source_splits,
        "requested_disasters": {
            name: sorted(values) for name, values in requested_groups.items()
        },
        "holdout_disasters": sorted(holdout_disasters),
        "split_audit": split_audit,
        "subsets": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in summary.items()},
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[gen] 完成：")
    for name, s in summary.items():
        print(f"  {name:8s}: buildings={s['buildings']:7d} changed={s['changed']:6d} "
              f"classes={s['class_counts']} -> {s['path']}")
    print(f"[gen] manifest → {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
