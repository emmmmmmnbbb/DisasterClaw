#!/usr/bin/env python3
"""
scripts/training/gen_xbd_yolo_dataset.py — xBD post_disaster 标注 → YOLO 检测数据集

为什么：现用 YOLO 是 RescueNet（低空斜拍）权重，在 xBD（卫星正射 1024²）上几乎检出为 0，
导致 VLN grounding 只能死靠开放词汇 VLM（俯视空间精度差→假到达、NE 大）。本脚本把 xBD
的 post_disaster building polygon + damage subtype 转成 YOLO 检测训练集，供微调一个域内检测器，
让 `vln_navigator.ground_with_yolo` 拿到像素级精确的目标点。

设计：
    - 仅用 post_disaster 标签（pre 无损伤分级）。
    - 训练池 = train + tier3（可选），独立测试 = test；train 池按瓦片随机切 train/val。
    - 每栋楼取 features.xy 的 POLYGON 外接 bbox（clamp 到图内），subtype → 4 类之一。
    - 图片用软链接（不复制 ~9000 张 PNG），labels 写 YOLO txt，最后写 data.yaml。

类别（对齐 perception.YOLO_LABEL_MAP 的中文标签）：
    0 no-damage / 1 minor-damage / 2 major-damage / 3 destroyed   （un-classified 跳过）

跨灾害留出集（P6 评测严谨性补丁）：
    train/test 共享同一批 10 种灾害事件（xBD 官方切分本身就是"同事件不同瓦片"），
    而 tier3 的 9 种灾害事件（joplin-tornado / lower-puna-volcano / moore-tornado /
    nepal-flooding / pinery-bushfire / portugal-wildfire / sunda-tsunami /
    tuscaloosa-tornado / woolsey-fire）与 train/test 完全不重叠，天然是一个未被利用
    的跨灾害留出集。若把 tier3 全部塞进训练池（旧默认行为），test 上的 mAP 只能衡量
    "同灾害事件内泛化"，不能衡量真正的跨灾害泛化。
    本脚本默认从 tier3 里排出 `--holdout-disasters` 指定的几种灾害事件，整体不参与
    任何训练/验证，单独落到 `<out>/holdout/` 作为 unseen-disaster 测试集；其余 tier3
    灾害事件仍并入训练池。

用法：
    python scripts/training/gen_xbd_yolo_dataset.py \
        --xbd-root /home/lc/datasets/xbd --out /home/lc/datasets/xbd_yolo \
        --splits train,tier3 --test-split test --val-frac 0.05 \
        --holdout-disasters nepal-flooding,moore-tornado,pinery-bushfire
    # 调试： --limit 200
    # 不留跨灾害集（旧行为）： --holdout-disasters ""
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from xbd_map import _parse_polygon_wkt  # noqa: E402

# subtype → (class_id, 英文类名)；un-classified / 其它 → 跳过
SUBTYPE_TO_CLASS: dict[str, int] = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}
CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def _bbox_from_poly(
    poly: list[tuple[float, float]], w: int, h: int
) -> tuple[float, float, float, float] | None:
    """POLYGON 顶点 → clamp 到图内的 (cx,cy,bw,bh) 归一化 YOLO bbox；退化则 None。"""
    if len(poly) < 3:
        return None
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(w), max(xs))
    y2 = min(float(h), max(ys))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:  # 退化/出界框
        return None
    cx = (x1 + x2) * 0.5 / w
    cy = (y1 + y2) * 0.5 / h
    return (cx, cy, bw / w, bh / h)


def _labels_for_tile(label_path: Path) -> tuple[list[str], dict[str, int]]:
    """读单张 post 标签 → YOLO 行列表 + 各类计数。"""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    w = int(meta.get("width") or meta.get("original_width") or 1024)
    h = int(meta.get("height") or meta.get("original_height") or 1024)
    feats = (data.get("features") or {}).get("xy") or []

    lines: list[str] = []
    counts: dict[str, int] = {}
    for f in feats:
        props = f.get("properties") or {}
        sub = (props.get("subtype") or "").strip()
        cid = SUBTYPE_TO_CLASS.get(sub)
        if cid is None:
            continue
        poly = _parse_polygon_wkt(f.get("wkt", ""))
        bbox = _bbox_from_poly(poly, w, h)
        if bbox is None:
            continue
        cx, cy, bw, bh = bbox
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        counts[sub] = counts.get(sub, 0) + 1
    return lines, counts


def _post_label_paths(xbd_root: Path, split: str) -> list[Path]:
    labels_dir = xbd_root / split / "labels"
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.glob("*post_disaster.json"))


_DISASTER_RE = re.compile(r"_\d+_(?:pre|post)_disaster$")


def disaster_of(label_path: Path) -> str:
    """从 `<disaster>_<tile idx>_post_disaster.json` 的 stem 里抽出灾害事件名。"""
    return _DISASTER_RE.sub("", label_path.stem)


def _emit(
    label_paths: list[tuple[Path, str]],  # (label_path, src_split)
    xbd_root: Path,
    out_root: Path,
    subset: str,  # train/val/test
    limit: int,
) -> dict:
    img_dir = out_root / subset / "images"
    lab_dir = out_root / subset / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_box = 0
    n_empty = 0
    cls_total: dict[str, int] = {}
    for label_path, src_split in label_paths:
        if limit and n_img >= limit:
            break
        tile_id = label_path.stem
        img_src = xbd_root / src_split / "images" / f"{tile_id}.png"
        if not img_src.exists():
            continue
        lines, counts = _labels_for_tile(label_path)
        if not lines:
            n_empty += 1
            continue  # YOLO 训练跳过纯空标（无可用建筑框）
        # 图软链（同名可能跨 split 重名概率极低，tile_id 唯一）
        link = img_dir / f"{tile_id}.png"
        if not link.exists():
            try:
                link.symlink_to(img_src.resolve())
            except FileExistsError:
                pass
        (lab_dir / f"{tile_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_img += 1
        n_box += len(lines)
        for k, v in counts.items():
            cls_total[k] = cls_total.get(k, 0) + v
    return {
        "subset": subset,
        "images": n_img,
        "boxes": n_box,
        "skipped_empty": n_empty,
        "class_counts": cls_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", default="/home/lc/datasets/xbd")
    ap.add_argument("--out", default="/home/lc/datasets/xbd_yolo")
    ap.add_argument("--splits", default="train,tier3", help="训练池 split（逗号分隔）")
    ap.add_argument("--test-split", default="test", help="独立测试 split（空=不生成）")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="每个 subset 最多张数（0=全部，调试用）")
    ap.add_argument(
        "--holdout-disasters",
        default="nepal-flooding,moore-tornado,pinery-bushfire",
        help=(
            "整体排出训练池、单独落到 <out>/holdout/ 的灾害事件名（逗号分隔，"
            "必须来自 tier3，且与 train/test 不重叠）；传空字符串关闭（旧行为，"
            "tier3 全部并入训练池）。"
        ),
    )
    args = ap.parse_args()

    xbd_root = Path(args.xbd_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    holdout_disasters = {
        d.strip() for d in args.holdout_disasters.split(",") if d.strip()
    }

    # 训练池（排出 holdout 灾害事件）
    pool: list[tuple[Path, str]] = []
    holdout_pool: list[tuple[Path, str]] = []
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        paths = _post_label_paths(xbd_root, sp)
        n_hold = 0
        for p in paths:
            if disaster_of(p) in holdout_disasters:
                holdout_pool.append((p, sp))
                n_hold += 1
            else:
                pool.append((p, sp))
        print(f"[gen] {sp}: {len(paths)} post labels（其中 {n_hold} 张划入 holdout）")
    rng.shuffle(pool)
    n_val = int(len(pool) * args.val_frac)
    val_pool = pool[:n_val]
    train_pool = pool[n_val:]

    stats = []
    stats.append(_emit(train_pool, xbd_root, out_root, "train", args.limit))
    stats.append(_emit(val_pool, xbd_root, out_root, "val", args.limit))
    if args.test_split:
        test_paths = [(p, args.test_split) for p in _post_label_paths(xbd_root, args.test_split)]
        stats.append(_emit(test_paths, xbd_root, out_root, "test", args.limit))
    if holdout_pool:
        stats.append(_emit(holdout_pool, xbd_root, out_root, "holdout", args.limit))
        holdout_by_disaster: dict[str, int] = {}
        for p, _sp in holdout_pool:
            d = disaster_of(p)
            holdout_by_disaster[d] = holdout_by_disaster.get(d, 0) + 1
        print(f"[gen] holdout 灾害事件分布: {holdout_by_disaster}")

    # data.yaml
    yaml_lines = [
        f"path: {out_root}",
        "train: train/images",
        "val: val/images",
    ]
    if args.test_split:
        yaml_lines.append("test: test/images")
    if holdout_pool:
        # 注意：ultralytics 只会给 train/val/test 自动拼 `path:` 前缀，自定义 key
        # （如 holdout）不会，所以这里必须写绝对路径，否则 `model.val(split="holdout")`
        # 会去当前工作目录找 holdout/images 而报 FileNotFoundError。
        yaml_lines.append(f"holdout: {out_root / 'holdout' / 'images'}")
    yaml_lines.append(f"nc: {len(CLASS_NAMES)}")
    yaml_lines.append("names:")
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {name}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("\n[gen] 完成：")
    for s in stats:
        print(f"  {s['subset']:5s}: imgs={s['images']:6d} boxes={s['boxes']:7d} "
              f"empty_skip={s['skipped_empty']:5d} classes={s['class_counts']}")
    print(f"[gen] data.yaml → {out_root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
