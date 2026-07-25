#!/usr/bin/env python3
"""
scripts/training/gen_floodnet_yolo_testset.py — FloodNet 分割标注 → 建筑框（二分类）测试集

背景（E14 跨数据集泛化，第二个数据集）：FloodNet 只有 2 档建筑标签（Flooded / Non-Flooded），
不是 xBD/RescueNet 的 4 档损伤体系——"被水淹"不等于"结构性损毁"（可能只是院子进水，房子完好），
所以本脚本**不**把 FloodNet 硬凑成 4 类跑 mAP（会引入标签语义不一致的假结论），而是产出一份
二分类 GT（0=undamaged 对应 Building Non-Flooded，1=flood-evidence 对应 Building Flooded），
配合 `scripts/benchmarks/eval_floodnet_evidence.py` 做"证据敏感度"评测：
检测器把 flooded 建筑判成 minor/major/destroyed（任一"有损伤"类）的比例，是否明显高于
non-flooded 建筑——这直接对应 backend/recheck.py 里 `EVIDENCE_CLASSES` 的假设
（水/受损同属"风险证据"）能否在真实无人机低空影像上站得住。

FloodNet mask 像素值（`test-label-img/*.png`，单通道）：
    0 Background / 1 Building-Flooded / 2 Building-Non-Flooded / 3 Road-Flooded /
    4 Road-Non-Flooded / 5 Water / 6 Tree / 7 Vehicle / 8 Pool / 9 Grass
本脚本只取 1、2（建筑类），每个连通域一个框。

用法：
    python scripts/training/gen_floodnet_yolo_testset.py \
        --floodnet-root /home/lc/datasets/floodnet/extracted/FloodNet-Supervised_v1.0/test \
        --out /home/lc/datasets/floodnet_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

MASK_VALUE_TO_LABEL = {1: "flooded", 2: "non_flooded"}
MIN_AREA_PX = 900


def _boxes_for_mask(mask_path: Path) -> list[dict]:
    arr = np.array(Image.open(mask_path))
    h, w = arr.shape[:2]
    boxes: list[dict] = []
    for value, label in MASK_VALUE_TO_LABEL.items():
        binary = arr == value
        if not binary.any():
            continue
        labeled, n = ndi.label(binary)
        for comp_id in range(1, n + 1):
            ys, xs = np.where(labeled == comp_id)
            if ys.size < MIN_AREA_PX:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            boxes.append(
                {
                    "label": label,
                    "bbox_xyxy": [x1, y1, x2 + 1, y2 + 1],
                    "bbox_norm": [
                        (x1 + x2 + 1) * 0.5 / w,
                        (y1 + y2 + 1) * 0.5 / h,
                        (x2 - x1 + 1) / w,
                        (y2 - y1 + 1) / h,
                    ],
                }
            )
    return boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--floodnet-root",
        default="/home/lc/datasets/floodnet/extracted/FloodNet-Supervised_v1.0/test",
    )
    ap.add_argument("--out", default="/home/lc/datasets/floodnet_eval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.floodnet_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted((root / "test-label-img").glob("*_lab.png"))
    if args.limit:
        mask_paths = mask_paths[: args.limit]

    manifest = []
    n_flooded = n_non_flooded = 0
    for mask_path in mask_paths:
        tile_id = mask_path.stem.replace("_lab", "")
        img_src = root / "test-org-img" / f"{tile_id}.jpg"
        if not img_src.exists():
            continue
        boxes = _boxes_for_mask(mask_path)
        if not boxes:
            continue
        link = img_dir / f"{tile_id}.jpg"
        if not link.exists():
            try:
                link.symlink_to(img_src.resolve())
            except FileExistsError:
                pass
        manifest.append({"tile_id": tile_id, "image": str(link), "boxes": boxes})
        n_flooded += sum(1 for b in boxes if b["label"] == "flooded")
        n_non_flooded += sum(1 for b in boxes if b["label"] == "non_flooded")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[gen] FloodNet test: tiles={len(manifest)} "
        f"flooded_boxes={n_flooded} non_flooded_boxes={n_non_flooded}"
    )
    print(f"[gen] manifest → {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
