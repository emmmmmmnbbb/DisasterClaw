#!/usr/bin/env python3
"""
scripts/training/gen_rescuenet_yolo_testset.py — RescueNet 分割标注 → YOLO 检测测试集

为什么（对应文档 E14"跨数据集泛化"）：现有 `xbd_yolov8s_1024` 检测器只在 xBD（卫星正射、
train/test 同源灾害事件）上验证过 mAP，此前 E13 的"跨灾害"其实是 xBD 内部同一数据集换灾种，
留出集也因训练切分 bug 被污染。RescueNet 是完全独立的第三方数据集：
    - 不同传感器/视角：低空无人机斜拍（vs xBD 卫星正射）；
    - 不同灾害事件：Hurricane Michael 低空实拍（vs xBD 训练池的灾害事件）；
    - 但类别体系恰好一致：Building No/Minor/Major Damage + Total Destruction 四档，
      与 backend/perception.py 的 4 类损伤标签直接对齐。
用这个数据集零样本（不重新训练）跑一次 mAP，是比"同数据集换灾种"更硬的跨域泛化证据：
如果 mAP 显著低于 xBD test，说明检测器学到的是"xBD 卫星正射域"特征，尚未验证对真实无人机
视角的泛化；如果保持一定水平，则是更有说服力的泛化证据。两种结果都如实报告。

标注格式：RescueNet 测试集（`test-label-img/*.png`）是单通道语义分割 mask，像素值 0~10：
    0 Background / 1 Water / 2 Building No-Damage / 3 Building Minor-Damage /
    4 Building Major-Damage / 5 Building Total-Destruction / 6 Road-Clear /
    7 Road-Blocked / 8 Vehicle / 9 Tree / 10 Pool
本脚本只取 2~5（4 类建筑损伤），对每个类别做连通域分析，每个连通域的外接框即一个 YOLO 目标框
（做法与 xBD"整栋建筑一个框"不同——RescueNet 无实例级标注，只能退化为"每块连通区域一个框"，
这是该数据集本身的标注粒度限制，非本脚本引入的近似）。

用法：
    python scripts/training/gen_rescuenet_yolo_testset.py \
        --rescuenet-root /home/lc/datasets/rescuenet --out /home/lc/datasets/rescuenet_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# RescueNet mask 像素值 → (YOLO class_id, 英文类名)，对齐 xBD 的 4 类损伤体系
MASK_VALUE_TO_CLASS: dict[int, tuple[int, str]] = {
    2: (0, "no-damage"),
    3: (1, "minor-damage"),
    4: (2, "major-damage"),
    5: (3, "destroyed"),
}
CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
MIN_AREA_PX = 900  # ~30x30px 连通域面积下限，滤掉分割噪声碎片


def _boxes_for_mask(mask_path: Path) -> tuple[list[str], dict[str, int]]:
    arr = np.array(Image.open(mask_path))
    h, w = arr.shape[:2]
    lines: list[str] = []
    counts: dict[str, int] = {}
    for value, (cid, name) in MASK_VALUE_TO_CLASS.items():
        binary = arr == value
        if not binary.any():
            continue
        labeled, n = ndi.label(binary)
        for comp_id in range(1, n + 1):
            ys, xs = np.where(labeled == comp_id)
            if ys.size < MIN_AREA_PX:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)
            cx = (x1 + x2 + 1) * 0.5 / w
            cy = (y1 + y2 + 1) * 0.5 / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            counts[name] = counts.get(name, 0) + 1
    return lines, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescuenet-root", default="/home/lc/datasets/rescuenet")
    ap.add_argument("--out", default="/home/lc/datasets/rescuenet_yolo")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.rescuenet_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    img_dir = out_root / "test" / "images"
    lab_dir = out_root / "test" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted((root / "test-label-img").glob("*_lab.png"))
    if args.limit:
        mask_paths = mask_paths[: args.limit]

    n_img = n_box = n_empty = 0
    cls_total: dict[str, int] = {}
    for mask_path in mask_paths:
        tile_id = mask_path.stem.replace("_lab", "")
        img_src = root / "test-org-img" / f"{tile_id}.jpg"
        if not img_src.exists():
            continue
        lines, counts = _boxes_for_mask(mask_path)
        if not lines:
            n_empty += 1
            continue
        link = img_dir / f"{tile_id}.jpg"
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

    yaml_lines = [
        f"path: {out_root}",
        "train: test/images",  # ultralytics 要求字段存在；本数据集只作零样本测试，不训练
        "val: test/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {i}: {name}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"[gen] RescueNet test: imgs={n_img} boxes={n_box} empty_skip={n_empty} classes={cls_total}")
    print(f"[gen] data.yaml → {out_root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
