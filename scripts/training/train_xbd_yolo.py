#!/usr/bin/env python3
"""
scripts/training/train_xbd_yolo.py — 在 xBD post_disaster 上微调 YOLOv8 损伤检测器

数据由 gen_xbd_yolo_dataset.py 产出（4 类：no/minor/major-damage、destroyed）。
目标：给 VLN grounding（vln_navigator.ground_with_yolo）一个域内、像素级精确的检测来源，
替代在 xBD 上几乎检出为 0 的 RescueNet 权重。

用法：
    python scripts/training/train_xbd_yolo.py \
        --data /home/lc/datasets/xbd_yolo/data.yaml \
        --model yolov8s.pt --imgsz 1024 --epochs 60 --batch 32 --device 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROJECT = REPO_ROOT / "runs" / "train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/lc/datasets/xbd_yolo/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default=str(DEFAULT_PROJECT))
    ap.add_argument("--name", default="xbd_yolov8s_1024")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # 卫星正射小目标：关掉会破坏尺度/朝向先验的强增广，保留轻度增广。
        mosaic=1.0,
        close_mosaic=10,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        plots=True,
        val=True,
    )
    print(f"[train] done → {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
