#!/usr/bin/env python3
"""
scripts/benchmarks/eval_floodnet_evidence.py — E14b：FloodNet 上的"受灾证据敏感度"评测

对应 docs 实施计划 E14（跨数据集泛化）第二部分。FloodNet 只有二档建筑标签（Flooded /
Non-Flooded），语义上不等价于 xBD 的 4 档损伤，所以不跑标准 mAP，而是检验一个更具体的问题：

    xBD 训练的检测器，把 FloodNet 的 flooded 建筑判成"有损伤"（minor/major/destroyed 任一）
    的比例，是否明显高于 non-flooded 建筑？

这直接对应 backend/recheck.py 的 EVIDENCE_CLASSES 假设——"水浸/受损都算风险证据"能否在
真实无人机低空影像（而不是训练检测器用的卫星正射影像）上站得住，是比单纯 mAP 更贴合
C2（不确定性驱动复核）主线的跨数据集证据。

做法：
    1. 用 xBD 检测器对每张 FloodNet 图跑推理（不设类别过滤，拿全部候选框）。
    2. 每个候选框与 GT 建筑框做 IoU 匹配（IoU>=阈值即算命中该建筑）。
    3. 统计：命中的 flooded / non-flooded 建筑里，被判成"有损伤类"的比例；
       以及检测器对建筑的整体召回率（IoU 匹配到任意候选框的比例，衡量域迁移下的定位能力）。

用法：
    python scripts/benchmarks/eval_floodnet_evidence.py \
        --manifest /home/lc/datasets/floodnet_eval/manifest.json \
        --weights runs/train/xbd_yolov8s_1024/weights/best.pt --device cuda:2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DAMAGE_CLASSES = {"minor-damage", "major-damage", "destroyed"}


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/lc/datasets/floodnet_eval/manifest.json")
    ap.add_argument("--weights", default="runs/train/xbd_yolov8s_1024/weights/best.pt")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou-match", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/benchmarks/e14_floodnet_evidence.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    class_names = model.names

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[: args.limit]

    stats = {
        "flooded": {"total": 0, "matched": 0, "matched_damage_cls": 0},
        "non_flooded": {"total": 0, "matched": 0, "matched_damage_cls": 0},
    }
    per_tile = []

    for i, entry in enumerate(manifest):
        img_path = entry["image"]
        result = model.predict(
            img_path, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        pred_boxes = []
        if result.boxes is not None:
            for xyxy, cls_id, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                pred_boxes.append(
                    {"bbox_xyxy": xyxy, "cls": class_names[int(cls_id)], "conf": float(conf)}
                )

        tile_matched = 0
        for gt in entry["boxes"]:
            label = "flooded" if gt["label"] == "flooded" else "non_flooded"
            stats[label]["total"] += 1
            best_iou, best_pred = 0.0, None
            for pred in pred_boxes:
                iou = _iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
                if iou > best_iou:
                    best_iou, best_pred = iou, pred
            if best_iou >= args.iou_match and best_pred is not None:
                stats[label]["matched"] += 1
                tile_matched += 1
                if best_pred["cls"] in DAMAGE_CLASSES:
                    stats[label]["matched_damage_cls"] += 1
        per_tile.append(
            {"tile_id": entry["tile_id"], "gt_boxes": len(entry["boxes"]), "matched": tile_matched}
        )
        if (i + 1) % 20 == 0:
            print(f"[eval] {i + 1}/{len(manifest)} tiles done")

    def _rate(label: str) -> dict:
        s = stats[label]
        recall = s["matched"] / s["total"] if s["total"] else 0.0
        damage_rate_among_matched = s["matched_damage_cls"] / s["matched"] if s["matched"] else 0.0
        return {**s, "recall": recall, "damage_rate_among_matched": damage_rate_among_matched}

    report = {
        "weights": args.weights,
        "manifest": args.manifest,
        "n_tiles": len(manifest),
        "iou_match": args.iou_match,
        "conf": args.conf,
        "flooded": _rate("flooded"),
        "non_flooded": _rate("non_flooded"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[eval] 报告 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
