#!/usr/bin/env python3
"""Evaluate strict-split xBD building localization (U-Net) checkpoints.

Reports Dice/IoU on val/test/holdout, proposal recall@IoU=0.5, and
building-only detection AP@0.5 (class-agnostic). Optionally, with
--change-ckpt + --xbd-root, attaches change_perception labels for a
4-class detection-style mAP comparable to YOLO metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SUBTYPE_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_boxes(mask: np.ndarray, min_area: int = 64) -> list[list[float]]:
    import cv2

    binary = (mask > 0).astype("uint8") * 255
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for lab in range(1, n_labels):
        x, y, w, h, area = stats[lab]
        if area < min_area:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def _proposal_recall(pred_boxes: list[list[float]], gt_boxes: list[list[float]], thr: float = 0.5) -> float:
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    hit = 0
    for gt in gt_boxes:
        if any(_iou(gt, pred) >= thr for pred in pred_boxes):
            hit += 1
    return hit / len(gt_boxes)


def _average_precision(
    preds: list[tuple[list[float], float]],
    gts: list[list[float]],
    thr: float = 0.5,
) -> float:
    """VOC-style AP for one class (or class-agnostic building). preds=(bbox, score)."""
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    matched = [False] * len(gts)
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)
    for i, (box, _score) in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            iou = _iou(box, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= thr and best_j >= 0:
            matched[best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, len(gts))
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recalls >= t
        ap += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 11.0


def _gt_damage_boxes_from_xbd(
    xbd_root: Path, pre_stem: str, min_area: int = 64,
) -> list[tuple[list[float], int]]:
    """Load post JSON for a pre tile stem → (bbox_xyxy, class_id) list."""
    from xbd_map import _parse_polygon_wkt

    post_stem = pre_stem.replace("_pre_disaster", "_post_disaster")
    label_path = None
    for split in ("train", "tier3", "test"):
        candidate = xbd_root / split / "labels" / f"{post_stem}.json"
        if candidate.exists():
            label_path = candidate
            break
    if label_path is None:
        return []
    data = json.loads(label_path.read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    w = int(meta.get("width") or meta.get("original_width") or 1024)
    h = int(meta.get("height") or meta.get("original_height") or 1024)
    out: list[tuple[list[float], int]] = []
    for feat in (data.get("features") or {}).get("xy") or []:
        props = feat.get("properties") or {}
        sub = (props.get("subtype") or "").strip()
        cid = SUBTYPE_TO_ID.get(sub)
        if cid is None:
            continue
        poly = _parse_polygon_wkt(feat.get("wkt", ""))
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(w), max(xs))
        y2 = min(float(h), max(ys))
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        out.append(([x1, y1, x2, y2], cid))
    return out


def _resolve_post_image(xbd_root: Path, pre_stem: str) -> Path | None:
    post_stem = pre_stem.replace("_pre_disaster", "_post_disaster")
    for split in ("train", "tier3", "test"):
        candidate = xbd_root / split / "images" / f"{post_stem}.png"
        if candidate.exists():
            return candidate
    return None


def _crop_pair(pre_img: Image.Image, post_img: Image.Image, bbox, margin: float = 0.25):
    w, h = pre_img.size
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    mx, my = bw * margin, bh * margin
    left = max(0.0, x1 - mx)
    top = max(0.0, y1 - my)
    right = min(float(w), x2 + mx)
    bottom = min(float(h), y2 + my)
    box = (int(left), int(top), max(int(left) + 1, int(right)), max(int(top) + 1, int(bottom)))
    pre = pre_img.crop(box).resize((96, 96), Image.BILINEAR)
    post = post_img.crop(box).resize((96, 96), Image.BILINEAR)
    return pre, post


def _eval_split(
    localizer,
    data_dir: Path,
    subset: str,
    limit: int = 0,
    change_model=None,
    xbd_root: Path | None = None,
) -> dict:
    img_dir = data_dir / subset / "images"
    mask_dir = data_dir / subset / "masks"
    paths = sorted(img_dir.glob("*.png"))
    if limit:
        paths = paths[:limit]
    if not paths:
        return {"n": 0}

    dices = []
    ious = []
    recalls = []
    n_pred = 0
    n_gt = 0
    all_pred_building: list[tuple[list[float], float]] = []
    all_gt_building: list[list[float]] = []
    # class-aware: per-image then pool — actually AP needs global list with image-local matching.
    # Keep per-image APs and average (mean AP) which is stable for reporting.
    building_aps = []
    class_aps = {name: [] for name in CLASS_NAMES}

    for img_path in paths:
        mask_path = mask_dir / img_path.name
        gt = np.array(Image.open(mask_path).convert("L")) > 127
        probs = localizer.predict_proba(img_path)
        pred = probs >= localizer.conf_threshold
        inter = float(np.logical_and(pred, gt).sum())
        union = float(np.logical_or(pred, gt).sum())
        pred_sum = float(pred.sum())
        gt_sum = float(gt.sum())
        dice = (2 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)
        dices.append(dice)
        ious.append(iou)

        from building_localization import mask_to_boxes

        dets = mask_to_boxes(
            probs,
            conf_threshold=localizer.conf_threshold,
            min_area=localizer.min_area,
            dilate_k=localizer.dilate_k,
        )
        pred_boxes = [d["bbox_xyxy"] for d in dets]
        pred_scores = [float(d.get("conf", 0.0)) for d in dets]
        gt_boxes = _mask_boxes(gt.astype("uint8") * 255)
        recalls.append(_proposal_recall(pred_boxes, gt_boxes, 0.5))
        n_pred += len(pred_boxes)
        n_gt += len(gt_boxes)
        building_aps.append(
            _average_precision(list(zip(pred_boxes, pred_scores)), gt_boxes, 0.5)
        )
        all_pred_building.extend(zip(pred_boxes, pred_scores))
        all_gt_building.extend(gt_boxes)

        if change_model is not None and xbd_root is not None:
            gt_labeled = _gt_damage_boxes_from_xbd(xbd_root, img_path.stem)
            post_path = _resolve_post_image(xbd_root, img_path.stem)
            if post_path is None:
                continue
            pre_img = Image.open(img_path).convert("RGB")
            post_img = Image.open(post_path).convert("RGB")
            labeled_preds: list[tuple[list[float], float, int]] = []
            for det in dets:
                bbox = det["bbox_xyxy"]
                try:
                    pre_c, post_c = _crop_pair(pre_img, post_img, bbox)
                    pred_cp = change_model.predict(pre_c, post_c)
                    probs_cp = pred_cp.class_probs
                    best = max(probs_cp.items(), key=lambda kv: float(kv[1]))
                    cid = SUBTYPE_TO_ID.get(str(best[0]))
                    if cid is None:
                        continue
                    labeled_preds.append((bbox, float(best[1]), cid))
                except Exception:
                    continue
            for cname in CLASS_NAMES:
                cid = SUBTYPE_TO_ID[cname]
                preds_c = [(b, s) for b, s, c in labeled_preds if c == cid]
                gts_c = [b for b, c in gt_labeled if c == cid]
                class_aps[cname].append(_average_precision(preds_c, gts_c, 0.5))

    result = {
        "n": len(paths),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "proposal_recall_iou50": float(np.mean(recalls)),
        "building_AP50_mean_per_image": float(np.mean(building_aps)) if building_aps else 0.0,
        "mean_pred_boxes": n_pred / max(1, len(paths)),
        "mean_gt_boxes": n_gt / max(1, len(paths)),
    }
    if change_model is not None:
        per_class = {
            name: float(np.mean(vals)) if vals else 0.0 for name, vals in class_aps.items()
        }
        result["damage_mAP50_mean_per_image"] = float(np.mean(list(per_class.values())))
        result["damage_AP50_per_class"] = per_class
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/lc/datasets/xbd_loc_strict_v1")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-event-disjoint", action="store_true")
    ap.add_argument(
        "--reference-yolo-manifest",
        default="/home/lc/datasets/xbd_yolo_strict_v1/manifest.json",
    )
    ap.add_argument(
        "--change-ckpt",
        default="",
        help="Optional change_perception checkpoint for 4-class detection mAP",
    )
    ap.add_argument("--xbd-root", default="/home/lc/datasets/xbd")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("split_audit") or {}
    if args.require_event_disjoint:
        from building_localization import assert_event_disjoint, assert_events_match_reference

        assert_event_disjoint(data_dir)
        assert_events_match_reference(manifest, Path(args.reference_yolo_manifest))

    from building_localization import load_building_localizer

    localizer = load_building_localizer(weights, device=args.device)

    change_model = None
    xbd_root = None
    if args.change_ckpt:
        import os

        os.environ["CHANGE_PERCEPTION_CKPT"] = str(Path(args.change_ckpt).expanduser().resolve())
        import change_perception as cp

        change_model = cp.get_change_perception()
        xbd_root = Path(args.xbd_root).expanduser().resolve()

    results = {}
    for split in ("val", "test", "holdout"):
        if not (data_dir / split / "images").exists():
            continue
        print(f"[eval-loc] evaluating {split} ...")
        results[split] = _eval_split(
            localizer,
            data_dir,
            split,
            limit=args.limit,
            change_model=change_model,
            xbd_root=xbd_root,
        )

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "data_dir": str(data_dir),
        "manifest_sha256": _sha256(manifest_path),
        "event_split": audit.get("events"),
        "change_ckpt": args.change_ckpt or None,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
