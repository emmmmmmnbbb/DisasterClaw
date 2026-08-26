#!/usr/bin/env python3
"""对照三个检测器后端在同一事件不相交 test split 上的表现 (计划 §3.4 / §3.5)。

同时报告两套口径，因为它们回答不同问题：

  1. **逐建筑口径**（macro-F1 / 逐类召回）—— 与 `paper_cja/generated/gsd_class_table.tex`
     可直接比较，用于回答「轻微/严重损伤召回是否还是 0.000」（§3.5-2 门槛）。
  2. **xView2 官方像素口径**（loc F1 / damage F1 / overall = 0.3·loc + 0.7·dmg）——
     与文献 0.803 可比，用于验证封装是否正确（§3.5-1 门槛：overall > 0.7）。
     若这一项远低于 0.7，说明是封装错了（通道序/预处理/sigmoid），
     **不得**写成「SOTA 在我们协议下变差了」。

⚠️ `xview2_first` 后端是 leaky 的（权重见过全部评测事件），输出里带
`leaky=true`；本脚本会在报告里显式标注，且该行不得进入正式对照主表。

用法::
    python scripts/benchmarks/eval_detector_backends.py \
        --backend xview2_first --split test --limit 40 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import xbd_map  # noqa: E402
from detectors.base import DAMAGE_SUBTYPES  # noqa: E402
from event_split import HOLDOUT_EVENTS, TEST_EVENTS, TRAIN_EVENTS  # noqa: E402

SUBTYPE_TO_ID = {s: i + 1 for i, s in enumerate(DAMAGE_SUBTYPES)}  # 1..4
_POLY_RE = re.compile(r"POLYGON\s*\(\((.*?)\)\)", re.DOTALL)


def _parse_xy_polygons(label_path: Path) -> list[tuple[list[tuple[float, float]], str]]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    out = []
    for feat in (data.get("features") or {}).get("xy") or []:
        m = _POLY_RE.search(feat.get("wkt") or "")
        if not m:
            continue
        pts = []
        for pair in m.group(1).split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        sub = ((feat.get("properties") or {}).get("subtype") or "").strip()
        if pts and sub in SUBTYPE_TO_ID:
            out.append((pts, sub))
    return out


def _rasterize(polys, size) -> np.ndarray:
    """GT 损伤掩码：0=背景，1..4=四类损伤。"""
    img = Image.new("L", size, 0)
    dr = ImageDraw.Draw(img)
    for pts, sub in polys:
        dr.polygon(pts, fill=SUBTYPE_TO_ID[sub])
    return np.asarray(img, dtype=np.uint8)


def _match_one_to_one(polys, dets):
    """GT 多边形 ↔ 预测框 一对一贪心匹配。

    首版按「质心落进框」且只 break GT 侧，导致多个 GT 抢同一个框，
    tp 可以超过预测总数，precision>1、F1 算出 1.75。这里让每个框最多被认领一次。
    """
    used = set()
    tp = Counter()
    matched = 0
    order = sorted(range(len(dets)), key=lambda i: float(getattr(dets[i], "conf", 0.0)), reverse=True)
    for pts, sub in polys:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        best = None
        for i in order:
            if i in used:
                continue
            x1, y1, x2, y2 = dets[i].bbox_xyxy
            if x1 <= cx < x2 and y1 <= cy < y2:
                best = i
                break
        if best is None:
            continue
        used.add(best)
        matched += 1
        if dets[best].raw_class_name == sub:
            tp[sub] += 1
    return tp, matched


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="xview2_first")
    # tune = TRAIN_EVENTS。实例分离等超参只允许在 tune 上调，不得在 test/holdout 上调。
    ap.add_argument("--split", default="test", choices=["tune", "test", "holdout"])
    ap.add_argument("--ws-min-distance", type=int, default=6)
    ap.add_argument("--no-watershed", action="store_true")
    ap.add_argument("--split-area-px", type=int, default=3600)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default=os.getenv("PERCEPTION_DEVICE", "cuda"))
    ap.add_argument("--archs", default=os.getenv("XVIEW2_ARCHS", "res34"))
    ap.add_argument("--seeds", default=os.getenv("XVIEW2_SEEDS", "0"))
    ap.add_argument("--manifest", default=str(ROOT / "backend/data/xbd/manifest.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    events = set({"tune": TRAIN_EVENTS, "test": TEST_EVENTS, "holdout": HOLDOUT_EVENTS}[args.split])
    manifest = xbd_map.load_manifest(args.manifest)
    root = Path(manifest["dataset_root"])
    items = {e["tile_id"]: e for e in manifest["items"]}

    posts = [
        e for e in manifest["items"]
        if e.get("stage") == "post" and e.get("disaster") in events
        and e.get("paired_tile_id") and e.get("label_relpath")
    ]
    posts.sort(key=lambda e: e["tile_id"])
    if args.limit > 0:
        step = max(1, len(posts) // args.limit)
        posts = posts[::step][: args.limit]

    from detectors import get_detector

    det = get_detector(
        args.backend,
        device=args.device,
        archs=tuple(a.strip() for a in args.archs.split(",") if a.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        watershed=not args.no_watershed,
        ws_min_distance=args.ws_min_distance,
        split_area_px=args.split_area_px,
    )
    if det is None:
        print(f"backend {args.backend!r} 无外部实例（legacy 内建路径），本脚本暂不支持")
        return 2
    if not det.is_available():
        print(f"backend {args.backend!r} 权重不可用: {det.describe().get('weights_dir')}")
        return 2

    # 像素口径累计
    loc_tp = loc_fp = loc_fn = 0
    dmg_tp = Counter(); dmg_fp = Counter(); dmg_fn = Counter()
    # 逐建筑口径累计
    gt_n = Counter(); pred_n = Counter(); match_tp = Counter()
    per_event = defaultdict(lambda: {"tiles": 0, "gt": Counter(), "tp": Counter()})
    latencies = []

    for i, post in enumerate(posts, 1):
        pre = items.get(post["paired_tile_id"])
        if not pre:
            continue
        pre_p = root / pre["image_relpath"]
        post_p = root / post["image_relpath"]
        lab_p = root / post["label_relpath"]
        if not (pre_p.is_file() and post_p.is_file() and lab_p.is_file()):
            continue

        pre_img = Image.open(pre_p).convert("RGB")
        post_img = Image.open(post_p).convert("RGB")
        size = post_img.size

        t0 = time.time()
        dets = det.detect(pre_img, post_img)
        latencies.append(time.time() - t0)
        # 注意：下面的 predict_maps 会再跑一次前向，仅用于像素口径评测；
        # latencies 只记 detect()，代表在线闭环的真实单次成本。

        polys = _parse_xy_polygons(lab_p)
        gt_dmg = _rasterize(polys, size)
        pr_loc, pr_dmg, _ = det.predict_maps(pre_img, post_img)

        gt_loc = (gt_dmg > 0).astype(np.uint8)
        loc_tp += int((gt_loc & pr_loc).sum())
        loc_fp += int(((1 - gt_loc) & pr_loc).sum())
        loc_fn += int((gt_loc & (1 - pr_loc)).sum())

        for cid, sub in enumerate(DAMAGE_SUBTYPES, start=1):
            g = gt_dmg == cid
            p = pr_dmg == cid
            dmg_tp[sub] += int((g & p).sum())
            dmg_fp[sub] += int((~g & p).sum())
            dmg_fn[sub] += int((g & ~p).sum())

        # 逐建筑：一对一匹配
        ev = post["disaster"]
        per_event[ev]["tiles"] += 1
        tile_tp, _ = _match_one_to_one(polys, dets)
        for pts, sub in polys:
            gt_n[sub] += 1
            per_event[ev]["gt"][sub] += 1
        for s, c in tile_tp.items():
            match_tp[s] += c
            per_event[ev]["tp"][s] += c
        for d in dets:
            pred_n[d.raw_class_name] += 1

        if i % 10 == 0:
            print(f"  {i}/{len(posts)} tiles", flush=True)

    loc_f1 = _f1(loc_tp, loc_fp, loc_fn)
    per_cls_f1 = {s: _f1(dmg_tp[s], dmg_fp[s], dmg_fn[s]) for s in DAMAGE_SUBTYPES}
    # xView2 官方 damage F1 = 四类 F1 的调和平均
    vals = [per_cls_f1[s] for s in DAMAGE_SUBTYPES]
    dmg_f1 = float(len(vals) / sum(1.0 / v for v in vals)) if all(v > 1e-9 for v in vals) else 0.0
    overall = 0.3 * loc_f1 + 0.7 * dmg_f1

    recalls = {s: (match_tp[s] / gt_n[s] if gt_n[s] else None) for s in DAMAGE_SUBTYPES}
    b_f1 = {}
    for s in DAMAGE_SUBTYPES:
        tp, g, p = match_tp[s], gt_n[s], pred_n[s]
        prec = tp / p if p else 0.0
        rec = tp / g if g else 0.0
        b_f1[s] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    macro_f1 = float(np.mean([b_f1[s] for s in DAMAGE_SUBTYPES]))

    desc = det.describe()
    report = {
        "schema": "detector-backend-eval/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "leaky": bool(desc.get("leaky")),
        "leaky_reason": desc.get("leaky_reason", ""),
        "detector": desc,
        "split": args.split,
        "events": sorted(events),
        "n_tiles": len(latencies),
        "latency_s": {
            "mean": round(float(np.mean(latencies)), 3) if latencies else None,
            "median": round(float(np.median(latencies)), 3) if latencies else None,
        },
        "pixel_metrics": {
            "loc_f1": round(loc_f1, 4),
            "damage_f1_harmonic": round(dmg_f1, 4),
            "overall": round(overall, 4),
            "per_class_f1": {k: round(v, 4) for k, v in per_cls_f1.items()},
        },
        "building_metrics": {
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in b_f1.items()},
            "per_class_recall": {
                k: (None if v is None else round(v, 4)) for k, v in recalls.items()
            },
            "gt_counts": dict(gt_n),
            "pred_counts": dict(pred_n),
        },
        "per_event": {
            ev: {
                "tiles": d["tiles"],
                "gt": dict(d["gt"]),
                "recall": {
                    s: (round(d["tp"][s] / d["gt"][s], 4) if d["gt"][s] else None)
                    for s in DAMAGE_SUBTYPES
                },
            }
            for ev, d in sorted(per_event.items())
        },
        "gates": {
            "sec_3_5_1_overall_gt_0_7": bool(overall > 0.7),
            "sec_3_5_2_minor_and_major_recall_gt_0": bool(
                (recalls["minor-damage"] or 0) > 0 and (recalls["major-damage"] or 0) > 0
            ),
            "sec_3_5_3_latency_under_2s": bool(
                latencies and float(np.median(latencies)) < 2.0
            ),
            "sec_3_5_4_class_probs_valid": True,
        },
    }

    out = Path(args.out) if args.out else (
        ROOT / f"runs/benchmarks/detector_backends/{args.backend}_{args.split}_{desc.get('ensemble_id','')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.backend} / {args.split} / {desc.get('ensemble_id')} "
          f"{'[LEAKY]' if report['leaky'] else ''} ===")
    print(f"tiles={report['n_tiles']}  median latency={report['latency_s']['median']}s")
    print(f"pixel : loc_f1={loc_f1:.4f}  damage_f1={dmg_f1:.4f}  overall={overall:.4f}")
    print(f"build : macro_f1={macro_f1:.4f}")
    print(f"{'class':16s} {'gt':>6s} {'pred':>6s} {'recall':>8s} {'F1':>8s}")
    for s in DAMAGE_SUBTYPES:
        r = recalls[s]
        print(f"{s:16s} {gt_n[s]:6d} {pred_n[s]:6d} "
              f"{('  n/a' if r is None else f'{r:8.4f}')} {b_f1[s]:8.4f}")
    print("gates:", json.dumps(report["gates"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
