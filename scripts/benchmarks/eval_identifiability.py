#!/usr/bin/env python3
"""P4 可识别性前置检查 (计划 §2.5-8 / §11.6 / E4 前置检查)。

回答一个问题：**在新观测模型 + SOTA 感知底座下，"降高再看一眼"到底改不改变答案？**

旧值（`AGENT_VQA_EXPERIMENT_STATUS.md` D6 holdout）：`n_flip=7`、`n_correctable=2`。
因为可纠正样本几乎为零，任何重观测策略之间必然无差异，E4 只能按分支 C
（不可识别）报告。本脚本重算这两个量。

## 测的是什么，不是什么

测**感知层的 headroom**：同一 ROI 在巡航档与下限档各观测一次，
证据（逐建筑损伤类）是否改变、是否变对。这是任何重观测策略能利用的**上界**——
若证据不变，则没有任何策略能靠重观测改善答案。

**不测** VLM 智能体的准确率。因此这里用规则式答案推导（等价于
`agent_vqa._rule_fallback`），刻意把 VLM 这一层剥掉，避免用 VLM 噪声
掩盖或伪造感知层的信号。

## 观测模型的关键细节

检测器要求 pre/post 同尺寸，但 post 必须**只携带该高度下真实可得的信息**：

    post: 按高度渲染到 1024 px（巡航 = 1.5 m/px）→ 再上采样到原生像素数
    pre : 直接按原生 GSD 渲染（灾前影像是已归档产品，与高度无关）

上采样 post 不增加信息，所以"巡航看得糊、下限看得清"这个信息不对称被如实保留；
若反过来把 pre 降到 post 的分辨率，就等于让降高同时恢复参考通道，
那是 review2 B2 批评的"撤销自己刚加的降质"的翻版。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import fov_ladder as FL  # noqa: E402
import mosaic as mosaic_mod  # noqa: E402
import xbd_map  # noqa: E402
from detectors.base import DAMAGE_SUBTYPES  # noqa: E402
from event_split import EVAL_EVENTS, HOLDOUT_EVENTS, TEST_EVENTS  # noqa: E402

_POLY_RE = re.compile(r"POLYGON\s*\(\((.*?)\)\)", re.DOTALL)

BEARINGS = ("北", "东北", "东", "东南", "南", "西南", "西", "西北")
SEVERE = ("major-damage", "destroyed")


def _gt_buildings(label_path: Path, entry: dict) -> list[dict]:
    """从 xBD 标注读 ROI 内建筑：像素质心 + 经纬质心 + subtype。"""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    p2g = entry.get("pixel_to_geo")
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
        if not pts or sub not in DAMAGE_SUBTYPES:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        lon = p2g["lon"][0] * cx + p2g["lon"][1] * cy + p2g["lon"][2]
        lat = p2g["lat"][0] * cx + p2g["lat"][1] * cy + p2g["lat"][2]
        out.append({"subtype": sub, "lon": float(lon), "lat": float(lat)})
    return out


def _view_to_geo(x: float, y: float, window: dict, out_px: int) -> tuple[float, float]:
    sw = (window["east"] - window["west"]) / out_px
    sh = (window["north"] - window["south"]) / out_px
    return window["west"] + x * sw, window["north"] - y * sh


def _observe(mo, det, entry, alt, roi_bounds, native_px_cap=4096):
    """在给定高度观测一次，返回落在 ROI 内的检测（带经纬质心）。"""
    b = entry["bounds"]
    clat = (b["north"] + b["south"]) / 2
    clon = (b["east"] + b["west"]) / 2

    post_img, meta = mo.render_for_alt(
        clat, clon, alt, stage="post", out_px=FL.SENSOR_PX,
        roi_tile_id=entry["tile_id"], enforce_roi=False,
    )
    native_px = int(min(native_px_cap, max(FL.SENSOR_PX, round(meta.span_m / FL.NATIVE_GSD_M))))
    pre_img, pre_meta = mo.render_for_alt(
        clat, clon, alt, stage="pre", out_px=native_px, enforce_roi=False,
    )
    if pre_meta.window != meta.window:
        raise RuntimeError("pre/post 窗口不一致")

    # post 上采样到原生像素数：不增加信息，只为满足 siamese 的同尺寸要求
    if post_img.size != pre_img.size:
        post_img = post_img.resize(pre_img.size, Image.BILINEAR)

    dets = det.detect(pre_img, post_img)
    kept = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        lon, lat = _view_to_geo((x1 + x2) / 2, (y1 + y2) / 2, meta.window, native_px)
        if not (roi_bounds["west"] <= lon <= roi_bounds["east"]
                and roi_bounds["south"] <= lat <= roi_bounds["north"]):
            continue
        kept.append({
            "lon": lon, "lat": lat, "subtype": d.raw_class_name,
            "conf": float(d.conf), "class_probs": dict(d.class_probs),
        })
    return kept, meta


def _match(gt: list[dict], dets: list[dict], tol_m: float = 12.0) -> list[dict | None]:
    """把每栋 GT 建筑匹配到最近的检测（一对一，米制阈值内）。"""
    used = set()
    out: list[dict | None] = []
    for g in gt:
        best, best_d = None, tol_m
        for i, d in enumerate(dets):
            if i in used:
                continue
            dy = (d["lat"] - g["lat"]) * 110_540.0
            dx = (d["lon"] - g["lon"]) * 111_320.0 * math.cos(math.radians(g["lat"]))
            dist = math.hypot(dx, dy)
            if dist < best_d:
                best, best_d = i, dist
        if best is None:
            out.append(None)
        else:
            used.add(best)
            out.append(dets[best])
    return out


def _bearing(from_lat, from_lon, to_lat, to_lon) -> str:
    north = (to_lat - from_lat) * 110_540.0
    east = (to_lon - from_lon) * 111_320.0 * math.cos(math.radians(from_lat))
    ang = math.degrees(math.atan2(east, north)) % 360.0
    return BEARINGS[int((ang + 22.5) // 45) % 8]


def _answers(items: list[dict], roi_center: tuple[float, float]) -> dict:
    """从一组建筑（GT 或检测）推导四类问题的答案。ROI-scoped。"""
    n_severe = sum(1 for x in items if x["subtype"] in SEVERE)
    n_destroyed = sum(1 for x in items if x["subtype"] == "destroyed")
    ans = {
        "presence_severe": "是" if n_severe > 0 else "否",
        "count_destroyed": "3+" if n_destroyed >= 3 else str(n_destroyed),
    }
    sev = [x for x in items if x["subtype"] in SEVERE]
    if sev:
        clat, clon = roi_center
        nearest = min(sev, key=lambda x: (x["lat"] - clat) ** 2 + (x["lon"] - clon) ** 2)
        ans["spatial_severe"] = _bearing(clat, clon, nearest["lat"], nearest["lon"])
    else:
        ans["spatial_severe"] = None
    return ans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "holdout", "eval"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default=os.getenv("PERCEPTION_DEVICE", "cuda"))
    ap.add_argument("--backend", default="xview2_first")
    ap.add_argument("--archs", default="res34")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--manifest", default=str(ROOT / "backend/data/xbd/manifest.json"))
    ap.add_argument("--roi-index", default=str(ROOT / "backend/data/xbd/roi_index.json"))
    ap.add_argument("--tiles-from", default="",
                    help="只评测该 Agent-VQA JSON 中出现的 tile_id")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    events = set(
        TEST_EVENTS if args.split == "test"
        else (HOLDOUT_EVENTS if args.split == "holdout" else EVAL_EVENTS)
    )
    manifest = xbd_map.load_manifest(args.manifest)
    root = Path(manifest["dataset_root"])
    cov = json.loads(Path(args.roi_index).read_text(encoding="utf-8"))["coverage"]
    allowed_tiles = None
    if args.tiles_from:
        source = json.loads(Path(args.tiles_from).read_text(encoding="utf-8"))
        allowed_tiles = {str(row["tile_id"]) for row in source.get("items", [])}
    mo = mosaic_mod.from_manifest(manifest)

    cands = [
        e for e in manifest["items"]
        if e.get("stage") == "post" and e.get("disaster") in events
        and e.get("label_relpath") and e.get("paired_tile_id")
        and float(cov.get(e["tile_id"], 0.0)) >= args.min_coverage
        and (allowed_tiles is None or e["tile_id"] in allowed_tiles)
    ]
    # 按事件分层采样。首版用全局等距步长，而候选按 tile_id 排序时同一事件是连续的，
    # 结果 40 个样本全落在 hurricane-michael（89 个候选），palu-tsunami（8 个）一个没取到。
    # 事件间方差是 review2 B1 与计划 11.6 第 5 条判据的核心，不能只覆盖一个事件。
    by_ev = defaultdict(list)
    for e in cands:
        by_ev[e["disaster"]].append(e)
    for v in by_ev.values():
        v.sort(key=lambda e: e["tile_id"])
    if args.limit > 0:
        n_ev = len(by_ev)
        quota = {ev: max(1, args.limit // n_ev) for ev in by_ev}
        # 事件候选不足时把余额让给其他事件
        left = args.limit - sum(min(quota[ev], len(v)) for ev, v in by_ev.items())
        picked = []
        for ev, v in sorted(by_ev.items()):
            take = min(quota[ev], len(v))
            step = max(1, len(v) // take)
            picked.append((ev, v[::step][:take]))
        if left > 0:
            for ev, v in sorted(by_ev.items(), key=lambda kv: -len(kv[1])):
                already = {e["tile_id"] for _, lst in picked for e in lst}
                extra = [e for e in v if e["tile_id"] not in already][:left]
                if extra:
                    picked.append((ev, extra))
                    left -= len(extra)
                if left <= 0:
                    break
        cands = [e for _, lst in picked for e in lst]
    cands.sort(key=lambda e: e["tile_id"])

    from detectors import get_detector

    det = get_detector(
        args.backend, device=args.device,
        archs=tuple(a.strip() for a in args.archs.split(",") if a.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
    )

    alt_cruise, alt_floor = FL.alt_cruise_m(), FL.alt_min_m()

    # 逐建筑
    b_flip = b_correctable = b_harmful = b_both_ok = b_neither = b_total = 0
    b_cruise_ok = b_floor_ok = 0
    # 逐问题
    q_flip = Counter(); q_correctable = Counter(); q_harmful = Counter()
    q_both = Counter(); q_neither = Counter(); q_total = Counter()
    per_event = defaultdict(lambda: {"rois": 0, "correctable": 0, "flip": 0})
    scenes = []

    for i, entry in enumerate(cands, 1):
        try:
            gt = _gt_buildings(root / entry["label_relpath"], entry)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {entry['tile_id']}: {exc}", flush=True)
            continue
        if not gt:
            continue
        rb = entry["bounds"]
        roi_center = ((rb["north"] + rb["south"]) / 2, (rb["east"] + rb["west"]) / 2)

        try:
            d_cruise, m_cruise = _observe(mo, det, entry, alt_cruise, rb)
            d_floor, m_floor = _observe(mo, det, entry, alt_floor, rb)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {entry['tile_id']} (render/detect): {exc}", flush=True)
            continue

        mc = _match(gt, d_cruise)
        mf = _match(gt, d_floor)
        s_flip = s_corr = 0
        for g, c, f in zip(gt, mc, mf):
            b_total += 1
            c_sub = c["subtype"] if c else None
            f_sub = f["subtype"] if f else None
            c_ok = c_sub == g["subtype"]
            f_ok = f_sub == g["subtype"]
            b_cruise_ok += int(c_ok)
            b_floor_ok += int(f_ok)
            if c_sub != f_sub:
                b_flip += 1
                s_flip += 1
            if not c_ok and f_ok:
                b_correctable += 1
                s_corr += 1
            elif c_ok and not f_ok:
                b_harmful += 1
            elif c_ok and f_ok:
                b_both_ok += 1
            else:
                b_neither += 1

        a_gt = _answers(gt, roi_center)
        a_c = _answers(d_cruise, roi_center)
        a_f = _answers(d_floor, roi_center)
        for k in a_gt:
            if a_gt[k] is None:
                continue
            q_total[k] += 1
            c_ok = a_c.get(k) == a_gt[k]
            f_ok = a_f.get(k) == a_gt[k]
            if a_c.get(k) != a_f.get(k):
                q_flip[k] += 1
            if not c_ok and f_ok:
                q_correctable[k] += 1
            elif c_ok and not f_ok:
                q_harmful[k] += 1
            elif c_ok and f_ok:
                q_both[k] += 1
            else:
                q_neither[k] += 1

        ev = entry["disaster"]
        per_event[ev]["rois"] += 1
        per_event[ev]["correctable"] += s_corr
        per_event[ev]["flip"] += s_flip
        scenes.append({
            "tile_id": entry["tile_id"], "event": ev, "n_gt": len(gt),
            "n_det_cruise": len(d_cruise), "n_det_floor": len(d_floor),
            "flip": s_flip, "correctable": s_corr,
            "xbd_fraction_cruise": round(m_cruise.xbd_fraction, 4),
        })
        if i % 5 == 0:
            print(f"  {i}/{len(cands)} ROIs  buildings={b_total} "
                  f"flip={b_flip} correctable={b_correctable}", flush=True)

    q_corr_total = int(sum(q_correctable.values()))
    report = {
        "schema": "identifiability-precheck/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "events": sorted(events),
        "n_rois": len(scenes),
        "observation_model": "mosaic_fov",
        "detector": det.describe(),
        "leaky": bool(det.describe().get("leaky")),
        "ladder": {
            "alt_cruise_m": round(alt_cruise, 2), "alt_floor_m": round(alt_floor, 2),
            "gsd_cruise_m": round(FL.eff_gsd_for_alt(alt_cruise), 3),
            "gsd_floor_m": round(FL.eff_gsd_for_alt(alt_floor), 3),
        },
        "building_level": {
            "n_buildings": b_total,
            "n_flip": b_flip,
            "n_correctable": b_correctable,
            "n_harmful": b_harmful,
            "n_both_correct": b_both_ok,
            "n_neither_correct": b_neither,
            "acc_cruise": round(b_cruise_ok / b_total, 4) if b_total else None,
            "acc_floor": round(b_floor_ok / b_total, 4) if b_total else None,
            "net_gain": b_correctable - b_harmful,
        },
        "question_level": {
            "n_questions": int(sum(q_total.values())),
            "n_flip": int(sum(q_flip.values())),
            "n_correctable": q_corr_total,
            "n_harmful": int(sum(q_harmful.values())),
            "n_both_correct": int(sum(q_both.values())),
            "n_neither_correct": int(sum(q_neither.values())),
            "by_type": {
                k: {
                    "n": q_total[k], "flip": q_flip[k], "correctable": q_correctable[k],
                    "harmful": q_harmful[k], "both_correct": q_both[k],
                    "neither_correct": q_neither[k],
                }
                for k in sorted(q_total)
            },
        },
        "per_event": {k: dict(v) for k, v in sorted(per_event.items())},
        "prior_values_old_system": {"n_flip": 7, "n_correctable": 2,
                                    "source": "AGENT_VQA_EXPERIMENT_STATUS.md D6 holdout"},
        "verdict": (
            "IDENTIFIABLE" if q_corr_total >= 20
            else ("MARGINAL" if q_corr_total >= 10 else "UNIDENTIFIABLE")
        ),
        "verdict_note": (
            "阈值是先验设定的可比较性下限，不是效果判据。"
            "UNIDENTIFIABLE 时 E4 仍按计划 3.4 分支 C 写，不得改判据。"
        ),
        "scenes": scenes,
    }

    out = Path(args.out) if args.out else (
        ROOT / f"runs/benchmarks/identifiability/precheck_{args.split}_{args.backend}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bl = report["building_level"]
    ql = report["question_level"]
    print(f"\n=== P4 可识别性前置检查 / {args.split} / {args.backend} "
          f"{'[LEAKY]' if report['leaky'] else ''} ===")
    print(f"ROI 场景 {report['n_rois']}  阶梯 {alt_cruise:.0f}m({report['ladder']['gsd_cruise_m']}m/px)"
          f" → {alt_floor:.0f}m({report['ladder']['gsd_floor_m']}m/px)")
    print(f"\n逐建筑 (n={bl['n_buildings']}):")
    print(f"  巡航正确率 {bl['acc_cruise']}   下限正确率 {bl['acc_floor']}")
    print(f"  n_flip={bl['n_flip']}  n_correctable={bl['n_correctable']}  "
          f"n_harmful={bl['n_harmful']}  净收益={bl['net_gain']}")
    print(f"\n逐问题 (n={ql['n_questions']}):")
    print(f"  n_flip={ql['n_flip']}  n_correctable={ql['n_correctable']}  "
          f"n_harmful={ql['n_harmful']}")
    for k, v in ql["by_type"].items():
        print(f"    {k:20s} n={v['n']:4d} flip={v['flip']:4d} "
              f"correctable={v['correctable']:3d} harmful={v['harmful']:3d}")
    print(f"\n旧系统: n_flip=7, n_correctable=2")
    print(f"判定: {report['verdict']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
