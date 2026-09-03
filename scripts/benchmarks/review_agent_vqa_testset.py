#!/usr/bin/env python3
"""scripts/benchmarks/review_agent_vqa_testset.py — Agent-VQA 题库审核 (D2, 计划 6.6).

执行 100% 自动几何与答案一致性检查, 不使用测试模型输出来决定保留哪些题。
审核记录写"模型辅助生成 + 作者抽查", 不得表述为纯人工审核。

检查项:
  1. schema 合法: 必填字段、question_type 合法、choices/answer 合法
  2. 几何一致性:
     - presence 正例: target 在巡航视场半径内
     - presence 负例: 视场+缓冲区内无 severe 目标
     - damage: target 在视场内, answer 与 subtype 对应
     - count: 分桶与视场内 severe 目标数一致
     - spatial: 方向由地理坐标计算, 与 answer 一致; target 在视场内
  3. 事件切分: 全部 disaster 属 EVAL_EVENTS, 无 LEAK_EVENTS
  4. 重复题检查
  5. 歧义标记完整性
  6. 输出 per-item 状态 + 聚合统计; 带歧义标志的题标记需作者检查

用法:
    python scripts/benchmarks/review_agent_vqa_testset.py \
        --in backend/data/benchmarks/agent_vqa_testset.json \
        --out runs/benchmarks/cja_agent_vqa/review.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import xbd_map  # noqa: E402
import fov_ladder as FL  # noqa: E402
from event_split import EVAL_EVENTS, LEAK_EVENTS, event_partition  # noqa: E402
from geo import latlon_to_meters  # noqa: E402

SUBTYPE_TO_CN_LEVEL = {
    "no-damage": "无损伤", "minor-damage": "轻微损伤",
    "major-damage": "严重损伤", "destroyed": "完全损毁",
}
SEVERE_SUBTYPES = ("major-damage", "destroyed")
DAMAGE_CHOICES = ["无损伤", "轻微损伤", "严重损伤", "完全损毁"]
COUNT_CHOICES = ["0", "1", "2", "3+"]
BEARING_NAMES = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]

CRUISE_ALT_M = FL.alt_cruise_m()
CRUISE_RADIUS_M = FL.span_m_for_alt(CRUISE_ALT_M) / 2.0
NEG_BUFFER_M = 25.0

VALID_QTYPES = ("presence", "damage", "count", "spatial")


def geodesic_m(a, b):
    n, e = latlon_to_meters(a[0], a[1], b[0], b[1])
    return math.hypot(n, e)


def bearing_name(north_m, east_m):
    ang = math.degrees(math.atan2(east_m, north_m)) % 360.0
    return BEARING_NAMES[int((ang + 22.5) // 45) % 8]


def load_buildings(entry, dataset_root):
    label_rel = entry.get("label_relpath")
    if not label_rel:
        return []
    label_path = dataset_root / label_rel
    if not label_path.exists():
        return []
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    feats = (data.get("features") or {}).get("lng_lat") or []
    out = []
    for f in feats:
        props = f.get("properties") or {}
        subtype = props.get("subtype")
        if subtype not in SUBTYPE_TO_CN_LEVEL:
            continue
        ring = xbd_map._parse_polygon_wkt(f.get("wkt", ""))
        if not ring:
            continue
        lon, lat = xbd_map._polygon_centroid(ring)
        out.append({"subtype": subtype, "lat": float(lat), "lon": float(lon),
                    "uid": str(props.get("uid") or "")})
    return out


def check_schema(it):
    errs = []
    for k in ("id", "scene_id", "tile_id", "disaster", "split", "question_type",
              "question", "choices", "answer", "start", "difficulty", "review"):
        if k not in it:
            errs.append(f"missing_field:{k}")
    if "observation_profile" not in it and "observation_model" not in it:
        errs.append("missing_field:observation_model")
    if it.get("observation_model") == "mosaic_fov_roi_scoped" and "roi" not in it:
        errs.append("missing_field:roi")
    if it.get("question_type") not in VALID_QTYPES:
        errs.append(f"invalid_question_type:{it.get('question_type')}")
    if it.get("answer") not in it.get("choices", []):
        errs.append("answer_not_in_choices")
    st = it.get("start", {})
    if not ({"lat", "lon", "alt"} <= set(st.keys())):
        errs.append("invalid_start")
    return errs


def check_geometry(it, buildings):
    """几何与答案一致性检查 (需传入该瓦片的全部建筑)。"""
    errs = []
    qt = it.get("question_type")
    start = (it["start"]["lat"], it["start"]["lon"])
    ans = it.get("answer")
    target = it.get("target")
    roi = it.get("roi") or {}
    bounds = roi.get("bounds")
    roi_scoped = it.get("observation_model") == "mosaic_fov_roi_scoped" and bounds

    def in_scope(building):
        if roi_scoped:
            return (
                bounds["south"] <= building["lat"] <= bounds["north"]
                and bounds["west"] <= building["lon"] <= bounds["east"]
            )
        return geodesic_m(start, (building["lat"], building["lon"])) <= CRUISE_RADIUS_M

    scope_center = start
    if roi_scoped:
        center = roi.get("center") or {}
        scope_center = (float(center["lat"]), float(center["lon"]))

    if qt == "presence":
        if ans == "是":
            if not target:
                errs.append("positive_without_target")
            else:
                if not in_scope(target):
                    errs.append("target_outside_roi" if roi_scoped else "target_outside_fov")
                if target.get("subtype") not in SEVERE_SUBTYPES:
                    errs.append("positive_target_not_severe")
        else:  # 否
            in_zone = [b for b in buildings if b["subtype"] in SEVERE_SUBTYPES and in_scope(b)]
            if in_zone:
                errs.append(f"negative_has_severe_in_buffer:{len(in_zone)}")

    elif qt == "damage":
        if not target:
            errs.append("damage_without_target")
        else:
            d = geodesic_m(start, (target["lat"], target["lon"]))
            if not in_scope(target):
                errs.append("target_outside_roi" if roi_scoped else f"target_outside_fov:{d:.1f}m")
            if not roi_scoped and d > 1.0:
                errs.append(f"damage_target_not_centered:{d:.1f}m")
            expected_marker = "roi_crosshair" if roi_scoped else "center_crosshair"
            if target.get("marker") != expected_marker:
                errs.append("damage_marker_missing")
            if not roi_scoped and "十字标记建筑" not in it.get("question", ""):
                errs.append("damage_question_marker_not_visible")
            expected = SUBTYPE_TO_CN_LEVEL.get(target.get("subtype", ""))
            if ans != expected:
                errs.append(f"answer_subtype_mismatch:ans={ans} expected={expected}")

    elif qt == "count":
        in_fov = [b for b in buildings if b["subtype"] in SEVERE_SUBTYPES and in_scope(b)]
        n = len(in_fov)
        expected = "3+" if n >= 3 else str(n)
        if ans != expected:
            errs.append(f"count_bucket_mismatch:ans={ans} expected={expected}(n={n})")

    elif qt == "spatial":
        if not target:
            errs.append("spatial_without_target")
        else:
            d = geodesic_m(scope_center, (target["lat"], target["lon"]))
            if not in_scope(target):
                errs.append("target_outside_roi" if roi_scoped else f"target_outside_fov:{d:.1f}m")
            n, e = latlon_to_meters(
                scope_center[0], scope_center[1], target["lat"], target["lon"],
            )
            expected = bearing_name(n, e)
            if ans != expected:
                errs.append(f"direction_mismatch:ans={ans} expected={expected}")
            nxy = it.get("evidence_norm_xy")
            if not (isinstance(nxy, list) and len(nxy) == 2
                    and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in nxy)):
                errs.append("invalid_evidence_norm_xy")

    return errs


def review(data, dataset_root, manifest_entries):
    entries_by_tile = {e["tile_id"]: e for e in manifest_entries}
    buildings_cache = {}

    def buildings_for(tile_id):
        if tile_id not in buildings_cache:
            buildings_cache[tile_id] = load_buildings(entries_by_tile.get(tile_id, {}), dataset_root)
        return buildings_cache[tile_id]

    per_item = []
    seen = set()
    for it in data.get("items", []):
        errs = check_schema(it)
        # 事件切分
        d = it.get("disaster", "")
        if d in LEAK_EVENTS:
            errs.append(f"event_in_leak_partition:{d}")
        elif d not in EVAL_EVENTS:
            errs.append(f"event_not_in_eval:{d}")
        # 几何
        if not errs:
            errs += check_geometry(it, buildings_for(it.get("tile_id", "")))
        # 重复
        key = (it.get("tile_id"), it.get("question_type"), it.get("question"), it.get("answer"))
        if key in seen:
            errs.append("duplicate")
        seen.add(key)
        # 歧义标志需作者检查
        flags = it.get("review", {}).get("ambiguity_flags", [])
        needs_author = bool(flags) or any("ambiguous" in e for e in errs)
        status = "approved" if not errs else "rejected"
        per_item.append({
            "id": it.get("id"), "tile_id": it.get("tile_id"),
            "question_type": it.get("question_type"), "answer": it.get("answer"),
            "errors": errs, "ambiguity_flags": flags,
            "needs_author_check": needs_author, "status": status,
        })

    n = len(per_item)
    approved = sum(1 for r in per_item if r["status"] == "approved")
    rejected = n - approved
    needs_author = sum(1 for r in per_item if r["needs_author_check"])
    err_counter = Counter()
    for r in per_item:
        for e in r["errors"]:
            err_counter[e.split(":")[0]] += 1
    by_qtype = {}
    for r in per_item:
        by_qtype.setdefault(r["question_type"], {"approved": 0, "rejected": 0})
        by_qtype[r["question_type"]][r["status"]] += 1
    return {
        "schema_version": "agent-vqa-review/1.1",
        "review_protocol": "模型辅助生成 + 作者抽查 (非纯人工审核); 100% 自动几何与答案一致性检查",
        "n": n, "approved": approved, "rejected": rejected,
        "needs_author_check": needs_author,
        "error_taxonomy": dict(err_counter),
        "by_question_type": by_qtype,
        "per_item": per_item,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="审核 Agent-VQA 题库")
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    dataset_root = xbd_map.resolve_dataset_root(args.dataset_root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    entries = [e for e in manifest.get("items", [])
               if str(e.get("stage")).lower() in {"post", "post_disaster"} and e.get("has_georef")]
    report = review(data, dataset_root, entries)

    out_path = Path(args.out) if args.out else Path(args.infile).with_suffix(".review.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 审核 {report['n']} 条题 -> {out_path}")
    print(f"     approved={report['approved']} rejected={report['rejected']} "
          f"needs_author={report['needs_author_check']}")
    print(f"     error_taxonomy={report['error_taxonomy']}")
    print(f"     by_question_type={report['by_question_type']}")
    return 0 if report["rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
