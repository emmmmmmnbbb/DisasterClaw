#!/usr/bin/env python3
"""scripts/benchmarks/gen_agent_vqa_testset_v2.py — Agent-VQA 题库 v2.0 (ROI-scoped)

对应 MOSAIC_FOV_SOTA_REVISION_PLAN.md 的 P5。相对 v1 的两处根本改动：

  1. **观测模型**：旧的「巡航 alt=30m → 半径 60m 圆形视场」换成 fov_ladder 的
     视场收缩模型（巡航 1330m / 1536m 跨度 → 下限 443m / 512m 跨度，= 一整瓦片）。
  2. **题目作用域**：旧的「当前视场」换成 **ROI-scoped**（Q2 决策）——
     每道题锚定在一张有标注的 post 瓦片（ROI，地理 bbox），答案只由 ROI 内的
     GT 建筑决定，与智能体的动作、高度无关。

四类问题（语义不变，作用域从 FOV 换成 ROI）：
  Q1 presence: ROI 内是否存在 {class}?            答案 是/否
  Q2 damage  : ROI 内标记建筑 {ref} 的损伤等级?    答案 4 分类
  Q3 count   : ROI 内有多少栋严重或完全损毁建筑?   答案 0/1/2/3+
  Q4 spatial : 最近的 {class} 位于 ROI 中心哪个方向? 答案 8 方向 + 归一化点
               —— 方位**相对 ROI 中心**而非 UAV 位置（MOSAIC_FOV_SOTA_REVISION_PLAN.md §2.3）

关键不变量（§2.5）：
  - ROI 必须 100% 真实 xBD 覆盖（巡航几何覆盖 ≥0.80 的场景池来自 roi_index.json）。
  - spatial 的 GT 方位相对 ROI 中心，智能体居中前后保持不变。
  - schema 从 agent-vqa/1.1 升到 agent-vqa/2.0，全部 SHA-256 重算。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import fov_ladder as FL  # noqa: E402
import xbd_map  # noqa: E402
from event_split import EVAL_EVENTS, LEAK_EVENTS, assert_eval_only, event_partition  # noqa: E402
from geo import latlon_to_meters, meters_to_latlon  # noqa: E402
from tile_consumption import (  # noqa: E402
    load_registry,
    register_tiles,
    sha256_file as registry_sha256_file,
    tile_ids as registry_tile_ids,
    write_registry,
)

SUBTYPE_TO_CLASS = {
    "no-damage": "无损伤建筑", "minor-damage": "轻微损伤建筑",
    "major-damage": "严重损伤建筑", "destroyed": "完全损毁建筑",
}
SUBTYPE_TO_CN_LEVEL = {
    "no-damage": "无损伤", "minor-damage": "轻微损伤",
    "major-damage": "严重损伤", "destroyed": "完全损毁",
}
DAMAGED_SUBTYPES = ("minor-damage", "major-damage", "destroyed")
SEVERE_SUBTYPES = ("major-damage", "destroyed")
PRIORITY_SUBTYPES = ("destroyed", "major-damage")

# 巡航几何（视场收缩模型）
ALT_CRUISE_M = FL.alt_cruise_m()
SPAN_CRUISE_M = FL.span_m_for_alt(ALT_CRUISE_M)          # 1536 m
TILE_SPAN_M = FL.TILE_SPAN_M                              # 512 m
AMBIGUITY_DIST_M = 18.0
CENTERED_START = False  # 由 main() 根据 --centered-start 设置：E4 从 ROI 中心出发

BEARING_NAMES = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
DAMAGE_CHOICES = ["无损伤", "轻微损伤", "严重损伤", "完全损毁"]
COUNT_CHOICES = ["0", "1", "2", "3+"]


def bearing_name(north_m: float, east_m: float) -> str:
    ang = math.degrees(math.atan2(east_m, north_m)) % 360.0
    return BEARING_NAMES[int((ang + 22.5) // 45) % 8]


def geodesic_m(a, b) -> float:
    n, e = latlon_to_meters(a[0], a[1], b[0], b[1])
    return math.hypot(n, e)


def difficulty_of(dist_m: float) -> str:
    if dist_m < TILE_SPAN_M * 0.25:
        return "easy"
    if dist_m <= TILE_SPAN_M * 0.45:
        return "medium"
    return "hard"


def polygon_area_m2(ring):
    if len(ring) < 3:
        return 0.0
    lat0 = sum(p[1] for p in ring) / len(ring)
    pts = [latlon_to_meters(lat0, ring[0][0], p[1], p[0]) for p in ring]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


@dataclass
class Building:
    subtype: str
    cls: str
    lat: float
    lon: float
    ring: list
    area_m2: float
    uid: str

    def centroid(self):
        return (self.lat, self.lon)


def tile_buildings(entry: dict, dataset_root: Path) -> list:
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
        if subtype not in SUBTYPE_TO_CLASS:
            continue
        ring = xbd_map._parse_polygon_wkt(f.get("wkt", ""))
        if not ring:
            continue
        lon, lat = xbd_map._polygon_centroid(ring)
        uid = str(props.get("uid") or f"{entry['tile_id']}_{len(out)}")
        out.append(Building(subtype, SUBTYPE_TO_CLASS[subtype], float(lat), float(lon),
                            ring, polygon_area_m2(ring), uid))
    return out


def roi_center(entry: dict):
    b = entry.get("bounds")
    return ((b["south"] + b["north"]) / 2.0, (b["west"] + b["east"]) / 2.0) if b else None


def in_roi(b: Building, bounds: dict) -> bool:
    """建筑质心是否落在 ROI（瓦片 bbox）内。ROI-scoped 答案的作用域。"""
    return (bounds["south"] <= b.lat <= bounds["north"]
            and bounds["west"] <= b.lon <= bounds["east"])


def sample_start(center, bounds, rng, offset_range=(200.0, 600.0)):
    """采样 UAV 巡航起点。

    --centered-start（E4 重观测机制）：起点 = ROI 中心，降高即放大居中 ROI。
    默认：偏移 ROI 中心以激活搜索通道（E5 端到端），但仍在巡航视场可达范围内。
    """
    if CENTERED_START:
        return center
    lo, hi = offset_range
    dist = rng.uniform(lo, hi)
    ang = math.radians(rng.uniform(0.0, 360.0))
    north, east = dist * math.cos(ang), dist * math.sin(ang)
    lat, lon = meters_to_latlon(center[0], center[1], north, east)
    # clamp 到「ROI 仍在巡航视场内」——距 ROI 中心不超过 span/2 - tile/2 的余量
    max_off = (SPAN_CRUISE_M - TILE_SPAN_M) / 2.0
    if dist > max_off:
        north, east = max_off * math.cos(ang), max_off * math.sin(ang)
        lat, lon = meters_to_latlon(center[0], center[1], north, east)
    return (lat, lon)


def norm_xy_from_roi_center(center, target):
    """目标相对 ROI 中心的归一化坐标 [0,1]。ROI 铺满时恰是图像坐标。"""
    n, e = latlon_to_meters(center[0], center[1], target[0], target[1])
    half = TILE_SPAN_M / 2.0
    x = max(0.0, min(1.0, 0.5 + e / (2.0 * half)))
    y = max(0.0, min(1.0, 0.5 - n / (2.0 * half)))
    return [round(x, 4), round(y, 4)]


@dataclass
class GenCounts:
    presence_pos: int = 0
    presence_neg: int = 0
    damage: dict = field(default_factory=lambda: {st: 0 for st in SUBTYPE_TO_CLASS})
    count: dict = field(default_factory=lambda: {c: 0 for c in COUNT_CHOICES})
    spatial: dict = field(default_factory=lambda: {d: 0 for d in BEARING_NAMES})
    rejected: int = 0


def _base_item(tile_id, disaster, split, qtype, question, choices, answer,
               start, center, target, difficulty, flags, roi_bounds):
    return {
        "id": "", "scene_id": tile_id, "tile_id": tile_id, "disaster": disaster,
        "split": split, "question_type": qtype, "question": question,
        "choices": choices, "answer": answer,
        "start": {"lat": round(start[0], 7), "lon": round(start[1], 7),
                  "alt": round(ALT_CRUISE_M, 2)},
        "roi": {
            "tile_id": tile_id,
            "center": {"lat": round(center[0], 7), "lon": round(center[1], 7)},
            "bounds": {k: round(v, 7) for k, v in roi_bounds.items()},
        },
        "target": target,
        "observation_model": "mosaic_fov_roi_scoped",
        "difficulty": difficulty,
        "review": {"status": "pending", "ambiguity_flags": flags, "author_checked": False},
    }


def gen_presence(tile_id, disaster, split, buildings, bounds, center, rng, counts):
    """Q1: ROI 内是否存在 {class}。答案只由 ROI 内建筑决定。"""
    items = []
    roi_blds = [b for b in buildings if in_roi(b, bounds)]
    if not roi_blds:
        return []
    for st in PRIORITY_SUBTYPES:
        cands = [b for b in roi_blds if b.subtype == st]
        if not cands:
            continue
        target = rng.choice(cands)
        flags = []
        near = [b for b in roi_blds if b is not target and b.subtype == target.subtype
                and geodesic_m(target.centroid(), b.centroid()) < AMBIGUITY_DIST_M]
        if near:
            flags.append("ambiguous_neighbour")
        start = sample_start(center, bounds, rng)
        items.append(_base_item(
            tile_id, disaster, split, "presence",
            f"标记区域内是否存在{target.cls}？", ["否", "是"], "是", start, center,
            {"lat": round(target.lat, 7), "lon": round(target.lon, 7), "subtype": target.subtype},
            {"distance": difficulty_of(geodesic_m(center, target.centroid())),
             "clutter": len(roi_blds) - 1, "edge_truncation": 0.0}, flags, bounds))
        counts.presence_pos += 1
        break
    # 负例：ROI 内无 severe 建筑
    if not any(b.subtype in SEVERE_SUBTYPES for b in roi_blds):
        start = sample_start(center, bounds, rng)
        items.append(_base_item(
            tile_id, disaster, split, "presence",
            "标记区域内是否存在完全损毁建筑？", ["否", "是"], "否", start, center, None,
            {"distance": "n/a", "clutter": len(roi_blds), "edge_truncation": 0.0},
            ["negative_by_geometry"], bounds))
        counts.presence_neg += 1
    return items


def gen_damage(tile_id, disaster, split, buildings, bounds, center, rng, counts):
    """Q2: ROI 内标记建筑 {ref} 的损伤等级。标记在渲染图上，ref 由 uid 给出。"""
    roi_blds = [b for b in buildings if in_roi(b, bounds)]
    if not roi_blds:
        return []
    available = sorted({b.subtype for b in roi_blds}, key=lambda st: (counts.damage[st], rng.random()))
    chosen = available[0]
    target = rng.choice([b for b in roi_blds if b.subtype == chosen])
    flags = []
    near = [b for b in roi_blds if b is not target and b.subtype in DAMAGED_SUBTYPES
            and geodesic_m(target.centroid(), b.centroid()) < AMBIGUITY_DIST_M]
    if near:
        flags.append("ambiguous_neighbour")
    start = sample_start(center, bounds, rng)
    items = [_base_item(
        tile_id, disaster, split, "damage",
        f"标记区域内标记建筑 {target.uid} 的损伤等级是什么？", DAMAGE_CHOICES,
        SUBTYPE_TO_CN_LEVEL[target.subtype], start, center,
        {"lat": round(target.lat, 7), "lon": round(target.lon, 7),
         "subtype": target.subtype, "ref_id": target.uid, "marker": "roi_crosshair"},
        {"distance": difficulty_of(geodesic_m(center, target.centroid())),
         "target_pixels": int(math.sqrt(max(target.area_m2, 1.0))),
         "clutter": len(roi_blds) - 1, "edge_truncation": 0.0}, flags, bounds)]
    counts.damage[target.subtype] += 1
    return items


def gen_count(tile_id, disaster, split, buildings, bounds, center, rng, counts):
    """Q3: ROI 内 severe/destroyed 数量分桶。答案由 ROI 内建筑决定，与动作无关。"""
    roi_blds = [b for b in buildings if in_roi(b, bounds)]
    if not roi_blds:
        return []
    severe = [b for b in roi_blds if b.subtype in SEVERE_SUBTYPES]
    n = len(severe)
    bucket = "3+" if n >= 3 else str(n)
    flags = []
    edge = [b for b in severe if geodesic_m(center, b.centroid()) > TILE_SPAN_M * 0.4]
    if edge:
        flags.append("edge_truncation")
    start = sample_start(center, bounds, rng)
    items = [_base_item(
        tile_id, disaster, split, "count",
        "标记区域内有多少栋严重或完全损毁建筑？", COUNT_CHOICES, bucket, start, center, None,
        {"distance": "n/a", "clutter": len(severe), "edge_truncation": 0.0}, flags, bounds)]
    counts.count[bucket] += 1
    return items


def gen_spatial(tile_id, disaster, split, buildings, bounds, center, rng, counts):
    """Q4: 最近的 {class} 位于 ROI 中心哪个方向。方向相对 ROI 中心（§2.3）。"""
    roi_blds = [b for b in buildings if in_roi(b, bounds)]
    cands = [b for b in roi_blds if b.subtype in PRIORITY_SUBTYPES]
    if not cands:
        return []
    # 优先选分布最少的方位
    proposals = []
    for _ in range(48):
        seed = rng.choice(cands)
        target = min(cands, key=lambda b: geodesic_m(center, b.centroid()))
        # 让目标选择有一定随机性，但保持按方位分层
        n, e = latlon_to_meters(center[0], center[1], target.lat, target.lon)
        proposals.append((counts.spatial[bearing_name(n, e)], rng.random(), target))
    _, _, target = min(proposals, key=lambda row: (row[0], row[1]))
    n, e = latlon_to_meters(center[0], center[1], target.lat, target.lon)
    direction = bearing_name(n, e)
    flags = []
    same_near = [b for b in roi_blds if b.subtype == target.subtype and b is not target
                 and geodesic_m(target.centroid(), b.centroid()) < AMBIGUITY_DIST_M * 1.5]
    if same_near:
        flags.append("ambiguous_neighbour")
    start = sample_start(center, bounds, rng)
    items = [_base_item(
        tile_id, disaster, split, "spatial",
        f"最近的{target.cls}位于标记区域中心哪个方向？", BEARING_NAMES, direction,
        start, center,
        {"lat": round(target.lat, 7), "lon": round(target.lon, 7), "subtype": target.subtype},
        {"distance": difficulty_of(geodesic_m(center, target.centroid())),
         "clutter": 0, "edge_truncation": 0.0}, flags, bounds)]
    items[0]["evidence_norm_xy"] = norm_xy_from_roi_center(center, target.centroid())
    counts.spatial[direction] += 1
    return items


# ── 校验与统计 ─────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def check_duplicates(items):
    seen, dups = set(), []
    for it in items:
        key = (it["tile_id"], it["question_type"], it["question"], it["answer"])
        if key in seen:
            dups.append(it["id"])
        seen.add(key)
    return dups


def assert_event_disjoint(disasters_used):
    leak = sorted(set(disasters_used) & set(LEAK_EVENTS))
    if leak:
        raise ValueError(f"事件切分泄漏: 题库包含 train/val 事件 {leak}")
    for d in disasters_used:
        assert_eval_only(d)


def stratify(items, disasters):
    by_qtype = {t: sum(1 for it in items if it["question_type"] == t)
                for t in ("presence", "damage", "count", "spatial")}
    by_answer = {}
    for it in items:
        by_answer.setdefault(it["question_type"], {}).setdefault(it["answer"], 0)
        by_answer[it["question_type"]][it["answer"]] += 1
    return {
        "total": len(items),
        "by_disaster": {d: sum(1 for it in items if it["disaster"] == d) for d in disasters},
        "by_question_type": by_qtype,
        "by_answer": by_answer,
        "with_ambiguity": sum(1 for it in items if it["review"]["ambiguity_flags"]),
    }


def gen_for_tile(entry, dataset_root, rng, counts):
    bounds = entry.get("bounds")
    if not bounds:
        return []
    buildings = tile_buildings(entry, dataset_root)
    if not buildings:
        return []
    tile_id = entry["tile_id"]
    disaster = entry.get("disaster") or "unknown"
    split = event_partition(disaster)
    if split in {"train", "val", "unknown"}:
        return []
    center = roi_center(entry)
    if not center:
        return []
    items = []
    items += gen_presence(tile_id, disaster, split, buildings, bounds, center, rng, counts)
    items += gen_damage(tile_id, disaster, split, buildings, bounds, center, rng, counts)
    items += gen_count(tile_id, disaster, split, buildings, bounds, center, rng, counts)
    items += gen_spatial(tile_id, disaster, split, buildings, bounds, center, rng, counts)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 Agent-VQA 题库 v2.0 (ROI-scoped)")
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--roi-index", default=str(BACKEND / "data" / "xbd" / "roi_index.json"))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--disasters", default=",".join(EVAL_EVENTS),
                    help="逗号分隔; 默认全部 EVAL_EVENTS")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--centered-start", action="store_true",
                    help="起点=ROI 中心（E4 重观测机制）；默认偏移以激活搜索通道（E5）")
    ap.add_argument("--exclude-registry", default="",
                    help="排除登记表中所有已消费/已分配 ROI")
    ap.add_argument("--eval-role", choices=["selection", "final", "boundary"], default="",
                    help="写入题库 manifest 的冻结角色")
    ap.add_argument("--update-registry", action="store_true",
                    help="生成成功后将所用 ROI 原子登记到 --exclude-registry")
    ap.add_argument("--n", type=int, default=200, help="目标题数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(BACKEND / "data" / "benchmarks" / "agent_vqa_testset_v2.json"))
    args = ap.parse_args()

    global CENTERED_START
    CENTERED_START = args.centered_start
    rng = random.Random(args.seed)
    dataset_root = xbd_map.resolve_dataset_root(args.dataset_root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cov = json.loads(Path(args.roi_index).read_text(encoding="utf-8"))["coverage"]
    wanted = {d.strip() for d in args.disasters.split(",") if d.strip()} or set(EVAL_EVENTS)
    registry_path = Path(args.exclude_registry) if args.exclude_registry else None
    registry = load_registry(registry_path) if registry_path else None
    excluded_tiles = registry_tile_ids(registry) if registry else set()
    if args.update_registry and (registry_path is None or not args.eval_role):
        print("[ERROR] --update-registry 需要 --exclude-registry 与 --eval-role", file=sys.stderr)
        return 2
    leak = wanted & set(LEAK_EVENTS)
    if leak:
        print(f"[ERROR] 禁止 train/val 事件: {sorted(leak)}", file=sys.stderr)
        return 2

    by_disaster = {}
    for e in manifest.get("items", []):
        if str(e.get("stage")).lower() not in {"post", "post_disaster"}:
            continue
        if not e.get("has_georef") or not e.get("bounds"):
            continue
        if not e.get("label_relpath"):
            continue
        d = e.get("disaster")
        if d not in wanted:
            continue
        if float(cov.get(e["tile_id"], 0.0)) < args.min_coverage:
            continue
        if e["tile_id"] in excluded_tiles:
            continue
        by_disaster.setdefault(d, []).append(e)
    for d in by_disaster:
        rng.shuffle(by_disaster[d])

    disasters = [d for d in by_disaster if by_disaster[d]]
    if not disasters:
        print("[ERROR] 没有符合条件的 ROI 瓦片（检查 --min-coverage 与 roi_index.json）", file=sys.stderr)
        return 1

    counts = GenCounts()
    items = []
    cursors = {d: 0 for d in disasters}
    exhausted = set()
    while len(items) < args.n and len(exhausted) < len(disasters):
        progressed = False
        for d in disasters:
            if len(items) >= args.n:
                break
            if d in exhausted:
                continue
            tiles = by_disaster[d]
            if cursors[d] >= len(tiles):
                exhausted.add(d)
                continue
            entry = tiles[cursors[d]]
            cursors[d] += 1
            new = gen_for_tile(entry, dataset_root, rng, counts)
            room = args.n - len(items)
            items += new[:room]
            progressed = True
        if not progressed:
            break

    for i, it in enumerate(items):
        it["id"] = f"{it['tile_id']}__{it['question_type']}_{i:04d}_{rng.randint(1000, 9999)}"

    disasters_used = sorted({it["disaster"] for it in items})
    try:
        assert_event_disjoint(disasters_used)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    dups = check_duplicates(items)
    if dups:
        print(f"[WARN] {len(dups)} 条重复题", file=sys.stderr)

    strat = stratify(items, disasters)
    manifest_sha = sha256_file(Path(args.manifest)) if Path(args.manifest).exists() else ""
    out = {
        "schema_version": "agent-vqa/2.0",
        "dataset_manifest_sha256": manifest_sha,
        "dataset_manifest_path": str(Path(args.manifest).relative_to(REPO_ROOT)) if Path(args.manifest).exists() else "",
        "dataset_root": str(dataset_root),
        "split_policy": "event-disjoint",
        "eval_role": args.eval_role or None,
        "consumption_registry": {
            "path": (
                str(registry_path.resolve().relative_to(REPO_ROOT))
                if registry_path and registry_path.exists()
                and registry_path.resolve().is_relative_to(REPO_ROOT)
                else (str(registry_path) if registry_path else "")
            ),
            "sha256": registry_sha256_file(registry_path) if registry_path else "",
            "n_excluded_tiles": len(excluded_tiles),
        },
        "split_audit": {
            "eval_events": list(EVAL_EVENTS), "leak_events": list(LEAK_EVENTS),
            "disasters_used": disasters_used,
            "leakage_check_passed": not (set(disasters_used) & set(LEAK_EVENTS)),
        },
        "observation_model": {
            "name": "mosaic_fov_roi_scoped",
            "sensor_px": FL.SENSOR_PX,
            "fov_deg": FL.FOV_DEG,
            "alt_cruise_m": round(ALT_CRUISE_M, 2),
            "alt_floor_m": round(FL.alt_min_m(), 2),
            "span_cruise_m": round(SPAN_CRUISE_M, 2),
            "span_floor_m": round(TILE_SPAN_M, 2),
            "gsd_cruise_m": round(FL.eff_gsd_for_alt(ALT_CRUISE_M), 3),
            "gsd_floor_m": round(FL.eff_gsd_for_alt(FL.alt_min_m()), 3),
            "roi_scoped": True,
            "roi_min_coverage": args.min_coverage,
        },
        "review_protocol": (
            "模型辅助生成 + 作者抽查; 100% 自动几何与答案一致性检查; "
            "所有带 ambiguity_flags 的题由作者检查; 不得表述为纯人工审核。"
        ),
        "stratification": strat,
        "gen_counts": {
            "presence_pos": counts.presence_pos, "presence_neg": counts.presence_neg,
            "damage": sum(counts.damage.values()), "damage_by_subtype": counts.damage,
            "count": counts.count, "spatial": sum(counts.spatial.values()),
            "spatial_by_direction": counts.spatial,
            "rejected": counts.rejected, "duplicates": len(dups),
        },
        "items_sha256": sha256_json(items),
        "seed": args.seed,
        "items": items,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.update_registry and registry_path is not None and registry is not None:
        registered = register_tiles(
            registry,
            {it["tile_id"] for it in items},
            eval_role=args.eval_role,
            source_run=str(out_path),
        )
        digest = write_registry(registry_path, registered)
        out["consumption_registry"]["sha256_after_registration"] = digest
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] v2.0 生成 {len(items)} 条题 -> {out_path}")
    print(f"     分层: {json.dumps(strat, ensure_ascii=False)}")
    print(f"     事件: {disasters_used} (leakage_check={out['split_audit']['leakage_check_passed']})")
    print(f"     spatial_by_direction: {counts.spatial}")
    if items:
        ex = items[0]
        print(f"     示例: [{ex['question_type']}] {ex['question']} -> {ex['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
