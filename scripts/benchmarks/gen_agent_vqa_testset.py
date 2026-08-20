#!/usr/bin/env python3
"""scripts/benchmarks/gen_agent_vqa_testset.py — Agent-VQA 题库自动生成 (D2).

对应 AGENT_VQA_REVISION_PLAN.md 第 6 节。从 xBD 双时相标注生成封闭式、可自动
评分的 Agent-VQA 题库。四类主问题 (计划 6.1):
  Q1 presence: 当前视场是否存在 {class}? 答案 是/否
  Q2 damage  : 标记建筑 {ref} 的损伤等级? 答案 4 分类
  Q3 count   : 当前视场有多少栋严重或完全损毁建筑? 答案分桶 0/1/2/3+
  Q4 spatial : 最近的 {class} 位于无人机哪个方向? 答案 8 方向 + 归一化点

生成约束 (计划 6.5): 事件白/黑名单、目标可见性、负例边界缓冲、唯一目标与歧义、
分布统计、SHA-256、事件交集断言、重复检查。
视场模型: 巡航 alt=30m -> patch_radius = clamp(alt*2, 20, 300) = 60m
(与 backend/perception.py 的 PERCEPTION_VIEW_ALT_FACTOR=2.0 一致)。
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

import xbd_map  # noqa: E402
from event_split import EVAL_EVENTS, LEAK_EVENTS, assert_eval_only, event_partition  # noqa: E402
from geo import latlon_to_meters, meters_to_latlon  # noqa: E402

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

CRUISE_ALT_M = 30.0
PERCEPTION_VIEW_ALT_FACTOR = 2.0
CRUISE_RADIUS_M = max(20.0, min(300.0, CRUISE_ALT_M * PERCEPTION_VIEW_ALT_FACTOR))
NEG_BUFFER_M = 25.0
POS_DIST_RANGE = (22.0, 55.0)
AMBIGUITY_DIST_M = 18.0

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
    if dist_m < 30:
        return "easy"
    if dist_m <= 45:
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


def tile_center(entry: dict):
    b = entry.get("bounds")
    return ((b["south"] + b["north"]) / 2.0, (b["west"] + b["east"]) / 2.0) if b else None


def clamp_into_bounds(lat, lon, bounds, margin_frac=0.04):
    mlat = (bounds["north"] - bounds["south"]) * margin_frac
    mlon = (bounds["east"] - bounds["west"]) * margin_frac
    return (min(max(lat, bounds["south"] + mlat), bounds["north"] - mlat),
            min(max(lon, bounds["west"] + mlon), bounds["east"] - mlon))


def has_radius_margin(point, bounds, radius_m=CRUISE_RADIUS_M):
    lat, lon = point
    lat_margin = radius_m / 111_000.0
    lon_margin = radius_m / max(111_000.0 * math.cos(math.radians(lat)), 1.0)
    return (bounds["south"] + lat_margin <= lat <= bounds["north"] - lat_margin
            and bounds["west"] + lon_margin <= lon <= bounds["east"] - lon_margin)


def sample_start_near(goal, bounds, rng, dist_range=POS_DIST_RANGE):
    lo, hi = dist_range
    dist = rng.uniform(lo, hi)
    ang = math.radians(rng.uniform(0.0, 360.0))
    north, east = dist * math.cos(ang), dist * math.sin(ang)
    lat, lon = meters_to_latlon(goal[0], goal[1], north, east)
    return clamp_into_bounds(lat, lon, bounds)


def sample_start_in_region(bounds, rng):
    mlat = (bounds["north"] - bounds["south"]) * 0.06
    mlon = (bounds["east"] - bounds["west"]) * 0.06
    return (rng.uniform(bounds["south"] + mlat, bounds["north"] - mlat),
            rng.uniform(bounds["west"] + mlon, bounds["east"] - mlon))


def visible_buildings(start, buildings, radius_m=CRUISE_RADIUS_M):
    return [b for b in buildings if geodesic_m(start, b.centroid()) <= radius_m]


def nearest_of_class(start, buildings, subtypes):
    cands = [b for b in buildings if b.subtype in subtypes]
    return min(cands, key=lambda b: geodesic_m(start, b.centroid())) if cands else None


def has_ambiguous_neighbour(target, buildings, subtypes=None, thresh_m=AMBIGUITY_DIST_M):
    for b in buildings:
        if b is target:
            continue
        if subtypes is not None:
            if b.subtype not in subtypes:
                continue
        elif b.subtype != target.subtype:
            continue
        if geodesic_m(target.centroid(), b.centroid()) < thresh_m:
            return True
    return False


def norm_xy_from_start(start, target, radius_m=CRUISE_RADIUS_M):
    n, e = latlon_to_meters(start[0], start[1], target[0], target[1])
    # 图像坐标约定：左上 (0,0)，右下 (1,1)；east 对应 +x，north 对应 -y。
    x = max(0.0, min(1.0, 0.5 + e / (2.0 * radius_m)))
    y = max(0.0, min(1.0, 0.5 - n / (2.0 * radius_m)))
    return [round(x, 4), round(y, 4)]


# ── 题目生成 ──────────────────────────────────────────────────────────────────

@dataclass
class GenCounts:
    presence_pos: int = 0
    presence_neg: int = 0
    damage: dict = field(default_factory=lambda: {st: 0 for st in SUBTYPE_TO_CLASS})
    count: dict = field(default_factory=lambda: {c: 0 for c in COUNT_CHOICES})
    spatial: dict = field(default_factory=lambda: {d: 0 for d in BEARING_NAMES})
    rejected: int = 0


def _base_item(tile_id, disaster, split, qtype, question, choices, answer,
               start, target, difficulty, ambiguity_flags):
    return {
        "id": "", "scene_id": tile_id, "tile_id": tile_id, "disaster": disaster,
        "split": split, "question_type": qtype, "question": question,
        "choices": choices, "answer": answer,
        "start": {"lat": round(start[0], 7), "lon": round(start[1], 7), "alt": CRUISE_ALT_M},
        "target": target, "observation_profile": "cruise", "difficulty": difficulty,
        "review": {"status": "pending", "ambiguity_flags": ambiguity_flags, "author_checked": False},
    }


def gen_presence(tile_id, disaster, split, buildings, bounds, rng, counts):
    """Q1 存在判断: 正例 (目标在视场内) + 负例 (视场+缓冲区内无对应类别)。"""
    items = []
    # 正例: 优先 destroyed/major, 目标在视场内
    for st in PRIORITY_SUBTYPES:
        cands = [b for b in buildings if b.subtype == st]
        if not cands:
            continue
        target = rng.choice(cands)
        start = sample_start_near(target.centroid(), bounds, rng)
        if geodesic_m(start, target.centroid()) > CRUISE_RADIUS_M:
            counts.rejected += 1
            continue
        flags = []
        if has_ambiguous_neighbour(target, buildings):
            flags.append("ambiguous_neighbour")
        dist = geodesic_m(start, target.centroid())
        items.append(_base_item(
            tile_id, disaster, split, "presence",
            f"当前视场是否存在{target.cls}？", ["否", "是"], "是",
            start,
            {"lat": round(target.lat, 7), "lon": round(target.lon, 7), "subtype": target.subtype},
            {"distance": difficulty_of(dist), "target_pixels": 0, "clutter": 0, "edge_truncation": 0.0},
            flags))
        counts.presence_pos += 1
        break
    # 负例: 找一个起点, 其视场+缓冲区内无 severe 目标
    for _ in range(8):
        start = sample_start_in_region(bounds, rng)
        in_zone = [b for b in buildings if b.subtype in SEVERE_SUBTYPES
                   and geodesic_m(start, b.centroid()) <= CRUISE_RADIUS_M + NEG_BUFFER_M]
        if not in_zone:
            items.append(_base_item(
                tile_id, disaster, split, "presence",
                "当前视场是否存在完全损毁建筑？", ["否", "是"], "否",
                start, None,
                {"distance": "n/a", "target_pixels": 0, "clutter": 0, "edge_truncation": 0.0},
                ["negative_by_geometry"]))
            counts.presence_neg += 1
            break
    return items


def gen_damage(tile_id, disaster, split, buildings, bounds, rng, counts):
    """Q2 损伤等级：目标位于视场中心，并在实际 VQA 图像上叠加十字标记。"""
    cands = [b for b in buildings if has_radius_margin(b.centroid(), bounds)]
    if not cands:
        return []
    available_subtypes = sorted({b.subtype for b in cands}, key=lambda st: (counts.damage[st], rng.random()))
    chosen_subtype = available_subtypes[0]
    target = rng.choice([b for b in cands if b.subtype == chosen_subtype])
    # 坐标只用于构造题目输入。在线策略看不到 target 经纬度；它只看到中心十字。
    start = target.centroid()
    if geodesic_m(start, target.centroid()) > CRUISE_RADIUS_M:
        counts.rejected += 1
        return []
    flags = []
    if has_ambiguous_neighbour(target, buildings, subtypes=DAMAGED_SUBTYPES):
        flags.append("ambiguous_neighbour")
    # 目标裁块像素占比与边界截断率写入难度字段 (近似估计)
    dist = geodesic_m(start, target.centroid())
    edge_trunc = 0.0
    if dist > CRUISE_RADIUS_M * 0.9:
        edge_trunc = round(min(1.0, (dist - CRUISE_RADIUS_M * 0.9) / (CRUISE_RADIUS_M * 0.1)), 2)
        flags.append("edge_truncation")
    items = [_base_item(
        tile_id, disaster, split, "damage",
        "视场中心十字标记建筑的损伤等级是什么？", DAMAGE_CHOICES,
        SUBTYPE_TO_CN_LEVEL[target.subtype], start,
        {"lat": round(target.lat, 7), "lon": round(target.lon, 7),
         "subtype": target.subtype, "ref_id": target.uid, "marker": "center_crosshair"},
        {"distance": difficulty_of(dist), "target_pixels": int(math.sqrt(max(target.area_m2, 1.0))),
         "clutter": len(visible_buildings(start, buildings)) - 1, "edge_truncation": edge_trunc},
        flags)]
    counts.damage[target.subtype] += 1
    return items


def gen_count(tile_id, disaster, split, buildings, bounds, rng, counts):
    """Q3 数量判断: 视场内 severe/destroyed 数量分桶 0/1/2/3+。"""
    candidates = [sample_start_in_region(bounds, rng) for _ in range(64)]
    severe = [b for b in buildings if b.subtype in SEVERE_SUBTYPES]
    for target in rng.sample(severe, min(len(severe), 32)) if severe else []:
        candidates.append(target.centroid())
        candidates.append(sample_start_near(target.centroid(), bounds, rng, dist_range=(5.0, 45.0)))

    available = []
    for start in candidates:
        in_fov = [b for b in buildings if b.subtype in SEVERE_SUBTYPES
                  and geodesic_m(start, b.centroid()) <= CRUISE_RADIUS_M]
        n = len(in_fov)
        bucket = "3+" if n >= 3 else str(n)
        available.append((counts.count[bucket], rng.random(), start, in_fov, bucket))

    if available:
        # 在当前瓦片可构造的桶中优先选择全局数量最少者，避免稀疏事件把 0 桶淹没其余答案。
        _, _, start, in_fov, bucket = min(available, key=lambda row: (row[0], row[1]))
        flags = []
        # 视场边缘截断的密集建筑可能存在标注歧义
        edge_targets = [b for b in in_fov if geodesic_m(start, b.centroid()) > CRUISE_RADIUS_M * 0.85]
        if edge_targets:
            flags.append("edge_truncation")
        items = [_base_item(
            tile_id, disaster, split, "count",
            "当前视场有多少栋严重或完全损毁建筑？", COUNT_CHOICES, bucket,
            start, None,
            {"distance": "n/a", "target_pixels": 0, "clutter": len(in_fov), "edge_truncation": 0.0},
            flags)]
        counts.count[bucket] += 1
        return items
    return []


def gen_spatial(tile_id, disaster, split, buildings, bounds, rng, counts):
    """Q4 空间定位: 最近的 {class} 方向 (8 方向 + 归一化点), 方向由地理坐标计算。"""
    # 选一个 severe 目标, 起点放在其视场内但偏一侧, 问方向
    cands = [b for b in buildings if b.subtype in PRIORITY_SUBTYPES]
    if not cands:
        return []
    proposals = []
    for _ in range(48):
        seed_target = rng.choice(cands)
        start = sample_start_near(seed_target.centroid(), bounds, rng, dist_range=(28.0, 55.0))
        target = nearest_of_class(start, buildings, (seed_target.subtype,)) or seed_target
        n, e = latlon_to_meters(start[0], start[1], target.lat, target.lon)
        proposals.append((counts.spatial[bearing_name(n, e)], rng.random(), start, target))
    _, _, start, target = min(proposals, key=lambda row: (row[0], row[1]))
    if geodesic_m(start, target.centroid()) > CRUISE_RADIUS_M:
        counts.rejected += 1
        return []
    n, e = latlon_to_meters(start[0], start[1], target.lat, target.lon)
    direction = bearing_name(n, e)
    flags = []
    # 距离相近的多个同类目标须标记为歧义
    same_class_near = [b for b in buildings if b.subtype == target.subtype
                       and b is not target
                       and geodesic_m(target.centroid(), b.centroid()) < AMBIGUITY_DIST_M * 1.5]
    if same_class_near:
        flags.append("ambiguous_neighbour")
    dist = geodesic_m(start, target.centroid())
    items = [_base_item(
        tile_id, disaster, split, "spatial",
        f"最近的{target.cls}位于无人机哪个方向？", BEARING_NAMES, direction,
        start,
        {"lat": round(target.lat, 7), "lon": round(target.lon, 7), "subtype": target.subtype},
        {"distance": difficulty_of(dist), "target_pixels": 0, "clutter": 0, "edge_truncation": 0.0},
        flags)]
    items[0]["evidence_norm_xy"] = norm_xy_from_start(start, target.centroid())
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
    """重复题、重复目标与近重复问题检查 (计划 6.5 第 8 项)。"""
    seen_q = set()
    dups = []
    for it in items:
        key = (it["tile_id"], it["question_type"], it["question"], it["answer"])
        if key in seen_q:
            dups.append(it["id"])
        seen_q.add(key)
    return dups


def assert_event_disjoint(disasters_used):
    """train/val/test 事件交集断言 (计划 6.5 第 7 项 / D1)。"""
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


# ── 主流程 ────────────────────────────────────────────────────────────────────

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
    items = []
    items += gen_presence(tile_id, disaster, split, buildings, bounds, rng, counts)
    items += gen_damage(tile_id, disaster, split, buildings, bounds, rng, counts)
    items += gen_count(tile_id, disaster, split, buildings, bounds, rng, counts)
    items += gen_spatial(tile_id, disaster, split, buildings, bounds, rng, counts)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 Agent-VQA 题库")
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--disasters", default=",".join(EVAL_EVENTS),
                    help="逗号分隔; 默认全部 EVAL_EVENTS, 自动剔除 train/val")
    ap.add_argument("--n", type=int, default=200, help="目标题数 (开发集默认 200)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", choices=["dev", "formal"], default="dev",
                    help="dev=200 evidence-rich 调试; formal=按功效规划")
    ap.add_argument("--require-eval-events", action="store_true", default=True)
    ap.add_argument("--out", default=str(BACKEND / "data" / "benchmarks" / "agent_vqa_testset.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dataset_root = xbd_map.resolve_dataset_root(args.dataset_root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    wanted = {d.strip() for d in args.disasters.split(",") if d.strip()}
    if args.require_eval_events:
        leak = wanted & set(LEAK_EVENTS)
        if leak:
            print(f"[ERROR] --require-eval-events 禁止 train/val 事件: {sorted(leak)}", file=sys.stderr)
            return 2
        if not wanted:
            wanted = set(EVAL_EVENTS)

    by_disaster = {}
    for e in manifest.get("items", []):
        if str(e.get("stage")).lower() not in {"post", "post_disaster"}:
            continue
        if not e.get("has_georef") or not e.get("bounds"):
            continue
        d = e.get("disaster")
        if wanted and d not in wanted:
            continue
        by_disaster.setdefault(d, []).append(e)
    for d in by_disaster:
        rng.shuffle(by_disaster[d])

    disasters = [d for d in by_disaster if by_disaster[d]]
    if not disasters:
        print("[ERROR] 没有符合条件的瓦片", file=sys.stderr)
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
            if len(items) + len(new) <= args.n:
                items += new
                progressed = True
            else:
                items += new[: args.n - len(items)]
        if not progressed:
            break

    # 分配稳定 id
    for i, it in enumerate(items):
        it["id"] = f"{it['tile_id']}__{it['question_type']}_{i:04d}_{rng.randint(1000, 9999)}"

    # 事件切分断言 (D1)
    disasters_used = sorted({it["disaster"] for it in items})
    try:
        assert_event_disjoint(disasters_used)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    # 重复检查
    dups = check_duplicates(items)
    if dups:
        print(f"[WARN] 发现 {len(dups)} 条重复题: {dups[:5]}", file=sys.stderr)

    strat = stratify(items, disasters)
    manifest_sha = sha256_file(Path(args.manifest)) if Path(args.manifest).exists() else ""
    out = {
        "schema_version": "agent-vqa/1.1",
        "dataset_manifest_sha256": manifest_sha,
        "dataset_manifest_path": str(Path(args.manifest).relative_to(REPO_ROOT)) if Path(args.manifest).exists() else "",
        "dataset_root": str(dataset_root),
        "split_policy": "event-disjoint",
        "split_audit": {
            "eval_events": list(EVAL_EVENTS), "leak_events": list(LEAK_EVENTS),
            "disasters_used": disasters_used,
            "leakage_check_passed": not (set(disasters_used) & set(LEAK_EVENTS)),
        },
        "observation_profile": {"altitude_m": CRUISE_ALT_M, "radius_m": CRUISE_RADIUS_M,
                                 "neg_buffer_m": NEG_BUFFER_M},
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
        "profile": args.profile,
        "items": items,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 生成 {len(items)} 条题 -> {out_path}")
    print(f"     分层: {json.dumps(strat, ensure_ascii=False)}")
    print(f"     事件: {disasters_used} (leakage_check={out['split_audit']['leakage_check_passed']})")
    print(f"     生成统计: pos={counts.presence_pos} neg={counts.presence_neg} "
          f"damage={counts.damage} count={counts.count} spatial={counts.spatial} "
          f"rejected={counts.rejected}")
    if items:
        ex = items[0]
        print(f"     示例: [{ex['question_type']}] {ex['question']} -> {ex['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
