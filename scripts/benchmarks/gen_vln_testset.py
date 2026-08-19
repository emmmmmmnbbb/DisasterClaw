#!/usr/bin/env python3
"""
scripts/benchmarks/gen_vln_testset.py — P4-1 VLN 评测题库自动生成

从 xBD 标注（label JSON 的 lng_lat 多边形 + subtype 损伤等级）自动生成
"指令 → GT 目标"的测试题库草稿，供 bench_vln_navigation.py 评测。

每条题：
    - instruction：自然语言导航指令（可带方向先验）
    - start：UAV 起点（取瓦片中心，alt=30）
    - landmarks：开放词汇目标短语序列（单 / 多地标）
    - goals：GT 目标（受损建筑质心 lat/lon + 损伤等级）
    - success_radius_m / shortest_path_m / difficulty / disaster

注意：本脚本只产"草稿"，建议人工校验剔除歧义题后再用于正式评测。

用法：
    python scripts/benchmarks/gen_vln_testset.py \
        --n 40 --seed 7 \
        --disasters palu-tsunami,mexico-earthquake,midwest-flooding,hurricane-michael \
        --out backend/data/benchmarks/vln_testset.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import xbd_map  # noqa: E402
from geo import latlon_to_meters, meters_to_latlon  # noqa: E402

# xBD subtype → 中文类别（与 perception.YOLO_LABEL_MAP / semantic_map 一致）
SUBTYPE_TO_CLASS = {
    "no-damage": "无损伤建筑",
    "minor-damage": "轻微损伤建筑",
    "major-damage": "严重损伤建筑",
    "destroyed": "完全损毁建筑",
}
DAMAGED_SUBTYPES = {"minor-damage", "major-damage", "destroyed"}
# 救援优先关注的目标类（生成单地标题时优先选）
PRIORITY_SUBTYPES = ["destroyed", "major-damage"]

_BEARING_NAMES = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]


def bearing_name(north_m: float, east_m: float) -> str:
    ang = math.degrees(math.atan2(east_m, north_m)) % 360.0  # 0=北 顺时针
    return _BEARING_NAMES[int((ang + 22.5) // 45) % 8]


def geodesic_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """a,b = (lat, lon) → 地表距离（米）。"""
    n, e = latlon_to_meters(a[0], a[1], b[0], b[1])
    return math.hypot(n, e)


def difficulty_of(dist_m: float) -> str:
    if dist_m < 60:
        return "easy"
    if dist_m <= 150:
        return "medium"
    return "hard"


def tile_buildings(dataset_root: Path, entry: dict) -> list[dict]:
    """读瓦片 label，返回受损建筑列表 [{subtype, class, lat, lon}]（仅受损类）。"""
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
    out: list[dict] = []
    for f in feats:
        props = f.get("properties") or {}
        subtype = props.get("subtype")
        if subtype not in DAMAGED_SUBTYPES:
            continue
        ring = xbd_map._parse_polygon_wkt(f.get("wkt", ""))
        if not ring:
            continue
        lon, lat = xbd_map._polygon_centroid(ring)  # (x=lon, y=lat)
        out.append({
            "subtype": subtype,
            "class": SUBTYPE_TO_CLASS[subtype],
            "lat": float(lat),
            "lon": float(lon),
        })
    return out


def tile_center(entry: dict) -> tuple[float, float] | None:
    b = entry.get("bounds")
    if not b:
        return None
    return ((b["south"] + b["north"]) / 2.0, (b["west"] + b["east"]) / 2.0)


_BUCKET_DIST = {"easy": (30.0, 55.0), "medium": (70.0, 140.0), "hard": (160.0, 250.0)}
# At the default 30 m altitude the perception radius is 60 m. Keep the target
# visible while placing the start outside the 25 m success radius.
_EVIDENCE_RICH_DIST = (32.0, 50.0)


def sample_start(
    goal: dict,
    bounds: dict,
    bucket: str,
    rng: random.Random,
    distance_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """按目标难度桶，从目标反推一个起点（随机方位 + 桶内随机距离），并 clamp 进瓦片范围。

    这样难度可控、起点必在 POST 覆盖内；最终难度仍按 clamp 后的真实距离判定。
    """
    lo, hi = distance_range or _BUCKET_DIST.get(bucket, (70.0, 140.0))
    dist = rng.uniform(lo, hi)
    ang = math.radians(rng.uniform(0.0, 360.0))
    north, east = dist * math.cos(ang), dist * math.sin(ang)
    lat, lon = meters_to_latlon(goal["lat"], goal["lon"], north, east)
    # clamp 进瓦片范围（留一点边距，避免贴边退化视场）
    mlat = (bounds["north"] - bounds["south"]) * 0.04
    mlon = (bounds["east"] - bounds["west"]) * 0.04
    lat = min(max(lat, bounds["south"] + mlat), bounds["north"] - mlat)
    lon = min(max(lon, bounds["west"] + mlon), bounds["east"] - mlon)
    return (lat, lon)


def _instruction(
    cls_phrase: str,
    direction: str | None,
    multi_second: str | None = None,
    rng: random.Random | None = None,
    rich: bool = False,
) -> str:
    """Template or spatially richer phrasing. Rich mode adds bearing/negation/distance."""
    rng = rng or random.Random(0)
    if multi_second:
        templates = [
            f"先到{cls_phrase}，再前往{multi_second}",
            f"先检查{cls_phrase}，随后转向{multi_second}",
        ]
        head = rng.choice(templates) if rich else templates[0]
    else:
        templates = [
            f"寻找{cls_phrase}",
            f"定位一栋{cls_phrase}，忽略完好建筑",
            f"前往最近的{cls_phrase}",
        ]
        head = rng.choice(templates) if rich else templates[0]
    if direction:
        if rich:
            return rng.choice([
                f"飞到{direction}侧{head}",
                f"在起点{direction}方向约数十米处{head}",
                f"不要向相反方位搜索，沿{direction}侧{head}",
            ])
        return f"飞到{direction}侧{head}"
    return head


def make_item(
    tile_id: str, disaster: str, start: tuple[float, float],
    goals: list[dict], landmarks: list[str], with_direction: bool, rng: random.Random,
    benchmark_profile: str = "standard",
) -> dict:
    # 最短路径：start→g1(→g2)
    pts = [start] + [(g["lat"], g["lon"]) for g in goals]
    sp = sum(geodesic_m(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    # 方向先验（按 start→首目标 的真实方位，保证先验正确）
    direction = None
    if with_direction:
        n, e = latlon_to_meters(start[0], start[1], goals[0]["lat"], goals[0]["lon"])
        direction = bearing_name(n, e)
    second = landmarks[1] if len(landmarks) > 1 else None
    instruction = _instruction(
        landmarks[0], direction, second, rng=rng, rich=benchmark_profile == "evidence-rich",
    )
    item = {
        "id": f"{tile_id}__{rng.randint(1000, 9999)}",
        "tile_id": tile_id,
        "instruction": instruction,
        "start": {"lat": round(start[0], 7), "lon": round(start[1], 7), "alt": 30.0},
        "landmarks": landmarks,
        "goals": [
            {"lat": round(g["lat"], 7), "lon": round(g["lon"], 7),
             "class": g["class"], "subtype": g["subtype"]}
            for g in goals
        ],
        "success_radius_m": 25,
        "shortest_path_m": round(sp, 1),
        "difficulty": difficulty_of(sp),
        "disaster": disaster,
        "with_direction": with_direction,
        "multi": len(goals) > 1,
    }
    if benchmark_profile == "evidence-rich":
        item.update({
            "benchmark_profile": benchmark_profile,
            "expected_evidence": True,
            "review": {
                "status": "pending",
                "reviewer": "model-assisted + author spot-check (not purely manual)",
                "checks": {
                    "target_visible_in_post_image": None,
                    "damage_label_unambiguous": None,
                    "instruction_unambiguous": None,
                    "start_and_goal_in_bounds": None,
                },
                "notes": "",
            },
        })
    return item


def gen_for_tile(
    entry: dict,
    dataset_root: Path,
    rng: random.Random,
    bucket: str,
    benchmark_profile: str = "standard",
) -> list[dict]:
    """为单个瓦片生成 0~2 条题（视可用受损建筑而定），起点按难度桶反推。"""
    bounds = entry.get("bounds")
    if not bounds:
        return []
    buildings = tile_buildings(dataset_root, entry)
    if benchmark_profile == "evidence-rich":
        buildings = [b for b in buildings if b["subtype"] in PRIORITY_SUBTYPES]
    if not buildings:
        return []
    tile_id = entry["tile_id"]
    disaster = entry.get("disaster") or "unknown"
    items: list[dict] = []

    # 1) 单地标
    for st in PRIORITY_SUBTYPES:
        cands = [b for b in buildings if b["subtype"] == st]
        if cands:
            g = rng.choice(cands)
            start = sample_start(
                g,
                bounds,
                bucket,
                rng,
                distance_range=_EVIDENCE_RICH_DIST if (benchmark_profile == "evidence-rich" and bucket == "easy") else None,
            )
            items.append(make_item(
                tile_id, disaster, start, [g], [g["class"]],
                with_direction=rng.random() < 0.5, rng=rng,
                benchmark_profile=benchmark_profile,
            ))
            break

    # 2) 多地标：evidence-rich 也按概率生成，以提升语言复杂度
    distinct: dict[str, dict] = {}
    for b in buildings:
        distinct.setdefault(b["subtype"], b)
    multi_ok = len(distinct) >= 2 and rng.random() < (0.35 if benchmark_profile == "evidence-rich" else 0.5)
    if multi_ok:
        two = list(distinct.values())[:2]
        start = sample_start(two[0], bounds, bucket, rng)
        two.sort(key=lambda b: geodesic_m(start, (b["lat"], b["lon"])))
        items.append(make_item(
            tile_id, disaster, start, two,
            [two[0]["class"], two[1]["class"]],
            with_direction=rng.random() < 0.5, rng=rng,
            benchmark_profile=benchmark_profile,
        ))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 VLN 评测题库草稿")
    ap.add_argument("--manifest", default=str(BACKEND / "data" / "xbd" / "manifest.json"))
    ap.add_argument("--dataset-root", default=None, help="xBD 数据集根（默认按 XBD_DATASET_ROOT / ~/datasets/xbd）")
    ap.add_argument("--disasters", default="palu-tsunami,mexico-earthquake,midwest-flooding,hurricane-michael")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--profile",
        choices=["standard", "evidence-rich"],
        default="standard",
        help=(
            "evidence-rich 仅生成 major/destroyed 单目标近距任务，使目标大概率出现在"
            "初始视场；每题仍须按 review 字段人工核验后方可用于论文。"
        ),
    )
    ap.add_argument(
        "--require-eval-events",
        action="store_true",
        help="拒绝 train/val 事件（与 --require-event-disjoint 同等强度）",
    )
    ap.add_argument("--out", default=str(BACKEND / "data" / "benchmarks" / "vln_testset.json"))
    args = ap.parse_args()

    from event_split import EVAL_EVENTS, LEAK_EVENTS, assert_eval_only

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

    # 按灾种分组 POST + georef 瓦片
    by_disaster: dict[str, list[dict]] = {}
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

    # 灾种轮转，逐瓦片生成，直到攒够 n 条
    disasters = [d for d in by_disaster if by_disaster[d]]
    if not disasters:
        print("[ERROR] 没有符合条件的瓦片，检查 --disasters / manifest。", file=sys.stderr)
        return 1

    items: list[dict] = []
    cursors = {d: 0 for d in disasters}
    exhausted: set[str] = set()
    buckets = ["easy", "medium", "hard"]
    bi = 0
    while len(items) < args.n and len(exhausted) < len(disasters):
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
            for it in gen_for_tile(
                entry,
                dataset_root,
                rng,
                buckets[bi % len(buckets)],
                benchmark_profile=args.profile,
            ):
                if args.require_eval_events:
                    assert_eval_only(it["disaster"])
                if len(items) < args.n:
                    items.append(it)
                    bi += 1

    # 统计
    def _count(key, val=None):
        return sum(1 for it in items if (it[key] == val if val is not None else it[key]))

    strat = {
        "total": len(items),
        "by_disaster": {d: sum(1 for it in items if it["disaster"] == d) for d in disasters},
        "by_difficulty": {k: sum(1 for it in items if it["difficulty"] == k)
                          for k in ("easy", "medium", "hard")},
        "with_direction": sum(1 for it in items if it["with_direction"]),
        "multi_landmark": sum(1 for it in items if it["multi"]),
    }

    out = {
        "generated_by": "gen_vln_testset.py",
        "seed": args.seed,
        "dataset_root": str(dataset_root),
        "success_radius_m": 25,
        "benchmark_profile": args.profile,
        "review_protocol": (
            "模型辅助核验（contact sheet + 几何检查）后由作者抽查；"
            "不得表述为纯人工审核。仅 review.status=approved 的题进入论文主实验。"
            if args.profile == "evidence-rich" else
            "自动生成草稿；正式评测前建议核验。"
        ),
        "stratification": strat,
        "items": items,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 生成 {len(items)} 条题 → {out_path}")
    print(f"     分层：{json.dumps(strat, ensure_ascii=False)}")
    if items:
        print(f"     示例：{items[0]['instruction']}  (goals={len(items[0]['goals'])}, "
              f"sp={items[0]['shortest_path_m']}m, {items[0]['difficulty']})")
    print("提示：这是草稿，请人工校验剔除歧义题后再用于正式评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
