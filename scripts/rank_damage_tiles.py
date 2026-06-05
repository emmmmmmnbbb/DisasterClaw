#!/usr/bin/env python
"""
rank_damage_tiles.py — 扫描 xBD manifest 里所有 POST 瓦片的 label 标注，
按 building 损毁严重程度排名，输出:

  backend/data/xbd/damage_ranking.json

打分规则（可通过命令行覆盖）:
    severity = destroyed * 5 + major * 3 + minor * 1

输出字段:
    rank, tile_id, disaster, disaster_type, center{lat,lon}, bounds,
    counts{destroyed, major_damage, minor_damage, no_damage, un_classified, total},
    severity, destroyed_ratio, damaged_ratio

用法:
    python scripts/rank_damage_tiles.py                    # 默认在仓库根目录运行
    python scripts/rank_damage_tiles.py --limit 100        # 只保留前 100 条
    python scripts/rank_damage_tiles.py --min-destroyed 30 # 过滤至少 30 栋 destroyed
    python scripts/rank_damage_tiles.py --force            # 忽略缓存，全量扫描

生成后，后端 /api/xbd/damage-ranking 会直接读取该文件。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DATA_DIR = REPO_ROOT / "backend" / "data" / "xbd"
DEFAULT_MANIFEST = BACKEND_DATA_DIR / "manifest.json"
DEFAULT_OUTPUT = BACKEND_DATA_DIR / "damage_ranking.json"

SEVERITY_WEIGHTS = {
    "destroyed": 5,
    "major-damage": 3,
    "minor-damage": 1,
    "no-damage": 0,
    "un-classified": 0,
}


def is_post(item: dict[str, Any]) -> bool:
    return str(item.get("stage") or "").lower() in {"post", "post_disaster"}


def compute_tile_score(label_path: Path, weights: dict[str, int]) -> tuple[int, collections.Counter]:
    """读取一个 post tile 的 label.json，按权重累计 subtype 数量。"""
    try:
        data = json.load(open(label_path, "r", encoding="utf-8"))
    except Exception:
        return 0, collections.Counter()
    features = (data.get("features") or {}).get("xy") or []
    cnt: collections.Counter[str] = collections.Counter()
    score = 0
    for feat in features:
        props = feat.get("properties") or {}
        if props.get("feature_type") != "building":
            continue
        sub = props.get("subtype")
        if not sub:
            continue
        cnt[sub] += 1
        score += int(weights.get(sub, 0))
    return score, cnt


def tile_center(entry: dict[str, Any]) -> tuple[float, float] | None:
    b = entry.get("bounds") or {}
    try:
        clat = (float(b["north"]) + float(b["south"])) / 2.0
        clon = (float(b["east"]) + float(b["west"])) / 2.0
    except Exception:
        return None
    return clat, clon


def build_ranking(
    manifest_path: Path,
    min_destroyed: int = 0,
    min_severity: int = 0,
    limit: int | None = None,
    require_georef: bool = True,
) -> dict[str, Any]:
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"]).expanduser()
    items = manifest.get("items", [])
    total_post = 0
    results: list[dict[str, Any]] = []
    for item in items:
        if not is_post(item):
            continue
        if require_georef and not item.get("has_georef"):
            continue
        total_post += 1
        label_rel = item.get("label_relpath")
        if not label_rel:
            continue
        label_path = dataset_root / label_rel
        if not label_path.exists():
            continue
        score, cnt = compute_tile_score(label_path, SEVERITY_WEIGHTS)
        destroyed = int(cnt.get("destroyed", 0))
        major = int(cnt.get("major-damage", 0))
        minor = int(cnt.get("minor-damage", 0))
        no_damage = int(cnt.get("no-damage", 0))
        unclass = int(cnt.get("un-classified", 0))
        total = destroyed + major + minor + no_damage + unclass
        if destroyed < min_destroyed or score < min_severity:
            continue
        center = tile_center(item)
        if center is None:
            continue
        damaged = destroyed + major + minor
        results.append(
            {
                "tile_id": item.get("tile_id"),
                "disaster": item.get("disaster"),
                "disaster_type": item.get("disaster_type"),
                "split": item.get("split"),
                "stage": item.get("stage"),
                "center": {"lat": center[0], "lon": center[1]},
                "bounds": item.get("bounds"),
                "gsd": item.get("gsd"),
                "capture_date": item.get("capture_date"),
                "counts": {
                    "destroyed": destroyed,
                    "major_damage": major,
                    "minor_damage": minor,
                    "no_damage": no_damage,
                    "un_classified": unclass,
                    "total_buildings": total,
                },
                "severity": score,
                "destroyed_ratio": round(destroyed / total, 4) if total else 0.0,
                "damaged_ratio": round(damaged / total, 4) if total else 0.0,
            }
        )

    results.sort(key=lambda r: (-r["severity"], -r["counts"]["destroyed"]))
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    if limit is not None:
        results = results[:limit]

    return {
        "generated_from": str(manifest_path),
        "dataset_root": str(dataset_root),
        "weights": SEVERITY_WEIGHTS,
        "total_post_tiles": total_post,
        "total_scored": len(results),
        "min_destroyed_filter": min_destroyed,
        "min_severity_filter": min_severity,
        "items": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=None, help="最多保留前 N 条；默认全部")
    ap.add_argument("--min-destroyed", type=int, default=0, help="destroyed 建筑数量下限")
    ap.add_argument("--min-severity", type=int, default=0, help="severity 分数下限")
    ap.add_argument("--no-georef", action="store_true", help="允许收录无 georef 的瓦片（默认排除）")
    ap.add_argument("--force", action="store_true", help="忽略输出缓存，强制重算")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    if args.output.exists() and not args.force:
        try:
            if args.output.stat().st_mtime >= args.manifest.stat().st_mtime:
                print(
                    f"[skip] {args.output} 已存在且比 manifest 新，--force 可强制重算。"
                )
                return 0
        except OSError:
            pass

    ranking = build_ranking(
        args.manifest,
        min_destroyed=args.min_destroyed,
        min_severity=args.min_severity,
        limit=args.limit,
        require_georef=not args.no_georef,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(ranking, fp, ensure_ascii=False, indent=2)
    print(
        f"[ok] wrote {args.output} — {ranking['total_scored']} tiles (from {ranking['total_post_tiles']} POST tiles)"
    )

    print("\nTop 10 preview:")
    print(
        f"{'rank':>4}  {'severity':>8}  {'dest':>4}  {'major':>5}  {'total':>5}  "
        f"{'dest%':>6}  {'disaster':<22}  tile_id"
    )
    for row in ranking["items"][:10]:
        c = row["counts"]
        print(
            f"{row['rank']:>4}  {row['severity']:>8}  {c['destroyed']:>4}  {c['major_damage']:>5}  "
            f"{c['total_buildings']:>5}  {row['destroyed_ratio']*100:>5.1f}%  "
            f"{(row.get('disaster') or '?'):<22}  {row['tile_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
