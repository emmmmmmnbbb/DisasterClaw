#!/usr/bin/env python3
"""Backfill the ROI registry from already inspected benchmark products."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from event_split import event_partition  # noqa: E402
from tile_consumption import load_registry, register_tiles, write_registry  # noqa: E402

DEFAULT_OUT = ROOT / "backend/data/benchmarks/tile_consumption_registry.json"
DEFAULT_SOURCES = (
    ROOT / "backend/data/benchmarks/agent_vqa_testset_v2.json",
    ROOT / "backend/data/benchmarks/agent_vqa_testset_v2_centered.json",
    ROOT / "backend/data/benchmarks/agent_vqa_testset_v2_holdout.json",
    ROOT / "backend/data/benchmarks/vln_recheck_eval_v2.json",
    ROOT / "backend/data/benchmarks/vln_recheck_testset.json",
    ROOT / "runs/benchmarks/identifiability/precheck_test_xview2_first.json",
)


_TILE_DISASTER = re.compile(r"^(?P<disaster>.+)_\d+_(?:pre|post)_disaster$")


def disaster_from_tile_id(tile_id: str) -> str:
    name = str(tile_id or "").strip()
    match = _TILE_DISASTER.match(name)
    if match:
        return match.group("disaster")
    return name.split("_")[0]


def eval_role_for_tile(tile_id: str) -> str | None:
    """VAL tiles are reserved for fitting; eval-event tiles already inspected are consumed.

    Train-event tiles never enter Agent-VQA generators, so they are skipped.
    """
    part = event_partition(disaster_from_tile_id(tile_id))
    if part == "val":
        return "fit"
    if part in {"test", "holdout"}:
        return "consumed"
    return None


def extract_tile_ids(data: object) -> set[str]:
    """Extract only explicit ROI/tile identifiers; never infer them from qids."""
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key in ("tile_id", "scene_id", "roi_tile_id"):
                raw = value.get(key)
                if isinstance(raw, str) and raw.strip():
                    found.add(raw.strip())
            for key in ("items", "scenes", "episodes"):
                if key in value:
                    walk(value[key])
        elif isinstance(value, list):
            for row in value:
                walk(row)

    walk(data)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="回填已消费 ROI 登记表")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sources", nargs="*", default=[str(p) for p in DEFAULT_SOURCES])
    ap.add_argument("--reset", action="store_true", help="忽略已有登记表并重建")
    args = ap.parse_args()

    out = Path(args.out)
    registry = {"schema": "tile-consumption-registry/1.0", "items": []}
    if out.is_file() and not args.reset:
        registry = load_registry(out)

    n_sources = 0
    for raw in args.sources:
        path = Path(raw)
        if not path.is_file():
            print(f"[skip] source not found: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tiles = extract_tile_ids(data)
        if not tiles:
            print(f"[skip] no explicit tile ids: {path}")
            continue
        source = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
        by_role: dict[str, set[str]] = {}
        skipped = 0
        for tile in tiles:
            role = eval_role_for_tile(tile)
            if role is None:
                skipped += 1
                continue
            by_role.setdefault(role, set()).add(tile)
        for role, role_tiles in sorted(by_role.items()):
            registry = register_tiles(
                registry, role_tiles, eval_role=role, source_run=source,
            )
            print(f"[add] {len(role_tiles):4d} {role:9s} <- {source}")
        if skipped:
            print(f"[skip] {skipped:4d} train-event tiles <- {source}")
        n_sources += 1

    digest = write_registry(out, registry)
    from collections import Counter
    roles = Counter(str(row.get("eval_role")) for row in registry["items"])
    print(f"[ok] {len(registry['items'])} unique tiles from {n_sources} sources ({dict(roles)})")
    print(f"[ok] registry={out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
