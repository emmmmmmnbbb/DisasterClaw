"""ROI consumption registry shared by Agent-VQA generators and audits."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA = "tile-consumption-registry/1.0"
VALID_ROLES = {"consumed", "fit", "selection", "final", "boundary"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def empty_registry() -> dict:
    return {
        "schema": SCHEMA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }


def load_registry(path: Path, *, allow_missing: bool = False) -> dict:
    if not path.is_file():
        if allow_missing:
            return empty_registry()
        raise FileNotFoundError(f"ROI consumption registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(
            f"registry schema mismatch: got {data.get('schema')!r}, expected {SCHEMA!r}"
        )
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("registry items must be a list")
    assert_role_disjoint(items)
    return data


def assert_role_disjoint(items: Iterable[dict]) -> None:
    """A tile may have several sources, but only one experimental role."""
    roles_by_tile: dict[str, set[str]] = {}
    for rec in items:
        tile_id = str(rec.get("tile_id") or "").strip()
        role = str(rec.get("eval_role") or "").strip()
        if not tile_id or role not in VALID_ROLES:
            raise ValueError(f"invalid registry row: {rec!r}")
        roles_by_tile.setdefault(tile_id, set()).add(role)
    overlaps = {tile: sorted(roles) for tile, roles in roles_by_tile.items() if len(roles) > 1}
    if overlaps:
        sample = dict(list(sorted(overlaps.items()))[:10])
        raise ValueError(f"ROI role overlap detected: {sample}")


def tile_ids(registry: dict, roles: Iterable[str] | None = None) -> set[str]:
    wanted = set(roles or VALID_ROLES)
    return {
        str(rec["tile_id"])
        for rec in registry.get("items", [])
        if str(rec.get("eval_role")) in wanted
    }


def register_tiles(
    registry: dict,
    tiles: Iterable[str],
    *,
    eval_role: str,
    source_run: str,
) -> dict:
    if eval_role not in VALID_ROLES:
        raise ValueError(f"invalid eval_role {eval_role!r}; expected one of {sorted(VALID_ROLES)}")
    by_tile = {str(r["tile_id"]): dict(r) for r in registry.get("items", [])}
    for tile in sorted({str(t).strip() for t in tiles if str(t).strip()}):
        current = by_tile.get(tile)
        if current and current.get("eval_role") != eval_role:
            raise ValueError(
                f"tile {tile!r} is already {current.get('eval_role')!r}, "
                f"cannot register as {eval_role!r}"
            )
        sources = set((current or {}).get("source_runs") or [])
        sources.add(str(source_run))
        by_tile[tile] = {
            "tile_id": tile,
            "eval_role": eval_role,
            "source_runs": sorted(sources),
        }
    out = {
        "schema": SCHEMA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": [by_tile[k] for k in sorted(by_tile)],
    }
    assert_role_disjoint(out["items"])
    return out


def write_registry(path: Path, registry: dict) -> str:
    assert_role_disjoint(registry.get("items", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)
