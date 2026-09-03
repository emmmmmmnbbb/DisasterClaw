from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "backend"))

from tile_consumption import (
    empty_registry,
    load_registry,
    register_tiles,
    sha256_file,
    tile_ids,
    write_registry,
)


def test_roles_are_mutually_exclusive(tmp_path):
    registry = register_tiles(
        empty_registry(), ["tile-a"], eval_role="selection", source_run="selection.json",
    )
    with pytest.raises(ValueError, match="already"):
        register_tiles(registry, ["tile-a"], eval_role="final", source_run="final.json")


def test_backfill_maps_val_to_fit_and_eval_to_consumed():
    from build_tile_consumption_registry import disaster_from_tile_id, eval_role_for_tile

    assert disaster_from_tile_id("hurricane-harvey_00000033_post_disaster") == "hurricane-harvey"
    assert disaster_from_tile_id("mexico-earthquake_00000001_post_disaster") == "mexico-earthquake"
    assert eval_role_for_tile("hurricane-harvey_00000033_post_disaster") == "fit"
    assert eval_role_for_tile("hurricane-michael_00000018_post_disaster") == "consumed"
    assert eval_role_for_tile("moore-tornado_00000001_post_disaster") == "consumed"
    assert eval_role_for_tile("guatemala-volcano_00000003_post_disaster") is None


def test_registry_roundtrip_and_hash(tmp_path):
    path = tmp_path / "registry.json"
    registry = register_tiles(
        empty_registry(), ["tile-b", "tile-a"], eval_role="consumed", source_run="old.json",
    )
    digest = write_registry(path, registry)
    loaded = load_registry(path)
    assert tile_ids(loaded) == {"tile-a", "tile-b"}
    assert digest == sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema"].endswith("/1.0")


def test_freeze_manifest_refuses_overwrite_and_records_hashes(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "freeze_recheck_manifest", HERE / "freeze_recheck_manifest.py",
    )
    freeze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze)

    def dump(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    sweep = dump("sweep.json", {"selected": {"entropy_trigger": 0.5, "min_info_gain": 0.05, "budget": 0.25}})
    selection = dump("selection.json", {"items": []})
    final = dump("final.json", {"items": []})
    fit = dump("fit.json", {"temperature": 1.2, "conformal_qhat_alpha01": 0.8})
    table = dump("table.json", {"schema": "fov-ladder-entropy/1.0", "bins": [{"gsd_m": 0.5}]})
    registry = dump("registry.json", {"schema": "tile-consumption-registry/1.0", "items": []})
    out = tmp_path / "frozen.json"
    sys.argv = [
        "freeze_recheck_manifest.py",
        "--selection-sweep", str(sweep),
        "--selection-testset", str(selection),
        "--final-testset", str(final),
        "--fit-report", str(fit),
        "--entropy-table", str(table),
        "--registry", str(registry),
        "--out", str(out),
    ]
    assert freeze.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "recheck-frozen-manifest/1.0"
    assert payload["entropy_table_sha256"]
    assert payload["testset_sha256"]
    with pytest.raises(FileExistsError):
        freeze.main()
