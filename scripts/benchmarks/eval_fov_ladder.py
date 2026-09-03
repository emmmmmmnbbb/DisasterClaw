#!/usr/bin/env python3
"""Paired building observations on the physical mosaic-FOV ladder.

This is the replacement for ``eval_gsd_ladder.py`` in formal experiments.
It never applies synthetic blur: every view is rendered by ``mosaic_fov`` at
cruise/mid/floor altitude and matched to the same ROI-scoped GT buildings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import fov_ladder as FL  # noqa: E402
import mosaic as mosaic_mod  # noqa: E402
import xbd_map  # noqa: E402
from detectors.base import DAMAGE_SUBTYPES  # noqa: E402
from event_split import EVAL_EVENTS, HOLDOUT_EVENTS, TEST_EVENTS, VAL_EVENTS  # noqa: E402
from eval_identifiability import _gt_buildings, _match, _observe  # noqa: E402
from recheck import entropy_uncertainty, fit_conformal_qhat  # noqa: E402
from tile_consumption import load_registry, register_tiles, write_registry  # noqa: E402

CLASS_NAMES = tuple(DAMAGE_SUBTYPES)
VIEW_ALTS = {
    "cruise": FL.alt_cruise_m(),
    "mid": FL.alt_for_span_tiles(2.0),
    "floor": FL.alt_min_m(),
}


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    powered = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / float(temperature))
    return _normalise(powered)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on validation probabilities only."""
    candidates = np.geomspace(0.25, 4.0, 241)
    losses = []
    for value in candidates:
        calibrated = apply_temperature(probs, float(value))
        losses.append(float(-np.log(calibrated[np.arange(len(labels)), labels]).mean()))
    return float(candidates[int(np.argmin(losses))])


def _metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    f1 = []
    for c in range(len(CLASS_NAMES)):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    onehot = np.eye(len(CLASS_NAMES))[y]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(np.mean(f1)),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "mean_entropy": float(np.mean([
            entropy_uncertainty(dict(zip(CLASS_NAMES, row))) for row in probs
        ])),
    }


def _missing_probs() -> dict[str, float]:
    return {name: float(name == "no-damage") for name in CLASS_NAMES}


def _view_record(det: dict | None) -> dict:
    probs = _missing_probs() if det is None else {
        name: float(det["class_probs"].get(name, 0.0)) for name in CLASS_NAMES
    }
    arr = _normalise(np.asarray([list(probs.values())]))[0]
    probs = {name: float(v) for name, v in zip(CLASS_NAMES, arr)}
    pred = CLASS_NAMES[int(np.argmax(arr))]
    return {
        "probs": probs,
        "pred": pred,
        "entropy": entropy_uncertainty(probs),
        "detected": det is not None,
    }


def _items_hash(rows: list[dict]) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_entropy_table(items: list[dict], fit_events: list[str]) -> dict:
    bins = []
    for view_name, alt in VIEW_ALTS.items():
        by_class: dict[str, list[float]] = defaultdict(list)
        all_entropy = []
        for item in items:
            view = item["views"][view_name]
            ent = float(view["entropy"])
            by_class[view["pred"]].append(ent)
            all_entropy.append(ent)
        by_pred = {
            name: {"mean_entropy": float(np.mean(vals)), "n": len(vals)}
            for name, vals in sorted(by_class.items()) if vals
        }
        by_pred["all"] = {
            "mean_entropy": float(np.mean(all_entropy)) if all_entropy else None,
            "n": len(all_entropy),
        }
        bins.append({
            "view": view_name,
            "alt_m": round(alt, 3),
            "gsd_m": round(FL.eff_gsd_for_alt(alt), 6),
            "by_pred_class": by_pred,
        })
    return {
        "schema": FL.ExpectedEntropyTable.SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_model": "mosaic_fov",
        "fit_events": sorted(fit_events),
        "fit_n": len(items),
        "items_sha256": _items_hash(items),
        "bins": bins,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="新 FOV 几何配对观测、校准与熵表拟合")
    ap.add_argument("--split", choices=["val", "eval", "test", "holdout"], default="val")
    ap.add_argument("--limit", type=int, default=0, help="ROI 上限，0=全部")
    ap.add_argument("--device", default=os.getenv("PERCEPTION_DEVICE", "cuda:0"))
    ap.add_argument("--backend", default="xview2_first")
    ap.add_argument("--archs", default="res34")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--manifest", default=str(BACKEND / "data/xbd/manifest.json"))
    ap.add_argument("--roi-index", default=str(BACKEND / "data/xbd/roi_index.json"))
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="非 val 运行使用的冻结温度")
    ap.add_argument("--out-dir", default=str(ROOT / "runs/benchmarks/paper_cja_mech_v1"))
    ap.add_argument("--table-out", default="",
                    help="额外写入在线控制器使用的冻结熵表（通常 backend/data/fov_entropy_table.json）")
    ap.add_argument("--registry", default="")
    ap.add_argument("--register-fit", action="store_true")
    ap.add_argument("--tiles-from", default="",
                    help="只评测该 Agent-VQA JSON 中出现的 tile_id")
    ap.add_argument("--shard", default="", help="ROI 分片 i/N；输出文件自动加 shard 后缀")
    args = ap.parse_args()

    if args.register_fit and args.split != "val":
        raise ValueError("--register-fit 只允许 --split val")

    events = {
        "val": set(VAL_EVENTS), "eval": set(EVAL_EVENTS),
        "test": set(TEST_EVENTS), "holdout": set(HOLDOUT_EVENTS),
    }[args.split]
    manifest = xbd_map.load_manifest(args.manifest)
    root = Path(manifest["dataset_root"])
    coverage = json.loads(Path(args.roi_index).read_text(encoding="utf-8"))["coverage"]
    allowed_tiles = None
    if args.tiles_from:
        source = json.loads(Path(args.tiles_from).read_text(encoding="utf-8"))
        allowed_tiles = {str(row["tile_id"]) for row in source.get("items", [])}
    candidates = [
        row for row in manifest["items"]
        if row.get("stage") == "post"
        and row.get("disaster") in events
        and row.get("label_relpath")
        and row.get("paired_tile_id")
        and float(coverage.get(row["tile_id"], 0.0)) >= args.min_coverage
        and (allowed_tiles is None or row["tile_id"] in allowed_tiles)
    ]
    candidates.sort(key=lambda row: (row["disaster"], row["tile_id"]))
    if args.limit:
        candidates = candidates[:args.limit]
    shard_suffix = ""
    if args.shard:
        shard_i, shard_n = (int(v) for v in args.shard.split("/", 1))
        if not 0 <= shard_i < shard_n:
            raise ValueError(f"invalid --shard {args.shard!r}")
        candidates = candidates[shard_i::shard_n]
        shard_suffix = f"_shard{shard_i}of{shard_n}"
    if not candidates:
        raise RuntimeError("没有符合 split/coverage 条件的 ROI")

    from detectors import get_detector

    detector = get_detector(
        args.backend,
        device=args.device,
        archs=tuple(v.strip() for v in args.archs.split(",") if v.strip()),
        seeds=tuple(int(v) for v in args.seeds.split(",") if v.strip()),
    )
    mosaic = mosaic_mod.from_manifest(manifest)
    items: list[dict] = []
    consumed_tiles: set[str] = set()
    for index, entry in enumerate(candidates, 1):
        gt = _gt_buildings(root / entry["label_relpath"], entry)
        if not gt:
            continue
        detected = {}
        for view_name, alt in VIEW_ALTS.items():
            rows, _meta = _observe(mosaic, detector, entry, alt, entry["bounds"])
            detected[view_name] = _match(gt, rows)
        for building_index, building in enumerate(gt):
            item = {
                "uid": f"{entry['tile_id']}::{building_index}",
                "tile_id": entry["tile_id"],
                "disaster": entry["disaster"],
                "split": args.split,
                "y": CLASS_NAMES.index(building["subtype"]),
                "subtype": building["subtype"],
                "views": {
                    name: _view_record(detected[name][building_index])
                    for name in VIEW_ALTS
                },
            }
            items.append(item)
        consumed_tiles.add(entry["tile_id"])
        if index % 5 == 0:
            print(f"[fov] {index}/{len(candidates)} ROIs, {len(items)} buildings", flush=True)

    if not items:
        raise RuntimeError("观测完成但没有可评测建筑")

    raw_cruise = np.stack([
        [float(row["views"]["cruise"]["probs"][name]) for name in CLASS_NAMES]
        for row in items
    ])
    labels = np.asarray([int(row["y"]) for row in items], dtype=np.int64)
    temperature = fit_temperature(raw_cruise, labels) if args.split == "val" else args.temperature
    for row in items:
        for view in row["views"].values():
            calibrated = apply_temperature(
                np.asarray([[float(view["probs"][name]) for name in CLASS_NAMES]]),
                temperature,
            )[0]
            view["probs"] = {name: float(v) for name, v in zip(CLASS_NAMES, calibrated)}
            view["pred"] = CLASS_NAMES[int(np.argmax(calibrated))]
            view["entropy"] = entropy_uncertainty(view["probs"])

    conformal_rows = [
        (row["views"]["cruise"]["probs"], row["subtype"]) for row in items
    ]
    qhat = fit_conformal_qhat(conformal_rows, alpha=0.1)
    fit_events = sorted({row["disaster"] for row in items})
    table = build_entropy_table(items, fit_events)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path = output_dir / f"fov_ladder_{args.split}{shard_suffix}_items.jsonl"
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items),
        encoding="utf-8",
    )
    table_path = output_dir / f"fov_entropy_table_{args.split}{shard_suffix}.json"
    table_path.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.table_out:
        online_table_path = Path(args.table_out)
        online_table_path.parent.mkdir(parents=True, exist_ok=True)
        online_table_path.write_text(
            json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    curve = []
    for view_name in VIEW_ALTS:
        probs = np.stack([
            [float(row["views"][view_name]["probs"][name]) for name in CLASS_NAMES]
            for row in items
        ])
        curve.append({
            "view": view_name,
            "alt_m": round(VIEW_ALTS[view_name], 3),
            "gsd_m": round(FL.eff_gsd_for_alt(VIEW_ALTS[view_name]), 6),
            **_metrics(probs, labels),
        })
    report = {
        "schema": "fov-ladder-eval/1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "events": fit_events,
        "n_rois": len(consumed_tiles),
        "n_buildings": len(items),
        "observation_model": "mosaic_fov",
        "detector": detector.describe(),
        "leaky": bool(detector.describe().get("leaky")),
        "temperature": temperature,
        "conformal_qhat_alpha01": qhat,
        "items_sha256": _items_hash(items),
        "curve": curve,
        "paths": {"items": str(items_path), "entropy_table": str(table_path)},
        "shard": args.shard or None,
    }
    if args.registry and args.register_fit:
        registry_path = Path(args.registry)
        registry = load_registry(registry_path, allow_missing=True)
        registry = register_tiles(
            registry, consumed_tiles, eval_role="fit", source_run=str(items_path),
        )
        report["registry_sha256"] = write_registry(registry_path, registry)
    (output_dir / f"fov_ladder_{args.split}{shard_suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    print(json.dumps({
        "split": args.split, "n_rois": len(consumed_tiles), "n_buildings": len(items),
        "temperature": temperature, "qhat": qhat, "curve": curve,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
