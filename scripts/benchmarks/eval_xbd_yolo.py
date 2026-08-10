#!/usr/bin/env python3
"""Evaluate a strict-split xBD YOLO checkpoint on val/test/holdout."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/lc/datasets/xbd_yolo_strict_v1/data.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-event-disjoint", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    manifest_path = data_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("split_audit") or {}
    if args.require_event_disjoint and (
        manifest.get("split_strategy") != "strict_event"
        or not audit.get("event_disjoint")
        or audit.get("overlaps")
    ):
        raise ValueError(f"not an event-disjoint dataset: {manifest_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ("val", "test", "holdout"):
        if split not in data_config:
            continue
        eval_yaml = out_path.parent / f"_eval_{split}.yaml"
        split_config = dict(data_config)
        split_config["val"] = data_config[split]
        eval_yaml.write_text(yaml.safe_dump(split_config, sort_keys=False), encoding="utf-8")
        metrics = model.val(
            data=str(eval_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            verbose=False,
        )
        results[split] = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "per_class_mAP50_95": [float(value) for value in metrics.box.maps],
        }
        eval_yaml.unlink(missing_ok=True)

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "data": str(data_path),
        "manifest_sha256": _sha256(manifest_path),
        "event_split": audit.get("events"),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
