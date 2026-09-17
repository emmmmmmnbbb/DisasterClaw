#!/usr/bin/env python3
"""Read-only P6 final identity gate; optionally write one launch receipt.

This does not import the app, render imagery, or load ChangeOS/Qwen weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from bench_agent_vqa import CONFIGS, source_fingerprint
from vlm_analyzer import AGENT_VQA_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate(protocol_path: Path, repeat: int, shard: int, out_dir: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "changeos-p6-final-protocol/1.0":
        raise ValueError("invalid P6 protocol schema")
    if repeat not in protocol["generation_repeats"]:
        raise ValueError(f"generation repeat {repeat} is not frozen")
    n_shards = int(protocol["shard_count"])
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} is outside 0/{n_shards}..{n_shards-1}/{n_shards}")
    if out_dir.exists():
        raise FileExistsError(f"final output already exists; no implicit resume: {out_dir}")

    for name, expected in (protocol.get("analysis_code_sha256") or {}).items():
        path = resolve(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen launch/analysis code changed: {name}")

    inputs = {}
    for name, hash_key in (
        ("testset", "testset_sha256"),
        ("review_report", "review_report_sha256"),
        ("frozen_manifest", "frozen_manifest_sha256"),
        ("changeos_weights", "changeos_weights_sha256"),
    ):
        path = resolve(str(protocol[name]))
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        actual = sha256(path)
        if actual != protocol[hash_key]:
            raise ValueError(f"{name} changed: {actual} != {protocol[hash_key]}")
        inputs[name] = {"path": str(path), "sha256": actual}

    taskset = json.loads(Path(inputs["testset"]["path"]).read_text(encoding="utf-8"))
    review = json.loads(Path(inputs["review_report"]["path"]).read_text(encoding="utf-8"))
    frozen = json.loads(Path(inputs["frozen_manifest"]["path"]).read_text(encoding="utf-8"))
    items = list(taskset.get("items") or [])
    qids = [str(item.get("id") or "") for item in items]
    if taskset.get("eval_role") != "final" or taskset.get("damage_label_mode") != "binary":
        raise ValueError("taskset must be final, binary")
    if len(items) != protocol["n_questions"] or len(set(qids)) != len(items) or not all(qids):
        raise ValueError("final item count or qid uniqueness changed")
    events = Counter(str(item.get("disaster") or "") for item in items)
    if set(events) != set(protocol["events"]) or dict(events) != protocol["event_question_counts"]:
        raise ValueError("final event composition changed")
    roi_ids = {(item.get("tile_id"), json_hash((item.get("roi") or {}).get("bounds")))
               for item in items}
    if len(roi_ids) != protocol["n_unique_roi"]:
        raise ValueError("final ROI count changed")
    if taskset.get("items_sha256") != json_hash(items):
        raise ValueError("internal items_sha256 is stale")
    statuses = {str(row.get("id") or ""): row for row in review.get("per_item") or []}
    if (review.get("schema_version") != "agent-vqa-review/2.0"
            or len(statuses) != len(items)
            or set(statuses) != set(qids)
            or any(statuses[qid].get("status") != "approved" for qid in qids)
            or review.get("approved") != len(items)
            or review.get("human_approved") != len(items)
            or review.get("auto_approved") != len(items)):
        raise ValueError("final review report no longer approves all questions")
    if (frozen.get("backend") != "changeos" or frozen.get("label_mode") != "binary"
            or frozen.get("testset_sha256") != inputs["testset"]["sha256"]
            or frozen.get("review_report_sha256") != inputs["review_report"]["sha256"]):
        raise ValueError("frozen manifest identity or label mode changed")

    configs = list(protocol["configs"])
    if (len(set(configs)) != len(configs)
            or any(config not in CONFIGS for config in configs)
            or json_hash({config: CONFIGS[config] for config in configs})
            != protocol["policy_config_sha256"]):
        raise ValueError("online policy definitions changed")
    if (source_fingerprint() != protocol["source_fingerprint"]
            or hashlib.sha256(AGENT_VQA_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
            != protocol["vlm_system_prompt_sha256"]):
        raise ValueError("online source or VLM prompt changed")
    if (int(protocol["rows_per_shard_per_repeat"])
            != len(items[shard::n_shards]) * len(configs)):
        raise ValueError("frozen rows-per-shard expectation changed")
    if (protocol["expected_total_online_rows"]
            != len(items) * len(configs) * len(protocol["generation_repeats"])):
        raise ValueError("frozen total online row count changed")
    return {
        "schema": "changeos-p6-final-preflight/1.0",
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "generation_repeat": repeat, "shard": f"{shard}/{n_shards}",
        "out_dir": str(out_dir), "n_questions": len(items[shard::n_shards]),
        "n_expected_rows": len(items[shard::n_shards]) * len(configs),
        "configs": configs, "source_fingerprint": protocol["source_fingerprint"],
        "inputs": inputs, "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    report = validate(args.protocol, args.repeat, args.shard, args.out_dir)
    if args.receipt:
        if args.receipt.exists() or args.receipt.parent != args.out_dir:
            raise ValueError("receipt must be a new file directly in the new output directory")
        args.out_dir.mkdir(parents=True, exist_ok=False)
        args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
