#!/usr/bin/env python3
"""Validate frozen P6 shards and summarize paired three-repeat policy outcomes.

Unlike report_agent_vqa.py, this script keys pairs by (repeat, qid), then
resamples whole ROI clusters with all their questions and repeats intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bench_agent_vqa import episode_seed
from audit_changeos_p7_costs import audit_row

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_seed(base: int, qid: str, repeat: int, step: int, role: str) -> int:
    payload = f"agent-vqa-generation|{base}|{qid}|{repeat}|{step}|{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def roi_key(item: dict) -> str:
    roi = item.get("roi") or {}
    bounds = json.dumps(roi.get("bounds") or {}, ensure_ascii=False, sort_keys=True)
    return str(item.get("tile_id") or "") + "|" + bounds


def load_protocol(path: Path) -> tuple[dict, dict[str, dict]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "changeos-p6-final-protocol/1.0":
        raise ValueError("invalid P6 protocol")
    for name, expected in (protocol.get("analysis_code_sha256") or {}).items():
        source = ROOT / name
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"frozen launch/analysis code changed: {name}")
    task_path = ROOT / protocol["testset"]
    if sha256(task_path) != protocol["testset_sha256"]:
        raise ValueError("frozen final testset changed")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    return protocol, {str(item["id"]): item for item in task["items"]}


def validate_run(run: Path, protocol: dict, items: dict[str, dict]) -> tuple[dict, list[dict]]:
    result_path, manifest_path, episode_path = (
        run / "results.json", run / "manifest.json", run / "episodes.jsonl"
    )
    if not all(path.is_file() for path in (result_path, manifest_path, episode_path)):
        raise ValueError(f"missing final result files: {run}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args = result.get("args") or {}
    repeat, shard, n_shards = (
        args.get("generation_repeat"), args.get("_shard_i"), args.get("_shard_n")
    )
    if repeat not in protocol["generation_repeats"] or not isinstance(shard, int) or n_shards != protocol["shard_count"]:
        raise ValueError(f"run repeat/shard is not frozen: {run}")
    if (args.get("configs") != ",".join(protocol["configs"])
            or args.get("testset") != protocol["testset"]
            or args.get("review_report") != protocol["review_report"]
            or args.get("frozen_manifest") != protocol["frozen_manifest"]
            or args.get("seed") != protocol["action_base_seed"]
            or args.get("generation_seed") != protocol["generation_base_seed"]
            or args.get("resume") is not False
            or args.get("allow_label_mismatch") is not False
            or args.get("limit") != 0 or args.get("split") or args.get("qtype")):
        raise ValueError(f"run command diverged from P6 protocol: {run}")
    if (result.get("valid_for_analysis") is not True
            or result.get("n_execution_errors") != 0
            or result.get("n_items") != len(list(items)[shard::n_shards])):
        raise ValueError(f"run has wrong size, execution errors, or invalid result: {run}")
    env = manifest.get("env") or {}
    if (env.get("source_fingerprint") != protocol["source_fingerprint"]
            or env.get("detector_backend") != "changeos"
            or env.get("damage_label_mode") != "binary"
            or env.get("vlm_provider") != "qwen_vl_local"
            or env.get("vlm_model") != "Qwen/Qwen2.5-VL-7B-Instruct"
            or float(env.get("vlm_top_p") or 0) != 0.9
            or float(env.get("vlm_repetition_penalty") or 0) != 1.1
            or str(env.get("agent_vqa_confidence_threshold") or "") != "0.5"
            or (env.get("perception_tool") or {}).get("weights_sha256")
            != protocol["changeos_weights_sha256"]
            or manifest.get("testset_sha256_16") != protocol["testset_sha256"]
            or manifest.get("review_report_sha256_16") != protocol["review_report_sha256"]
            or manifest.get("frozen_manifest_sha256_16") != protocol["frozen_manifest_sha256"]
            or manifest.get("vlm_system_prompt_sha256") != protocol["vlm_system_prompt_sha256"]
            or manifest.get("generation_repeat") != repeat
            or manifest.get("generation_base_seed") != protocol["generation_base_seed"]):
        raise ValueError(f"run executable/perception/generation identity differs: {run}")
    modes = manifest.get("agent_vqa_answer_modes") or {}
    if modes != {config: protocol["answer_mode"] for config in protocol["configs"]}:
        raise ValueError(f"answer mode changed: {run}")
    expected_qids = set(list(items)[shard::n_shards])
    expected_keys = {(config, qid) for config in protocol["configs"] for qid in expected_qids}
    rows = []
    seen = set()
    for line in episode_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        config, qid = str(row.get("config") or ""), str(row.get("qid") or "")
        key = (config, qid)
        if key not in expected_keys or key in seen:
            raise ValueError(f"unexpected/duplicate final episode key: {run} {key}")
        seen.add(key)
        item = items[qid]
        if (row.get("question_type") != item["question_type"]
                or row.get("disaster") != item["disaster"]
                or row.get("tile_id") != item["tile_id"]
                or row.get("answer_mode") != protocol["answer_mode"]
                or row.get("action_seed") != episode_seed(protocol["action_base_seed"], config, qid)
                or (row.get("ok") is not True and row.get("reason_code") != "out_of_coverage")):
            raise ValueError(f"episode identity or execution differs: {run} {key}")
        if row.get("ok") is True and (
            not row.get("trajectory") or not row.get("generation_seeds")
        ):
            raise ValueError(f"completed hybrid episode lacks recorded VLM generation: {run} {key}")
        observed_seeds = []
        for step, trajectory in enumerate(row.get("trajectory") or []):
            actual = trajectory.get("generation_seed")
            if actual is None:
                continue
            observed_seeds.append(actual)
            role = str(trajectory.get("generation_call_role") or "")
            if (trajectory.get("generation_repeat") != repeat
                    or actual != generation_seed(protocol["generation_base_seed"], qid, repeat, step, role)):
                raise ValueError(f"generation call key mismatch: {run} {key} step={step}")
        if row.get("generation_seeds") != observed_seeds:
            raise ValueError(f"generation seed summary differs from trajectory: {run} {key}")
        rows.append(row)
    if seen != expected_keys or len(rows) != protocol["rows_per_shard_per_repeat"]:
        raise ValueError(f"final shard incomplete: {run}")
    return {
        "run": str(run), "repeat": repeat, "shard": shard,
        "n_rows": len(rows), "n_questions": len(expected_qids),
        "results_sha256": sha256(result_path),
        "manifest_sha256": sha256(manifest_path),
        "episodes_sha256": sha256(episode_path), "valid_for_analysis": True,
    }, rows


def _accuracy(rows: list[dict]) -> float:
    return sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0


def bootstrap_roi(deltas: dict[str, list[float]], n_boot: int, seed: int) -> dict:
    rois = sorted(deltas)
    if not rois or n_boot < 100:
        raise ValueError("ROI bootstrap requires data and at least 100 draws")
    point = sum(sum(values) for values in deltas.values()) / sum(len(values) for values in deltas.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled = [rng.choice(rois) for _ in rois]
        values = [delta for roi in sampled for delta in deltas[roi]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "n_roi": len(rois), "n_paired_repeat_questions": sum(map(len, deltas.values())),
        "mean_difference": point,
        "ci95": [draws[int(0.025 * n_boot)], draws[min(int(0.975 * n_boot), n_boot - 1)]],
        "ci98_75_bonferroni_four_primary": [
            draws[int(0.00625 * n_boot)], draws[min(int(0.99375 * n_boot), n_boot - 1)]
        ],
        "n_boot": n_boot, "bootstrap_seed": seed,
        "unit": "ROI cluster; all qids and three repeats retained within a sampled ROI",
    }


def summarize(rows: list[dict], protocol: dict, items: dict[str, dict], n_boot: int = 5000) -> dict:
    by_config = defaultdict(list)
    by_repeat_config = defaultdict(list)
    indexed = {}
    for row in rows:
        repeat = int(row["_p6_repeat"])
        config, qid = str(row["config"]), str(row["qid"])
        key = (repeat, config, qid)
        if key in indexed:
            raise ValueError(f"duplicate across P6 shards: {key}")
        indexed[key] = row
        by_config[config].append(row)
        by_repeat_config[(repeat, config)].append(row)
    all_qids = set(items)
    all_keys = {(repeat, config, qid)
                for repeat in protocol["generation_repeats"]
                for config in protocol["configs"] for qid in all_qids}
    if set(indexed) != all_keys:
        raise ValueError("full P6 matrix is incomplete or inconsistent")
    by_policy = {}
    for config in protocol["configs"]:
        series = by_config[config]
        motion = [audit_row(row, items[str(row["qid"])]["start"]) for row in series]
        by_policy[config] = {
            "n_repeat_questions": len(series), "n_unique_questions": len(all_qids),
            "accuracy": _accuracy(series),
            "accuracy_by_repeat": {
                str(repeat): _accuracy(by_repeat_config[(repeat, config)])
                for repeat in protocol["generation_repeats"]
            },
            "mean_executed_reobservations_per_question_repeat":
                sum(int(row.get("n_reobservations") or 0) for row in series) / len(series),
            "n_motion_complete": sum(bool(entry["trajectory_motion_complete"]) for entry in motion),
            "mean_state_derived_total_horizontal_m":
                sum(entry["total_horizontal_m"] for entry in motion) / len(motion),
            "mean_state_derived_total_vertical_m":
                sum(entry["total_vertical_m"] for entry in motion) / len(motion),
            "mean_state_derived_search_horizontal_m":
                sum(entry["search_horizontal_m"] for entry in motion) / len(motion),
            "mean_state_derived_reobserve_horizontal_m":
                sum(entry["reobserve_horizontal_m"] for entry in motion) / len(motion),
            "mean_episode_wall_s": sum(float(row.get("wall_s") or 0) for row in series) / len(series),
            "by_question_type": {
                qtype: {"n_repeat_questions": len(sub), "accuracy": _accuracy(sub)}
                for qtype in sorted({row["question_type"] for row in series})
                if (sub := [row for row in series if row["question_type"] == qtype])
            },
            "by_event": {
                event: {"n_repeat_questions": len(sub), "accuracy": _accuracy(sub)}
                for event in protocol["events"]
                if (sub := [row for row in series if row["disaster"] == event])
            },
        }
    paired = {}
    for baseline in ("A0_HOLD", "A1_RANDOM", "A2_ALWAYS", "A3U_RAW_ENTROPY"):
        deltas: dict[str, list[float]] = defaultdict(list)
        by_repeat = {}
        for repeat in protocol["generation_repeats"]:
            repeat_deltas = []
            for qid in sorted(all_qids):
                task = indexed[(repeat, "T1_TASK", qid)]
                base = indexed[(repeat, baseline, qid)]
                delta = float(bool(task.get("correct"))) - float(bool(base.get("correct")))
                deltas[roi_key(items[qid])].append(delta)
                repeat_deltas.append(delta)
            by_repeat[str(repeat)] = sum(repeat_deltas) / len(repeat_deltas)
        paired[f"T1_TASK_vs_{baseline}"] = {
            "difference_by_repeat": by_repeat,
            "roi_cluster_bootstrap": bootstrap_roi(deltas, n_boot, seed=42),
        }
    return {
        "schema": "changeos-p6-final-report/1.0",
        "status": "VALIDATED_COMPLETE",
        "n_online_rows": len(rows), "n_questions": len(all_qids),
        "n_repeats": len(protocol["generation_repeats"]),
        "by_policy": by_policy, "primary_paired_contrasts": paired,
        "cost_note": "actions and episode wall time are observed; horizontal/vertical totals are trajectory-state geometric estimates, not measured UAV flight time; component latencies remain unmeasured",
        "inference_note": "three events only; ROI bootstrap reflects within-event ROI/question variability, not unseen-event generalization",
        "budget_note": "realized policy operating points; no randomized reobservation-cap sweep",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="validate only supplied shards; no inferential report")
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    protocol, items = load_protocol(args.protocol)
    audits = []
    rows = []
    run_keys = set()
    for run in args.runs:
        audit, run_rows = validate_run(run, protocol, items)
        key = (audit["repeat"], audit["shard"])
        if key in run_keys:
            raise ValueError(f"duplicate repeat/shard run: {key}")
        run_keys.add(key)
        audits.append(audit)
        rows.extend({**row, "_p6_repeat": audit["repeat"]} for row in run_rows)
    if not args.partial and (len(run_keys) != protocol["shard_count"] * len(protocol["generation_repeats"])
                             or len(rows) != protocol["expected_total_online_rows"]):
        raise ValueError("P6 full report requires all 12 valid repeat/shard runs")
    report = {
        "schema": "changeos-p6-final-audit-report/1.0",
        "protocol": str(args.protocol), "protocol_sha256": sha256(args.protocol),
        "runs": audits, "n_validated_rows": len(rows),
        "partial": args.partial,
        "analysis": None if args.partial else summarize(rows, protocol, items, args.n_boot),
    }
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite P6 report: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_runs": len(audits), "n_validated_rows": len(rows),
                      "partial": args.partial, "status": "PASS"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
