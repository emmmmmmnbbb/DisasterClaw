#!/usr/bin/env python3
"""Paired, live fixed-view / changed-view T1 recheck diagnostic.

The intervention is installed only in this runner.  At a real T1 recheck
motion call, it re-perceives the unchanged pose, calls the normal answer path
with the *next step's* seed, restores the live semantic map, and lets the
unchanged T1 branch continue.  The fixed answer is never fed to the policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_vqa import (GenerationContext, build_evidence_from_perception,
                       derive_generation_seed)  # noqa: E402
from bench_agent_vqa import CONFIGS, apply_config, episode_seed, source_fingerprint  # noqa: E402
from vlm_analyzer import AGENT_VQA_SYSTEM_PROMPT, VLMAnalyzer  # noqa: E402
from semantic_map import SemanticMap  # noqa: E402


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clone_semantic_map(source: SemanticMap | None) -> SemanticMap | None:
    """Copy runtime map layers without attempting to pickle its RLock."""
    if source is None:
        return None
    clone = SemanticMap(source.origin_lat, source.origin_lon,
                        source.cell_size_m, source.instruction)
    with source._lock:
        clone.created_at = source.created_at
        clone.step_count = source.step_count
        clone._object_cells = copy.deepcopy(source._object_cells)
        clone._explored = set(source._explored)
        clone._current_fov = set(source._current_fov)
    return clone


def reference_rows(root: Path) -> dict[tuple[int, str], dict]:
    rows = {}
    for repeat in range(3):
        for shard in range(4):
            path = root / (f"p6_final_three_event_v1_20260915_repeat{repeat}"
                           f"_shard{shard}of4/episodes.jsonl")
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                if row.get("config") == "T1_TASK":
                    rows[(repeat, row["qid"])] = row
    return rows


def run_one(app, item: dict, repeat: int, frozen: dict) -> dict:
    qid = item["id"]
    apply_config(app, CONFIGS["T1_TASK"])
    app.VLN_ENTROPY_TABLE = str(frozen.get("entropy_table_path", app.VLN_ENTROPY_TABLE))
    app.VLN_CONFORMAL_QHAT = float(frozen.get("qhat", app.VLN_CONFORMAL_QHAT))
    app.VLN_CONFORMAL_ALPHA = float(frozen.get("conformal_alpha", app.VLN_CONFORMAL_ALPHA))
    app.VLN_RECHECK_TEMPERATURE = float(frozen.get("temperature", app.VLN_RECHECK_TEMPERATURE))
    app.VLN_RECHECK_MIN_INFO_GAIN = float(frozen.get("min_info_gain", app.VLN_RECHECK_MIN_INFO_GAIN))
    app.AGENT_VQA_GENERATION_BASE_SEED = 42000
    app.AGENT_VQA_GENERATION_REPEAT = repeat
    app.VLN_RECHECK_RANDOM_SEED = episode_seed(42, "T1_TASK", qid)
    original_factory = app._make_agent_vqa_controller
    original_execute = app.execute_action
    original_answer = VLMAnalyzer.answer_image_question
    forks: list[dict] = []
    calls: dict[int, list[dict]] = defaultdict(list)
    holder: dict = {}
    phase = "changed"

    def answer_audit(self, *args, **kwargs):
        result = original_answer(self, *args, **kwargs)
        seed = kwargs.get("generation_seed")
        calls[seed].append({
            "phase": phase,
            "prompt_sha256": hashlib.sha256(result["prompt"].encode()).hexdigest(),
            "image_sha256": hashlib.sha256(kwargs["image_bytes"]).hexdigest(),
            "model": result.get("model"),
            "temperature": result.get("temperature"),
            "max_tokens": result.get("max_tokens"),
            "generation_seed": seed,
        })
        return result

    def factory(source, task_context=None):
        ctl = original_factory(source, task_context)
        holder["ctl"] = ctl
        original_reobserve = ctl._reobserve

        def reobserve(result, spec, evidence):
            holder["pending"] = {
                "result": result, "spec": spec, "evidence": evidence,
                "memory": copy.deepcopy(ctl.evidence_memory),
                "position": ctl._pos(), "step": len(ctl.trajectory),
                "pre_answer": ctl.answer_history[-1].answer,
                "task_context": task_context, "done": False,
            }
            try:
                return original_reobserve(result, spec, evidence)
            finally:
                holder.pop("pending", None)

        ctl._reobserve = reobserve
        return ctl

    def execute_intercept(action, params, *args, **kwargs):
        nonlocal phase
        pending = holder.get("pending")
        if action == "fly_relative" and pending and not pending["done"]:
            pending["done"] = True
            ctl = holder["ctl"]
            live_map = app.state.semantic_map
            app.state.semantic_map = clone_semantic_map(live_map)
            pre_state = {
                "position": pending["position"],
                "step": pending["step"], "qid": qid, "repeat": repeat,
            }
            pre_memory = {
                "tracks": pending["memory"]._tracks,
                "damage_tracks": pending["memory"]._damage_tracks,
                "observation_ids": pending["memory"]._observation_ids,
                "coverage": pending["memory"]._coverage_by_observation,
            }
            before_fallback, before_degraded = ctl.fallback_used, ctl.degraded_reason
            fixed_result = None
            fixed_answer = None
            fixed_evidence = None
            try:
                phase = "fixed"
                fixed_result = ctl._perceive()
                if fixed_result is None:
                    raise RuntimeError("fixed_view_perception_returned_none")
                fixed_evidence = build_evidence_from_perception(
                    fixed_result, pending["spec"], fixed_result.patch_id,
                    pending["task_context"],
                )
                fixed_evidence = pending["memory"].update(pending["spec"], fixed_evidence)
                next_step = pending["step"] + 1
                context = GenerationContext(
                    qid=qid, repeat=repeat, step=next_step,
                    call_role="candidate_answer",
                    seed=derive_generation_seed(42000, qid, repeat, next_step,
                                                "candidate_answer"),
                )
                fixed_answer = ctl._candidate_answer(
                    qid, pending["spec"], fixed_evidence, fixed_result, context,
                )
            finally:
                app.state.semantic_map = live_map
                ctl.fallback_used = before_fallback
                ctl.degraded_reason = before_degraded
                phase = "changed"
            forks.append({
                "qid": qid, "repeat": repeat,
                "recheck_index": sum(1 for fork in forks if fork["qid"] == qid),
                "pre_step_index": pending["step"],
                "next_step_index": pending["step"] + 1,
                "pre_answer": pending["pre_answer"],
                "pre_observation_id": pending["result"].patch_id,
                "fixed_observation_id": fixed_result.patch_id,
                "fixed_answer": fixed_answer.answer,
                "fixed_abstain": fixed_answer.abstain,
                "fixed_reason_code": fixed_answer.reason_code,
                "fixed_predicted_objects": fixed_evidence.to_dict().get("objects", []),
                "fixed_evidence": fixed_evidence.to_dict(),
                "same_pre_recheck_state_hash": digest(pre_state),
                "same_pre_recheck_memory_hash": digest(pre_memory),
                "generation_seed": context.seed,
                "prompt_template_sha256": hashlib.sha256(
                    AGENT_VQA_SYSTEM_PROMPT.encode()).hexdigest(),
                "planned_motion": dict(params or {}),
            })
        return original_execute(action, params, *args, **kwargs)

    app._make_agent_vqa_controller = factory
    app.execute_action = execute_intercept
    VLMAnalyzer.answer_image_question = answer_audit
    start = time.time()
    try:
        report = app.run_agent_vqa_episode_headless(
            item["question"], item["start"], item=item, source="bench")
    finally:
        app._make_agent_vqa_controller = original_factory
        app.execute_action = original_execute
        VLMAnalyzer.answer_image_question = original_answer
    trajectory = report.get("trajectory") or []
    for fork in forks:
        index = fork["next_step_index"]
        changed = trajectory[index] if index < len(trajectory) else {}
        seed = fork["generation_seed"]
        matching_calls = calls.get(seed, [])
        fixed_calls = [call for call in matching_calls if call["phase"] == "fixed"]
        changed_calls = [call for call in matching_calls if call["phase"] == "changed"]
        pre_seed = derive_generation_seed(42000, qid, repeat,
                                          fork["pre_step_index"], "candidate_answer")
        pre_calls = [call for call in calls.get(pre_seed, [])
                     if call["phase"] == "changed"]
        fork.update({
            "gt_answer": item["answer"], "question_type": item["question_type"],
            "event": item.get("disaster"), "roi_id": digest(
                [item.get("tile_id"), (item.get("roi") or {}).get("bounds")]),
            "changed_observation_id": changed.get("observation_id", ""),
            "changed_answer": changed.get("candidate_answer", ""),
            "changed_decision": changed.get("decision", ""),
            "changed_predicted_objects": (changed.get("evidence") or {}).get("objects", []),
            "fixed_correct": bool(fork["fixed_answer"] and not fork["fixed_abstain"]
                                  and fork["fixed_answer"] == item["answer"]),
            "changed_correct": bool(changed.get("candidate_answer")
                                    and changed.get("candidate_answer") == item["answer"]),
            "fixed_call": fixed_calls[-1] if fixed_calls else None,
            "changed_call": changed_calls[-1] if changed_calls else None,
            "pre_call": pre_calls[-1] if pre_calls else None,
            "same_fixed_image_as_pre": bool(pre_calls and fixed_calls)
                                       and pre_calls[-1]["image_sha256"]
                                       == fixed_calls[-1]["image_sha256"],
            "matched_seed": len(fixed_calls) == 1 and len(changed_calls) == 1,
            "matched_template": bool(fixed_calls and changed_calls)
                                and fixed_calls[-1]["model"] == changed_calls[-1]["model"]
                                and fixed_calls[-1]["temperature"] == changed_calls[-1]["temperature"]
                                and fixed_calls[-1]["max_tokens"] == changed_calls[-1]["max_tokens"],
        })
    return {"qid": qid, "repeat": repeat, "ok": report.get("ok"),
            "error": report.get("error", ""),
            "answer": report.get("answer"),
            "trajectory": trajectory,
            "degraded_reason": report.get("degraded_reason", ""),
            "wall_s": round(time.time() - start, 2),
            "n_reobservations": sum(t.get("decision") == "reobserve" for t in trajectory),
            "forks": forks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference-runs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if file_digest(args.testset) != protocol["testset_sha256"]:
        raise ValueError("frozen final testset hash mismatch")
    if source_fingerprint() != protocol["source_fingerprint"]:
        raise ValueError("frozen online source fingerprint mismatch")
    frozen_path = Path(protocol["frozen_manifest"])
    frozen_path = frozen_path if frozen_path.is_absolute() else ROOT / frozen_path
    if file_digest(frozen_path) != protocol["frozen_manifest_sha256"]:
        raise ValueError("frozen policy manifest hash mismatch")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    items = json.loads(args.testset.read_text(encoding="utf-8"))["items"]
    refs = reference_rows(args.reference_runs)
    selected = [item for item in items[args.shard::4]
                if refs[(args.repeat, item["id"])].get("n_reobservations", 0) > 0]
    if args.limit:
        selected = selected[:args.limit]
    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema": "revision-matched-view/1.0", "variant": "live_strict_reperception",
        "testset": str(args.testset), "testset_sha256": file_digest(args.testset),
        "protocol": str(args.protocol), "protocol_sha256": file_digest(args.protocol),
        "source_fingerprint": source_fingerprint(),
        "runner_sha256": file_digest(Path(__file__)),
        "frozen_manifest": str(frozen_path), "frozen_manifest_sha256": file_digest(frozen_path),
        "repeat": args.repeat, "shard": args.shard,
        "n_selected_reference_episodes": len(selected),
        "reference_rechecks": sum(refs[(args.repeat, item["id"])]["n_reobservations"]
                                  for item in selected),
        "answer_mode": "hybrid", "generation_base_seed": 42000,
        "generation_parameters": {"temperature": 0.1, "max_tokens": 300,
                                  "top_p": 0.9, "repetition_penalty": 1.1},
        "sampling_note": "Only reference T1 episodes with executed rechecks are rerun; all live rechecks are forked.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                  indent=2) + "\n", encoding="utf-8")
    import app  # noqa: E402
    counts = Counter()
    with (args.out_dir / "episodes.jsonl").open("w", encoding="utf-8") as output:
        for i, item in enumerate(selected, 1):
            row = run_one(app, item, args.repeat, frozen)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            counts["episodes"] += 1
            counts["forks"] += len(row["forks"])
            counts["errors"] += not bool(row["ok"])
            counts["invalid_pairs"] += sum(
                not (fork["matched_seed"] and fork["matched_template"]
                     and fork["same_fixed_image_as_pre"] and fork["changed_observation_id"])
                for fork in row["forks"]
            )
            print(f"[{i}/{len(selected)}] {item['id']} forks={len(row['forks'])} "
                  f"wall={row['wall_s']}s", flush=True)
    print(json.dumps(dict(counts), ensure_ascii=False), flush=True)
    expected = manifest["reference_rechecks"]
    return 0 if (not counts["errors"] and not counts["invalid_pairs"]
                 and counts["forks"] == expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
