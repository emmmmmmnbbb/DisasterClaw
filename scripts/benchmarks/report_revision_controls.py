#!/usr/bin/env python3
"""Audit and summarize the two revision controls; never edit manuscript files."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def rows(paths: list[Path], name: str) -> list[dict]:
    out = []
    for path in paths:
        with (path / name).open(encoding="utf-8") as source:
            out.extend(json.loads(line) for line in source if line.strip())
    return out


def reference_rows(root: Path) -> dict[tuple[str, int, str], dict]:
    out = {}
    for repeat in range(3):
        for shard in range(4):
            path = root / (f"p6_final_three_event_v1_20260915_repeat{repeat}"
                           f"_shard{shard}of4/episodes.jsonl")
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                config = row.get("config")
                if config in {"A0_HOLD", "T1_TASK"}:
                    key = (config, repeat, row["qid"])
                    if key in out:
                        raise ValueError(f"duplicate reference row: {key}")
                    out[key] = {
                        "correct": bool(row.get("correct")),
                        "question_type": row.get("question_type"),
                        "recheck_steps": [i for i, step in enumerate(row.get("trajectory") or [])
                                          if step.get("decision") == "reobserve"],
                    }
    if len(out) != 960:
        raise ValueError(f"reference A0/T1 row count is {len(out)}, not 960")
    return out


def aggregate(values: list[dict], arm: str) -> dict:
    subset = [row for row in values if row.get("arm") == arm]
    by_type = {}
    for qtype in sorted({row["question_type"] for row in subset}):
        group = [row for row in subset if row["question_type"] == qtype]
        by_type[qtype] = {"n": len(group),
                          "accuracy": sum(bool(row["correct"]) for row in group) / len(group)}
    return {"n": len(subset),
            "accuracy": sum(bool(row["correct"]) for row in subset) / len(subset)
            if subset else None, "by_type": by_type}


def roi_bootstrap(forks: list[dict], draws: int = 10000, seed: int = 20260920) -> list[float]:
    clusters = defaultdict(list)
    for row in forks:
        clusters[row["roi_id"]].append(row)
    ids = sorted(clusters)
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = [row for roi in rng.choices(ids, k=len(ids)) for row in clusters[roi]]
        deltas.append(sum(int(row["changed_correct"]) - int(row["fixed_correct"])
                          for row in sample) / len(sample))
    deltas.sort()
    return [deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]]


def paired_gap(reference: dict, predictions: list[dict], config: str,
               arm: str, draws: int = 5000) -> dict:
    selected = [row for row in predictions if row["arm"] == arm]
    joined = [(row, reference[(config, row["repeat"], row["qid"])])
              for row in selected]
    clusters = defaultdict(list)
    for row, ref in joined:
        clusters[row["roi_id"]].append(int(ref["correct"]) - int(row["correct"]))
    cluster_ids = sorted(clusters)
    rng = random.Random(20260920)
    bootstrap = []
    for _ in range(draws):
        sample = [value for roi in rng.choices(cluster_ids, k=len(cluster_ids))
                  for value in clusters[roi]]
        bootstrap.append(sum(sample) / len(sample))
    bootstrap.sort()
    return {
        "n": len(joined),
        "n_roi_clusters": len(cluster_ids),
        "visual_accuracy": sum(ref["correct"] for _, ref in joined) / len(joined),
        "shortcut_accuracy": sum(row["correct"] for row, _ in joined) / len(joined),
        "paired_accuracy_gap": sum(int(ref["correct"]) - int(row["correct"])
                                   for row, ref in joined) / len(joined),
        "roi_cluster_95pct_ci": [bootstrap[int(0.025 * draws)],
                                 bootstrap[int(0.975 * draws) - 1]],
    }


def matched_summary(episode_rows: list[dict], expected: int | None) -> dict:
    forks = [fork for episode in episode_rows for fork in episode["forks"]]
    valid = [row for row in forks if row["changed_observation_id"]
             and row["matched_seed"] and row["matched_template"]
             and row["same_fixed_image_as_pre"]]
    counts = Counter((bool(row["fixed_correct"]), bool(row["changed_correct"]))
                     for row in valid)
    by_type = {}
    for qtype in sorted({row["question_type"] for row in valid}):
        group = [row for row in valid if row["question_type"] == qtype]
        by_type[qtype] = {
            "n": len(group),
            "fixed_accuracy": sum(row["fixed_correct"] for row in group) / len(group),
            "changed_accuracy": sum(row["changed_correct"] for row in group) / len(group),
            "paired_delta": sum(int(row["changed_correct"]) - int(row["fixed_correct"])
                                for row in group) / len(group),
            "help": sum(not row["fixed_correct"] and row["changed_correct"] for row in group),
            "harm": sum(row["fixed_correct"] and not row["changed_correct"] for row in group),
        }
    return {
        "n_episode_rows": len(episode_rows), "n_forks": len(forks),
        "n_valid_pairs": len(valid), "expected_reference_rechecks": expected,
        "n_roi_clusters": len({row["roi_id"] for row in valid}),
        "n_missing_or_unmatched": len(forks) - len(valid),
        "n_execution_errors": sum(not row["ok"] for row in episode_rows),
        "fixed_accuracy": sum(row["fixed_correct"] for row in valid) / len(valid) if valid else None,
        "changed_accuracy": sum(row["changed_correct"] for row in valid) / len(valid) if valid else None,
        "paired_delta": sum(int(row["changed_correct"]) - int(row["fixed_correct"])
                            for row in valid) / len(valid) if valid else None,
        "roi_cluster_95pct_ci": roi_bootstrap(valid) if valid else None,
        "four_cells": {
            "wrong_wrong": counts[(False, False)], "help_wrong_correct": counts[(False, True)],
            "harm_correct_wrong": counts[(True, False)], "correct_correct": counts[(True, True)],
        },
        "answer_disagreement": sum(row["fixed_answer"] != row["changed_answer"]
                                   for row in valid) / len(valid) if valid else None,
        "by_question_type": by_type,
    }


def shortcut_summary(predictions: list[dict], manifests: list[dict]) -> dict:
    keyed = {(row["qid"], row["repeat"], row["arm"]) for row in predictions}
    if len(keyed) != len(predictions):
        raise ValueError("duplicate shortcut predictions")
    final_model = manifests[0]["S0_final_majority"]
    dev_model = manifests[0]["S0_dev_majority"]
    unique = {}
    for row in predictions:
        unique[row["qid"]] = row
    s0_rows = []
    for row in unique.values():
        qtype = row["question_type"]
        gt = row["gt_answer"]
        s0_rows.extend([
            {**row, "arm": "S0_FINAL_ORACLE",
             "correct": final_model["by_type"][qtype]["answer"] == gt},
            {**row, "arm": "S0_DEV_FITTED",
             "correct": dev_model["by_type"][qtype]["answer"] == gt},
            {**row, "arm": "S0_FINAL_OVERALL_ORACLE",
             "correct": final_model["overall"]["answer"] == gt},
            {**row, "arm": "S0_DEV_OVERALL_FITTED",
             "correct": dev_model["overall"]["answer"] == gt},
        ])
    all_rows = predictions + s0_rows
    return {
        "n_predictions": len(predictions), "n_unique_questions": len(unique),
        "call_errors": sum(row["status"] == "call_error" for row in predictions),
        "invalid_outputs": sum(row["status"] not in ("valid", "call_error")
                               for row in predictions),
        "arms": {arm: aggregate(all_rows, arm) for arm in
                 ("S0_FINAL_ORACLE", "S0_DEV_FITTED",
                  "S0_FINAL_OVERALL_ORACLE", "S0_DEV_OVERALL_FITTED",
                  "S1_LANGUAGE", "S2_METADATA")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--shortcut-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-runs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    matched_manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                         for path in args.matched_dirs]
    shortcut_manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                          for path in args.shortcut_dirs]
    if len({m["testset_sha256"] for m in matched_manifests + shortcut_manifests}) != 1:
        raise ValueError("testset hashes differ")
    if len({(m["repeat"], m["shard"]) for m in matched_manifests}) != len(matched_manifests):
        raise ValueError("duplicate matched shards")
    if len({m["shard"] for m in shortcut_manifests}) != len(shortcut_manifests):
        raise ValueError("duplicate shortcut shards")
    if (len(matched_manifests) != 12
            or {(m["repeat"], m["shard"]) for m in matched_manifests}
            != {(repeat, shard) for repeat in range(3) for shard in range(4)}):
        raise ValueError("matched-view report requires all 3 repeats x 4 shards")
    if len(shortcut_manifests) != 4 or {m["shard"] for m in shortcut_manifests} != set(range(4)):
        raise ValueError("shortcut report requires all 4 shards")
    if len({m["runner_sha256"] for m in matched_manifests}) != 1:
        raise ValueError("matched-view runner changed across shards")
    if len({m["runner_sha256"] for m in shortcut_manifests}) != 1:
        raise ValueError("shortcut runner changed across shards")
    episodes = rows(args.matched_dirs, "episodes.jsonl")
    predictions = rows(args.shortcut_dirs, "predictions.jsonl")
    reference = reference_rows(args.reference_runs)
    expected = sum(m["reference_rechecks"] for m in matched_manifests)
    if expected != 161:
        raise ValueError(f"reference T1 recheck count changed: {expected} != 161")
    report = {
        "schema": "revision-controls-report/1.0",
        "testset_sha256": matched_manifests[0]["testset_sha256"],
        "matched_shards": len(matched_manifests),
        "shortcut_shards": len(shortcut_manifests),
        "matched": matched_summary(episodes, expected),
        "shortcuts": shortcut_summary(predictions, shortcut_manifests),
        "paired_shortcut_gaps": {
            f"{config}_minus_{arm}": paired_gap(reference, predictions, config, arm)
            for config in ("A0_HOLD", "T1_TASK")
            for arm in ("S1_LANGUAGE", "S2_METADATA")
        },
        "reference_recheck_schedule_audit": {
            "n_episodes": len(episodes),
            "n_same_step_indices": sum(
                reference[("T1_TASK", row["repeat"], row["qid"])]["recheck_steps"]
                == [i for i, step in enumerate(row.get("trajectory") or [])
                    if step.get("decision") == "reobserve"]
                for row in episodes),
        },
    }
    if (report["matched"]["n_forks"] != expected
            or report["matched"]["n_valid_pairs"] != expected
            or report["matched"]["n_execution_errors"]
            or report["shortcuts"]["n_predictions"] != 960
            or report["shortcuts"]["n_unique_questions"] != 160
            or report["shortcuts"]["call_errors"]):
        raise ValueError("incomplete or invalid experiment outputs; refusing final report")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m, s = report["matched"], report["shortcuts"]
    lines = [
        "# 返修补实验：固定视角配对与无图像捷径对照", "",
        f"- Matched valid pairs: {m['n_valid_pairs']} / {m['n_forks']} "
        f"(reference rechecks: {m['expected_reference_rechecks']})",
        f"- Changed vs fixed accuracy: {m['changed_accuracy']:.4f} vs {m['fixed_accuracy']:.4f}",
        f"- Paired delta: {m['paired_delta']:.4f}; ROI cluster 95% CI: "
        f"{m['roi_cluster_95pct_ci']}",
        f"- Help / harm: {m['four_cells']['help_wrong_correct']} / "
        f"{m['four_cells']['harm_correct_wrong']}", "",
        f"- Original T1 recheck-step schedule reproduced: "
        f"{report['reference_recheck_schedule_audit']['n_same_step_indices']} / "
        f"{report['reference_recheck_schedule_audit']['n_episodes']} episodes", "",
        "| control | n | accuracy |", "| --- | ---: | ---: |",
    ]
    for arm, record in s["arms"].items():
        lines.append(f"| {arm} | {record['n']} | {record['accuracy']:.4f} |")
    lines.extend(["", "| paired gap | n | visual - shortcut | ROI 95% CI |",
                  "| --- | ---: | ---: | --- |"])
    for name, value in report["paired_shortcut_gaps"].items():
        lines.append(f"| {name} | {value['n']} | {value['paired_accuracy_gap']:.4f} | "
                     f"{value['roi_cluster_95pct_ci']} |")
    lines.extend(["", "S0_FINAL_ORACLE / S0_FINAL_OVERALL_ORACLE 使用 final 标签选择多数类，"
                  "仅供描述，不是部署基线。",
                  "S0_DEV_FITTED / S0_DEV_OVERALL_FITTED 使用独立开发集选择多数类。", "",
                  "S1/S2 使用同一 Qwen 模型、答案词表、生成参数与精确匹配评分；"
                  "无图像提示词采用 answer/abstain JSON schema，不运行主系统的感知或 hybrid 规则覆盖。",
                  "配对实验为逐次 recheck 的候选答案比较，不等同于从分叉点独立完成整条 episode。",
                  "",
                  "本报告不修改论文正文或论文图片。"])
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    methods = [
        "# Supplementary experiment protocol and boundaries", "",
        "## Fixed-view matched-call", "",
        "- Unit: each T1_TASK executed recheck transition in the frozen final set (3 repeats).",
        "- Branch point: immediately before the live fly_relative recheck action.",
        "- Fixed arm: clone the pre-recheck semantic map and evidence memory, remain at the old pose, "
        "rerun the same perception pipeline on the old view, and call the ordinary hybrid answer path.",
        "- Changed arm: execute the original T1 action, perceive the new view, and call the same answer path.",
        "- Both arms use the same qid/repeat/next-step generation seed, Qwen model and generation settings. "
        "The system/template path is shared; full prompts intentionally differ because the perception evidence differs.",
        "- The fixed answer is diagnostic only and never fed back into T1 control. Each record stores "
        "pre-state/memory hashes, old/fixed/changed observation IDs, rendered-image and prompt hashes, "
        "and predicted objects.",
        "- The strict validity gate requires the fixed-arm marked image to be byte-identical to the "
        "pre-recheck marked image, plus exactly one fixed and one changed model call at a shared seed.",
        "- Candidate-answer correctness is scored at the next transition; it is not terminal episode accuracy. "
        "Perception nondeterminism may remain, even when old-view pixels are identical.", "",
        "## Shortcut controls", "",
        "- S0 final task-majority uses final labels solely as a descriptive oracle; S0 development-fitted "
        "uses a question/ROI-disjoint development set.",
        "- S1 permits question text, answer choices, and task type only.",
        "- S2 additionally permits runtime-visible ROI bounds/tile, initial vehicle pose, and for damage "
        "only the marked target reference and coordinate; it excludes images, detector outputs, GT, "
        "target subtype, and spatial true-nearest target coordinates.",
        "- S1/S2 use the same Qwen2.5-VL-7B-Instruct checkpoint, answer vocabulary, exact-match "
        "scoring, temperature, top-p, repetition penalty, and token limit as the main answer model, "
        "with the same qid/repeat/step-0 seed schedule. A modality-appropriate text-only JSON prompt "
        "replaces the main visual schema; the hybrid rule path is not used.",
        "- Every final question is evaluated in three aligned repeats. Invalid JSON or out-of-vocabulary "
        "answers count as incorrect; call errors invalidate the report.", "",
        "## Statistics", "",
        "- Recheck transitions are paired within decision point. Report the 2×2 correctness cells, "
        "help/harm, disagreement, task-wise differences, and an ROI-cluster bootstrap interval.",
        "- Shortcut results report overall and task-wise accuracy; A0/T1 gaps are paired by "
        "question and repeat, with ROI-cluster bootstrap intervals.",
        "- All 12 matched-view and 4 shortcut shards, 161 valid forks, and 960 valid shortcut "
        "predictions are required for a final report.",
    ]
    (args.out_dir / "protocol_notes.md").write_text("\n".join(methods) + "\n", encoding="utf-8")
    print(json.dumps({"matched": m, "shortcuts": s}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
