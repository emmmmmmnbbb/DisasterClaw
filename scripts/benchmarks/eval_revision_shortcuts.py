#!/usr/bin/env python3
"""Final-set non-image shortcut controls with a frozen input whitelist."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from agent_vqa import (BEARING_CHOICES, COUNT_CHOICES, DAMAGE_CHOICES,
                       PRESENCE_CHOICES, derive_generation_seed, parse_question,
                       task_context_from_item)  # noqa: E402
from llm_client import get_client  # noqa: E402

CHOICES = {"presence": PRESENCE_CHOICES, "damage": DAMAGE_CHOICES,
           "count": COUNT_CHOICES, "spatial": BEARING_CHOICES}
SYSTEM = (
    "你正在执行灾害巡检问答的无图像捷径对照实验。你没有图像，也没有探测器输出。"
    "只能使用用户消息列出的字段，不得假称看到了建筑或影像。"
    "只输出单个 JSON 对象，字段 answer（必须逐字选自候选答案）、"
    "abstain（布尔值）；若信息不足，abstain=true，仍填写一个候选答案。"
    "不要输出 Markdown、思维链或额外说明。"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def winner(counts: Counter[str]) -> str:
    if not counts:
        return ""
    top = max(counts.values())
    return sorted(answer for answer, n in counts.items() if n == top)[0]


def majority_fit(items: list[dict]) -> dict:
    by_type = defaultdict(Counter)
    overall = Counter()
    for item in items:
        by_type[item["question_type"]][item["answer"]] += 1
        overall[item["answer"]] += 1
    return {"by_type": {key: {"answer": winner(value), "counts": dict(value)}
                        for key, value in sorted(by_type.items())},
            "overall": {"answer": winner(overall), "counts": dict(overall)}}


def allowed_payload(item: dict, arm: str) -> dict:
    spec = parse_question(item["question"])
    if spec.question_type != item["question_type"]:
        raise ValueError(f"question type mismatch: {item['id']}")
    payload = {"question": item["question"],
               "choices": list(CHOICES[spec.question_type]),
               "question_type": spec.question_type}
    if arm == "S2_METADATA":
        context = task_context_from_item(item, spec)
        payload.update({
            "roi_tile_id": context.roi_tile_id,
            "roi_bounds": context.roi_bounds,
            "initial_vehicle_state": {key: item["start"][key]
                                      for key in ("lat", "lon", "alt")},
        })
        if spec.question_type == "damage":
            payload["marked_target"] = {
                "ref_id": context.target_ref_id,
                "lat": context.target_lat, "lon": context.target_lon,
            }
    return payload


def parse_answer(raw: str, choices: list[str]) -> tuple[str, bool, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        try:
            obj = json.loads(text[left:right + 1]) if left >= 0 and right > left else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return "", True, "invalid_json"
    answer = str(obj.get("answer", "")).strip()
    if answer not in choices:
        return "", True, "out_of_vocabulary"
    if not isinstance(obj.get("abstain"), bool):
        return "", True, "invalid_abstain"
    return answer, obj["abstain"], "valid"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--dev-testset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha(args.testset) != protocol["testset_sha256"]:
        raise ValueError("frozen final testset hash mismatch")
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    final = json.loads(args.testset.read_text(encoding="utf-8"))["items"]
    dev = json.loads(args.dev_testset.read_text(encoding="utf-8"))["items"]
    if {item["id"] for item in final} & {item["id"] for item in dev}:
        raise ValueError("development and final question IDs overlap")
    final_majority = majority_fit(final)
    dev_majority = majority_fit(dev)
    selected = final[args.shard::4]
    if args.limit:
        selected = selected[:args.limit]
    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema": "revision-shortcut-controls/1.0",
        "testset": str(args.testset), "testset_sha256": sha(args.testset),
        "dev_testset": str(args.dev_testset), "dev_testset_sha256": sha(args.dev_testset),
        "protocol": str(args.protocol), "protocol_sha256": sha(args.protocol),
        "runner_sha256": sha(Path(__file__)), "shard": args.shard,
        "n_questions": len(selected), "generation_repeats": [0, 1, 2],
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "temperature": 0.1,
        "max_tokens": 300, "top_p": 0.9, "repetition_penalty": 1.1,
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "input_boundary": {
            "S1_LANGUAGE": ["question", "choices", "question_type"],
            "S2_METADATA": ["S1 fields", "roi_tile_id", "roi_bounds",
                            "initial_vehicle_state", "damage_only_marked_target_ref_and_position"],
            "excluded": ["image", "detector_prediction", "gt_label", "target_subtype",
                         "spatial_true_nearest_target", "scoring_fields"],
        },
        "S0_final_majority_role": "descriptive_oracle_not_deployable",
        "S0_dev_majority_role": "development_fitted_control",
        "S0_final_majority": final_majority,
        "S0_dev_majority": dev_majority,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    client = get_client(module="vlm")
    counts = Counter()
    with (args.out_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for i, item in enumerate(selected, 1):
            qtype = item["question_type"]
            for repeat in range(3):
                seed = derive_generation_seed(42000, item["id"], repeat, 0,
                                              "candidate_answer")
                for arm in ("S1_LANGUAGE", "S2_METADATA"):
                    payload = allowed_payload(item, arm)
                    user_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    start = time.time()
                    try:
                        raw = client.chat(
                            [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user_prompt}],
                            temperature=0.1, max_tokens=300, seed=seed)
                        answer, abstain, status = parse_answer(raw, payload["choices"])
                        error = ""
                    except Exception as exc:
                        raw, answer, abstain, status = "", "", True, "call_error"
                        error = f"{type(exc).__name__}: {exc}"
                    row = {
                        "qid": item["id"], "repeat": repeat, "arm": arm,
                        "question_type": qtype, "event": item.get("disaster"),
                        "roi_id": digest([item.get("tile_id"),
                                          (item.get("roi") or {}).get("bounds")]),
                        "gt_answer": item["answer"], "answer": answer,
                        "abstain": abstain, "correct": bool(answer == item["answer"] and not abstain),
                        "status": status, "error": error, "raw_output": raw,
                        "allowed_input": payload,
                        "prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
                        "generation_seed": seed, "wall_s": round(time.time() - start, 2),
                        "S0_final_majority_answer": final_majority["by_type"][qtype]["answer"],
                        "S0_dev_majority_answer": dev_majority["by_type"][qtype]["answer"],
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    counts["calls"] += 1
                    counts["call_errors"] += bool(error)
            print(f"[{i}/{len(selected)}] {item['id']} calls={counts['calls']}", flush=True)
    print(json.dumps(dict(counts), ensure_ascii=False), flush=True)
    return 0 if not counts["call_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
