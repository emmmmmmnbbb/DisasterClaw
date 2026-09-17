"""Hermetic guards for the frozen P6 launch and repeat-aware statistics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from preflight_changeos_p6_final import validate
from bench_agent_vqa import episode_seed
from report_changeos_p6_final import bootstrap_roi, generation_seed, summarize, validate_run

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "cja_en/review/changeos_p6_final_protocol_20260915.json"


def test_live_final_preflight_is_read_only(tmp_path):
    out = tmp_path / "not_created"
    report = validate(PROTOCOL, repeat=2, shard=3, out_dir=out)
    assert report["status"] == "PASS"
    assert report["n_expected_rows"] == 280
    assert not out.exists()
    with pytest.raises(ValueError, match="not frozen"):
        validate(PROTOCOL, repeat=3, shard=3, out_dir=out)
    out.mkdir()
    with pytest.raises(FileExistsError, match="no implicit resume"):
        validate(PROTOCOL, repeat=2, shard=3, out_dir=out)


def test_preflight_rejects_changed_frozen_analysis_code(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["analysis_code_sha256"]["scripts/benchmarks/report_changeos_p6_final.py"] = "0" * 64
    altered = tmp_path / "altered_protocol.json"
    altered.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen launch/analysis code changed"):
        validate(altered, repeat=0, shard=0, out_dir=tmp_path / "new_run")


def test_roi_bootstrap_preserves_repeat_question_count():
    result = bootstrap_roi({"roi-a": [1.0, 0.0, 0.0], "roi-b": [0.0, 0.0, 0.0]},
                           n_boot=500, seed=42)
    assert result["n_roi"] == 2
    assert result["n_paired_repeat_questions"] == 6
    assert result["mean_difference"] == pytest.approx(1 / 6)


def test_summary_pairs_within_repeat_not_qid_only():
    configs = ["A0_HOLD", "A1_RANDOM", "A2_ALWAYS", "A3U_RAW_ENTROPY", "T1_TASK"]
    items = {
        f"q{i}": {"tile_id": f"tile-{i // 2}", "roi": {"bounds": {"west": i // 2}},
                  "start": {"lat": 0.0, "lon": 0.0, "alt": 30.0},
                  "question_type": "presence", "disaster": "event"}
        for i in range(4)
    }
    protocol = {"configs": configs, "generation_repeats": [0, 1, 2], "events": ["event"]}
    rows = []
    for repeat in range(3):
        for config in configs:
            for qid in items:
                correct = config == "T1_TASK" and repeat == 0 and qid == "q0"
                rows.append({"_p6_repeat": repeat, "config": config, "qid": qid,
                             "correct": correct, "question_type": "presence",
                             "disaster": "event", "n_reobservations": 0, "wall_s": 1.0})
    report = summarize(rows, protocol, items, n_boot=500)
    test = report["primary_paired_contrasts"]["T1_TASK_vs_A0_HOLD"]
    assert report["n_online_rows"] == 60
    assert test["difference_by_repeat"] == {"0": 0.25, "1": 0.0, "2": 0.0}
    assert test["roi_cluster_bootstrap"]["n_paired_repeat_questions"] == 12
    assert test["roi_cluster_bootstrap"]["mean_difference"] == pytest.approx(1 / 12)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate across P6 shards"):
        summarize(rows, protocol, items, n_boot=500)


def test_strict_shard_audit_rejects_generation_mismatch(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    task = json.loads((ROOT / protocol["testset"]).read_text(encoding="utf-8"))
    items = {item["id"]: item for item in task["items"]}
    qids = list(items)[0::4]
    run = tmp_path / "run"
    run.mkdir()
    args = {
        "generation_repeat": 0, "_shard_i": 0, "_shard_n": 4,
        "configs": ",".join(protocol["configs"]),
        "testset": protocol["testset"], "review_report": protocol["review_report"],
        "frozen_manifest": protocol["frozen_manifest"],
        "seed": 42, "generation_seed": 42000, "resume": False,
        "allow_label_mismatch": False, "limit": 0, "split": "", "qtype": "",
    }
    result = {"args": args, "valid_for_analysis": True, "n_execution_errors": 0,
              "n_items": len(qids)}
    manifest = {
        "env": {
            "source_fingerprint": protocol["source_fingerprint"],
            "detector_backend": "changeos", "damage_label_mode": "binary",
            "vlm_provider": "qwen_vl_local", "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "vlm_top_p": 0.9, "vlm_repetition_penalty": 1.1,
            "agent_vqa_confidence_threshold": "0.5",
            "perception_tool": {"weights_sha256": protocol["changeos_weights_sha256"]},
        },
        "testset_sha256_16": protocol["testset_sha256"],
        "review_report_sha256_16": protocol["review_report_sha256"],
        "frozen_manifest_sha256_16": protocol["frozen_manifest_sha256"],
        "vlm_system_prompt_sha256": protocol["vlm_system_prompt_sha256"],
        "generation_repeat": 0, "generation_base_seed": 42000,
        "agent_vqa_answer_modes": {c: "hybrid" for c in protocol["configs"]},
    }
    rows = []
    for config in protocol["configs"]:
        for qid in qids:
            item = items[qid]
            seed = generation_seed(42000, qid, 0, 0, "candidate_answer")
            rows.append({
                "config": config, "qid": qid, "question_type": item["question_type"],
                "disaster": item["disaster"], "tile_id": item["tile_id"],
                "answer_mode": "hybrid", "action_seed": episode_seed(42, config, qid),
                "ok": True, "generation_seeds": [seed],
                "trajectory": [{"generation_seed": seed, "generation_repeat": 0,
                                "generation_call_role": "candidate_answer"}],
            })
    (run / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    episodes = run / "episodes.jsonl"
    episodes.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    audit, parsed = validate_run(run, protocol, items)
    assert audit["n_rows"] == 280 and len(parsed) == 280
    rows[0]["trajectory"][0]["generation_seed"] += 1
    episodes.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generation call key mismatch"):
        validate_run(run, protocol, items)
