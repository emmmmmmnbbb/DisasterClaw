"""Agent-VQA benchmark configuration and offline-scoring regression tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json


HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("bench_agent_vqa")
reporter = _load("report_agent_vqa")


def _fake_app():
    return SimpleNamespace(
        VLN_ENTROPY_TABLE="default-table.json",
        VLN_CONFORMAL_QHAT=0.9,
        VLN_CONFORMAL_ALPHA=0.1,
    )


def test_policy_configs_have_distinct_effective_identity() -> None:
    identities = {}
    for name in ("A1_RANDOM", "A2_ALWAYS", "A3_ENTROPY", "A4_CONFORMAL", "A5_EXPECTED"):
        app = _fake_app()
        bench.apply_config(app, bench.CONFIGS[name])
        identities[name] = bench.effective_config(app)

    assert identities["A1_RANDOM"]["trigger_mode"] == "random"
    assert identities["A2_ALWAYS"]["trigger_mode"] == "fixed"
    assert identities["A3_ENTROPY"]["uncertainty_mode"] == "entropy"
    assert identities["A4_CONFORMAL"]["trigger_mode"] == "conformal"
    assert identities["A5_EXPECTED"]["trigger_mode"] == "info_gain"
    assert len({(v["trigger_mode"], v["uncertainty_mode"]) for v in identities.values()}) == 5
    assert all(v["oracle"] is False for v in identities.values())


def test_online_oracle_config_is_rejected() -> None:
    try:
        bench.apply_config(_fake_app(), bench.CONFIGS["O_REF"])
    except ValueError as exc:
        assert "离线" in str(exc) or "offline" in str(exc).lower()
    else:
        raise AssertionError("O_REF must not execute as an ordinary online policy")


def test_reobserve_pair_scoring_uses_before_and_after() -> None:
    item = {"id": "q1", "answer": "是", "question_type": "presence"}
    run = {
        "ok": True,
        "answer": {"answer": "是", "abstain": False, "decision": "answer", "confidence": 0.8},
        "trajectory": [
            {"candidate_answer": "否", "decision": "reobserve"},
            {"candidate_answer": "是", "decision": "answer"},
        ],
    }
    row = bench.score_episode(run, item)
    assert row["answer_corrected"] is True
    assert row["answer_harmed"] is False
    assert row["reobserve_pairs"][0]["before"] == "否"
    assert row["reobserve_pairs"][0]["after"] == "是"
    assert row["trajectory"] == run["trajectory"]


def test_score_episode_persists_trajectory_and_skip_audit() -> None:
    item = {"id": "q_audit", "answer": "是", "question_type": "presence"}
    run = {
        "ok": True, "n_steps": 1,
        "answer": {
            "answer": "是", "abstain": False, "decision": "answer",
            "confidence": 0.9,
            "evidence": {"source": "detector", "target_subtype": "destroyed"},
        },
        "trajectory": [
            {"candidate_answer": "是", "decision": "answer",
             "reobserve_kind": "skip", "reobserve_reason": "把握足够",
             "uncertainty": 0.2, "evidence": {"target_label": "完全损毁建筑"}},
        ],
    }
    row = bench.score_episode(run, item)
    assert row["trajectory"][0]["reobserve_kind"] == "skip"
    assert row["n_reobserve_skips"] == 1
    assert row["n_reobservations"] == 0
    assert row["evidence"]["source"] == "detector"
    agg = bench.aggregate([row])
    assert agg["n_reobserve_skips"] == 1
    assert agg["n_reobservations"] == 0


def test_schema_diagnostics_are_preserved_for_invalid_outputs() -> None:
    item = {"id": "q_bad", "answer": "是", "question_type": "presence"}
    run = {
        "ok": True,
        "answer": {
            "answer": "", "abstain": True, "decision": "abstain",
            "reason_code": "invalid_output", "confidence": 0.0,
            "schema_errors": ["missing_evidence_source"],
            "raw_model_output": '{"answer":"是"}',
        },
        "trajectory": [],
    }
    row = bench.score_episode(run, item)
    assert row["schema_errors"] == ["missing_evidence_source"]
    assert row["raw_model_output"] == '{"answer":"是"}'
    agg = bench.aggregate([row])
    assert agg["invalid_schema_errors"] == {"missing_evidence_source": 1}
    assert agg["failure_taxonomy"] == {"invalid_output": 1}


def test_schema_error_aggregation_removes_dynamic_payload() -> None:
    rows = [
        {"ok": True, "correct": False, "abstain": True,
         "schema_errors": ["invalid_evidence_source:自然语言一"]},
        {"ok": True, "correct": False, "abstain": True,
         "schema_errors": ["invalid_evidence_source:自然语言二"]},
    ]
    assert bench.aggregate(rows)["invalid_schema_errors"] == {
        "invalid_evidence_source": 2,
    }


def test_report_aggregate_separates_invalid_output_from_abstention() -> None:
    rows = [
        {"ok": True, "correct": False, "abstain": True,
         "reason_code": "invalid_output", "n_reobservations": 0},
        {"ok": True, "correct": False, "abstain": True,
         "reason_code": "budget_exhausted", "n_reobservations": 0},
    ]
    agg = reporter.aggregate(rows)
    assert agg["failure_taxonomy"] == {"invalid_output": 1, "abstain": 1}


def test_resume_rows_remain_available_for_aggregation(tmp_path: Path) -> None:
    rows = [
        {"qid": "q1", "config": "A0_HOLD", "correct": True},
        {"qid": "q2", "config": "A0_HOLD", "correct": False},
    ]
    fp = tmp_path / "episodes.jsonl"
    fp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    done_rows = bench.load_completed_rows(fp)
    resumed = [done_rows[("A0_HOLD", qid)] for qid in ("q1", "q2")]
    assert len(resumed) == 2
    assert sum(bool(r["correct"]) for r in resumed) == 1


def test_hindsight_oracle_is_gt_bounded_and_offline() -> None:
    hold = [
        {"qid": "q1", "correct": False, "abstain": False, "confidence": 0.8, "n_steps": 1},
        {"qid": "q2", "correct": True, "abstain": False, "confidence": 0.8, "n_steps": 1},
        {"qid": "q3", "correct": True, "abstain": False, "confidence": 0.6, "n_steps": 1},
        {"qid": "q4", "correct": False, "abstain": True, "confidence": 0.2, "n_steps": 1},
    ]
    always = [
        {"qid": "q1", "correct": True, "abstain": False, "confidence": 0.7, "n_steps": 2},
        {"qid": "q2", "correct": False, "abstain": False, "confidence": 0.7, "n_steps": 2},
        {"qid": "q3", "correct": True, "abstain": False, "confidence": 0.9, "n_steps": 2},
        {"qid": "q4", "correct": False, "abstain": False, "confidence": 0.5, "n_steps": 2},
    ]
    rows, diag = reporter.hindsight_oracle_rows(hold, always)
    by_qid = {r["qid"]: r for r in rows}
    assert all(r["oracle_offline_only"] for r in rows)
    assert by_qid["q1"]["correct"] is True and by_qid["q1"]["oracle_source"] == "A2_ALWAYS"
    assert by_qid["q2"]["correct"] is True and by_qid["q2"]["oracle_source"] == "A0_HOLD"
    assert by_qid["q3"]["correct"] is True and by_qid["q3"]["oracle_source"] == "A0_HOLD"
    assert by_qid["q4"]["correct"] is False and by_qid["q4"]["oracle_source"] == "A2_ALWAYS"
    assert diag["n_correctable"] == 1
    assert diag["n_harmful"] == 1
    assert diag["n_both_correct"] == 1
    assert diag["n_neither_correct"] == 1
    assert diag["online_deployable"] is False
