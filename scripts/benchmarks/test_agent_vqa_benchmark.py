"""Agent-VQA benchmark configuration and offline-scoring regression tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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


def test_hindsight_oracle_is_gt_bounded_and_offline() -> None:
    hold = [
        {"qid": "q1", "correct": False, "abstain": False, "confidence": 0.8, "n_steps": 1},
        {"qid": "q2", "correct": True, "abstain": False, "confidence": 0.8, "n_steps": 1},
    ]
    always = [
        {"qid": "q1", "correct": True, "abstain": False, "confidence": 0.7, "n_steps": 2},
        {"qid": "q2", "correct": False, "abstain": False, "confidence": 0.7, "n_steps": 2},
    ]
    rows, diag = reporter.hindsight_oracle_rows(hold, always)
    assert all(r["correct"] and r["oracle_offline_only"] for r in rows)
    assert diag["n_correctable"] == 1
    assert diag["n_harmful"] == 1
    assert diag["online_deployable"] is False
