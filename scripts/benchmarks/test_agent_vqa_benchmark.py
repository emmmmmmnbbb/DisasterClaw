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
task_auditor = _load("audit_task_policy")


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


def test_score_episode_collapses_damage_levels_in_binary_mode(monkeypatch) -> None:
    monkeypatch.setenv("DAMAGE_LABEL_MODE", "binary")
    item = {"id": "q_bin", "answer": "严重损伤", "question_type": "damage"}
    run = {
        "ok": True,
        "answer": {
            "answer": "损伤", "abstain": False,
            "decision": "answer", "confidence": 0.8,
        },
        "trajectory": [
            {"candidate_answer": "无损伤", "decision": "reobserve"},
            {"candidate_answer": "轻微损伤", "decision": "answer"},
        ],
    }
    row = bench.score_episode(run, item)
    assert row["correct"] is True
    assert row["gt_answer"] == "损伤"
    assert row["pred_answer"] == "损伤"
    assert row["gt_answer_original"] == "严重损伤"
    assert row["pred_answer_original"] == "损伤"
    assert row["reobserve_pairs"] == [{
        "before": "无损伤", "after": "损伤",
        "before_correct": False, "after_correct": True,
    }]
    assert row["answer_corrected"] is True


def test_manifest_snapshot_freezes_shared_changeos_tool(tmp_path: Path, monkeypatch) -> None:
    weights = tmp_path / "changeos.pt"
    weights.write_bytes(b"fixed-changeos-weights")
    monkeypatch.setenv("DETECTOR_BACKEND", "changeos")
    monkeypatch.setenv("CHANGEOS_WEIGHTS", str(weights))
    monkeypatch.setenv("DAMAGE_LABEL_MODE", "binary")
    snapshot = bench.env_snapshot()
    assert snapshot["detector_backend"] == "changeos"
    assert snapshot["damage_label_mode"] == "binary"
    assert len(snapshot["source_fingerprint"]) == 64
    assert snapshot["perception_tool"] == {
        "name": "ChangeOS",
        "weights": str(weights),
        "weights_sha256": bench.file_hash(weights),
        "frozen": True,
        "policy_shared": True,
        "evaluation_role": "fixed_external_perception_tool",
    }


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


def test_out_of_coverage_is_policy_failure_not_execution_error() -> None:
    rows = [{
        "ok": False, "correct": False, "abstain": True,
        "reason_code": "out_of_coverage", "n_reobservations": 0,
    }]
    assert bench.aggregate(rows)["failure_taxonomy"] == {"out_of_coverage": 1}
    assert reporter.aggregate(rows)["failure_taxonomy"] == {"out_of_coverage": 1}


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


def test_episode_seed_is_stable_and_varies_by_item_and_config() -> None:
    a = bench.episode_seed(42, "A1_RANDOM", "q1")
    assert a == bench.episode_seed(42, "A1_RANDOM", "q1")
    assert a != bench.episode_seed(42, "A1_RANDOM", "q2")
    assert a != bench.episode_seed(42, "A2_ALWAYS", "q1")
    draws = {
        bench.episode_seed(42, "A1_RANDOM", f"q{i}") for i in range(32)
    }
    assert len(draws) == 32


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


def test_motion_ablation_configs_are_distinct() -> None:
    modes = {}
    for name in ("AB_HOLD", "AB_NOOP", "AB_CENTER", "AB_DESCEND", "AB_FULL", "AB_WIDE"):
        app = _fake_app()
        bench.apply_config(app, bench.CONFIGS[name])
        modes[name] = app.VLN_RECHECK_MOTION_MODE
    assert modes == {
        "AB_HOLD": "hold",
        "AB_NOOP": "no_op",
        "AB_CENTER": "center_only",
        "AB_DESCEND": "descend_only",
        "AB_FULL": "descend_center",
        "AB_WIDE": "wide_roi",
    }


def test_report_includes_mcnemar_and_event_cluster_bootstrap() -> None:
    rows_a = [
        {"qid": f"a{i}", "disaster": "event-a", "correct": False}
        for i in range(20)
    ] + [{"qid": "b1", "disaster": "event-b", "correct": True}]
    rows_b = [
        {"qid": f"a{i}", "disaster": "event-a", "correct": True}
        for i in range(20)
    ] + [{"qid": "b1", "disaster": "event-b", "correct": False}]
    mc = reporter.mcnemar_exact(rows_a, rows_b)
    clustered = reporter.event_cluster_bootstrap_correctness(
        rows_a, rows_b, n_boot=100, seed=1,
    )
    item = reporter.paired_bootstrap_correctness(rows_a, rows_b, n_boot=100, seed=1)
    assert mc["b_correct_a_wrong"] == 20 and mc["a_correct_b_wrong"] == 1
    assert clustered["n_events"] == 2
    assert clustered["ci95"] != item["ci95"]


def test_hindsight_oracle_accepts_matched_reobserve_arm() -> None:
    hold = [{"qid": "q1", "correct": False, "abstain": False, "confidence": 0.8, "n_steps": 1}]
    matched = [{"qid": "q1", "correct": True, "abstain": False, "confidence": 0.7, "n_steps": 2}]
    rows, diag = reporter.hindsight_oracle_rows(hold, matched, always_name="A2_FIXED_MATCHED")
    assert rows[0]["oracle_source"] == "A2_FIXED_MATCHED"
    assert diag["always_config"] == "A2_FIXED_MATCHED"


def test_configs_declare_answer_mode_explicitly() -> None:
    """raw 消融必须是纯 VLM，主闭环默认 hybrid，规则基线独立为 deterministic。"""
    assert bench.CONFIGS["V0_RAW"]["answer_mode"] == "vlm"
    assert bench.CONFIGS["D0_RULE"]["answer_mode"] == "deterministic"
    for name in ("A0_HOLD", "A1_RANDOM", "A2_ALWAYS", "A3_ENTROPY", "A5_EXPECTED"):
        app = _fake_app()
        bench.apply_config(app, bench.CONFIGS[name])
        assert bench.effective_config(app)["answer_mode"] == "hybrid"
    app = _fake_app()
    bench.apply_config(app, bench.CONFIGS["V0_RAW"])
    assert bench.effective_config(app)["answer_mode"] == "vlm"


def test_pure_vlm_arms_differ_from_hybrid_twins_only_by_answer_mode() -> None:
    """纯 VLM 臂必须与对应 hybrid 配置逐字段相同，否则无法归因到回答模式。"""
    for vlm_name, hybrid_name in (("V2_STATE_VLM", "V2_STATE"), ("A0_VLM", "A0_HOLD")):
        vlm_app, hybrid_app = _fake_app(), _fake_app()
        bench.apply_config(vlm_app, bench.CONFIGS[vlm_name])
        bench.apply_config(hybrid_app, bench.CONFIGS[hybrid_name])
        vlm_cfg, hybrid_cfg = bench.effective_config(vlm_app), bench.effective_config(hybrid_app)
        assert vlm_cfg["answer_mode"] == "vlm"
        assert hybrid_cfg["answer_mode"] == "hybrid"
        assert {k: v for k, v in vlm_cfg.items() if k != "answer_mode"} == \
               {k: v for k, v in hybrid_cfg.items() if k != "answer_mode"}


def _write_run(root: Path, name: str, rows: list[dict], modes: dict) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "results.json").write_text(
        json.dumps({"valid_for_analysis": True}), encoding="utf-8")
    (run / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps({"agent_vqa_answer_modes": modes}), encoding="utf-8")
    return run


def test_answer_mode_conflict_across_runs_is_detected(tmp_path: Path) -> None:
    rows = [{"qid": "q1", "config": "A0_HOLD", "correct": True}]
    run_a = _write_run(tmp_path, "a", rows, {"A0_HOLD": "hybrid"})
    run_b = _write_run(tmp_path, "b", rows, {"A0_HOLD": "deterministic"})
    _, conflicts = reporter.answer_mode_by_config([run_a, run_b], {"A0_HOLD": rows})
    assert conflicts == {"A0_HOLD": ["deterministic", "hybrid"]}


def test_report_separates_rule_baseline_and_refuses_mode_conflict(
    tmp_path: Path, monkeypatch,
) -> None:
    rows_rule = [{"qid": "q1", "config": "D0_RULE", "correct": True,
                  "answer_mode": "deterministic", "n_reobservations": 0}]
    rows_policy = [{"qid": "q1", "config": "A0_HOLD", "correct": False,
                    "answer_mode": "hybrid", "n_reobservations": 0}]
    run = _write_run(tmp_path, "ok", rows_rule + rows_policy,
                     {"D0_RULE": "deterministic", "A0_HOLD": "hybrid"})
    out = tmp_path / "reports"
    monkeypatch.setattr("sys.argv", [
        "report_agent_vqa.py", "--runs", str(run), "--out", str(out),
    ])
    assert reporter.main() == 0
    meta = json.loads((out / "config_meta.json").read_text(encoding="utf-8"))
    assert meta["rule_baseline_configs"] == ["D0_RULE"]
    assert meta["model_driven_configs"] == ["A0_HOLD"]
    rule_report = json.loads((out / "rule_baseline.json").read_text(encoding="utf-8"))
    assert rule_report["aggregate"]["D0_RULE"]["accuracy"] == 1.0
    assert "A0_HOLD" not in rule_report["aggregate"]
    paired = json.loads((out / "paired_tests.json").read_text(encoding="utf-8"))
    assert paired["D0_RULE_vs_A0_HOLD"]["comparable_as_policy"] is False

    # 同一配置名混入多种回答模式 → 拒绝聚合
    bad_rows = [{"qid": "q1", "config": "A0_HOLD", "correct": True,
                 "answer_mode": "deterministic"}]
    bad = _write_run(tmp_path, "bad", bad_rows, {"A0_HOLD": "hybrid"})
    monkeypatch.setattr("sys.argv", [
        "report_agent_vqa.py", "--runs", str(bad), "--out", str(tmp_path / "r2"),
    ])
    assert reporter.main() == 3


def test_score_episode_flight_time_uses_method_speeds() -> None:
    item = {"id": "q_time", "answer": "是", "question_type": "presence"}
    run = {
        "ok": True, "n_steps": 2,
        "answer": {"answer": "是", "abstain": False, "decision": "answer", "confidence": 0.8},
        "trajectory": [
            {
                "candidate_answer": "否", "decision": "reobserve",
                "reobserve_params": {"north_m": 40.0, "east_m": 0.0, "up_m": -443.4},
            },
            {"candidate_answer": "是", "decision": "answer"},
        ],
    }
    row = bench.score_episode(run, item)
    from recheck import reobserve_flight_time_s
    assert row["reobserve_flight_time_s"] == round(reobserve_flight_time_s(40.0, 443.4), 3)


def test_task_policy_audit_accepts_explained_actions_and_rejects_roi_loss() -> None:
    base_step = {
        "observation_id": "obs1", "reobserve_kind": "recheck",
        "reobserve_reason": "positive task-conditioned utility",
        "reobserve_params": {"north_m": 0.0, "east_m": 0.0, "up_m": -10.0},
        "policy_metrics": {
            "allow": True, "utility": 0.2, "predicted_roi_coverage": 1.0,
        },
        "evidence": {
            "history_observation_ids": ["obs1"], "objects": [],
            "target_visible": True,
        },
    }
    row = {
        "qid": "q1", "question_type": "count", "correct": True, "ok": True,
        "n_reobservations": 1, "trajectory": [base_step],
    }
    good = task_auditor.audit([row])
    assert good["valid_for_analysis"] and good["n_violations"] == 0

    bad = json.loads(json.dumps(row))
    bad["trajectory"][0]["policy_metrics"]["predicted_roi_coverage"] = 0.7
    rejected = task_auditor.audit([bad])
    assert not rejected["valid_for_analysis"]
    assert rejected["violations"][0]["code"] == "context_roi_below_minimum"
