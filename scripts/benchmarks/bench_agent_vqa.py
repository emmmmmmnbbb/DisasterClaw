#!/usr/bin/env python3
"""scripts/benchmarks/bench_agent_vqa.py — Agent-VQA 主评测脚本 (计划 9.1 / E1-E5)

读 Agent-VQA 题库 (gen_agent_vqa_testset.py 产出) → 对每个配置逐题调用
app.run_agent_vqa_episode_headless (真实感知/问答/搜索/重观测链路) → 由 report + GT
计算答案准确率 / 弃答率 / 翻转-纠错-损害 / 预算效用 / fallback 统计 → 落
runs/benchmarks/cja_agent_vqa/<run_id>/。

配置 (V0_RAW / V1_STRUCT / V2_STATE / A0_HOLD / A1_RANDOM / A2_ALWAYS /
A3_ENTROPY / A4_CONFORMAL / A5_EXPECTED / O_REF) 通过 import app 后直接改 app
模块级全局切换 (AGENT_VQA_* 与 VLN_RECHECK_* 开关)。

在线字段 (控制器可读) 与离线评分字段 (仅本脚本读 GT 计算) 严格分离：
  - 在线: answer / confidence / decision / reason_code / trajectory / fallback_used
  - 离线: correct / abstain_should / answer_corrected / answer_harmed (本脚本填充)

每题即时落盘 (episodes.jsonl)，支持 --resume 续跑 (跳过已完成 (config, qid))。
manifest 记录 env / git commit / 数据 hash / prompt / 阈值，结果自描述可追溯。

用法 (先 source .env 让本地 VLM/planner 配置生效):
    cd backend && set -a && source ../.env && set +a && \\
      python ../scripts/benchmarks/bench_agent_vqa.py \\
        --configs V0_RAW,D0_RULE,A0_HOLD,A3_ENTROPY --limit 8 --split test
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))  # 必须在 import app 之前

from recheck import reobserve_flight_time_s  # noqa: E402
from vlm_analyzer import AGENT_VQA_SYSTEM_PROMPT  # noqa: E402

DEFAULT_TESTSET = BACKEND / "data" / "benchmarks" / "agent_vqa_testset_v2.json"
RUNS_DIR = REPO_ROOT / "runs" / "benchmarks" / "cja_agent_vqa"

# 配置 → (evidence_level, max_search, max_reobs, recheck_trigger[, recheck_extra][, oracle])
# evidence_level: raw / struct / state  (计划 9.1 图像/状态列)
# max_search / max_reobs: 0 = 不可移动 / 不可重观测
# recheck_trigger: 控制 reobserve_fn 的 RecheckController 触发模式
CONFIGS = {
    "V0_RAW": {"evidence_level": "raw", "max_search": 0, "max_reobs": 0,
               "answer_mode": "vlm", "recheck_trigger": "threshold",
               "desc": "静态 VLM 基线 (仅图像)"},
    "D0_RULE": {"evidence_level": "struct", "max_search": 0, "max_reobs": 0,
                "answer_mode": "deterministic", "recheck_trigger": "threshold",
                "desc": "结构化证据确定性回答基线"},
    "V1_STRUCT": {"evidence_level": "struct", "max_search": 0, "max_reobs": 0,
                  "recheck_trigger": "threshold", "desc": "图像 + 结构化感知"},
    "V2_STATE": {"evidence_level": "state", "max_search": 0, "max_reobs": 0,
                 "recheck_trigger": "threshold", "desc": "V1 + STMR + 历史"},
    # 纯 VLM 臂: 与同名 hybrid 配置逐字段相同, 只把 answer_mode 换成 vlm。
    # 用途是在结构化证据下发后单独检验 VLM 是否仍然坍缩; hybrid 会用规则答案
    # 覆盖 VLM, 因此不能用来判断 VLM 本身的能力或坍缩。
    "V2_STATE_VLM": {"evidence_level": "state", "max_search": 0, "max_reobs": 0,
                     "answer_mode": "vlm", "recheck_trigger": "threshold",
                     "desc": "V2_STATE 的纯 VLM 回答臂 (无规则覆盖)"},
    "A0_VLM": {"evidence_level": "state", "max_search": 6, "max_reobs": 0,
               "answer_mode": "vlm", "recheck_trigger": "threshold",
               "desc": "A0_HOLD 的纯 VLM 回答臂 (坍缩诊断用)"},
    "A0_HOLD": {"evidence_level": "state", "max_search": 6, "max_reobs": 0,
                "recheck_trigger": "threshold", "desc": "Agent 单观测基线 (可搜索不可重观测)"},
    "A1_RANDOM": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                  "recheck_trigger": "random", "recheck_extra": {"random_prob": 0.5},
                  "desc": "随机重观测 (预算匹配对照)"},
    "A2_ALWAYS": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                  "recheck_trigger": "fixed", "desc": "总是重观测 (额外观测上限对照)"},
    "A3_ENTROPY": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                   "recheck_trigger": "threshold",
                   "recheck_extra": {"uncertainty_mode": "entropy"},
                   "desc": "校准熵驱动主动策略"},
    "T1_TASK": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                "recheck_trigger": "task_conditioned",
                "recheck_extra": {
                    "uncertainty_mode": "entropy", "trigger": 0.5,
                    "min_roi_coverage": 0.98, "cost_weight": 0.05,
                    "coverage_weight": 1.0, "cost_scale_s": 60.0,
                    "min_utility": 0.05,
                },
                "desc": "任务条件化效用策略（证据增益-动作成本-覆盖损失）"},
    "A3U_RAW_ENTROPY": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                        "recheck_trigger": "threshold",
                        "recheck_extra": {"uncertainty_mode": "entropy_raw"},
                        "desc": "未校准熵驱动主动策略"},
    "A4_CONFORMAL": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                     "recheck_trigger": "conformal",
                     "recheck_extra": {"uncertainty_mode": "entropy"},
                     "desc": "共形集合不确定性对照"},
    "A5_EXPECTED": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                    "recheck_trigger": "info_gain",
                    "recheck_extra": {"uncertainty_mode": "entropy"},
                    "desc": "验证集期望收益 (泄漏安全条件策略)"},
    "A1_RANDOM_MATCHED": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "random",
        "desc": "随机分配、动作总数与参考策略严格相等",
    },
    "A2_FIXED_MATCHED": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "same_items",
        "desc": "固定下降、逐题预算与参考策略相等",
    },
    "A2_ALL_REOBSERVE": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "desc": "所有题复观测（无预算上限参考）",
    },
    "AB_HOLD": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 0,
        "recheck_trigger": "fixed", "recheck_extra": {"motion_mode": "hold"},
        "desc": "动作消融：保持",
    },
    "AB_NOOP": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "same_items",
        "recheck_extra": {"motion_mode": "no_op", "force_reobserve_invalid": True},
        "desc": "动作消融：同姿态重复观测",
    },
    "AB_CENTER": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "same_items",
        "recheck_extra": {"motion_mode": "center_only", "force_reobserve_invalid": True},
        "desc": "动作消融：仅居中",
    },
    "AB_DESCEND": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "same_items",
        "recheck_extra": {"motion_mode": "descend_only", "force_reobserve_invalid": True},
        "desc": "动作消融：仅下降",
    },
    "AB_FULL": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "same_items",
        "recheck_extra": {"motion_mode": "descend_center", "force_reobserve_invalid": True},
        "desc": "动作消融：下降并居中",
    },
    "AB_WIDE": {
        "evidence_level": "state", "max_search": 6, "max_reobs": 2,
        "recheck_trigger": "fixed", "matched_baseline": "same_items",
        "recheck_extra": {"motion_mode": "wide_roi", "force_reobserve_invalid": True},
        "desc": "动作消融：仅在完整保留查询 ROI 时下降",
    },
    "O_REF": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
              "recheck_trigger": "threshold", "offline_only": True,
              "desc": "离线 hindsight oracle 参照 (由 A0/A2 配对结果合成, 禁止部署)"},
}


def apply_config(app, cfg: dict) -> None:
    """运行期切换 app 模块级 Agent-VQA 与 recheck 开关。未显式设置的字段回落默认值。"""
    if cfg.get("offline_only"):
        raise ValueError("O_REF 是 report_agent_vqa.py 从 A0_HOLD/A2_ALWAYS 合成的离线参照")
    app.AGENT_VQA_EVIDENCE_LEVEL = cfg["evidence_level"]
    app.AGENT_VQA_ANSWER_MODE = cfg.get("answer_mode", "hybrid")
    app.AGENT_VQA_MAX_SEARCH_STEPS = int(cfg["max_search"])
    app.AGENT_VQA_MAX_REOBSERVATIONS = int(cfg["max_reobs"])
    # reobserve_fn 内部新建 RecheckController 时读这些 app 级开关
    app.VLN_RECHECK_TRIGGER = cfg["recheck_trigger"]
    app.VLN_UNCERTAINTY_MODE = "heuristic"
    extra = cfg.get("recheck_extra") or {}
    app.VLN_UNCERTAINTY_MODE = extra.get("uncertainty_mode", "heuristic")
    app.VLN_RECHECK_MIN_INFO_GAIN = float(extra.get("min_info_gain", 0.05))
    app.VLN_RECHECK_THRESHOLD = float(extra.get("trigger", 0.5))
    app.VLN_RECHECK_RANDOM_PROB = float(extra.get("random_prob", 0.5))
    app.VLN_RECHECK_RANDOM_SEED = int(extra.get("random_seed", 0))
    app.VLN_RECHECK_MOTION_MODE = extra.get("motion_mode", "descend_center")
    app.AGENT_VQA_FORCE_REOBSERVE_INVALID = bool(extra.get("force_reobserve_invalid", False))
    app.VLN_RECHECK_TEMPERATURE = float(extra.get("temperature", 1.0))
    app.VLN_TASK_MIN_ROI_COVERAGE = float(extra.get("min_roi_coverage", 0.98))
    app.VLN_TASK_COST_WEIGHT = float(extra.get("cost_weight", 0.05))
    app.VLN_TASK_COVERAGE_WEIGHT = float(extra.get("coverage_weight", 1.0))
    app.VLN_TASK_COST_SCALE_S = float(extra.get("cost_scale_s", 60.0))
    app.VLN_TASK_MIN_UTILITY = float(extra.get("min_utility", 0.05))
    app.VLN_ENTROPY_TABLE = str(extra.get("entropy_table_path", app.VLN_ENTROPY_TABLE))
    app.VLN_CONFORMAL_QHAT = float(extra.get("conformal_qhat", 0.9))
    app.VLN_CONFORMAL_ALPHA = float(extra.get("conformal_alpha", 0.1))
    # 真值 oracle 不在在线控制器中运行；O_REF 只能由报告脚本离线合成。
    app.AGENT_VQA_ORACLE = False


def effective_config(app) -> dict:
    """记录实际传入控制器的开关，防止消融名称与运行行为不一致。"""
    return {
        "evidence_level": app.AGENT_VQA_EVIDENCE_LEVEL,
        "answer_mode": app.AGENT_VQA_ANSWER_MODE,
        "max_search": app.AGENT_VQA_MAX_SEARCH_STEPS,
        "max_reobs": app.AGENT_VQA_MAX_REOBSERVATIONS,
        "trigger_mode": app.VLN_RECHECK_TRIGGER,
        "uncertainty_mode": app.VLN_UNCERTAINTY_MODE,
        "min_info_gain": app.VLN_RECHECK_MIN_INFO_GAIN,
        "trigger": app.VLN_RECHECK_THRESHOLD,
        "random_prob": app.VLN_RECHECK_RANDOM_PROB,
        "random_seed": app.VLN_RECHECK_RANDOM_SEED,
        "motion_mode": app.VLN_RECHECK_MOTION_MODE,
        "force_reobserve_invalid": app.AGENT_VQA_FORCE_REOBSERVE_INVALID,
        "temperature": app.VLN_RECHECK_TEMPERATURE,
        "min_roi_coverage": app.VLN_TASK_MIN_ROI_COVERAGE,
        "cost_weight": app.VLN_TASK_COST_WEIGHT,
        "coverage_weight": app.VLN_TASK_COVERAGE_WEIGHT,
        "cost_scale_s": app.VLN_TASK_COST_SCALE_S,
        "min_utility": app.VLN_TASK_MIN_UTILITY,
        "entropy_table_path": app.VLN_ENTROPY_TABLE,
        "conformal_qhat": app.VLN_CONFORMAL_QHAT,
        "conformal_alpha": app.VLN_CONFORMAL_ALPHA,
        "oracle": app.AGENT_VQA_ORACLE,
    }


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return bool(out.strip())
    except Exception:
        return True


def file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint() -> str:
    """Hash the executable Agent-VQA path, including uncommitted source files."""
    relpaths = (
        "backend/agent_vqa.py", "backend/app.py", "backend/perception.py",
        "backend/recheck.py", "backend/detectors/base.py",
        "backend/detectors/changeos.py", "backend/vlm_analyzer.py",
        "backend/llm_client.py", "backend/local_qwen_vl.py", "backend/ml_runtime.py",
        "backend/config.py",
        "scripts/benchmarks/bench_agent_vqa.py",
    )
    h = hashlib.sha256()
    for rel in relpaths:
        path = REPO_ROOT / rel
        h.update(rel.encode("utf-8") + b"\0")
        if path.is_file():
            h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def episode_seed(base_seed: int, config: str, qid: str) -> int:
    """Stable per-episode seed, independent of shard order and process count."""
    payload = f"agent-vqa|{int(base_seed)}|{config}|{qid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def load_completed_rows(path: Path) -> dict[tuple[str, str], dict]:
    """Load the latest durable row for each (config, qid) resume key."""
    rows: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        qid = rec.get("qid") or ""
        cfg = rec.get("config") or ""
        if qid and cfg:
            rows[(cfg, qid)] = rec
    return rows


def env_snapshot() -> dict:
    import importlib.metadata
    detector_backend = os.environ.get("DETECTOR_BACKEND", "legacy")
    changeos_weights = Path(os.environ.get(
        "CHANGEOS_WEIGHTS", REPO_ROOT / "backend/outputs/changeos/changeos_r34.pt",
    )).expanduser()
    info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": os.environ.get("DATASET_MODE", "xbd"),
        "perception_device": os.environ.get("PERCEPTION_DEVICE", "cuda"),
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "source_fingerprint": source_fingerprint(),
        "llm_model": os.environ.get("LLM_MODEL", ""),
        "vlm_provider": os.environ.get("VLM_PROVIDER", "vlm"),
        "vlm_model": (
            os.environ.get("BASE_MODEL")
            or os.environ.get("VLM_LOCAL_MODEL")
            or os.environ.get("VLM_MODEL")
            or "Qwen/Qwen2.5-VL-7B-Instruct"
        ),
        "vlm_top_p": float(os.environ.get("VLM_LOCAL_TOP_P", "0.9") or "0.9"),
        "vlm_repetition_penalty": float(
            os.environ.get("VLM_LOCAL_REPETITION_PENALTY", "1.1") or "1.1"
        ),
        "agent_vqa_confidence_threshold": os.environ.get("AGENT_VQA_CONFIDENCE_THRESHOLD", "0.5"),
        "detector_backend": detector_backend,
        "damage_label_mode": os.environ.get("DAMAGE_LABEL_MODE", "four_class"),
    }
    if detector_backend.strip().lower() in {"changeos", "changeos_r34", "binary_changeos"}:
        info["perception_tool"] = {
            "name": "ChangeOS",
            "weights": str(changeos_weights),
            "weights_sha256": file_hash(changeos_weights),
            "frozen": True,
            "policy_shared": True,
            "evaluation_role": "fixed_external_perception_tool",
        }
    info["packages"] = {}
    for pkg in ("numpy", "scipy", "torch", "torchvision", "ultralytics", "transformers"):
        try:
            info["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return info


def score_episode(report: dict, item: dict) -> dict:
    """离线评分: 在线 report + GT → correct / abstain_should / flip。

    在线控制器不读 GT; 所有 GT 比较只在此函数完成 (计划 7.4 / E4)。
    """
    gt_answer_original = item.get("answer", "")
    ans = (report or {}).get("answer") or {}
    pred_original = ans.get("answer", "")
    binary_damage = (
        os.getenv("DAMAGE_LABEL_MODE", "four_class").strip().lower() in {"binary", "damaged"}
        and item.get("question_type") == "damage"
    )

    def _collapse_answer(value: str) -> str:
        if not binary_damage:
            return value
        if value in {"无损伤", "没损伤", "未损伤", "no-damage"}:
            return "无损伤"
        if value in {"轻微损伤", "严重损伤", "完全损毁", "损伤", "damaged"}:
            return "损伤"
        return value

    gt_answer = _collapse_answer(gt_answer_original)
    pred = _collapse_answer(pred_original)
    abstain = bool(ans.get("abstain"))
    decision = ans.get("decision", "")
    correct = (not abstain) and bool(pred) and pred == gt_answer
    # 弃答是否"应该": GT 为否定类 (如 presence=否 / count=0) 且模型 abstain 视为合理保守
    abstain_should = abstain and gt_answer in {"否", "0"}
    traj = (report or {}).get("trajectory", [])
    preds = [
        _collapse_answer(t.get("candidate_answer"))
        for t in traj if t.get("candidate_answer")
    ]
    flipped = len(set(preds)) > 1
    reobserve_pairs = []
    for i, step in enumerate(traj[:-1]):
        if step.get("decision") != "reobserve":
            continue
        before = _collapse_answer(step.get("candidate_answer") or "")
        after = _collapse_answer(traj[i + 1].get("candidate_answer") or "")
        reobserve_pairs.append({
            "before": before,
            "after": after,
            "before_correct": bool(before) and before == gt_answer,
            "after_correct": bool(after) and after == gt_answer,
        })
    answer_corrected = any(not p["before_correct"] and p["after_correct"] for p in reobserve_pairs)
    answer_harmed = any(p["before_correct"] and not p["after_correct"] for p in reobserve_pairs)
    n_correcting_reobservations = sum(
        1 for p in reobserve_pairs if not p["before_correct"] and p["after_correct"]
    )
    n_harming_reobservations = sum(
        1 for p in reobserve_pairs if p["before_correct"] and not p["after_correct"]
    )
    n_reobserve_skips = sum(1 for t in traj if t.get("reobserve_kind") == "skip")
    reobserve_params = [
        dict(t.get("reobserve_params") or {}) for t in traj
        if t.get("decision") == "reobserve"
    ]
    reobserve_executed = [
        dict(t.get("reobserve_executed") or {}) for t in traj
        if t.get("decision") == "reobserve"
    ]
    motion_rows = (
        reobserve_executed
        if reobserve_executed and all(bool(p) for p in reobserve_executed)
        else reobserve_params
    )
    horizontal_m = sum(
        (float(p.get("north_m", 0.0)) ** 2 + float(p.get("east_m", 0.0)) ** 2) ** 0.5
        for p in motion_rows
    )
    vertical_m = sum(abs(float(p.get("up_m", 0.0))) for p in motion_rows)
    difficulty = item.get("difficulty", "")
    difficulty_band = difficulty.get("distance", "") if isinstance(difficulty, dict) else difficulty
    ans_evidence = ans.get("evidence") or {}
    return {
        "qid": item.get("id") or "",
        "config": report.get("config", "") if report else "",
        "question_type": item.get("question_type", ""),
        "difficulty": difficulty_band,
        "disaster": item.get("disaster", ""),
        "split": item.get("split", ""),
        "tile_id": item.get("tile_id", ""),
        "question": item.get("question", ""),
        "gt_answer": gt_answer,
        "gt_answer_original": gt_answer_original,
        "pred_answer": pred,
        "pred_answer_original": pred_original,
        "abstain": abstain,
        "decision": decision,
        "reason_code": ans.get("reason_code", ""),
        "schema_errors": ans.get("schema_errors", []),
        "raw_model_output": ans.get("raw_model_output", ""),
        "confidence": ans.get("confidence"),
        "correct": correct,
        "abstain_should": abstain_should,
        "flipped": flipped,
        "initial_pred_answer": preds[0] if preds else "",
        "reobserve_pairs": reobserve_pairs,
        "n_reobservations": len(reobserve_pairs),
        "n_reobserve_skips": n_reobserve_skips,
        "reobserve_horizontal_m": round(horizontal_m, 3),
        "reobserve_vertical_m": round(vertical_m, 3),
        "reobserve_motion_source": (
            "executed" if motion_rows is reobserve_executed else "requested_fallback"
        ),
        "reobserve_flight_time_s": round(
            reobserve_flight_time_s(horizontal_m, vertical_m), 3,
        ),
        "entropy_table_loaded": any(bool(t.get("entropy_table_loaded")) for t in traj),
        "entropy_fallback_used": any(bool(t.get("entropy_fallback_used")) for t in traj),
        "answer_corrected": answer_corrected,
        "answer_harmed": answer_harmed,
        "n_correcting_reobservations": n_correcting_reobservations,
        "n_harming_reobservations": n_harming_reobservations,
        "n_steps": report.get("n_steps", 0) if report else 0,
        "fallback_used": report.get("fallback_used", False) if report else False,
        "degraded_reason": report.get("degraded_reason", "") if report else "",
        "ok": report.get("ok", False) if report else False,
        "error": report.get("error", "") if report else "",
        "wall_s": report.get("wall_s", 0.0) if report else 0.0,
        "trajectory": traj,
        "evidence": ans_evidence,
        "generation_seeds": [
            t.get("generation_seed") for t in traj
            if t.get("generation_seed") is not None
        ],
        "generation_repeat": next(
            (t.get("generation_repeat") for t in traj
             if t.get("generation_seed") is not None),
            None,
        ),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    answered = [r for r in rows if not r.get("abstain")]
    abstained = [r for r in rows if r.get("abstain")]
    correct = [r for r in answered if r.get("correct")]
    flipped = [r for r in rows if r.get("flipped")]
    fallback = [r for r in rows if r.get("fallback_used")]
    agg = {
        "n": n,
        "answered": len(answered),
        "abstained": len(abstained),
        "abstain_rate": round(len(abstained) / n, 4),
        "accuracy": round(len(correct) / n, 4),                     # 全题口径 (弃答算错)
        "answer_acc": round(len(correct) / len(answered), 4) if answered else None,
        "flip_rate": round(len(flipped) / n, 4),
        "fallback_rate": round(len(fallback) / n, 4),
        "n_steps_mean": round(sum(r.get("n_steps", 0) for r in rows) / n, 2),
        "n_reobservations": sum(int(r.get("n_reobservations", 0) or 0) for r in rows),
        "n_reobserve_skips": sum(int(r.get("n_reobserve_skips", 0) or 0) for r in rows),
        "n_triggered": sum(1 for r in rows if int(r.get("n_reobservations", 0) or 0) > 0),
        "reobserve_horizontal_m": round(sum(float(r.get("reobserve_horizontal_m", 0.0)) for r in rows), 3),
        "reobserve_vertical_m": round(sum(float(r.get("reobserve_vertical_m", 0.0)) for r in rows), 3),
        "reobserve_flight_time_s": round(sum(float(r.get("reobserve_flight_time_s", 0.0)) for r in rows), 3),
        "entropy_table_loaded": any(bool(r.get("entropy_table_loaded")) for r in rows),
        "entropy_fallback_used": any(bool(r.get("entropy_fallback_used")) for r in rows),
    }
    by_type = {}
    for qt in ("presence", "damage", "count", "spatial"):
        sub = [r for r in rows if r.get("question_type") == qt]
        if sub:
            by_type[qt] = {
                "n": len(sub),
                "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4),
                "abstain_rate": round(sum(1 for r in sub if r.get("abstain")) / len(sub), 4),
            }
    agg["by_question_type"] = by_type
    by_event = {}
    for d in sorted({r.get("disaster") for r in rows if r.get("disaster")}):
        sub = [r for r in rows if r.get("disaster") == d]
        by_event[d] = {
            "n": len(sub),
            "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4),
        }
    agg["by_event"] = by_event
    by_diff = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r.get("difficulty") == d]
        if sub:
            by_diff[d] = {"n": len(sub),
                          "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4)}
    agg["by_difficulty"] = by_diff
    fail = {}
    invalid_schema = {}
    policy_failure_reasons = {"out_of_coverage"}
    for r in rows:
        reason = str(r.get("reason_code") or "")
        if reason in policy_failure_reasons:
            # A policy can leave the available imagery footprint.  Keep that
            # episode as a scored failure without invalidating the whole run.
            key = reason
        elif r.get("ok") and not r.get("correct"):
            if r.get("reason_code") == "invalid_output":
                key = "invalid_output"
            else:
                key = "abstain" if r.get("abstain") else "wrong_answer"
        elif not r.get("ok"):
            key = "execution_error"
        else:
            continue
        fail[key] = fail.get(key, 0) + 1
        for err in r.get("schema_errors") or []:
            code = str(err).split(":", 1)[0]
            invalid_schema[code] = invalid_schema.get(code, 0) + 1
    agg["failure_taxonomy"] = fail
    agg["invalid_schema_errors"] = invalid_schema
    return agg


def md_table(per_config: dict) -> str:
    cols = ["config", "n", "accuracy", "abstain_rate", "n_steps_mean", "n_reobservations", "n_reobserve_skips"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for name, rec in per_config.items():
        a = rec.get("agg", {})
        lines.append("| " + " | ".join(str(a.get(c, "")) for c in cols[1:]) + " |")
        # 把 config 名插到行首
        lines[-1] = f"| {name} " + lines[-1][1:]
    return "\n".join(lines)


def _normalise_label_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    return "binary" if m in {"binary", "damaged", "damage"} else "four_class"


def _detector_label_mode(backend: str) -> str:
    """检测后端输出的损伤标签空间：二分类 ChangeOS 或四分类其余后端。"""
    if str(backend or "").strip().lower() in {"changeos", "changeos_r34", "binary_changeos"}:
        return "binary"
    return "four_class"


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent-VQA 评测 (E1-E5 消融成绩单)")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--configs", default="V0_RAW,D0_RULE,A0_HOLD,A3_ENTROPY")
    ap.add_argument("--limit", type=int, default=0, help="每个配置最多跑前 N 题 (0=全部)")
    ap.add_argument("--split", default="", help="只跑某 split train/val/test (空=全部)")
    ap.add_argument("--qtype", default="", help="只跑某题型 presence/damage/count/spatial (空=全部)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generation-seed", type=int, default=42000,
                    help="VLM generation base seed; independent of --seed/action RNG")
    ap.add_argument("--generation-repeat", type=int, default=0,
                    help="repeat index included in each per-call generation seed")
    ap.add_argument("--frozen-manifest", default="",
                    help="冻结配置 JSON；校验题库 hash 并注入 T/qhat/阈值/熵表")
    ap.add_argument("--review-report", default="",
                    help="agent-vqa-review/2.0 审核报告；final 题库必须提供且每题 overall approved")
    ap.add_argument("--matched-reference", default="T1_TASK",
                    help="同预算基线的参考配置（必须在 matched 配置前运行）")
    ap.add_argument("--matched-budget-frac", type=float, default=-1.0,
                    help="无参考结果时用于 smoke 的固定预算比例；正式评测保持 -1")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true",
                    help="续跑: 跳过 episodes.jsonl 已有的 (config,qid)")
    ap.add_argument("--allow-crash-resume", action="store_true",
                    help="允许 final 题库在进程崩溃后续跑；只跳过已完成 (config,qid)，不重跑")
    ap.add_argument("--allow-label-mismatch", action="store_true",
                    help="覆盖题库 damage_label_mode 与检测后端标签空间的兼容性检查"
                         "（仅在明确知道自己在做什么时使用；误用会产生一张看似正常的无效表）")
    ap.add_argument("--shard", default="",
                    help="分片并行: 'i/N' 只跑 items[i::N] (用于多 GPU 按题分片, "
                         "保持每片内 (config,qid) 配对完整)")
    args = ap.parse_args()

    # 按题分片 (多 GPU 并行): 在过滤后取 items[i::N]
    if args.shard:
        try:
            i_str, n_str = args.shard.split("/")
            shard_i, shard_n = int(i_str), int(n_str)
            assert 0 <= shard_i < shard_n
        except Exception:
            print(f"[ERROR] --shard 格式应为 i/N (如 0/4), 得到 {args.shard!r}", file=sys.stderr)
            return 2
        args._shard_i, args._shard_n = shard_i, shard_n
    else:
        args._shard_i, args._shard_n = 0, 1

    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            print(f"[ERROR] 未知配置 {c}, 可选 {list(CONFIGS)}", file=sys.stderr)
            return 2
    if "O_REF" in configs:
        print("[ERROR] O_REF 不运行在线 episode；请先跑 A0_HOLD,A2_ALWAYS，再用 "
              "report_agent_vqa.py 合成离线 hindsight oracle。", file=sys.stderr)
        return 2

    testset_path = Path(args.testset)
    if not testset_path.is_file():
        print(f"[ERROR] 题库不存在: {testset_path}", file=sys.stderr)
        return 2
    testset = json.loads(testset_path.read_text(encoding="utf-8"))

    # P0 fail-fast：二分类后端（ChangeOS）与四分类题库不兼容。damage 题虽在
    # score_episode 里折叠了，presence/count/spatial 仍会系统性错配（题面问
    # “严重或完全损毁”，检测只给“受损”），必须阻断，而不是跑出一张看似正常的无效表。
    detector_backend = os.environ.get("DETECTOR_BACKEND", "legacy")
    backend_label_mode = _detector_label_mode(detector_backend)
    testset_label_mode = _normalise_label_mode(testset.get("damage_label_mode"))
    env_label_mode = _normalise_label_mode(os.environ.get("DAMAGE_LABEL_MODE", "four_class"))
    label_problems = []
    if testset_label_mode != backend_label_mode:
        label_problems.append(
            f"题库 damage_label_mode={testset_label_mode!r} 与检测后端 "
            f"{detector_backend!r} 的标签空间 {backend_label_mode!r} 不兼容"
        )
    if env_label_mode != testset_label_mode:
        label_problems.append(
            f"DAMAGE_LABEL_MODE={env_label_mode!r} 与题库 {testset_label_mode!r} 不一致"
        )
    if label_problems and not args.allow_label_mismatch:
        for p in label_problems:
            print(f"[ERROR] {p}", file=sys.stderr)
        print(
            "[ERROR] 二分类 ChangeOS 后端必须配套 --damage-label-mode binary 生成的题库，"
            "并设置 DAMAGE_LABEL_MODE=binary（含 DETECTOR_BACKEND=changeos）；"
            "确认无误可用 --allow-label-mismatch 覆盖。",
            file=sys.stderr,
        )
        return 2

    if testset.get("eval_role") == "final":
        if args.resume and not args.allow_crash_resume:
            print(
                "[ERROR] final 题库默认禁止 --resume；崩溃续跑请同时传 --allow-crash-resume",
                file=sys.stderr,
            )
            return 3
        if args.resume and args.allow_crash_resume:
            print(
                "[WARN] crash-resume on final testset: skip completed (config,qid) only",
                file=sys.stderr,
            )
        if not args.frozen_manifest:
            print("[ERROR] final 题库必须提供 --frozen-manifest", file=sys.stderr)
            return 3
        if not args.review_report:
            print("[ERROR] final 题库必须提供 --review-report，自动检查不能替代人工审核",
                  file=sys.stderr)
            return 3
    frozen = {}
    if args.frozen_manifest:
        frozen_path = Path(args.frozen_manifest)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        # P2 防护：冻结 manifest 里的 qhat / temperature 是在特定标签空间上拟合的；
        # 二分类后端不得静默复用四分类 qhat（conformal 覆盖语义会被破坏）。
        frozen_label_mode = _normalise_label_mode(frozen.get("label_mode"))
        if (
            frozen_label_mode
            and frozen_label_mode != backend_label_mode
            and not args.allow_label_mismatch
        ):
            print(
                f"[ERROR] 冻结 manifest label_mode={frozen_label_mode!r} 与检测后端 "
                f"{detector_backend!r} 的标签空间 {backend_label_mode!r} 不兼容；"
                "请为 ChangeOS 重新拟合 binary qhat 并重新冻结。",
                file=sys.stderr,
            )
            return 2
        if testset.get("eval_role") == "final":
            expected_hash = str(frozen.get("testset_sha256") or frozen.get("testset_sha256_16") or "")
            actual_hash = file_hash(testset_path)
            if expected_hash and not actual_hash.startswith(expected_hash) and not expected_hash.startswith(actual_hash):
                print(
                    f"[ERROR] 冻结 manifest 与题库 hash 不一致: {expected_hash} != {actual_hash}",
                    file=sys.stderr,
                )
                return 3
    items = testset.get("items", [])
    if args.split:
        items = [it for it in items if it.get("split") == args.split]
    if args.qtype:
        items = [it for it in items if it.get("question_type") == args.qtype]
    review_report = None
    if args.review_report:
        review_path = Path(args.review_report)
        review_report = json.loads(review_path.read_text(encoding="utf-8"))
        if review_report.get("schema_version") != "agent-vqa-review/2.0":
            print("[ERROR] --review-report 必须使用 agent-vqa-review/2.0", file=sys.stderr)
            return 3
        statuses = {str(r.get("id")): r.get("status") for r in review_report.get("per_item", [])}
        not_approved = [str(it.get("id")) for it in items if statuses.get(str(it.get("id"))) != "approved"]
        if not_approved:
            print(f"[ERROR] {len(not_approved)} 题未同时通过自动与人工审核，拒绝运行",
                  file=sys.stderr)
            return 3
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        print("[ERROR] 题库为空 (过滤后无题)。", file=sys.stderr)
        return 2

    # 应用分片 (多 GPU 并行): 在 limit/split/qtype 过滤之后取片
    if args._shard_n > 1:
        items = items[args._shard_i :: args._shard_n]
        print(f"[bench] shard {args._shard_i}/{args._shard_n}: 本片 {len(items)} 题")

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    if args._shard_n > 1:
        run_id += f"_shard{args._shard_i}of{args._shard_n}"
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] run_id={run_id}  out_dir={out_dir}")
    print(f"[bench] configs={configs} items={len(items)} split={args.split or '*'} qtype={args.qtype or '*'}")

    # 续跑: 收集已完成 (config, qid)
    done_keys: set = set()
    done_rows: dict[tuple[str, str], dict] = {}
    raw_path = out_dir / "episodes.jsonl"
    if args.resume and raw_path.is_file():
        done_rows = load_completed_rows(raw_path)
        done_keys = set(done_rows)
        print(f"[bench] resume: 跳过 {len(done_keys)} 条已完成记录")

    print("[bench] 正在导入 app (首次会加载/预热感知 + 本地 VLM, 可能耗时数分钟)...")
    t_import = time.time()
    import app  # noqa: E402
    app.AGENT_VQA_GENERATION_BASE_SEED = int(args.generation_seed)
    app.AGENT_VQA_GENERATION_REPEAT = int(args.generation_repeat)
    if frozen:
        app.VLN_ENTROPY_TABLE = str(frozen.get("entropy_table_path", app.VLN_ENTROPY_TABLE))
        app.VLN_CONFORMAL_QHAT = float(frozen.get("qhat", app.VLN_CONFORMAL_QHAT))
        app.VLN_CONFORMAL_ALPHA = float(frozen.get("conformal_alpha", app.VLN_CONFORMAL_ALPHA))
        app.VLN_RECHECK_TEMPERATURE = float(frozen.get("temperature", app.VLN_RECHECK_TEMPERATURE))
        app.VLN_RECHECK_MIN_INFO_GAIN = float(
            frozen.get("min_info_gain", app.VLN_RECHECK_MIN_INFO_GAIN)
        )
    print(f"[bench] app ready in {time.time() - t_import:.1f}s")

    raw_fp = open(raw_path, "a" if args.resume else "w", encoding="utf-8")
    per_config: dict = {}
    reference_budget_by_qid: dict[str, int] = {}

    for cfg_name in configs:
        cfg = CONFIGS[cfg_name]
        apply_config(app, cfg)
        if frozen:
            # apply_config resets defaults; frozen values are the only allowed
            # source of final-evaluation policy hyperparameters.
            app.VLN_ENTROPY_TABLE = str(frozen.get("entropy_table_path", app.VLN_ENTROPY_TABLE))
            app.VLN_CONFORMAL_QHAT = float(frozen.get("qhat", app.VLN_CONFORMAL_QHAT))
            app.VLN_CONFORMAL_ALPHA = float(
                frozen.get("conformal_alpha", app.VLN_CONFORMAL_ALPHA)
            )
            app.VLN_RECHECK_TEMPERATURE = float(
                frozen.get("temperature", app.VLN_RECHECK_TEMPERATURE)
            )
            app.VLN_RECHECK_MIN_INFO_GAIN = float(
                frozen.get("min_info_gain", app.VLN_RECHECK_MIN_INFO_GAIN)
            )
            if cfg_name in {"A3_ENTROPY", "A3U_RAW_ENTROPY"}:
                app_module_trigger = frozen.get("entropy_trigger")
                if app_module_trigger is not None:
                    # RecheckConfig reads VLN_RECHECK_TRIGGER_THRESHOLD through
                    # the existing trigger field below.
                    app.VLN_RECHECK_THRESHOLD = float(app_module_trigger)
        matched_budget: dict[str, int] | None = None
        if cfg.get("matched_baseline"):
            if reference_budget_by_qid:
                if cfg["matched_baseline"] == "same_items":
                    matched_budget = dict(reference_budget_by_qid)
                else:
                    total_actions = sum(reference_budget_by_qid.values())
                    qids = [
                        it.get("id") or f"{it.get('tile_id','')}_{it.get('question_type','')}_{i}"
                        for i, it in enumerate(items)
                    ]
                    rng = random.Random(args.seed)
                    rng.shuffle(qids)
                    matched_budget = {qid: 0 for qid in qids}
                    for action_index in range(total_actions):
                        matched_budget[qids[action_index % len(qids)]] += 1
            elif args.matched_budget_frac >= 0:
                qids = [
                    it.get("id") or f"{it.get('tile_id','')}_{it.get('question_type','')}_{i}"
                    for i, it in enumerate(items)
                ]
                k = int(round(len(qids) * args.matched_budget_frac))
                rng = random.Random(args.seed)
                rng.shuffle(qids)
                selected = set(qids[:k])
                matched_budget = {qid: int(qid in selected) for qid in qids}
            else:
                raise RuntimeError(
                    f"{cfg_name} 需要先运行参考配置 {args.matched_reference}，"
                    "或仅在 smoke 中显式传 --matched-budget-frac"
                )
        base_effective = effective_config(app)
        print(f"\n[bench] ===== {cfg_name} ({cfg['desc']}) "
              f"evidence={cfg['evidence_level']} search={cfg['max_search']} reobs={cfg['max_reobs']} =====")
        cfg_rows: list = []
        for idx, item in enumerate(items):
            qid = item.get("id") or f"{item.get('tile_id','')}_{item.get('question_type','')}_{idx}"
            if (cfg_name, qid) in done_keys:
                cfg_rows.append(done_rows[(cfg_name, qid)])
                print(f"  [{cfg_name}] {idx + 1}/{len(items)} skip (done)")
                continue
            t0 = time.time()
            try:
                action_seed = episode_seed(args.seed, cfg_name, qid)
                # Each episode constructs a fresh RecheckController.  A constant
                # seed would repeat the same first Bernoulli draw for every item,
                # collapsing A1_RANDOM into an always/never policy.
                app.VLN_RECHECK_RANDOM_SEED = action_seed
                if matched_budget is not None:
                    app.AGENT_VQA_MAX_REOBSERVATIONS = int(matched_budget.get(qid, 0))
                else:
                    app.AGENT_VQA_MAX_REOBSERVATIONS = int(cfg["max_reobs"])
                start = item["start"]
                report = app.run_agent_vqa_episode_headless(
                    item["question"], start, item=item, source="bench",
                )
                report["config"] = cfg_name
                report["wall_s"] = round(time.time() - t0, 2)
            except Exception as exc:
                report = {"ok": False, "error": f"crash: {exc}",
                          "question": item["question"], "config": cfg_name,
                          "wall_s": round(time.time() - t0, 2)}
                traceback.print_exc()
            row = score_episode(report, item)
            row["action_seed"] = episode_seed(args.seed, cfg_name, qid)
            row["answer_mode"] = base_effective["answer_mode"]
            row["evidence_level"] = base_effective["evidence_level"]
            cfg_rows.append(row)
            raw_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_fp.flush()
            mark = "OK " if row.get("correct") else ("abs" if row.get("abstain") else "miss")
            print(f"  [{cfg_name}] {idx + 1}/{len(items)} {mark} "
                  f"pred={row['pred_answer']!r} gt={row['gt_answer']!r} "
                  f"steps={row['n_steps']} {row['wall_s']}s :: {item['question'][:24]}")
        # 续跑读回的旧行可能缺字段，按本配置的实际运行开关补齐，保证同一配置
        # 的所有行都带 answer_mode，报告端才能校验是否混跑。
        for row in cfg_rows:
            row.setdefault("answer_mode", base_effective["answer_mode"])
            row.setdefault("evidence_level", base_effective["evidence_level"])
        per_config[cfg_name] = {
            "agg": aggregate(cfg_rows), "switches": cfg,
            "effective_switches": base_effective,
        }
        if cfg_name == args.matched_reference:
            reference_budget_by_qid = {
                str(row.get("qid")): int(row.get("n_reobservations", 0) or 0)
                for row in cfg_rows
            }

    raw_fp.close()

    execution_errors = sum(
        int((rec.get("agg") or {}).get("failure_taxonomy", {}).get("execution_error", 0))
        for rec in per_config.values()
    )
    valid_for_analysis = execution_errors == 0

    results = {
        "run_id": run_id,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "env": env_snapshot(),
        "testset": str(args.testset),
        "testset_sha256_16": file_hash(testset_path),
        "n_items": len(items),
        "valid_for_analysis": valid_for_analysis,
        "n_execution_errors": execution_errors,
        "configs": {c: per_config.get(c, {}) for c in configs},
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # manifest (计划 9.4): env + git + 数据 hash + prompt + 阈值
    manifest = {
        "run_id": run_id,
        "env": results["env"],
        "testset": results["testset"],
        "testset_sha256_16": results["testset_sha256_16"],
        "review_report": str(args.review_report or ""),
        "review_report_sha256_16": file_hash(Path(args.review_report)) if args.review_report else "",
        "configs": {c: CONFIGS[c] for c in configs},
        "frozen_manifest": str(args.frozen_manifest or ""),
        "frozen_manifest_sha256_16": file_hash(Path(args.frozen_manifest)) if args.frozen_manifest else "",
        "valid_for_analysis": valid_for_analysis,
        "n_execution_errors": execution_errors,
        "agent_vqa_confidence_threshold": os.environ.get("AGENT_VQA_CONFIDENCE_THRESHOLD", "0.5"),
        "agent_vqa_evidence_levels": {c: CONFIGS[c]["evidence_level"] for c in configs},
        "agent_vqa_answer_modes": {
            c: CONFIGS[c].get("answer_mode", "hybrid") for c in configs
        },
        "vlm_system_prompt": AGENT_VQA_SYSTEM_PROMPT,
        "vlm_system_prompt_sha256": hashlib.sha256(
            AGENT_VQA_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "generation_seed_protocol": "sha256(base_seed,qid,repeat,step,call_role)",
        "generation_base_seed": int(args.generation_seed),
        "generation_repeat": int(args.generation_repeat),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    table = md_table(per_config)
    summary_md = (
        f"# Agent-VQA 评测成绩单 — {run_id}\n\n"
        f"- 题库: `{args.testset}` ({len(items)} 题, split={args.split or '*'}, "
        f"qtype={args.qtype or '*'})\n"
        f"- 设备: {results['env'].get('gpu' if 'gpu' in results['env'] else 'perception_device', 'cpu')}\n"
        f"- git: {results['env'].get('git_commit', '')[:12]}"
        f"{' (dirty)' if results['env'].get('git_dirty') else ''}\n\n"
        f"## 主消融表 (E1-E5)\n\n{table}\n\n"
        f"> accuracy=全题口径 (弃答算错); answer_acc=仅作答题; abstain_rate 越低越好;\n"
        f"> n_reobservations=策略实际触发的重观测次数; n_reobserve_skips=策略被询问但决定跳过。\n"
        f"> 两者同时为 0 且 max_reobs>0，说明动作通道未接通，结果不可用于主动 VQA 结论。\n"
        f"> 在线字段与离线评分严格分离; correct/corrected/harmed 仅由 GT 离线计算。\n"
    )
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print("\n" + table)
    print(f"\n[bench] 完成。结果目录: {out_dir}")
    print(f"[bench]   - results.json   (机器可读全量指标)")
    print(f"[bench]   - summary.md      (成绩单表格)")
    print(f"[bench]   - episodes.jsonl  (每条 episode 明细)")
    print(f"[bench]   - manifest.json   (env/git/hash/prompt/阈值)")
    if not valid_for_analysis:
        print(f"[bench] INVALID: {execution_errors} 条执行错误；本运行不得进入统计分析。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
