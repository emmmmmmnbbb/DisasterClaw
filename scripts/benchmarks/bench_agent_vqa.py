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
        --configs V0_RAW,A0_HOLD,A3_ENTROPY --limit 8 --split test
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))  # 必须在 import app 之前

DEFAULT_TESTSET = BACKEND / "data" / "benchmarks" / "agent_vqa_testset.json"
RUNS_DIR = REPO_ROOT / "runs" / "benchmarks" / "cja_agent_vqa"

# 配置 → (evidence_level, max_search, max_reobs, recheck_trigger[, recheck_extra][, oracle])
# evidence_level: raw / struct / state  (计划 9.1 图像/状态列)
# max_search / max_reobs: 0 = 不可移动 / 不可重观测
# recheck_trigger: 控制 reobserve_fn 的 RecheckController 触发模式
CONFIGS = {
    "V0_RAW": {"evidence_level": "raw", "max_search": 0, "max_reobs": 0,
               "recheck_trigger": "threshold", "desc": "静态 VLM 基线 (仅图像)"},
    "V1_STRUCT": {"evidence_level": "struct", "max_search": 0, "max_reobs": 0,
                  "recheck_trigger": "threshold", "desc": "图像 + 结构化感知"},
    "V2_STATE": {"evidence_level": "state", "max_search": 0, "max_reobs": 0,
                 "recheck_trigger": "threshold", "desc": "V1 + STMR + 历史"},
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
    "A4_CONFORMAL": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                     "recheck_trigger": "conformal",
                     "recheck_extra": {"uncertainty_mode": "entropy"},
                     "desc": "共形集合不确定性对照"},
    "A5_EXPECTED": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
                    "recheck_trigger": "info_gain",
                    "recheck_extra": {"uncertainty_mode": "entropy"},
                    "desc": "验证集期望收益 (泄漏安全条件策略)"},
    "O_REF": {"evidence_level": "state", "max_search": 6, "max_reobs": 2,
              "recheck_trigger": "threshold", "offline_only": True,
              "desc": "离线 hindsight oracle 参照 (由 A0/A2 配对结果合成, 禁止部署)"},
}


def apply_config(app, cfg: dict) -> None:
    """运行期切换 app 模块级 Agent-VQA 与 recheck 开关。未显式设置的字段回落默认值。"""
    if cfg.get("offline_only"):
        raise ValueError("O_REF 是 report_agent_vqa.py 从 A0_HOLD/A2_ALWAYS 合成的离线参照")
    app.AGENT_VQA_EVIDENCE_LEVEL = cfg["evidence_level"]
    app.AGENT_VQA_MAX_SEARCH_STEPS = int(cfg["max_search"])
    app.AGENT_VQA_MAX_REOBSERVATIONS = int(cfg["max_reobs"])
    # reobserve_fn 内部新建 RecheckController 时读这些 app 级开关
    app.VLN_RECHECK_TRIGGER = cfg["recheck_trigger"]
    app.VLN_UNCERTAINTY_MODE = "heuristic"
    extra = cfg.get("recheck_extra") or {}
    app.VLN_UNCERTAINTY_MODE = extra.get("uncertainty_mode", "heuristic")
    app.VLN_RECHECK_MIN_INFO_GAIN = float(extra.get("min_info_gain", 0.05))
    app.VLN_RECHECK_RANDOM_PROB = float(extra.get("random_prob", 0.5))
    app.VLN_RECHECK_RANDOM_SEED = int(extra.get("random_seed", 0))
    app.VLN_ENTROPY_TABLE = str(extra.get("entropy_table_path", app.VLN_ENTROPY_TABLE))
    app.VLN_CONFORMAL_QHAT = float(extra.get("conformal_qhat", 0.9))
    app.VLN_CONFORMAL_ALPHA = float(extra.get("conformal_alpha", 0.1))
    # 真值 oracle 不在在线控制器中运行；O_REF 只能由报告脚本离线合成。
    app.AGENT_VQA_ORACLE = False


def effective_config(app) -> dict:
    """记录实际传入控制器的开关，防止消融名称与运行行为不一致。"""
    return {
        "evidence_level": app.AGENT_VQA_EVIDENCE_LEVEL,
        "max_search": app.AGENT_VQA_MAX_SEARCH_STEPS,
        "max_reobs": app.AGENT_VQA_MAX_REOBSERVATIONS,
        "trigger_mode": app.VLN_RECHECK_TRIGGER,
        "uncertainty_mode": app.VLN_UNCERTAINTY_MODE,
        "min_info_gain": app.VLN_RECHECK_MIN_INFO_GAIN,
        "random_prob": app.VLN_RECHECK_RANDOM_PROB,
        "random_seed": app.VLN_RECHECK_RANDOM_SEED,
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
    return h.hexdigest()[:16]


def env_snapshot() -> dict:
    import importlib.metadata
    info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": os.environ.get("DATASET_MODE", "xbd"),
        "perception_device": os.environ.get("PERCEPTION_DEVICE", "cuda"),
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "llm_model": os.environ.get("LLM_MODEL", ""),
        "vlm_model": os.environ.get("VLM_MODEL", ""),
        "agent_vqa_confidence_threshold": os.environ.get("AGENT_VQA_CONFIDENCE_THRESHOLD", "0.5"),
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
    gt_answer = item.get("answer", "")
    ans = (report or {}).get("answer") or {}
    pred = ans.get("answer", "")
    abstain = bool(ans.get("abstain"))
    decision = ans.get("decision", "")
    correct = (not abstain) and bool(pred) and pred == gt_answer
    # 弃答是否"应该": GT 为否定类 (如 presence=否 / count=0) 且模型 abstain 视为合理保守
    abstain_should = abstain and gt_answer in {"否", "0"}
    traj = (report or {}).get("trajectory", [])
    preds = [t.get("candidate_answer") for t in traj if t.get("candidate_answer")]
    flipped = len(set(preds)) > 1
    reobserve_pairs = []
    for i, step in enumerate(traj[:-1]):
        if step.get("decision") != "reobserve":
            continue
        before = step.get("candidate_answer") or ""
        after = traj[i + 1].get("candidate_answer") or ""
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
    difficulty = item.get("difficulty", "")
    difficulty_band = difficulty.get("distance", "") if isinstance(difficulty, dict) else difficulty
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
        "pred_answer": pred,
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
    for r in rows:
        if r.get("ok") and not r.get("correct"):
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
    cols = ["config", "n", "accuracy", "abstain_rate", "flip_rate", "n_steps_mean"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for name, rec in per_config.items():
        a = rec.get("agg", {})
        lines.append("| " + " | ".join(str(a.get(c, "")) for c in cols[1:]) + " |")
        # 把 config 名插到行首
        lines[-1] = f"| {name} " + lines[-1][1:]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent-VQA 评测 (E1-E5 消融成绩单)")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--configs", default="V0_RAW,A0_HOLD,A3_ENTROPY")
    ap.add_argument("--limit", type=int, default=0, help="每个配置最多跑前 N 题 (0=全部)")
    ap.add_argument("--split", default="", help="只跑某 split train/val/test (空=全部)")
    ap.add_argument("--qtype", default="", help="只跑某题型 presence/damage/count/spatial (空=全部)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true",
                    help="续跑: 跳过 episodes.jsonl 已有的 (config,qid)")
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
    items = testset.get("items", [])
    if args.split:
        items = [it for it in items if it.get("split") == args.split]
    if args.qtype:
        items = [it for it in items if it.get("question_type") == args.qtype]
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
    raw_path = out_dir / "episodes.jsonl"
    if args.resume and raw_path.is_file():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                qid = rec.get("qid") or ""
                cfg = rec.get("config") or ""
                if qid and cfg:
                    done_keys.add((cfg, qid))
            except Exception:
                pass
        print(f"[bench] resume: 跳过 {len(done_keys)} 条已完成记录")

    print("[bench] 正在导入 app (首次会加载/预热感知 + 本地 VLM, 可能耗时数分钟)...")
    t_import = time.time()
    import app  # noqa: E402
    print(f"[bench] app ready in {time.time() - t_import:.1f}s")

    raw_fp = open(raw_path, "a" if args.resume else "w", encoding="utf-8")
    per_config: dict = {}

    for cfg_name in configs:
        cfg = CONFIGS[cfg_name]
        apply_config(app, cfg)
        print(f"\n[bench] ===== {cfg_name} ({cfg['desc']}) "
              f"evidence={cfg['evidence_level']} search={cfg['max_search']} reobs={cfg['max_reobs']} =====")
        cfg_rows: list = []
        for idx, item in enumerate(items):
            qid = item.get("id") or f"{item.get('tile_id','')}_{item.get('question_type','')}_{idx}"
            if (cfg_name, qid) in done_keys:
                print(f"  [{cfg_name}] {idx + 1}/{len(items)} skip (done)")
                continue
            t0 = time.time()
            try:
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
            cfg_rows.append(row)
            raw_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_fp.flush()
            mark = "OK " if row.get("correct") else ("abs" if row.get("abstain") else "miss")
            print(f"  [{cfg_name}] {idx + 1}/{len(items)} {mark} "
                  f"pred={row['pred_answer']!r} gt={row['gt_answer']!r} "
                  f"steps={row['n_steps']} {row['wall_s']}s :: {item['question'][:24]}")
        per_config[cfg_name] = {
            "agg": aggregate(cfg_rows), "switches": cfg,
            "effective_switches": effective_config(app),
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
        "configs": {c: CONFIGS[c] for c in configs},
        "valid_for_analysis": valid_for_analysis,
        "n_execution_errors": execution_errors,
        "agent_vqa_confidence_threshold": os.environ.get("AGENT_VQA_CONFIDENCE_THRESHOLD", "0.5"),
        "agent_vqa_evidence_levels": {c: CONFIGS[c]["evidence_level"] for c in configs},
        "vlm_system_prompt": "见 backend/vlm_analyzer.py:AGENT_VQA_SYSTEM_PROMPT",
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
        f"> flip_rate=重观测后答案翻转比例; n_steps_mean=平均动作步数 (越少越经济)。\n"
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
