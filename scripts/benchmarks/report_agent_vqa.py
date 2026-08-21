#!/usr/bin/env python3
"""scripts/benchmarks/report_agent_vqa.py — Agent-VQA 汇总报告 (计划 9.2)

读取一个或多个 bench_agent_vqa.py run 目录的 episodes.jsonl → 产出:
  - aggregate.json: 总体 / 分题型 / 分事件 / 分难度指标
  - paired_tests.json: 配对配置差值 + bootstrap 95% CI (计划 E1 配对统计)
  - event_breakdown.csv: 分事件指标表
  - failure_taxonomy.csv: fallback / 错误类型统计

配对统计要求同一题目在两个配置下都跑过; 按题目 id 配对, 对正确性差值做
非参数 bootstrap (重采样题目), 报告均值差与 95% CI, excludes_zero 标记显著性。

用法:
    python scripts/benchmarks/report_agent_vqa.py \\
        --runs runs/benchmarks/cja_agent_vqa/20260101_120000 \\
        --out runs/benchmarks/cja_agent_vqa/20260101_120000/reports
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_rows(run_dir: Path) -> list[dict]:
    fp = run_dir / "episodes.jsonl"
    if not fp.is_file():
        return []
    rows = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(rows):
    """总体 + 分题型 + 分事件 + 分难度 + 翻转/纠错/损害 + fallback。"""
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
        "accuracy": round(len(correct) / n, 4),
        "answer_acc": round(len(correct) / len(answered), 4) if answered else None,
        "flip_rate": round(len(flipped) / n, 4),
        "fallback_rate": round(len(fallback) / n, 4),
        "n_steps_mean": round(sum(r.get("n_steps", 0) for r in rows) / n, 2),
        "confidence_mean": _mean([r.get("confidence") for r in rows]),
        "n_reobservations": sum(int(r.get("n_reobservations", 0) or 0) for r in rows),
        "n_reobserve_skips": sum(int(r.get("n_reobserve_skips", 0) or 0) for r in rows),
    }
    # 分题型
    by_type = {}
    for qt in ("presence", "damage", "count", "spatial"):
        sub = [r for r in rows if r.get("question_type") == qt]
        if sub:
            by_type[qt] = {
                "n": len(sub),
                "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4),
                "abstain_rate": round(sum(1 for r in sub if r.get("abstain")) / len(sub), 4),
                "flip_rate": round(sum(1 for r in sub if r.get("flipped")) / len(sub), 4),
            }
    agg["by_question_type"] = by_type
    # 分事件
    by_event = {}
    for d in sorted({r.get("disaster") for r in rows if r.get("disaster")}):
        sub = [r for r in rows if r.get("disaster") == d]
        by_event[d] = {
            "n": len(sub),
            "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4),
            "abstain_rate": round(sum(1 for r in sub if r.get("abstain")) / len(sub), 4),
        }
    agg["by_event"] = by_event
    # 分难度
    by_diff = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r.get("difficulty") == d]
        if sub:
            by_diff[d] = {
                "n": len(sub),
                "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4),
            }
    agg["by_difficulty"] = by_diff
    # 重观测前后必须用相邻观测对判定，不能用“最终是否正确”倒推纠错/损害。
    triggered = [r for r in rows if int(r.get("n_reobservations", 0) or 0) > 0]
    flip_corrected = sum(int(r.get("n_correcting_reobservations", 0) or 0) for r in rows)
    flip_harmed = sum(int(r.get("n_harming_reobservations", 0) or 0) for r in rows)
    n_reobservations = sum(int(r.get("n_reobservations", 0) or 0) for r in rows)
    agg["flip_matrix"] = {
        "n_flip": len(flipped),
        "n_triggered": len(triggered),
        "n_reobservations": n_reobservations,
        "n_corrected": flip_corrected,
        "n_harmed": flip_harmed,
        "n_neutral": n_reobservations - flip_corrected - flip_harmed,
    }
    # 失败分类
    fail = defaultdict(int)
    for r in rows:
        if r.get("ok") and not r.get("correct"):
            if r.get("reason_code") == "invalid_output":
                fail["invalid_output"] += 1
            else:
                fail["abstain" if r.get("abstain") else "wrong_answer"] += 1
        elif not r.get("ok"):
            fail["execution_error"] += 1
    agg["failure_taxonomy"] = dict(fail)
    return agg


def paired_bootstrap_correctness(rows_a, rows_b, n_boot=2000, seed=42):
    """配对正确性差值 bootstrap (计划 E1 配对统计)。

    rows_a / rows_b 必须按同一 qid 配对; 对 per-item correct(0/1) 差值做
    非参数 bootstrap 重采样, 返回均值差 + 95% CI + excludes_zero。
    """
    by_qid_a = {r.get("qid"): r for r in rows_a}
    pairs = []
    for rb in rows_b:
        ra = by_qid_a.get(rb.get("qid"))
        if ra is None:
            continue
        ca = 1.0 if ra.get("correct") else 0.0
        cb = 1.0 if rb.get("correct") else 0.0
        pairs.append(cb - ca)  # b - a
    if not pairs:
        return {"n_paired": 0}
    rng = random.Random(seed)
    n = len(pairs)
    point = sum(pairs) / n
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(pairs[i] for i in idx) / n)
    draws.sort()
    lo = draws[int(0.025 * n_boot)]
    hi = draws[min(int(0.975 * n_boot), n_boot - 1)]
    return {
        "n_paired": n,
        "mean_difference": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n_boot": n_boot,
    }


def budget_utility_curve(rows):
    """预算效用曲线: 按动作步数分桶的累计准确率 (计划 9.2)。"""
    by_steps = defaultdict(list)
    for r in rows:
        s = r.get("n_steps", 0)
        by_steps[s].append(1.0 if r.get("correct") else 0.0)
    curve = []
    for s in sorted(by_steps):
        vals = by_steps[s]
        curve.append({"n_steps": s, "n": len(vals),
                      "accuracy": round(sum(vals) / len(vals), 4)})
    return curve


def risk_coverage_curve(rows):
    """风险覆盖曲线: 按置信度阈值分桶的覆盖率/准确率 (计划 9.2)。

    coverage = 置信度 >= 阈值的题目比例; accuracy = 该子集准确率。
    """
    confs = [r.get("confidence") for r in rows if r.get("confidence") is not None]
    if not confs:
        return []
    out = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        sub = [r for r in rows if (r.get("confidence") or 0) >= thr]
        if not sub:
            continue
        out.append({
            "threshold": thr,
            "coverage": round(len(sub) / len(rows), 4),
            "accuracy": round(sum(1 for r in sub if r.get("correct")) / len(sub), 4),
            "n": len(sub),
        })
    return out


def hindsight_oracle_rows(hold_rows, always_rows):
    """用成对 A0/A2 结果构造离线观测选择 oracle；绝不回送在线控制器。"""
    hold = {r.get("qid"): r for r in hold_rows}
    always = {r.get("qid"): r for r in always_rows}
    rows = []
    correctable = harmful = both_correct = neither_correct = 0
    for qid in sorted(set(hold) & set(always)):
        h, a = hold[qid], always[qid]
        hc, ac = bool(h.get("correct")), bool(a.get("correct"))
        if not hc and ac:
            chosen, source = a, "A2_ALWAYS"
            correctable += 1
        elif hc and not ac:
            chosen, source = h, "A0_HOLD"
            harmful += 1
        elif hc and ac:
            chosen = min((h, a), key=lambda r: (r.get("n_steps", 0), -(r.get("confidence") or 0)))
            source = "A0_HOLD" if chosen is h else "A2_ALWAYS"
            both_correct += 1
        else:
            chosen = max((h, a), key=lambda r: (not r.get("abstain"), r.get("confidence") or 0))
            source = "A0_HOLD" if chosen is h else "A2_ALWAYS"
            neither_correct += 1
        rec = dict(chosen)
        rec["config"] = "O_REF"
        rec["oracle_source"] = source
        rec["oracle_offline_only"] = True
        rows.append(rec)
    diagnostics = {
        "definition": "GT-informed hindsight selection between paired A0_HOLD and A2_ALWAYS outcomes",
        "online_deployable": False,
        "n_paired": len(rows),
        "n_correctable": correctable,
        "n_harmful": harmful,
        "n_both_correct": both_correct,
        "n_neither_correct": neither_correct,
    }
    return rows, diagnostics


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent-VQA 汇总报告 (配对统计 + 分层指标)")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="一个或多个 bench_agent_vqa run 目录 (含 episodes.jsonl)")
    ap.add_argument("--out", default="", help="报告输出目录 (默认 <首个run>/reports)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    for r in run_dirs:
        if not r.is_dir():
            print(f"[ERROR] run 目录不存在: {r}", file=sys.stderr)
            return 2
        result_path = r / "results.json"
        if not result_path.is_file():
            print(f"[ERROR] run 缺少 results.json，无法验证有效性: {r}", file=sys.stderr)
            return 2
        try:
            result_meta = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ERROR] 无法解析 {result_path}: {exc}", file=sys.stderr)
            return 2
        if result_meta.get("valid_for_analysis") is not True:
            print(f"[ERROR] run 已标记 valid_for_analysis=false，拒绝聚合: {r}", file=sys.stderr)
            return 2

    out_dir = Path(args.out) if args.out else (run_dirs[0] / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按配置聚合所有 run 的行
    by_config = defaultdict(list)
    for rd in run_dirs:
        for row in load_rows(rd):
            by_config[row.get("config", "")].append(row)

    oracle_diagnostics = {"available": False, "reason": "requires paired A0_HOLD and A2_ALWAYS"}
    if by_config.get("A0_HOLD") and by_config.get("A2_ALWAYS"):
        oracle_rows, oracle_diagnostics = hindsight_oracle_rows(
            by_config["A0_HOLD"], by_config["A2_ALWAYS"],
        )
        if oracle_rows:
            by_config["O_REF"] = oracle_rows
            oracle_diagnostics["available"] = True

    configs = sorted(by_config)
    if not configs:
        print("[ERROR] 没有可汇总的 episode 数据。", file=sys.stderr)
        return 2

    # aggregate.json
    aggregate_all = {c: aggregate(by_config[c]) for c in configs}
    (out_dir / "aggregate.json").write_text(
        json.dumps(aggregate_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "oracle_diagnostics.json").write_text(
        json.dumps(oracle_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    # paired_tests.json: 所有配置两两配对
    paired = {}
    for i, a in enumerate(configs):
        for b in configs[i + 1:]:
            paired[f"{b}_vs_{a}"] = paired_bootstrap_correctness(
                by_config[a], by_config[b], n_boot=args.n_boot, seed=args.seed)
    (out_dir / "paired_tests.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")

    # event_breakdown.csv
    ev_rows = []
    for c in configs:
        for ev, rec in (by_config[c] and aggregate(by_config[c]).get("by_event", {}) or {}).items():
            ev_rows.append([c, ev, rec["n"], rec["accuracy"], rec["abstain_rate"]])
    write_csv(out_dir / "event_breakdown.csv",
              ["config", "event", "n", "accuracy", "abstain_rate"], ev_rows)

    # failure_taxonomy.csv
    fail_rows = []
    for c in configs:
        for k, v in aggregate(by_config[c]).get("failure_taxonomy", {}).items():
            fail_rows.append([c, k, v])
    write_csv(out_dir / "failure_taxonomy.csv",
              ["config", "failure_type", "count"], fail_rows)

    # 预算效用 + 风险覆盖曲线 (每个配置)
    curves = {c: {
        "budget_utility": budget_utility_curve(by_config[c]),
        "risk_coverage": risk_coverage_curve(by_config[c]),
    } for c in configs}
    (out_dir / "curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台速览
    print(f"[report] 配置: {configs}")
    print(f"[report] 指标 (accuracy / abstain_rate / flip_rate):")
    for c in configs:
        a = aggregate_all[c]
        print(f"  {c}: n={a['n']} acc={a['accuracy']} abst={a['abstain_rate']} "
              f"flip={a['flip_rate']} steps={a['n_steps_mean']}")
    print(f"\n[report] 配对显著性 (excludes_zero=True 即 95% CI 不含 0):")
    for k, v in paired.items():
        if v.get("n_paired"):
            print(f"  {k}: diff={v['mean_difference']} CI={v['ci95']} "
                  f"sig={v['excludes_zero']} (n={v['n_paired']})")
    print(f"\n[report] 完成。报告目录: {out_dir}")
    print(f"[report]   - aggregate.json / paired_tests.json / curves.json")
    print(f"[report]   - event_breakdown.csv / failure_taxonomy.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
