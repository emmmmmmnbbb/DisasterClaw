#!/usr/bin/env python
"""
bench_exception_rate.py — Bench-C: 100 次连续任务的异常事件率统计。

按 --seed 抽样 100 个任务（默认 T1-T5 等概率），逐次执行并把结果分类：

    flight_timeout      action_result.success=False 且 message 包含 timeout/超时/未抵达，
                        或单步 wall_ms > FLIGHT_BUDGET_S * 1000
    perception_failure  perception_result.degraded=True 或 detect_disaster step 失败
    report_format_error ai_execution_report 缺 summary / steps，或 summary 含 "LLM 不可用" /
                        "部分失败"，或步骤 wall_ms 缺失
    success             以上都没命中

输出: runs/benchmarks/<run-id>/exception.json
    - 每次任务的分类与原始时间戳
    - 4 类计数 + 总成功率
    - 累计成功率 CDF（用于绘图）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import socketio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    SLA,
    dump_json,
    ensure_run_dir,
    env_snapshot,
    get_tasks,
    now_run_id,
    seeded_task_sequence,
)


FLIGHT_BUDGET_S = 20.0
TASK_TIMEOUT_S = 90.0


class ExceptionClassifier:
    """Subscribes to socket events for ONE task and classifies the outcome."""

    def __init__(self, sio: socketio.Client):
        self.sio = sio
        self.reset()

        @sio.on("ai_plan_result")
        def _on_plan(data):
            self.plan_payload = data

        @sio.on("action_result")
        def _on_action_result(data):
            self.action_results.append(data)

        @sio.on("perception_result")
        def _on_perception(data):
            self.perception_results.append(data)

        @sio.on("ai_execution_report")
        def _on_ai_report(data):
            self.report = data
            self.report_arrived = True

        @sio.on("execution_report")
        def _on_exec_report(data):
            self.report = data
            self.report_arrived = True

    def reset(self):
        self.plan_payload: dict | None = None
        self.action_results: list[dict] = []
        self.perception_results: list[dict] = []
        self.report: dict | None = None
        self.report_arrived = False
        self.t_submit: float = 0.0

    def submit(self, task_text: str) -> None:
        self.reset()
        self.t_submit = time.time()
        self.sio.emit("ai_task", {"task": task_text})

    def wait(self, timeout_s: float = TASK_TIMEOUT_S) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline and not self.report_arrived:
            time.sleep(0.05)

    def classify(self) -> dict:
        elapsed_s = time.time() - self.t_submit
        report = self.report or {}
        summary = (report.get("summary") or "").lower()

        # ---- 1. flight_timeout ----
        timeout_hit = False
        timeout_reasons = []
        for ar in self.action_results:
            r = ar.get("result", {}) or {}
            msg = (r.get("message") or "").lower()
            if not r.get("success") and any(
                kw in msg for kw in ("timeout", "超时", "未抵达", "未到达", "abort")
            ):
                timeout_hit = True
                timeout_reasons.append(msg)
            wall_ms = ar.get("wall_ms")
            if wall_ms is not None and wall_ms > FLIGHT_BUDGET_S * 1000:
                if ar.get("action") in ("fly_to_geo", "fly_relative", "hover"):
                    timeout_hit = True
                    timeout_reasons.append(f"{ar.get('action')} wall_ms={wall_ms}>{int(FLIGHT_BUDGET_S*1000)}")
        if elapsed_s > TASK_TIMEOUT_S - 1 and not self.report_arrived:
            timeout_hit = True
            timeout_reasons.append("hard task timeout")

        if timeout_hit:
            return {
                "category": "flight_timeout",
                "elapsed_s": elapsed_s,
                "reasons": timeout_reasons,
            }

        # ---- 2. perception_failure ----
        perc_fail = False
        perc_reasons = []
        for p in self.perception_results:
            if p.get("degraded"):
                perc_fail = True
                perc_reasons.append(f"degraded: {p.get('degraded_reason')}")
        for ar in self.action_results:
            if ar.get("action") == "detect_disaster" and not (ar.get("result") or {}).get("success"):
                perc_fail = True
                perc_reasons.append(f"detect_disaster failed: {(ar.get('result') or {}).get('message')}")
        if perc_fail:
            return {
                "category": "perception_failure",
                "elapsed_s": elapsed_s,
                "reasons": perc_reasons,
            }

        # ---- 3. report_format_error ----
        fmt_reasons = []
        if not self.report_arrived:
            fmt_reasons.append("no execution_report received")
        elif "summary" not in report or "steps" not in report:
            fmt_reasons.append("report missing summary/steps")
        elif "llm 不可用" in summary or "部分失败" in summary:
            fmt_reasons.append(f"summary marker: {report.get('summary')}")
        # detect missing wall_ms instrumentation (means an old/incompatible backend)
        elif self.action_results and all(ar.get("wall_ms") is None for ar in self.action_results):
            fmt_reasons.append("backend not emitting wall_ms (instrumentation lost)")
        if fmt_reasons:
            return {
                "category": "report_format_error",
                "elapsed_s": elapsed_s,
                "reasons": fmt_reasons,
            }

        # ---- 4. success ----
        return {
            "category": "success",
            "elapsed_s": elapsed_s,
            "n_actions": len(self.action_results),
            "n_perception": len(self.perception_results),
            "summary": report.get("summary"),
        }


def run_bench(args: argparse.Namespace) -> dict:
    tasks = get_tasks()
    sequence = seeded_task_sequence(args.total, seed=args.seed, ids=list(tasks.keys()))

    sio = socketio.Client(reconnection=False, logger=False, engineio_logger=False)
    classifier = ExceptionClassifier(sio)
    # Werkzeug dev server only supports long-polling for socket.io
    sio.connect(args.url, transports=["polling"])
    print(f"[bench-c] connected to {args.url}, total={args.total} seed={args.seed}")

    classifications: list[dict] = []
    counts = {"flight_timeout": 0, "perception_failure": 0, "report_format_error": 0, "success": 0}

    cumulative_success: list[int] = []
    n_success = 0

    try:
        for i, tid in enumerate(sequence, start=1):
            t = tasks[tid]
            print(f"\n[{i}/{args.total}] task={tid} {t['name']}")
            classifier.submit(t["task_text"])
            classifier.wait(timeout_s=args.timeout)
            cls = classifier.classify()
            cls["index"] = i
            cls["task_id"] = tid
            cls["task_name"] = t["name"]
            classifications.append(cls)
            counts[cls["category"]] = counts.get(cls["category"], 0) + 1
            if cls["category"] == "success":
                n_success += 1
            cumulative_success.append(n_success)
            print(
                f"   -> {cls['category']} ({cls.get('elapsed_s', 0):.1f}s)"
                + (f"   reasons={cls.get('reasons')}" if cls.get('reasons') else "")
            )
            time.sleep(args.cooldown)
    finally:
        try:
            sio.disconnect()
        except Exception:
            pass

    total = len(classifications)
    rates = {k: v / max(total, 1) for k, v in counts.items()}
    cdf = [
        {"index": i + 1, "cum_success": cs, "cum_success_rate": cs / (i + 1)}
        for i, cs in enumerate(cumulative_success)
    ]

    return {
        "bench": "exception_rate",
        "args": {
            "url": args.url,
            "total": args.total,
            "seed": args.seed,
            "timeout": args.timeout,
            "cooldown": args.cooldown,
        },
        "sla_ms": SLA,
        "flight_budget_s": FLIGHT_BUDGET_S,
        "task_timeout_s": TASK_TIMEOUT_S,
        "env": env_snapshot(),
        "counts": counts,
        "rates": rates,
        "success_rate": rates.get("success", 0.0),
        "classifications": classifications,
        "cdf": cdf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:5011")
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cooldown", type=float, default=1.5)
    ap.add_argument("--timeout", type=float, default=TASK_TIMEOUT_S)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or now_run_id()
    out_dir = ensure_run_dir(run_id)
    print(f"[bench-c] run_id={run_id} out_dir={out_dir}")

    result = run_bench(args)
    result["run_id"] = run_id

    out_path = out_dir / "exception.json"
    dump_json(out_path, result)
    print(f"\n[bench-c] wrote {out_path}")
    print(f"[bench-c] success_rate={result['success_rate']*100:.1f}%  counts={result['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
