#!/usr/bin/env python
"""
bench_task_latency.py — Bench-A: 5 个典型巡查任务的全链路毫秒耗时。

通过 python-socketio 客户端连后端，对每个任务：
    1) 提交 ai_task （或 manual sequence，用 --bypass-planner）
    2) 订阅 task_started / ai_plan_result / action_result / perception_result /
       ai_execution_report / execution_report 事件
    3) 提取 5 个时间戳：
        t_submit                   客户端发出指令的时刻
        t_plan_done (ai 模式)      LLM 规划完成
        t_first_action             第一个 action_result 到达
        t_first_perception         第一帧 perception_result 到达
        t_report                   ai_execution_report / execution_report 到达
    4) 派生 4 个阶段耗时：
        plan_ms, fly_ms, perception_ms, report_ms
    5) 每个任务跑 N=10 次（先 1 次 warm-up 不计），输出 mean / p50 / p95。

用法:
    python scripts/benchmarks/bench_task_latency.py --run-id 20260507_bench
        --tasks T1 T2 T3 T4 T5 --repeats 10
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
    short_action_chain,
    stat_summary,
)


class TaskRunner:
    """Single-shot bench client: submits one task, blocks until report or timeout."""

    def __init__(self, url: str, timeout_s: float = 120.0):
        self.url = url
        self.timeout_s = timeout_s
        self.sio = socketio.Client(reconnection=False, logger=False, engineio_logger=False)
        self._reset()

        @self.sio.event
        def connect():
            pass

        @self.sio.on("task_started")
        def _on_task_started(data):
            self.events["task_started"] = time.time_ns()

        @self.sio.on("ai_plan_result")
        def _on_plan(data):
            self.events["ai_plan_result"] = time.time_ns()
            self.plan_payload = data

        @self.sio.on("action_result")
        def _on_action_result(data):
            now = time.time_ns()
            self.action_results.append({"ts_ns": now, "data": data})
            if "first_action" not in self.events:
                self.events["first_action"] = now

        @self.sio.on("perception_result")
        def _on_perception(data):
            now = time.time_ns()
            self.perception_results.append({"ts_ns": now, "data": data})
            if "first_perception" not in self.events:
                self.events["first_perception"] = now

        @self.sio.on("ai_execution_report")
        def _on_ai_report(data):
            self.events["report"] = time.time_ns()
            self.report = data

        @self.sio.on("execution_report")
        def _on_exec_report(data):
            self.events["report"] = time.time_ns()
            self.report = data

    def _reset(self):
        self.events: dict[str, int] = {}
        self.action_results: list[dict] = []
        self.perception_results: list[dict] = []
        self.plan_payload: dict | None = None
        self.report: dict | None = None

    def connect(self):
        # Werkzeug's dev server does NOT speak the websocket upgrade — force
        # long polling so we behave the same way the frontend does.
        self.sio.connect(self.url, transports=["polling"])

    def disconnect(self):
        try:
            self.sio.disconnect()
        except Exception:
            pass

    def run_task_ai(self, task_text: str) -> dict:
        self._reset()
        submit_ns = time.time_ns()
        self.events["submit"] = submit_ns
        self.sio.emit("ai_task", {"task": task_text})
        return self._wait_for_report()

    def run_task_manual_sequence(self, label: str, steps: list[dict]) -> dict:
        """Bypass the planner: invoke each step via execute_action directly.
        We still rely on the backend to emit events normally.

        IMPORTANT: backend's execute_action only handles a *single* action per
        socket call, so "manual" mode here just wraps each action in its own
        emit and aggregates results client-side. Use --bypass-planner sparingly.
        """
        self._reset()
        submit_ns = time.time_ns()
        self.events["submit"] = submit_ns
        for step in steps:
            self.sio.emit("execute_action", {
                "action": step["action"],
                "params": step["params"],
            })
            time.sleep(0.05)
        return self._wait_for_report()

    def _wait_for_report(self) -> dict:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if "report" in self.events:
                break
            time.sleep(0.05)
        return self._summarize()

    def _summarize(self) -> dict:
        ev = self.events
        submit = ev.get("submit")
        result = {
            "submit_ns": submit,
            "task_started_ns": ev.get("task_started"),
            "plan_done_ns": ev.get("ai_plan_result"),
            "first_action_ns": ev.get("first_action"),
            "first_perception_ns": ev.get("first_perception"),
            "report_ns": ev.get("report"),
            "n_action_results": len(self.action_results),
            "n_perception_results": len(self.perception_results),
            "report": self.report,
            "plan_payload_meta": (
                {
                    "summary": self.plan_payload.get("summary") if isinstance(self.plan_payload, dict) else None,
                    "plan_wall_ms": self.plan_payload.get("plan_wall_ms") if isinstance(self.plan_payload, dict) else None,
                    "n_steps": len(self.plan_payload.get("steps", [])) if isinstance(self.plan_payload, dict) else 0,
                }
                if self.plan_payload
                else None
            ),
            "perception_timings": [
                (p.get("data") or {}).get("timings") for p in self.perception_results
            ],
        }
        result["timed_out"] = "report" not in ev
        # Derived phase ms (None if event missing)
        def diff_ms(a, b):
            if a is None or b is None:
                return None
            return max(0, (b - a) // 1_000_000)
        plan_ms = diff_ms(submit, ev.get("ai_plan_result"))
        fly_ms = diff_ms(ev.get("ai_plan_result") or ev.get("first_action"),
                          ev.get("first_perception"))
        perception_ms = None
        if ev.get("first_perception") and self.perception_results:
            t = (self.perception_results[0].get("data") or {}).get("timings")
            if isinstance(t, dict):
                perception_ms = int(t.get("total_ms") or 0)
        report_ms = diff_ms(submit, ev.get("report"))
        result["phases_ms"] = {
            "plan_ms": plan_ms,
            "submit_to_first_action_ms": diff_ms(submit, ev.get("first_action")),
            "submit_to_first_perception_ms": diff_ms(submit, ev.get("first_perception")),
            "fly_to_perception_ms": fly_ms,
            "perception_compute_ms": perception_ms,
            "submit_to_report_ms": report_ms,
            "total_ms": report_ms,
        }
        return result


def run_bench(args: argparse.Namespace) -> dict:
    tasks = get_tasks()
    selected = [k for k in args.tasks if k in tasks] or list(tasks.keys())

    runner = TaskRunner(args.url, timeout_s=args.timeout)
    runner.connect()

    runs: dict[str, list[dict]] = {tid: [] for tid in selected}
    try:
        for tid in selected:
            t = tasks[tid]
            print(f"\n=== {tid} {t['name']} ===")
            print(f"   task_text: {t['task_text']}")
            print(f"   steps    : {short_action_chain(t['steps'])}")
            for rep in range(args.repeats + (1 if args.warmup else 0)):
                is_warmup = bool(args.warmup) and rep == 0
                tag = "warm" if is_warmup else f"rep{rep}"
                print(f"   [{tag}] running...", flush=True)
                if args.bypass_planner:
                    rec = runner.run_task_manual_sequence(t["label"], t["steps"])
                else:
                    rec = runner.run_task_ai(t["task_text"])
                rec.update({"task_id": tid, "rep": rep, "is_warmup": is_warmup})
                if not is_warmup:
                    runs[tid].append(rec)
                phs = rec.get("phases_ms") or {}
                print(
                    f"     plan={phs.get('plan_ms')}ms first_perc={phs.get('submit_to_first_perception_ms')}ms "
                    f"total={phs.get('total_ms')}ms timed_out={rec['timed_out']}"
                )
                # small inter-task gap to let UAV settle
                time.sleep(args.cooldown)
    finally:
        runner.disconnect()

    # ---- aggregate ----
    aggregate: dict[str, dict] = {}
    for tid, recs in runs.items():
        valid = [r for r in recs if not r["timed_out"]]
        agg_phases: dict[str, dict] = {}
        for key in ("plan_ms", "submit_to_first_action_ms", "submit_to_first_perception_ms",
                    "fly_to_perception_ms", "perception_compute_ms", "total_ms"):
            vals = [r["phases_ms"].get(key) for r in valid if r["phases_ms"].get(key) is not None]
            agg_phases[key] = stat_summary(vals)
        aggregate[tid] = {
            "task_name": tasks[tid]["name"],
            "task_label": tasks[tid]["label"],
            "n_runs": len(recs),
            "n_valid": len(valid),
            "n_timed_out": len(recs) - len(valid),
            "phases": agg_phases,
            "sla_compliance": {
                "total_under_30s": sum(
                    1 for r in valid if (r["phases_ms"].get("total_ms") or 1e9) <= SLA["total_ms"]
                ),
                "plan_under_3s": sum(
                    1 for r in valid if (r["phases_ms"].get("plan_ms") or 1e9) <= SLA["plan_ms"]
                ),
                "first_perception_under_10s": sum(
                    1 for r in valid
                    if (r["phases_ms"].get("submit_to_first_perception_ms") or 1e9)
                    <= SLA["first_perception_ms"]
                ),
                "report_under_25s": sum(
                    1 for r in valid
                    if (r["phases_ms"].get("submit_to_report_ms") or 1e9) <= SLA["report_ms"]
                ),
            },
        }

    return {
        "bench": "task_latency",
        "args": {
            "url": args.url,
            "tasks": selected,
            "repeats": args.repeats,
            "bypass_planner": args.bypass_planner,
            "warmup": bool(args.warmup),
            "timeout_s": args.timeout,
        },
        "sla_ms": SLA,
        "env": env_snapshot(),
        "runs": runs,
        "aggregate": aggregate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:5011")
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3", "T4", "T5"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="run a warm-up rep per task (default on)")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--bypass-planner", action="store_true",
                    help="emit raw execute_action steps instead of going through the LLM planner")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="seconds to wait between tasks (UAV settle + log flush)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-task hard timeout in seconds")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or now_run_id()
    out_dir = ensure_run_dir(run_id)
    print(f"[bench-a] run_id={run_id} out_dir={out_dir}")

    result = run_bench(args)
    result["run_id"] = run_id

    out_path = out_dir / "task_latency.json"
    dump_json(out_path, result)
    print(f"\n[bench-a] wrote {out_path}")
    print(f"[bench-a] aggregated tasks: {list(result['aggregate'].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
