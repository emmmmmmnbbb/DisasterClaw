#!/usr/bin/env python3
"""scripts/benchmarks/probe_esarbench.py — ESARBench 离线语言条件受害者定位探针。

不跑 UE5/AirSim 仿真，只用 ESARBench 公开的 task_data.json（600 条任务定义：
语言指令 + 起点 + 受害者/线索/信号的三维坐标）。把"在候选点中按受害者概率排序"
做成纯文本任务喂给 LLM，考察 MLLM 的语言条件空间推理能否跨基准（损伤评估→搜救）
落地到"定位"这一最小可测能力上。

产物：
  - per-task 排序 + 指标 JSON
  - 汇总：MRR / P@K / 预算命中率，并给出几何基线（随机 / 就近 / 聚类密度）对照。

用法：
    python scripts/benchmarks/probe_esarbench.py --limit 20 --model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TASK_DATA = "/tmp/esar_task_data.json"


def _load_api_key() -> str:
    for p in ("/home/lc/.config/litellm/env",):
        if not Path(p).is_file():
            continue
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def _chat(model: str, prompt: str, api_key: str, temperature: float = 0.0) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def build_candidates(task: dict) -> tuple[list[dict], dict]:
    """返回 (候选点列表, start)。候选点带内部字段 kind='target'|'cue'|'fire'。"""
    pts = []
    for t in task.get("targets", []):
        pts.append({"kind": "target", "loc": t["location"]})
    for c in task.get("cues", []):
        pts.append({"kind": "cue", "loc": c["location"]})
    for f in task.get("fire", []):
        pts.append({"kind": "fire", "loc": f["location"]})
    start = task.get("start_location", {})
    return pts, start


def build_prompt(task: dict, pts: list[dict], order: list[int]) -> str:
    start = task.get("start_location", {})
    lines = [
        "You are assisting a UAV search-and-rescue mission.",
        "Below is a natural-language description of the situation, the UAV's start position,",
        "and a list of candidate points of interest (3D coordinates in simulation world units).",
        "",
        f"Description: {task.get('prompt', '')}",
        f"UAV start: x={start.get('x')}, y={start.get('y')}, z={start.get('z')}",
        "",
        "Candidate points:",
    ]
    for i in order:
        loc = pts[i]["loc"]
        lines.append(f"point_{i}: x={loc.get('x')}, y={loc.get('y')}, z={loc.get('z')}")
    lines += [
        "",
        "Rank the points from most likely to contain a victim to least likely.",
        "Output ONLY a comma-separated list of point ids in rank order. For example:",
        "3,7,1,0,5",
    ]
    return "\n".join(lines)


def parse_ranking(text: str, n: int) -> list[int]:
    """从模型输出里按出现顺序抽取整数 id，去重并过滤越界，缺的补在末尾。"""
    seen, out = set(), []
    for tok in text.replace(",", " ").split():
        t = tok.strip()
        if t.isdigit():
            i = int(t)
            if 0 <= i < n and i not in seen:
                seen.add(i)
                out.append(i)
    for i in range(n):
        if i not in seen:
            out.append(i)
    return out


def _k_for(task: dict) -> int:
    return max(1, len(task.get("targets", [])))


def metrics_for(ranking: list[int], pts: list[dict], k: int) -> dict:
    victim_ranks = [r for r, i in enumerate(ranking) if pts[i]["kind"] == "target"]
    first = min(victim_ranks) + 1 if victim_ranks else None
    mrr = 1.0 / first if first else 0.0
    topk = ranking[:k]
    p_at_k = sum(1 for i in topk if pts[i]["kind"] == "target") / len(topk)
    n_targets = len([p for p in pts if p["kind"] == "target"])
    hit = 1.0 if any(pts[i]["kind"] == "target" for i in topk) else 0.0
    recall = sum(1 for i in topk if pts[i]["kind"] == "target") / n_targets
    return {"mrr": mrr, "p_at_k": p_at_k, "hit": hit, "recall_at_k": recall}


def baseline_rankings(pts: list[dict], start: dict) -> dict:
    n = len(pts)
    idx = list(range(n))
    # 随机
    rng = random.Random(0)
    random_order = idx[:]
    rng.shuffle(random_order)
    # 就近（离起点近的优先）
    nearest = sorted(idx, key=lambda i: _dist(pts[i]["loc"], start))
    # 聚类密度（到 3 近邻平均距离小的优先，受害者倾向成群）
    def density(i):
        ds = sorted(_dist(pts[i]["loc"], pts[j]["loc"]) for j in range(n) if j != i)
        return sum(ds[:3]) / max(1, len(ds[:3]))
    cluster = sorted(idx, key=density)
    return {"random": random_order, "nearest_start": nearest, "cluster_density": cluster}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-data", default=DEFAULT_TASK_DATA)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = json.loads(Path(args.task_data).read_text(encoding="utf-8"))
    tasks = tasks[: args.limit]
    api_key = _load_api_key()
    if not api_key:
        print("[ERROR] 未找到 DEEPSEEK_API_KEY")
        return 2

    rng = random.Random(args.seed)
    agg = {"model": [], "random": [], "nearest_start": [], "cluster_density": []}
    per_task = []
    for t_idx, task in enumerate(tasks):
        pts, start = build_candidates(task)
        n = len(pts)
        k = _k_for(task)
        order = list(range(n))
        rng.shuffle(order)  # 打乱候选点输入顺序，避免模型照抄
        prompt = build_prompt(task, pts, order)
        raw = _chat(args.model, prompt, api_key)
        ranking = parse_ranking(raw, n)
        m = metrics_for(ranking, pts, k)
        bl = baseline_rankings(pts, start)
        row = {
            "task_id": task.get("task_id"),
            "map": task.get("map_name"),
            "difficulty": task.get("difficulty_level"),
            "n_candidates": n,
            "n_targets": len(task.get("targets", [])),
            "prompt": task.get("prompt"),
            "raw_output": raw,
            "model": m,
        }
        for name, br in bl.items():
            bm = metrics_for(br, pts, k)
            row[name] = bm
            agg[name].append(bm)
        agg["model"].append(m)
        per_task.append(row)
        print(f"[{t_idx + 1}/{len(tasks)}] task={task.get('task_id')} "
              f"MRR={m['mrr']:.3f} P@K={m['p_at_k']:.3f} hit={m['hit']:.0f} "
              f"rand_MRR={bl['random'] and metrics_for(bl['random'], pts, k)['mrr']:.3f}")

    summary = {}
    for name, rows in agg.items():
        summary[name] = {
            "mrr": round(statistics.mean(r["mrr"] for r in rows), 4),
            "p_at_k": round(statistics.mean(r["p_at_k"] for r in rows), 4),
            "hit": round(statistics.mean(r["hit"] for r in rows), 4),
            "recall_at_k": round(statistics.mean(r["recall_at_k"] for r in rows), 4),
        }
    print("\n==== 汇总 (n=%d) ====" % len(tasks))
    for name in ("model", "random", "nearest_start", "cluster_density"):
        s = summary[name]
        print(f"  {name:16s} MRR={s['mrr']:.4f}  P@K={s['p_at_k']:.4f}  "
              f"hit={s['hit']:.4f}  recall@K={s['recall_at_k']:.4f}")

    out = args.out or f"/tmp/esarbench_probe_{args.limit}.json"
    Path(out).write_text(json.dumps({"summary": summary, "per_task": per_task},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] 结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
