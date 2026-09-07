#!/usr/bin/env python3
"""no_move_reask.py — "不动重问" 对照实验（位置不动、同一观测重复问 VLM）。

目的：把"复观测带来的答案变化"中来自【VLM 采样随机性】的成分与来自【真实运动/新图像】
的成分分开。对每题取第一步的初始观测（同一张图、同一份结构化证据），连续调用 K 次
与论文实验完全相同的 VLM 答题接口（vlm_answer_fn → VLMAnalyzer.answer_image_question，
temperature=0.1，evidence_level=state），统计答案翻转率。

关键边界（务必遵守）：
  - 不改 backend/ 与 scripts/ 的任何源码；本脚本只读 app/agent_vqa 内部函数。
  - 不移动、不重观测：只用 step 0 的观测，重复采样，不执行 fly_relative / search / reobserve。
  - 输出只写 cja_en/ 之下，绝不写 runs/ 或覆盖原始实验目录。
  - 只读冻结 final 题库 (agent_vqa_final_v2.json)，不读取条目 answer 之外的在线决策信息。

运行方式（需 source .env 以启用本地 Qwen，且需要空闲 GPU）：
    cd backend && set -a && source ../.env && set +a && \
      PERCEPTION_DEVICE=cuda:0 VLM_LOCAL_DEVICE=cuda:0 HF_HUB_OFFLINE=1 \
      python ../cja_en/scripts/no_move_reask.py --limit 4 --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))  # 必须在 import app 之前

DEFAULT_TESTSET = BACKEND / "data" / "benchmarks" / "agent_vqa_final_v2.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="不动重问对照：同一观测重复问 VLM 测翻转率")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题 (0=全部)")
    ap.add_argument("--qtype", default="", help="只跑某题型 presence/damage/count/spatial")
    ap.add_argument("--shard", default="", help="分片并行: 'i/N' 只跑 items[i::N]")
    ap.add_argument("--k", type=int, default=5, help="每题重复问 VLM 次数")
    ap.add_argument("--out-dir", default=str(ROOT / "cja_en" / "runs" / "no_move_reask"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--evidence-level", default="state",
                    help="证据层级 raw/struct/state；论文机制实验为 state")
    args = ap.parse_args()

    testset_path = Path(args.testset)
    if not testset_path.is_file():
        print(f"[ERROR] 题库不存在: {testset_path}", file=sys.stderr)
        return 2
    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    items = testset.get("items", [])
    if args.qtype:
        items = [it for it in items if it.get("question_type") == args.qtype]
    if args.limit > 0:
        items = items[: args.limit]
    if args.shard:
        i_str, n_str = args.shard.split("/")
        shard_i, shard_n = int(i_str), int(n_str)
        assert 0 <= shard_i < shard_n
        items = items[shard_i::shard_n]
    if not items:
        print("[ERROR] 过滤后无题", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if args.tag:
        out_dir = out_dir.with_name(out_dir.name + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[reask] items={len(items)} k={args.k} evidence={args.evidence_level} out={out_dir}")

    print("[reask] 正在 import app（首次加载感知 + 本地 VLM，可能数分钟）...", file=sys.stderr)
    t0 = time.time()
    import app  # noqa: E402
    from agent_vqa import parse_question, parse_vlm_json_output  # noqa: E402

    app.AGENT_VQA_EVIDENCE_LEVEL = args.evidence_level
    print(f"[reask] app ready in {time.time() - t0:.1f}s", file=sys.stderr)

    # 复用论文实验的控制器闭包：ctl._vlm 即 vlm_answer_fn（同 prompt/证据/温度 0.1）
    ctl = app._make_agent_vqa_controller("bench")

    raw_path = out_dir / "episodes.jsonl"
    raw_fp = raw_path.open("w", encoding="utf-8")

    records = []
    skipped = []
    for idx, item in enumerate(items):
        qid = item.get("id") or f"{item.get('tile_id','')}_{item.get('question_type','')}_{idx}"
        question = item["question"]
        start = item["start"]
        gt = item.get("answer")

        try:
            lat = float(start["lat"])
            lon = float(start["lon"])
            alt = float(start.get("alt", app.state.hover_altitude_m))
            entry = app.xbd_store.find_tile_containing(lat, lon, stage_priority=("post_disaster",))
            if entry is None:
                skipped.append({"qid": qid, "reason": "start_not_covered"})
                continue
            app.state.activate_xbd_tile(entry)
            app.state.adapter.reset_origin(lat, lon, alt=alt)
            app.state._sync_world_from_adapter()

            # 语义地图：与 run_agent_vqa_episode 一致（evidence_level=state 会引用）
            if app.SEMANTIC_MAP_ENABLED:
                snap = app.state.adapter.snapshot()
                app.state.semantic_map = app.SemanticMap(
                    origin_lat=float(snap["lat"]), origin_lon=float(snap["lon"]),
                    cell_size_m=app.SEMANTIC_MAP_CELL_M, instruction=question,
                )
            else:
                app.state.semantic_map = None

            spec = parse_question(question)
            result = ctl._perceive()          # step 0 观测（含检测器 + 感知）
            if result is None:
                skipped.append({"qid": qid, "reason": "out_of_coverage"})
                continue
            img = ctl._img(result)            # 初始图像字节（不动）
            if not img:
                skipped.append({"qid": qid, "reason": "no_patch_bytes"})
                continue

            samples = []
            for k in range(args.k):
                text = ctl._vlm(img, result, spec, qid)      # 同一观测重复采样
                ans = parse_vlm_json_output(text, spec, qid)
                samples.append({
                    "answer": ans.answer,
                    "decision": ans.decision,
                    "confidence": ans.confidence,
                    "abstain": bool(ans.abstain),
                    "reason_code": ans.reason_code,
                })
            rec = {
                "qid": qid,
                "question_type": item.get("question_type"),
                "disaster": item.get("disaster"),
                "gt_answer": gt,
                "k": args.k,
                "samples": samples,
            }
            records.append(rec)
            raw_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_fp.flush()
            answers = [s["answer"] for s in samples]
            n_uniq = len(set(answers))
            mark = "flip" if n_uniq > 1 else "stable"
            print(f"  [{idx + 1}/{len(items)}] {mark} uniq={n_uniq} gt={gt!r} "
                  f"answers={answers} :: {question[:20]}")
        except Exception as exc:
            skipped.append({"qid": qid, "reason": f"error: {exc}"})
            import traceback
            traceback.print_exc()
            print(f"  [{idx + 1}/{len(items)}] ERROR {exc}", file=sys.stderr)

    raw_fp.close()

    # ── 汇总 ──────────────────────────────────────────────
    summary = {"n_items": len(items), "n_run": len(records), "n_skipped": len(skipped),
               "k": args.k, "evidence_level": args.evidence_level, "skipped": skipped}
    if records:
        flips = []
        per_type = Counter()
        flips_by_type = Counter()
        for r in records:
            answers = [s["answer"] for s in r["samples"]]
            is_flip = len(set(answers)) > 1
            flips.append(is_flip)
            qt = r["question_type"]
            per_type[qt] += 1
            if is_flip:
                flips_by_type[qt] += 1
        summary["flip_rate"] = sum(flips) / len(flips)
        summary["flip_count"] = sum(flips)
        summary["by_type"] = {
            qt: {"n": per_type[qt], "flips": flips_by_type[qt],
                 "flip_rate": flips_by_type[qt] / per_type[qt] if per_type[qt] else 0.0}
            for qt in sorted(per_type)
        }
        # 决策翻转（answer / continue_search / reobserve / abstain 是否跨采样变化）
        dec_flips = sum(
            1 for r in records
            if len({s["decision"] for s in r["samples"]}) > 1
        )
        summary["decision_flip_count"] = dec_flips
        summary["decision_flip_rate"] = dec_flips / len(records)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[reask] 汇总:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[reask] 完成：{len(records)} 题运行，{len(skipped)} 题跳过 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
