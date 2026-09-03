#!/usr/bin/env python3
"""scripts/benchmarks/export_confidence_calibration.py — 生成重观测置信标定表。

从 paper_cja_mech_final 两 shard 的 episodes.jsonl 计算：
  - 置信度档 -> 作答数 / 正确率（置信度与正确率单调正相关）
  - 高置信尾（conf>=0.9）下单观测 A0 与共形重观测 A4 的正确率
  - 重观测抬升置信度的次数占比与均值（写入表注）

产物：paper_cja/generated/confidence_calibration_table.tex
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SHARDS = [
    REPO / "runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_shard0of2/episodes.jsonl",
    REPO / "runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_shard1of2/episodes.jsonl",
]
OUT = REPO / "paper_cja/generated/confidence_calibration_table.tex"


def load_rows():
    rows = []
    for fp in SHARDS:
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()

    # 1) 置信度档 -> 作答数 / 正确率（全作答，弃答 conf=0 不参与分档）
    hist = defaultdict(lambda: [0, 0])
    for r in rows:
        c = r.get("confidence")
        if c is None or c <= 0.0:
            continue
        b = round(c, 2)
        hist[b][0] += 1
        hist[b][1] += 1 if r.get("correct") else 0
    buckets = [0.80, 0.90, 0.95, 0.99]
    bucket_rows = []
    for b in buckets:
        n, ok = hist.get(b, (0, 0))
        acc = ok / n if n else 0.0
        bucket_rows.append((b, n, acc))

    # 2) 高置信尾 conf>=0.9 的单观测 vs 共形重观测
    def high_conf_acc(cfg):
        sub = [r for r in rows if r.get("config") == cfg and (r.get("confidence") or 0) >= 0.9]
        n = len(sub)
        acc = sum(1 for r in sub if r.get("correct")) / n if n else 0.0
        return n, acc

    a0_n, a0_acc = high_conf_acc("A0_HOLD")
    a4_n, a4_acc = high_conf_acc("A4_CONFORMAL")

    # 3) 重观测抬升置信度的次数与均值
    up = down = same = 0
    inis, fins = [], []
    for r in rows:
        if int(r.get("n_reobservations", 0) or 0) <= 0:
            continue
        cs = [t.get("confidence") for t in (r.get("trajectory") or []) if t.get("confidence") is not None]
        if len(cs) < 2:
            continue
        a, b = cs[0], cs[-1]
        if b > a + 1e-3:
            up += 1
        elif b < a - 1e-3:
            down += 1
        else:
            same += 1
        inis.append(a)
        fins.append(b)
    n_reobs = up + down + same
    mean_ini = statistics.mean(inis) if inis else 0.0
    mean_fin = statistics.mean(fins) if fins else 0.0

    lines = [
        "\\begin{tabular}{lrr}",
        "\\hline",
        "分组 & 作答数 & 正确率 \\\\",
        "\\hline",
    ]
    for b, n, acc in bucket_rows:
        lines.append(f"置信度 {b:.2f} & {n} & {acc * 100:.1f}\\% \\\\")
    lines += [
        "\\hline",
        f"conf$\\ge$0.9 单观测 (A0) & {a0_n} & {a0_acc * 100:.1f}\\% \\\\",
        f"conf$\\ge$0.9 共形重观测 (A4) & {a4_n} & {a4_acc * 100:.1f}\\% \\\\",
        "\\hline",
        "\\end{tabular}",
    ]
    tex = "\n".join(lines) + "\n"
    OUT.write_text(tex, encoding="utf-8")
    print(f"[export] {OUT}")
    print(f"  buckets: {bucket_rows}")
    print(f"  conf>=0.9: A0 n={a0_n} acc={a0_acc:.4f}; A4 n={a4_n} acc={a4_acc:.4f}")
    print(f"  reobs: up={up} down={down} same={same} (n={n_reobs}); "
          f"mean conf {mean_ini:.3f} -> {mean_fin:.3f}")


if __name__ == "__main__":
    main()
