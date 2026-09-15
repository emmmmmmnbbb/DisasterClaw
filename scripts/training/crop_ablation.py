#!/usr/bin/env python3
"""crop 上下文边距消融 —— 判定 minor/major 塌缩是否由"建筑紧裁剪裁掉周围环境"引起。

做法：从 strict_v1 训练集按类别上限采样一个平衡子集，随机 75/25 分成
mini-train / mini-val，写成临时 jsonl。随后用同一份数据、只改
`--context-margin`（0.25=1.5× / 0.5=2× / 1.0=3×）各训一遍，
对比 minor / major / destroyed 的 recall。其它超参完全一致。

只做相对比较（crop 是唯一变量），绝对精度不代表正式训练。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def build_balanced_subset(src: Path, cap_per_class: int, seed: int) -> list[dict]:
    records = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_class: dict[int, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_class[int(r["class_id"])].append(r)

    rng = random.Random(seed)
    balanced: list[dict] = []
    for cid in range(len(CLASS_NAMES)):
        pool = by_class.get(cid, [])
        rng.shuffle(pool)
        balanced.extend(pool[:cap_per_class])

    rng.shuffle(balanced)
    return balanced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/lc/datasets/xbd_change_strict_v1/train.jsonl")
    ap.add_argument("--out-dir", default="/tmp/xbd_crop_ablation")
    ap.add_argument("--cap-per-class", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = build_balanced_subset(Path(args.src), args.cap_per_class, args.seed)
    n = len(records)
    n_val = int(n * args.val_frac)
    val, train = records[:n_val], records[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    (out / "val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8")

    def dist(rs: list[dict]) -> dict[str, int]:
        c = collections.Counter(int(r["class_id"]) for r in rs)
        return {CLASS_NAMES[k]: v for k, v in sorted(c.items())}

    print(f"[ablation] total={n} train={len(train)} val={len(val)}")
    print(f"[ablation] train_dist={dist(train)}")
    print(f"[ablation] val_dist={dist(val)}")
    print(f"[ablation] 已写入 {out}/train.jsonl 与 {out}/val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
