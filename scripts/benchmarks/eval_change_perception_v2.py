#!/usr/bin/env python3
"""scripts/benchmarks/eval_change_perception_v2.py — C1/C2/H2 复核证据的可复现评测。

对应 paper_cja/REVIEW_PACKAGE.md 的三条 CRITICAL 意见：
    C1：损伤分类器在训练/验证集上的逐类表现从未汇报 → 本脚本产出训练曲线 +
        train/val/test 三个切分的 macro-F1、逐类召回。
    C2：未做类别不平衡处理、未与任何基线对标 → 本脚本对比
        (a) 类别加权 vs 未加权（旧 checkpoint），
        (b) 事件不相交切分 vs 官方标准（非事件不相交）切分下的同一架构，
        用于把"评测协议更难"与"架构/训练配方偏简单"两个因素分开。
    H2：差分注意力 vs 拼接融合，此前登记但从未报告 → 本脚本产出真实对比。

用法：
    python scripts/benchmarks/eval_change_perception_v2.py \
        --out runs/benchmarks/paper_cja_v2/change_perception_v2_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from change_perception import (  # noqa: E402
    XbdChangeDataset, ChangeMultiTaskNet, _run_epoch, _make_transform,
    CLASS_NAMES, macro_f1_from_logits,
)

STRICT_DIR = Path("/home/lc/datasets/xbd_change_strict_v1")
STANDARD_DIR = Path("/home/lc/datasets/xbd_change_standard_v1")
CKPT_DIR = BACKEND / "outputs/change_perception"


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ChangeMultiTaskNet(
        dropout_p=ckpt.get("dropout_p", 0.0),
        use_diff_attention=ckpt.get("use_diff_attention", False),
        pretrained=False,
    ).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, ckpt


def evaluate(model, jsonl_path: Path, device: str, sample_n: int = 6000, seed: int = 0) -> dict:
    recs = [json.loads(l) for l in jsonl_path.open(encoding="utf-8")]
    random.Random(seed).shuffle(recs)
    recs = recs[:sample_n]
    ds = XbdChangeDataset.__new__(XbdChangeDataset)
    ds.records = recs
    ds.augment = False
    ds._transform = _make_transform()
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6)
    _, acc, logits, labels = _run_epoch(model, loader, device, optimizer=None)
    macro_f1, f1s = macro_f1_from_logits(logits, labels)
    per_class = {}
    preds = logits.argmax(dim=-1)
    for c, name in enumerate(CLASS_NAMES):
        support = int((labels == c).sum())
        pred_n = int((preds == c).sum())
        tp = int(((preds == c) & (labels == c)).sum())
        rec = tp / support if support else 0.0
        per_class[name] = {"support": support, "pred_n": pred_n, "recall": round(rec, 4), "f1": round(f1s[c], 4)}
    return {"n": len(recs), "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4), "per_class": per_class}


_EPOCH_RE = re.compile(
    r"epoch (\d+)/(\d+): train_loss=([\d.]+) train_acc=([\d.]+) train_macroF1=([\d.]+) "
    r"val_loss=([\d.]+) val_acc=([\d.]+) val_macroF1=([\d.]+)"
)


def parse_training_log(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(3)), "train_acc": float(m.group(4)), "train_macro_f1": float(m.group(5)),
                "val_loss": float(m.group(6)), "val_acc": float(m.group(7)), "val_macro_f1": float(m.group(8)),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-n", type=int, default=6000)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs/benchmarks/paper_cja_v2/change_perception_v2_report.json"))
    args = ap.parse_args()
    device = args.device

    report: dict = {"checkpoints": {}, "training_curves": {}}

    ckpts = {
        "old_unweighted_diff_attention": CKPT_DIR / "strict_diff_attention_seed0.pt",
        "v2_weighted_diff_attention": CKPT_DIR / "strict_diff_attention_seed0_v2.pt",
        "v2_weighted_concat": CKPT_DIR / "strict_baseline_seed0_v2.pt",
        "standard_split_weighted_concat": CKPT_DIR / "standard_baseline_seed0.pt",
    }
    split_dirs = {
        "old_unweighted_diff_attention": STRICT_DIR,
        "v2_weighted_diff_attention": STRICT_DIR,
        "v2_weighted_concat": STRICT_DIR,
        "standard_split_weighted_concat": STANDARD_DIR,
    }
    for name, ckpt_path in ckpts.items():
        if not ckpt_path.is_file():
            print(f"[skip] {name}: {ckpt_path} 不存在")
            continue
        model, ckpt = load_model(ckpt_path, device)
        data_dir = split_dirs[name]
        entry = {
            "ckpt_path": str(ckpt_path),
            "use_diff_attention": bool(ckpt.get("use_diff_attention", False)),
            "class_weighted": bool(ckpt.get("class_weighted", False)),
            "logged_best_val_acc": ckpt.get("best_val_acc"),
            "logged_best_val_macro_f1": ckpt.get("best_val_macro_f1"),
            "data_dir": str(data_dir),
            "splits": {},
        }
        for split in ("train", "val", "test"):
            p = data_dir / f"{split}.jsonl"
            if p.is_file():
                print(f"[eval] {name} on {split} ({data_dir.name})...")
                entry["splits"][split] = evaluate(model, p, device, sample_n=args.sample_n)
        report["checkpoints"][name] = entry
        del model
        torch.cuda.empty_cache()

    for name, log_path in {
        "v2_weighted_diff_attention": CKPT_DIR / "logs_v2/diff_attention.log",
        "v2_weighted_concat": CKPT_DIR / "logs_v2/baseline.log",
        "standard_split_weighted_concat": CKPT_DIR / "logs_v2/standard_baseline.log",
    }.items():
        rows = parse_training_log(log_path)
        if rows:
            report["training_curves"][name] = rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
