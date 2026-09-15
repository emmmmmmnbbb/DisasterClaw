#!/usr/bin/env python3
"""Freeze the ChangeOS (binary) reobservation manifest.

The four-class freeze path (``freeze_recheck_manifest.py``) requires a selection
sweep, a four-class FOV entropy table and a four-class temperature — none of
which apply to the frozen binary ChangeOS backend.  ChangeOS applies its own
softmax temperature internally and the A5 info-gain strategy is a four-class
construct, so this manifest records only what the binary arm needs:

  - ``qhat`` fitted on the binary label space (``eval_changeos_conformal.py``)
  - ``label_mode`` = ``binary`` so ``bench_agent_vqa.py`` fails closed on any
    binary/four-class qhat mismatch
  - ``entropy_table_path`` = "" (A5_EXPECTED must be excluded from the run)

Usage::

    python scripts/benchmarks/freeze_changeos_manifest.py \
        --conformal-report runs/benchmarks/changeos_conformal/val_binary.json \
        --final-testset backend/data/benchmarks/agent_vqa_final_v2_binary.json \
        --review-report runs/benchmarks/agent_vqa_binary/review.json \
        --out runs/benchmarks/changeos_binary/frozen_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="冻结 ChangeOS 二分类重观测配置")
    ap.add_argument("--conformal-report", required=True)
    ap.add_argument("--final-testset", required=True)
    ap.add_argument("--review-report", required=True)
    ap.add_argument("--out", default=str(
        ROOT / "runs/benchmarks/changeos_binary/frozen_manifest.json"
    ))
    args = ap.parse_args()

    conformal = json.loads(Path(args.conformal_report).read_text(encoding="utf-8"))
    if conformal.get("label_mode") != "binary":
        raise ValueError(
            f"--conformal-report 不是二分类拟合: label_mode={conformal.get('label_mode')!r}"
        )
    qhat = float(conformal["conformal_qhat_alpha01"])

    final_testset = Path(args.final_testset)
    review_report = Path(args.review_report)

    payload = {
        "schema": "recheck-frozen-manifest/1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "backend": "changeos",
        "label_mode": "binary",
        "leaky": False,
        "temperature": 1.0,          # ChangeOS 内部 softmax，不做 recheck 级温度标定
        "qhat": qhat,
        "conformal_alpha": 0.1,
        "entropy_trigger": 0.5,      # A3_ENTROPY 阈值触发；与四分类默认一致
        "min_info_gain": 0.05,       # 未使用（A5_EXPECTED 不适用于二分类）
        "entropy_table_path": "",    # A5_EXPECTED 在二分类运行中必须排除
        "conformal_report": str(Path(args.conformal_report).resolve()),
        "conformal_report_sha256": sha256(Path(args.conformal_report)),
        "final_testset": str(final_testset.resolve()),
        "testset_sha256": sha256(final_testset),
        "review_report": str(review_report.resolve()),
        "review_report_sha256": sha256(review_report),
        "excluded_configs_note": (
            "A5_EXPECTED (info_gain) 依赖四分类 FOV 熵表，二分类运行必须排除；"
            "A3/A3U/A4 的熵/共形触发在二分类 class_probs 上定义良好。"
        ),
    }
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"frozen manifest already exists; refusing overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
