#!/usr/bin/env python3
"""Freeze selection artifacts before the one-shot final Agent-VQA run."""

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
    ap = argparse.ArgumentParser(description="冻结主动复观测最终评测配置")
    ap.add_argument("--selection-sweep", required=True)
    ap.add_argument("--selection-testset", required=True)
    ap.add_argument("--final-testset", required=True)
    ap.add_argument("--fit-report", required=True)
    ap.add_argument("--entropy-table", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--backend", default="xview2_first")
    ap.add_argument("--leaky", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default=str(
        ROOT / "runs/benchmarks/paper_cja_mech_v1/frozen_manifest.json"
    ))
    args = ap.parse_args()

    paths = {
        name: Path(value) for name, value in {
            "selection_sweep": args.selection_sweep,
            "selection_testset": args.selection_testset,
            "final_testset": args.final_testset,
            "fit_report": args.fit_report,
            "entropy_table": args.entropy_table,
            "registry": args.registry,
        }.items()
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing freeze inputs: " + ", ".join(missing))

    sweep = json.loads(paths["selection_sweep"].read_text(encoding="utf-8"))
    fit = json.loads(paths["fit_report"].read_text(encoding="utf-8"))
    table = json.loads(paths["entropy_table"].read_text(encoding="utf-8"))
    if table.get("schema") != "fov-ladder-entropy/1.0":
        raise ValueError(f"invalid entropy table schema: {table.get('schema')!r}")
    selected = sweep.get("selected") or {}
    payload = {
        "schema": "recheck-frozen-manifest/1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "backend": args.backend,
        "leaky": bool(args.leaky),
        "temperature": float(fit["temperature"]),
        "qhat": float(fit["conformal_qhat_alpha01"]),
        "conformal_alpha": 0.1,
        "entropy_trigger": float(selected.get("entropy_trigger", 0.5)),
        "min_info_gain": float(selected.get("min_info_gain", 0.05)),
        "budget": float(selected.get("budget", 0.25)),
        "entropy_table_path": str(paths["entropy_table"].resolve()),
        "entropy_table_sha256": sha256(paths["entropy_table"]),
        "selection_testset": str(paths["selection_testset"].resolve()),
        "selection_testset_sha256": sha256(paths["selection_testset"]),
        "final_testset": str(paths["final_testset"].resolve()),
        "testset_sha256": sha256(paths["final_testset"]),
        "registry_sha256": sha256(paths["registry"]),
        "selection_sweep_sha256": sha256(paths["selection_sweep"]),
        "fit_report_sha256": sha256(paths["fit_report"]),
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
