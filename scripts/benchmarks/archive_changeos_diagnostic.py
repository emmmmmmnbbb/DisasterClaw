#!/usr/bin/env python3
"""Create a self-contained provenance archive for the 2026-09-13 diagnostic run.

The benchmark was executed from a dirty worktree, so the Git commit alone cannot
reconstruct it.  This script copies the exact executable sources and result
records, writes a sanitized environment description, and hashes every archived
file.  It intentionally records model weights by identity/hash rather than
duplicating the large checkpoints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "runs/benchmarks/changeos_binary/archive/diagnostic_20260913"
)

SOURCE_FILES = (
    ".env.example",
    "backend/agent_vqa.py",
    "backend/app.py",
    "backend/change_perception.py",
    "backend/config.py",
    "backend/detectors/__init__.py",
    "backend/detectors/base.py",
    "backend/detectors/changeos.py",
    "backend/llm_client.py",
    "backend/local_qwen_vl.py",
    "backend/perception.py",
    "backend/recheck.py",
    "backend/vlm_analyzer.py",
    "scripts/benchmarks/bench_agent_vqa.py",
    "scripts/benchmarks/eval_changeos_conformal.py",
    "scripts/benchmarks/freeze_changeos_manifest.py",
    "scripts/benchmarks/gen_agent_vqa_testset_v2.py",
    "scripts/benchmarks/report_agent_vqa.py",
    "scripts/benchmarks/review_agent_vqa_testset.py",
    "scripts/benchmarks/run_changeos_fixed_shards.sh",
    "scripts/benchmarks/rerun_shards_01.sh",
    "scripts/models/download_changeos.sh",
)

RESULT_INPUTS = (
    "runs/benchmarks/changeos_binary/frozen_manifest.json",
    "runs/benchmarks/changeos_binary/full_fixed_20260913_report",
    "runs/benchmarks/changeos_binary/full_fixed_20260913_shard0of4",
    "runs/benchmarks/changeos_binary/full_fixed_20260913_shard1of4",
    "runs/benchmarks/changeos_binary/full_fixed_20260913_shard2of4",
    "runs/benchmarks/changeos_binary/full_fixed_20260913_shard3of4",
    "runs/benchmarks/changeos_conformal/val_binary.json",
    "runs/benchmarks/agent_vqa_binary/review.json",
    "backend/data/benchmarks/agent_vqa_v2_binary.json",
    "backend/data/benchmarks/agent_vqa_final_v2_binary.json",
    "backend/data/benchmarks/tile_consumption_registry.json",
)

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
    / QWEN_REVISION
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(args: list[str]) -> str:
    proc = subprocess.run(
        args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return proc.stdout


def copy_input(relative: str, out: Path) -> None:
    source = ROOT / relative
    if not source.exists():
        raise FileNotFoundError(source)
    target = out / "payload" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def tree_manifest(root: Path) -> list[dict]:
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return records


def qwen_file_manifest() -> list[dict]:
    if not QWEN_SNAPSHOT.is_dir():
        return []
    records = []
    for path in sorted(p for p in QWEN_SNAPSHOT.iterdir() if p.is_file()):
        resolved = path.resolve()
        record = {
            "name": path.name,
            "bytes": resolved.stat().st_size,
            "resolved_blob": resolved.name,
        }
        # Hugging Face LFS cache blob names are the content SHA-256.  Small
        # metadata files may use Git object IDs, so compute those directly.
        if len(resolved.name) == 64 and all(c in "0123456789abcdef" for c in resolved.name):
            record["sha256"] = resolved.name
            record["hash_source"] = "huggingface_cache_blob_name"
        else:
            record["sha256"] = sha256(resolved)
            record["hash_source"] = "computed"
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty archive: {out}")
    out.mkdir(parents=True, exist_ok=True)

    for relative in RESULT_INPUTS + SOURCE_FILES:
        copy_input(relative, out)

    audit = out / "audit"
    audit.mkdir()
    (audit / "git_status.txt").write_text(
        run_text(["git", "status", "--short"]), encoding="utf-8"
    )
    (audit / "git_diff.patch").write_text(
        run_text(["git", "diff", "--binary", "--", ".", ":(exclude)cja_en/main.pdf"]),
        encoding="utf-8",
    )
    (audit / "pip_freeze.txt").write_text(
        run_text([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8"
    )
    (audit / "nvidia_smi.txt").write_text(
        run_text(["nvidia-smi"]), encoding="utf-8"
    )

    sys.path.insert(0, str(ROOT / "backend"))
    import config  # noqa: PLC0415
    import vlm_analyzer  # noqa: PLC0415

    provider_name = config.MODULE_CONFIG["vlm"].get("provider") or config.ACTIVE_PROVIDER
    provider = dict(config.PROVIDERS[provider_name])
    provider.pop("api_key", None)
    prompt = vlm_analyzer.AGENT_VQA_SYSTEM_PROMPT
    (audit / "agent_vqa_system_prompt.txt").write_text(prompt, encoding="utf-8")

    changeos_weights = ROOT / "backend/outputs/changeos/changeos_r34.pt"
    shard_audit = []
    for shard in range(4):
        rel = f"runs/benchmarks/changeos_binary/full_fixed_20260913_shard{shard}of4"
        episode_file = ROOT / rel / "episodes.jsonl"
        rows = [json.loads(line) for line in episode_file.read_text(encoding="utf-8").splitlines() if line]
        shard_audit.append({
            "shard": shard,
            "episodes": len(rows),
            "unique_config_qid": len({(row.get("config"), row.get("qid")) for row in rows}),
            "execution_errors": sum(not bool(row.get("ok", True)) for row in rows),
            "episodes_sha256": sha256(episode_file),
        })

    original_manifest = json.loads(
        (ROOT / "runs/benchmarks/changeos_binary/frozen_manifest.json").read_text(encoding="utf-8")
    )
    provenance = {
        "schema": "changeos-agent-vqa-diagnostic-archive/1.0",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_role": "development_diagnostic",
        "eligible_as_untouched_final": False,
        "ineligibility_reasons": [
            "questions have been inspected and used for strategy diagnosis",
            "human review is pending",
            "Qwen sampling was not controlled by a per-call generation seed",
        ],
        "git": {
            "commit": run_text(["git", "rev-parse", "HEAD"]).strip(),
            "dirty": bool(run_text(["git", "status", "--porcelain"]).strip()),
        },
        "testset": {
            "path": "backend/data/benchmarks/agent_vqa_v2_binary.json",
            "sha256": sha256(ROOT / "backend/data/benchmarks/agent_vqa_v2_binary.json"),
            "eval_role": "development_diagnostic",
        },
        "review": {
            "path": "runs/benchmarks/agent_vqa_binary/review.json",
            "sha256": sha256(ROOT / "runs/benchmarks/agent_vqa_binary/review.json"),
            "human_approved": 0,
            "human_pending": 160,
            "needs_author_check": 74,
        },
        "changeos": {
            "role": "fixed_external_perception_tool",
            "weights_path": str(changeos_weights),
            "weights_bytes": changeos_weights.stat().st_size,
            "weights_sha256": sha256(changeos_weights),
            "download_url": "https://github.com/Z-Zheng/ChangeOS/releases/download/v0.2/changeos_r34.pt",
            "official_training_data": "xBD training data",
            "pretraining_exposure": (
                "event exposure is present for xBD event-level claims; exact tile overlap of "
                "the released R34 file is not documented in its release metadata"
            ),
            "official_sources": [
                "https://github.com/Z-Zheng/ChangeOS",
                "https://doi.org/10.1016/j.rse.2021.112636",
            ],
            "license_observation": (
                "setup.py classifies the package as Apache Software License, but the official "
                "repository root inspected on 2026-09-13 did not expose a LICENSE file; this "
                "archive therefore records only the weight hash and does not redistribute it"
            ),
            "frozen_manifest_leaky_field": original_manifest.get("leaky"),
            "leaky_field_interpretation": "configuration declaration, not proof of checkpoint training exposure",
        },
        "qwen": {
            "role": "fixed_answer_generator",
            "model_id": QWEN_MODEL_ID,
            "revision": QWEN_REVISION,
            "snapshot_path": str(QWEN_SNAPSHOT),
            "provider": provider_name,
            "provider_config_sanitized": provider,
            "answer_temperature": 0.1,
            "top_p": provider.get("top_p"),
            "repetition_penalty": provider.get("repetition_penalty"),
            "do_sample": True,
            "per_call_generation_seed": False,
            "files": qwen_file_manifest(),
        },
        "prompt": {
            "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "system_prompt_archive": "audit/agent_vqa_system_prompt.txt",
            "user_prompt_builder": "backend/vlm_analyzer.py:VLMAnalyzer.answer_image_question",
            "user_prompt_builder_file_sha256": sha256(ROOT / "backend/vlm_analyzer.py"),
        },
        "run": {
            "base_action_seed": 42,
            "generation_seed_controlled": False,
            "configs": [
                "V1_STRUCT", "V2_STATE", "A0_HOLD", "A1_RANDOM", "A2_ALWAYS",
                "A3_ENTROPY", "A3U_RAW_ENTROPY", "A4_CONFORMAL",
            ],
            "shards": shard_audit,
            "source_fingerprint": "7f10d5f1fe2766558e4b2c44f58e9344426b8c31e15eeab2a86ba2dff23bcf35",
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    files = tree_manifest(out)
    (out / "SHA256SUMS.json").write_text(
        json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Hash manifest itself separately because a file cannot contain its own stable hash.
    (out / "SHA256SUMS.sha256").write_text(
        f"{sha256(out / 'SHA256SUMS.json')}  SHA256SUMS.json\n", encoding="ascii"
    )
    print(out)
    print(f"archived_files={len(files)}")
    print(f"archive_bytes={sum(item['bytes'] for item in files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
