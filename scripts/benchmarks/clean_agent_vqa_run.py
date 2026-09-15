#!/usr/bin/env python3
"""Create a traceable, de-duplicated Agent-VQA run without changing raw data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import bench_agent_vqa as bench


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_aggregate(rows: list[dict]) -> dict:
    """Keep policy boundary exits scored without treating them as run crashes."""
    result = bench.aggregate(rows)
    out_of_coverage = sum(
        1 for row in rows if row.get("reason_code") == "out_of_coverage"
    )
    if out_of_coverage:
        failures = dict(result.get("failure_taxonomy") or {})
        remaining = int(failures.get("execution_error", 0)) - out_of_coverage
        if remaining > 0:
            failures["execution_error"] = remaining
        else:
            failures.pop("execution_error", None)
        failures["out_of_coverage"] = out_of_coverage
        result["failure_taxonomy"] = failures
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    src, dst = args.source, args.destination
    episodes = src / "episodes.jsonl"
    if not episodes.is_file():
        raise SystemExit(f"missing {episodes}")
    if dst.exists():
        raise SystemExit(f"destination already exists: {dst}")

    rows = []
    seen = set()
    raw_count = 0
    for line in episodes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw_count += 1
        row = json.loads(line)
        key = (str(row.get("config") or ""), str(row.get("qid") or ""))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        rows.append(row)

    old_results = json.loads((src / "results.json").read_text(encoding="utf-8"))
    configs = list((old_results.get("configs") or {}).keys())
    by_config = {name: [row for row in rows if row.get("config") == name] for name in configs}
    expected = int(old_results.get("n_items") or 0)
    complete = bool(configs and expected) and all(len(by_config[name]) == expected for name in configs)
    aggregates = {name: analysis_aggregate(by_config[name]) for name in configs}
    execution_errors = sum(
        int(aggregates[name].get("failure_taxonomy", {}).get("execution_error", 0))
        for name in configs
    )
    valid = complete and execution_errors == 0

    dst.mkdir(parents=True)
    with (dst / "episodes.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = dict(old_results)
    results["run_id"] = f"{old_results.get('run_id', src.name)}_clean"
    results["valid_for_analysis"] = valid
    results["n_execution_errors"] = execution_errors
    results["cleaned_from"] = str(src)
    for name in configs:
        rec = dict((old_results.get("configs") or {}).get(name) or {})
        rec["agg"] = aggregates[name]
        results["configs"][name] = rec
    (dst / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = results["run_id"]
    manifest["valid_for_analysis"] = valid
    manifest["n_execution_errors"] = execution_errors
    manifest["cleaned_from"] = str(src)
    manifest["raw_episodes_sha256"] = sha256(episodes)
    (dst / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dst / "summary.md").write_text(bench.md_table(results["configs"]) + "\n", encoding="utf-8")

    provenance = {
        "schema": "agent-vqa-run-cleanup/1.0",
        "source": str(src),
        "source_episodes_sha256": sha256(episodes),
        "selection": "first durable row per (config, qid)",
        "raw_rows": raw_count,
        "unique_rows": len(rows),
        "duplicates_removed": raw_count - len(rows),
        "expected_rows": expected * len(configs),
        "complete": complete,
        "execution_errors": execution_errors,
        "policy_out_of_coverage": sum(
            1 for row in rows if row.get("reason_code") == "out_of_coverage"
        ),
        "valid_for_analysis": valid,
    }
    (dst / "cleanup_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
