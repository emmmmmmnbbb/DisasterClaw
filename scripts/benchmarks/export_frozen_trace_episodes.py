#!/usr/bin/env python3
"""Export the two frozen matched-view episodes requested for Figure 3."""

from __future__ import annotations

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cja_en" / "review" / "frozen_trace_raw"
CASES = {
    "damage_help": {
        "qid": "hurricane-michael_00000435_post_disaster__damage_0001_9430",
        "repeat": 0,
        "recheck_index": 1,
        "source": ROOT / "runs/benchmarks/changeos_binary/revision_matched_view_repeat0_shard1of4_full_20260920/episodes.jsonl",
        "log": ROOT / "runs/benchmarks/changeos_binary/revision_matched_logs_full_20260920/repeat0_shard1.log",
    },
    "spatial_harm": {
        "qid": "hurricane-michael_00000306_post_disaster__spatial_0092_5452",
        "repeat": 0,
        "recheck_index": 0,
        "source": ROOT / "runs/benchmarks/changeos_binary/revision_matched_view_repeat0_shard0of4_full_20260920/episodes.jsonl",
        "log": ROOT / "runs/benchmarks/changeos_binary/revision_matched_logs_full_20260920/repeat0_shard0.log",
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in CASES.items():
        selected = []
        for line in spec["source"].read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("qid") != spec["qid"] or row.get("repeat") != spec["repeat"]:
                continue
            selected.append(row)
        if len(selected) != 1:
            raise RuntimeError(f"expected one episode for {name}, found {len(selected)}")
        episode = selected[0]
        forks = [f for f in episode.get("forks", []) if f.get("recheck_index") == spec["recheck_index"]]
        if len(forks) != 1:
            raise RuntimeError(f"expected one fork for {name}, found {len(forks)}")
        payload = {
            "episode": episode,
            "selected_fork": forks[0],
            "source_episodes_jsonl": str(spec["source"]),
            "source_run_log": str(spec["log"]),
        }
        (OUT / f"{name}_episode.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(spec["log"], OUT / f"{name}_run.log")
        (OUT / f"{name}_source.txt").write_text(
            f"episodes_jsonl={spec['source']}\nrun_log={spec['log']}\n"
            "No separate observation PNG or detection-overlay artifact exists in the source run directory.\n",
            encoding="utf-8",
        )
    print(f"exported {len(CASES)} frozen trace cases to {OUT}")


if __name__ == "__main__":
    main()
