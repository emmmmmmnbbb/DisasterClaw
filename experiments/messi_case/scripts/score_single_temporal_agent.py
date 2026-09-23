"""Score single-temporal controller replay against the separate MESSI truth file."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["full_descend", "full_train_paths", "full_test_paths"])
    args = ap.parse_args()
    outputs = {}
    for split in args.splits:
        gt = {r["sample_id"]: r["answer"] for r in load(ROOT / "manifests" / f"{split}_ground_truth.jsonl")}
        samples = {r["sample_id"]: r for r in load(ROOT / "manifests" / f"{split}_samples.jsonl")}
        episodes = load(ROOT / "runs" / f"single_temporal_{split}" / "episodes.jsonl")
        by_sample = {}
        for row in episodes:
            by_sample.setdefault(row["sample_id"], {})[row["arm"]] = row
        if set(by_sample) != set(gt) or any(set(v) != {"hold", "agent"} for v in by_sample.values()):
            raise RuntimeError(f"incomplete or unmatched paired runs: {split}")
        part = {}
        for arm in ("hold", "agent"):
            rows = [v[arm] for v in by_sample.values()]
            correct = sum(r["answer"] == gt[r["sample_id"]] for r in rows)
            part[arm] = {"n": len(rows), "correct": correct,
                         "accuracy": correct / len(rows) if rows else None,
                         "rechecks": sum(r["rechecks"] for r in rows),
                         "stop_reasons": dict(Counter(r["stop_reason"] for r in rows))}
        transitions = Counter()
        corrected = harmed = 0
        for sid, arms in by_sample.items():
            h, a = arms["hold"]["answer"], arms["agent"]["answer"]
            transitions[f"{h}->{a}"] += 1
            if h != gt[sid] and a == gt[sid]: corrected += 1
            if h == gt[sid] and a != gt[sid]: harmed += 1
        strata = Counter(samples[sid].get("scene_group_id", "") for sid in by_sample)
        outputs[split] = {"n": len(gt), "arms": part,
                          "answer_transitions": dict(transitions),
                          "corrected": corrected, "harmed": harmed,
                          "scene_groups": len(strata)}
    out = {"experiment": "single_temporal_disasterclaw_messi",
           "truth_source": "separate MESSI vegetation annotations; never supplied to online replay",
           "strata": outputs}
    path = ROOT / "results/single_temporal_agent_summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
