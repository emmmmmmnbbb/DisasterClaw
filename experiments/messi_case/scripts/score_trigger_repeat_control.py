"""Score equal-budget high-repeat vs low-view control on triggered MESSI episodes."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("full_descend", "full_train_paths", "full_test_paths")


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answer(event):
    return event["answer"]


def main():
    records = []
    for split in SPLITS:
        gt = {r["sample_id"]: r["answer"] for r in read(ROOT / "manifests" / f"{split}_ground_truth.jsonl")}
        original = read(ROOT / "runs" / f"single_temporal_{split}" / "episodes.jsonl")
        hold = {r["sample_id"]: r for r in original if r["arm"] == "hold"}
        agent = {r["sample_id"]: r for r in original if r["arm"] == "agent" and r["rechecks"] == 1}
        repeat_path = ROOT / "runs" / f"single_temporal_trigger_repeat_high_{split}" / "episodes.jsonl"
        repeated = {r["sample_id"]: r for r in read(repeat_path) if r["arm"] == "repeat_high"}
        if set(agent) != set(repeated):
            raise RuntimeError(f"Triggered IDs differ in {split}: {len(agent)} vs {len(repeated)}")
        for sid, low_row in agent.items():
            h = hold[sid]
            repeat = repeated[sid]
            high_first = h["events"][0]["prediction"]
            if high_first != repeat["events"][0]["prediction"]:
                raise RuntimeError(f"First high prediction mismatch: {sid}")
            if len(repeat["events"]) != 4 or repeat["events"][3]["image"] != "high_crop":
                raise RuntimeError(f"Repeat-high control did not reacquire the high image: {sid}")
            if len(low_row["events"]) != 4 or low_row["events"][3]["image"] != "low_crop":
                raise RuntimeError(f"Agent did not acquire the low image: {sid}")
            truth = gt[sid]
            low_answer = low_row["answer"]
            repeated_answer = repeat["answer"]
            first_answer = h["answer"]
            records.append({
                "sample_id": sid, "split": split, "truth": truth,
                "initial_high_answer": first_answer,
                "initial_high_prediction": high_first,
                "agent_low_answer": low_answer,
                "repeat_high_answer": repeated_answer,
                "initial_correct": first_answer == truth,
                "low_correct": low_answer == truth,
                "repeat_high_correct": repeated_answer == truth,
                "low_vs_initial": "corrected" if first_answer != truth and low_answer == truth else
                                  "harmed" if first_answer == truth and low_answer != truth else
                                  "unchanged_correct" if low_answer == truth else "unchanged_incorrect",
                "low_vs_repeat_high": "low_correct_repeat_wrong" if low_answer == truth and repeated_answer != truth else
                                      "low_wrong_repeat_correct" if low_answer != truth and repeated_answer == truth else
                                      "both_correct" if low_answer == truth else "both_wrong",
                "high_repeat_trace": repeat,
                "low_agent_trace": low_row,
                "hold_trace": h,
            })
    cats = Counter(r["low_vs_initial"] for r in records)
    low_vs_repeat = Counter(r["low_vs_repeat_high"] for r in records)
    n = len(records)
    summary = {
        "n_triggered": n,
        "initial_high_correct": sum(r["initial_correct"] for r in records),
        "repeat_high_correct": sum(r["repeat_high_correct"] for r in records),
        "real_low_correct": sum(r["low_correct"] for r in records),
        "real_low_vs_initial": dict(cats),
        "real_low_vs_repeat_high": dict(low_vs_repeat),
        "reobservations_per_arm": 1,
        "answer_calls_per_arm": 2,
        "high_repeat_matches_initial_answer": all(
            r["repeat_high_answer"] == r["initial_high_answer"] for r in records
        ),
        "ir_yamim_reuses_prior_qwen_exploratory_data": True,
        "examples": {},
    }
    correction = next(r for r in records if r["low_vs_initial"] == "corrected")
    harm = next(r for r in records if r["low_vs_initial"] == "harmed")
    for name, row in (("correction", correction), ("harm", harm)):
        summary["examples"][name] = {
            "sample_id": row["sample_id"], "split": row["split"], "truth": row["truth"],
            "initial_high_answer": row["initial_high_answer"],
            "real_low_answer": row["agent_low_answer"],
            "repeat_high_answer": row["repeat_high_answer"],
            "high_first_prediction": row["initial_high_prediction"],
            "low_agent_trace": row["low_agent_trace"],
            "repeat_high_trace": row["high_repeat_trace"],
        }
    (ROOT / "results/single_temporal_trigger_control.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (ROOT / "results/single_temporal_triggered_paired_records.json").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, ensure_ascii=False, indent=2))
    for name, row in summary["examples"].items():
        print(name, row["sample_id"], "truth", row["truth"], "initial", row["initial_high_answer"],
              "repeat-high", row["repeat_high_answer"], "real-low", row["real_low_answer"])


if __name__ == "__main__":
    main()
