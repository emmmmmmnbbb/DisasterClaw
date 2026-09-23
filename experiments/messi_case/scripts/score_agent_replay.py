"""Score completed replay logs against the separately stored reference answers."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["formal", "exhaustive_descend", "exhaustive_paths",
                                    "full_descend", "full_train_paths", "full_test_paths"], default="formal")
ap.add_argument("--variant", choices=["self_reported", "logits"], default="self_reported")
args = ap.parse_args()
gt = {x["sample_id"]: x["answer"] for x in map(json.loads,
    (ROOT / f"manifests/{args.split}_ground_truth.jsonl").read_text().splitlines())}
samples = {x["sample_id"]: x for x in map(json.loads,
    (ROOT / f"manifests/{args.split}_samples.jsonl").read_text().splitlines())}
assert set(samples) == set(gt)
run_dir = f"agent_replay_logits_{args.split}" if args.variant == "logits" else f"agent_replay_{args.split}"
rows = [json.loads(x) for x in (ROOT / "runs" / run_dir / "episodes.jsonl").read_text().splitlines()]
assert len(rows) == 2 * len(gt)
by_id = {}
for row in rows:
    sid, arm = row["sample_id"], row["arm"]
    assert sid in gt and arm in ("hold", "agent")
    assert (sid, arm) not in by_id
    by_id[sid, arm] = row
assert set(by_id) == {(sid, arm) for sid in gt for arm in ("hold", "agent")}
for sid in gt:
    hold, agent = by_id[sid, "hold"], by_id[sid, "agent"]
    assert hold["model_calls"][0]["raw"] == agent["model_calls"][0]["raw"]
    assert [r["image_key"] for r in hold["image_reads"]] == ["high_marked"]
    assert [r["image_key"] for r in agent["image_reads"]] == ["high_marked"] + (["low_crop"] if agent["rechecks"] else [])

summary = {}
for arm in ("hold", "agent"):
    episodes = [by_id[sid, arm] for sid in gt]
    summary[arm] = {"correct": sum(e["answer"] == gt[e["sample_id"]] for e in episodes),
                    "n": len(episodes), "accuracy": sum(e["answer"] == gt[e["sample_id"]] for e in episodes)/len(episodes),
                    "rechecks": sum(e["rechecks"] for e in episodes),
                    "stop_reasons": dict(Counter(e["stop_reason"] for e in episodes)),
                    "first_answer_invalid": sum(e["model_calls"][0]["answer"] is None for e in episodes)}
summary["paired"] = {
    "corrected": sum(by_id[s,"hold"]["answer"] != gt[s] and by_id[s,"agent"]["answer"] == gt[s] for s in gt),
    "harmed": sum(by_id[s,"hold"]["answer"] == gt[s] and by_id[s,"agent"]["answer"] != gt[s] for s in gt),
    "action_taken": sum(by_id[s,"agent"]["rechecks"] for s in gt),
    "no_action": sum(1-by_id[s,"agent"]["rechecks"] for s in gt),
}
summary["by_action"] = {}
for action in (0,1):
    ids = [sid for sid in gt if by_id[sid,"agent"]["rechecks"] == action]
    summary["by_action"][str(action)] = {
        "n": len(ids),
        "hold_correct": sum(by_id[s,"hold"]["answer"] == gt[s] for s in ids),
        "agent_correct": sum(by_id[s,"agent"]["answer"] == gt[s] for s in ids),
        "corrected": sum(by_id[s,"hold"]["answer"] != gt[s] and by_id[s,"agent"]["answer"] == gt[s] for s in ids),
        "harmed": sum(by_id[s,"hold"]["answer"] == gt[s] and by_id[s,"agent"]["answer"] != gt[s] for s in ids),
        "gate_reasons": dict(Counter(by_id[s,"agent"]["trajectory"][0]["reobserve_reason"] for s in ids)),
    }
summary["by_truth_class"] = {}
for label in ("yes", "no"):
    ids = [sid for sid in gt if gt[sid] == label]
    summary["by_truth_class"][label] = {
        "n": len(ids),
        "hold_correct": sum(by_id[s,"hold"]["answer"] == label for s in ids),
        "agent_correct": sum(by_id[s,"agent"]["answer"] == label for s in ids),
        "rechecks": sum(by_id[s,"agent"]["rechecks"] for s in ids),
    }
summary["balanced_accuracy"] = {
    arm: sum(summary["by_truth_class"][label][f"{arm}_correct"] /
             summary["by_truth_class"][label]["n"] for label in ("yes","no")) / 2
    for arm in ("hold", "agent")
}
summary["by_sequence"] = {}
for seq in sorted({s["sequence_id"] for s in samples.values()}):
    ids = [sid for sid in gt if samples[sid]["sequence_id"] == seq]
    summary["by_sequence"][seq] = {
        "n": len(ids),
        "hold_correct": sum(by_id[s,"hold"]["answer"] == gt[s] for s in ids),
        "agent_correct": sum(by_id[s,"agent"]["answer"] == gt[s] for s in ids),
        "rechecks": sum(by_id[s,"agent"]["rechecks"] for s in ids),
        "corrected": sum(by_id[s,"hold"]["answer"] != gt[s] and by_id[s,"agent"]["answer"] == gt[s] for s in ids),
        "harmed": sum(by_id[s,"hold"]["answer"] == gt[s] and by_id[s,"agent"]["answer"] != gt[s] for s in ids),
    }
summary["per_sample"] = [{"sample_id": s, "gt": gt[s],
                            "sequence_id": samples[s]["sequence_id"],
                            "hold": by_id[s,"hold"]["answer"],
                            "agent": by_id[s,"agent"]["answer"],
                            "rechecks": by_id[s,"agent"]["rechecks"],
                            "gate_reason": by_id[s,"agent"]["trajectory"][0]["reobserve_reason"]}
                           for s in gt]
out = ROOT / "results" / (f"agent_replay_logits_{args.split}_summary.json" if args.variant == "logits"
                            else f"agent_replay_{args.split}_summary.json")
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
