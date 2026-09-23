"""Combine the two vegetation strata without treating ROIs as independent sites."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("exhaustive_descend", "exhaustive_paths")


def csv_rows(path):
    return list(csv.DictReader(path.open()))


def load_variant(prefix):
    rows = {}
    for split in SPLITS:
        path = ROOT / "results" / f"agent_replay_{prefix}{split}_summary.json"
        if not path.exists():
            return None
        rows[split] = json.loads(path.read_text())
    total = {"n": sum(rows[s]["hold"]["n"] for s in SPLITS),
             "hold_correct": sum(rows[s]["hold"]["correct"] for s in SPLITS),
             "agent_correct": sum(rows[s]["agent"]["correct"] for s in SPLITS),
             "rechecks": sum(rows[s]["agent"]["rechecks"] for s in SPLITS),
             "corrected": sum(rows[s]["paired"]["corrected"] for s in SPLITS),
             "harmed": sum(rows[s]["paired"]["harmed"] for s in SPLITS)}
    total["hold_accuracy"] = total["hold_correct"]/total["n"]
    total["agent_accuracy"] = total["agent_correct"]/total["n"]
    return {"by_stratum": {s: {"n": rows[s]["hold"]["n"],
                               "hold_correct": rows[s]["hold"]["correct"],
                               "agent_correct": rows[s]["agent"]["correct"],
                               "rechecks": rows[s]["agent"]["rechecks"],
                               "corrected": rows[s]["paired"]["corrected"],
                               "harmed": rows[s]["paired"]["harmed"]}
                           for s in SPLITS}, "combined": total}


def main():
    fixed = {}
    for split in SPLITS:
        rows = csv_rows(ROOT / "results" / f"{split}_summary.csv")
        comparison = json.loads((ROOT / "results" / f"{split}_comparison.json").read_text())
        fixed[split] = {"n": int(rows[0]["planned_n"]),
                        "correct": {r["branch"]: int(r["correct"]) for r in rows},
                        "C_vs_B_help": comparison["C_B_help"],
                        "C_vs_B_harm": comparison["C_B_harm"]}
    n = sum(v["n"] for v in fixed.values())
    combined_fixed = {"n": n, "correct": {branch: sum(v["correct"][branch] for v in fixed.values())
                                        for branch in "ABC"},
                      "C_vs_B_help": sum(v["C_vs_B_help"] for v in fixed.values()),
                      "C_vs_B_harm": sum(v["C_vs_B_harm"] for v in fixed.values())}
    combined_fixed["accuracy"] = {b: combined_fixed["correct"][b]/n for b in "ABC"}
    out = {"scope": "public annotated 100-to-30-m MESSI vegetation pairs",
           "fixed_observation": {"by_stratum": fixed, "combined": combined_fixed},
           "self_reported_agent": load_variant(""),
           "token_entropy_agent": load_variant("logits_")}
    path = ROOT / "results/public_sample_summary.json"
    path.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
