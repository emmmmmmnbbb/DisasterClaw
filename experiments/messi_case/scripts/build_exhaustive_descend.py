"""Include every unambiguous fixed-grid region in public MESSI descend pairs.

This is a coverage expansion, separate from the original 12-case selection.
The existing 100_0001 pilot sequence stays excluded from formal scoring.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ("100_0001", "100_0002", "100_0003", "100_0004", "100_0042", "100_0043")
PILOT = "100_0001"
SPLIT = "exhaustive_descend"


def main():
    selections, exclusions, counts = [], [], {}
    for seq in SEQUENCES:
        path = ROOT / "derived" / seq / "vegetation" / "candidates.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Generate candidates first: {path}")
        candidates = [json.loads(line) for line in path.read_text().splitlines()]
        if len(candidates) != 24:
            raise RuntimeError(f"Expected all 24 fixed grid cells for {seq}; got {len(candidates)}")
        counts[seq] = dict(Counter(x["status"] for x in candidates))
        for x in candidates:
            reason = None
            if seq == PILOT:
                reason = "pilot_sequence_used_for_prompt_development"
            elif x["status"] == "review":
                reason = "annotation_fractions_ambiguous_or_discordant"
            if reason:
                exclusions.append({"sequence_id": seq, "sample_id": x["id"],
                                   "reason": reason, "candidate_status": x["status"]})
                continue
            assert x["status"] in ("candidate_positive", "candidate_negative")
            selections.append({"sequence": seq, "candidate_id": x["id"],
                               "answer": x["status"].removeprefix("candidate_"),
                               "split": SPLIT, "scene_group_id": "Agamim_site_"+seq[-4:],
                               "question": "Does the indicated ground region contain visible vegetation, including grass, trees, or shrubs?",
                               "evidence": "MESSI vegetation (class 12) mask agreement at both heights; automatic reference"})
    manifest_dir = ROOT / "manifests"
    path = manifest_dir / f"{SPLIT}_selection.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(selections, ensure_ascii=False, indent=2) + "\n")
    with (manifest_dir / f"{SPLIT}_exclusions.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("sequence_id", "sample_id", "reason", "candidate_status"))
        w.writeheader(); w.writerows(exclusions)
    print(json.dumps({"selected": len(selections), "excluded": len(exclusions),
                      "sequence_counts": counts,
                      "selection_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
