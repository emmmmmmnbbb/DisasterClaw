"""Freeze all annotation-consistent public Agamim path frame pairs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "exhaustive_paths"
QUESTION = "Does the indicated ground region contain visible vegetation, including grass, trees, or shrubs?"


def main():
    candidates = [json.loads(line) for line in
                  (ROOT / "derived/public_paths/candidates.jsonl").read_text().splitlines()]
    selected = []
    for x in candidates:
        if x["status"] not in ("candidate_positive", "candidate_negative"):
            continue
        selected.append({"sequence": x["sequence"], "candidate_id": x["id"],
                         "answer": x["status"].removeprefix("candidate_"),
                         "split": SPLIT, "scene_group_id": "Agamim_"+x["sequence"],
                         "question": QUESTION,
                         "evidence": "MESSI vegetation (class 12) mask agreement at both heights; automatic reference"})
    path = ROOT / "manifests" / f"{SPLIT}_selection.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps({"paired": len(candidates), "selected": len(selected),
                      "statuses": dict(Counter(x["status"] for x in candidates)),
                      "selection_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
