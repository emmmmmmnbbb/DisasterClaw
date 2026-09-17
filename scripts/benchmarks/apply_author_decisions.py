#!/usr/bin/env python3
"""把作者决策 CSV 回写进题集的 review.human_review 字段。

decision 词表：approve -> approved / reject -> rejected / adjust -> rejected(带 note)。
human_review 要求 status 与 author_checked 同时为真才计入人工通过
（review_agent_vqa_testset.py 的逻辑）。回写到一个新文件，不改原候选。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ANSWER_EN = {
    "是": "yes", "否": "no", "损伤": "damaged", "无损伤": "no-damage",
    "北": "north", "东北": "northeast", "东": "east", "东南": "southeast",
    "南": "south", "西南": "southwest", "西": "west", "西北": "northwest",
}
READ_ONLY_FIELDS = ("sheet", "panel", "question_type", "event", "ambiguity_flags")
DECISIONS = {"approve": "approved", "reject": "rejected", "adjust": "rejected"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {
            "qid", "decision", "reviewer", "answer", *READ_ONLY_FIELDS,
        }.issubset(reader.fieldnames):
            raise ValueError(f"author decisions CSV missing required columns: {path}")
        return list(reader)


def apply_decisions(data: dict, decisions: list[dict], template: list[dict]) -> dict:
    items = data.get("items") or []
    by_id = {str(item.get("id") or ""): item for item in items}
    template_by_id = {str(row.get("qid") or ""): row for row in template}
    if len(by_id) != len(items) or len(template_by_id) != len(template):
        raise ValueError("candidate or review template has duplicate qids")
    seen = set()
    for row in decisions:
        qid = str(row.get("qid") or "").strip()
        if not qid or qid in seen or qid not in by_id or qid not in template_by_id:
            raise ValueError(f"missing, duplicate, or unknown decision qid: {qid!r}")
        seen.add(qid)
        original = template_by_id[qid]
        if any(row.get(field) != original.get(field) for field in READ_ONLY_FIELDS):
            raise ValueError(f"review-sheet metadata changed for {qid}")
        item = by_id[qid]
        if (
            original.get("answer") != item.get("answer")
            or original.get("question_type") != item.get("question_type")
            or original.get("event") != item.get("disaster")
            or original.get("ambiguity_flags") != ",".join(
                (item.get("review") or {}).get("ambiguity_flags") or []
            )
        ):
            raise ValueError(f"review template does not match candidate item {qid}")
        expected = ANSWER_EN.get(str(item.get("answer") or ""), item.get("answer"))
        if row.get("answer") not in {item.get("answer"), expected}:
            raise ValueError(f"answer label changed rather than translated for {qid}")
        decision = str(row.get("decision") or "").strip().lower()
        reviewer = str(row.get("reviewer") or "").strip()
        if decision not in DECISIONS or not reviewer:
            raise ValueError(f"invalid/missing author decision or reviewer for {qid}")
        note = str(row.get("note") or "").strip()
        if decision == "adjust" and not note:
            raise ValueError(f"adjust decision requires a note for {qid}")
        hr = item.setdefault("review", {}).setdefault("human_review", {})
        hr["status"] = DECISIONS[decision]
        hr["author_checked"] = True
        hr["reviewer"] = reviewer
        # The CSV does not contain the original time of the human inspection.
        # Do not infer it from the time at which this import script ran.
        hr["reviewed_at"] = ""
        hr["note"] = note
    if seen != set(by_id) or seen != set(template_by_id):
        raise ValueError("decision CSV must cover every candidate and template qid")
    data["items_sha256"] = sha256_json(items)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    decisions_path = Path(args.decisions)
    testset_path = Path(args.testset)
    template_path = Path(args.template)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"reviewed testset already exists: {out}")
    data = json.loads(testset_path.read_text(encoding="utf-8"))
    data = apply_decisions(data, read_csv(decisions_path), read_csv(template_path))
    data["author_decisions_receipt"] = {
        "source_testset": str(testset_path),
        "source_testset_sha256": sha256(testset_path),
        "decisions_csv": str(decisions_path),
        "decisions_csv_sha256": sha256(decisions_path),
        "review_template": str(template_path),
        "review_template_sha256": sha256(template_path),
        "role": "CSV-author decisions mechanically imported; not independent visual verification",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    c = Counter(it["review"]["human_review"]["status"] for it in data["items"])
    print(f"回写完成 -> {out}")
    print(f"human_review status: {dict(c)}  决策行: {len(data['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
