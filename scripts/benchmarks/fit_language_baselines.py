#!/usr/bin/env python3
"""Fit leakage-controlled majority and question-only Agent-VQA baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_question(question: str) -> str:
    """Remove only opaque target identifiers; retain the natural-language task."""
    return " ".join(UUID_RE.sub("<TARGET_ID>", str(question or "")).split())


def _winner(counts: Counter[str]) -> tuple[str, list[str]]:
    if not counts:
        return "", []
    top = max(counts.values())
    ties = sorted(answer for answer, count in counts.items() if count == top)
    return ties[0], ties


def fit(items: list[dict]) -> dict:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_template: dict[str, Counter[str]] = defaultdict(Counter)
    template_type: dict[str, str] = {}
    for item in items:
        qtype = str(item.get("question_type") or "")
        answer = str(item.get("answer") or "")
        template = normalize_question(item.get("question", ""))
        if not qtype or not answer or not template:
            continue
        by_type[qtype][answer] += 1
        by_template[template][answer] += 1
        previous = template_type.setdefault(template, qtype)
        if previous != qtype:
            raise ValueError(f"normalized template crosses question types: {template!r}")

    majority = {}
    for qtype, counts in sorted(by_type.items()):
        answer, ties = _winner(counts)
        majority[qtype] = {
            "answer": answer, "ties": ties, "counts": dict(sorted(counts.items())),
            "n": sum(counts.values()),
        }
    question_only = {}
    for template, counts in sorted(by_template.items()):
        answer, ties = _winner(counts)
        question_only[template] = {
            "question_type": template_type[template], "answer": answer,
            "ties": ties, "counts": dict(sorted(counts.items())),
            "n": sum(counts.values()),
        }
    return {"majority_by_question_type": majority, "question_only_by_template": question_only}


def evaluate(items: list[dict], model: dict) -> dict:
    rows = []
    for item in items:
        qtype = str(item.get("question_type") or "")
        template = normalize_question(item.get("question", ""))
        majority = model["majority_by_question_type"].get(qtype, {}).get("answer", "")
        template_rec = model["question_only_by_template"].get(template)
        question_answer = template_rec.get("answer", "") if template_rec else majority
        gt = str(item.get("answer") or "")
        rows.append({
            "qid": str(item.get("id") or ""), "question_type": qtype,
            "normalized_question": template, "gt_answer": gt,
            "majority_answer": majority, "majority_correct": majority == gt,
            "question_only_answer": question_answer,
            "question_only_correct": question_answer == gt,
            "question_only_fallback": template_rec is None,
        })

    def aggregate(field: str) -> dict:
        out = {"n": len(rows), "accuracy": sum(bool(r[field]) for r in rows) / len(rows) if rows else 0.0}
        out["by_question_type"] = {}
        for qtype in sorted({r["question_type"] for r in rows}):
            selected = [r for r in rows if r["question_type"] == qtype]
            out["by_question_type"][qtype] = {
                "n": len(selected),
                "accuracy": sum(bool(r[field]) for r in selected) / len(selected),
            }
        return out

    return {
        "majority": aggregate("majority_correct"),
        "question_only": aggregate("question_only_correct"),
        "question_only_fallbacks": sum(r["question_only_fallback"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-testset", type=Path, required=True)
    parser.add_argument("--eval-testset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fit-split", default="development")
    args = parser.parse_args()
    fit_data = json.loads(args.fit_testset.read_text(encoding="utf-8"))
    eval_data = json.loads(args.eval_testset.read_text(encoding="utf-8"))
    model = fit(fit_data.get("items", []))
    result = evaluate(eval_data.get("items", []), model)
    artifact = {
        "schema": "agent-vqa-language-baselines/1.0",
        "fit_testset": str(args.fit_testset),
        "fit_testset_sha256": sha256(args.fit_testset),
        "fit_split": args.fit_split,
        "eval_testset": str(args.eval_testset),
        "eval_testset_sha256": sha256(args.eval_testset),
        "same_data_diagnostic_only": sha256(args.fit_testset) == sha256(args.eval_testset),
        "features": ["normalized_question_text", "question_type"],
        "excluded_features": ["image", "coordinates", "target_subtype", "detector_evidence", "target_uuid"],
        "model": model,
        "evaluation": result,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "language_baselines.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Agent-VQA language-only baselines", "",
        f"- Same-data diagnostic only: `{str(artifact['same_data_diagnostic_only']).lower()}`",
        f"- Normalized templates: {len(model['question_only_by_template'])}",
        f"- Question-only fallbacks: {result['question_only_fallbacks']}", "",
        "| baseline | n | accuracy |", "| --- | ---: | ---: |",
        f"| majority by question type | {result['majority']['n']} | {result['majority']['accuracy']:.4f} |",
        f"| question only | {result['question_only']['n']} | {result['question_only']['accuracy']:.4f} |",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "majority_accuracy": result["majority"]["accuracy"],
        "question_only_accuracy": result["question_only"]["accuracy"],
        "templates": len(model["question_only_by_template"]),
        "same_data_diagnostic_only": artifact["same_data_diagnostic_only"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
