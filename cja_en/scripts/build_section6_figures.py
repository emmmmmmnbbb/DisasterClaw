#!/usr/bin/env python3
"""Build Section 6 figures from the audited offline experiment records.

This script does not run a model. It verifies the two source files against the
frozen source hashes, derives the plotted values, and writes PGFPlots
sources plus a small provenance manifest beneath ``cja_en/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cja_en"
RUN = ROOT / "runs/benchmarks/paper_cja_mech_v1"
EXPECTED_HASHES = {
    "runs/benchmarks/paper_cja_mech_v1/budget_allocation.json":
        "09315295843e905a3de77f38a9b6ad9ba04500f75a3353d4f30dcdce2e1b4d2d",
    "runs/benchmarks/paper_cja_mech_v1/final_fov/fov_ladder_eval_items.jsonl":
        "d825ecaf07595e63997775ec53de627d504d201d4b9a606bb33ba7f7a3753958",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audited_read(relative_path: str, *, jsonl: bool = False):
    path = ROOT / relative_path
    expected = EXPECTED_HASHES.get(relative_path)
    if not expected:
        raise RuntimeError(f"Source is absent from the audit: {relative_path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Source changed after audit: {relative_path}")
    raw = path.read_text(encoding="utf-8")
    if jsonl:
        return [json.loads(line) for line in raw.splitlines() if line.strip()], actual
    return json.loads(raw), actual


def write(relative_path: str, content: str) -> str:
    path = OUT / relative_path
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return sha256(path)


def build_view_change_figure(rows: list[dict]) -> tuple[str, dict[str, int]]:
    corrected = harmed = wrong_to_wrong = 0
    for row in rows:
        truth = int(row["y"])
        cruise = max(row["views"]["cruise"]["probs"], key=row["views"]["cruise"]["probs"].get)
        floor = max(row["views"]["floor"]["probs"], key=row["views"]["floor"]["probs"].get)
        labels = ["no-damage", "minor-damage", "major-damage", "destroyed"]
        cruise_id, floor_id = labels.index(cruise), labels.index(floor)
        if cruise_id == floor_id:
            continue
        if cruise_id != truth and floor_id == truth:
            corrected += 1
        elif cruise_id == truth and floor_id != truth:
            harmed += 1
        else:
            wrong_to_wrong += 1
    counts = {
        "incorrect_to_correct": corrected,
        "correct_to_incorrect": harmed,
        "incorrect_to_different_incorrect": wrong_to_wrong,
        "changed_total": corrected + harmed + wrong_to_wrong,
    }
    tex = rf"""
\definecolor{{cjaBlue}}{{HTML}}{{0077BB}}
\definecolor{{cjaOrange}}{{HTML}}{{EE7733}}
\definecolor{{cjaGray}}{{HTML}}{{777777}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=.70\linewidth,
  height=5.2cm,
  xbar,
  bar width=11pt,
  xmin=0,
  xmax=215,
  xlabel={{Buildings (n)}},
  ytick={{0,1,2}},
  yticklabels={{Changed but still incorrect,Harmed,Corrected}},
  nodes near coords,
  nodes near coords align={{horizontal}},
  enlarge y limits=.24,
  axis line style={{black!65}},
  tick style={{black!65}},
  xmajorgrids=true,
  grid style={{black!12}},
]
\addplot+[draw=black!60,fill=cjaBlue] coordinates {{({corrected},2)}};
\addplot+[draw=black!60,fill=cjaOrange] coordinates {{({harmed},1)}};
\addplot+[draw=black!60,fill=cjaGray] coordinates {{({wrong_to_wrong},0)}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return tex, counts


def build_budget_figure(summary: dict) -> str:
    labels = {
        "none": "No reobservation",
        "random": "Random",
        "entropy_uncal": "Raw entropy",
        "entropy_cal": "Calibrated entropy",
        "expected_gain": "Expected entropy reduction",
        "conformal": "APS set size",
        "oracle": "Label-informed diagnostic",
    }
    styles = {
        "none": "black,densely dashed,mark=none",
        "random": "gray!75!black,mark=square*",
        "entropy_uncal": "orange!85!black,mark=triangle*",
        "entropy_cal": "blue!75!black,mark=*",
        "expected_gain": "teal!80!black,mark=diamond*",
        "conformal": "violet!85!black,mark=x",
        "oracle": "red!70!black,dashed,mark=o",
    }
    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=.98\linewidth,height=7.0cm,",
        r"  xlabel={Reobservation fraction $b$},ylabel={Macro-F1},",
        r"  xmin=0,xmax=1,ymin=.60,ymax=.71,",
        r"  xtick={0,.1,.25,.5,1},ytick={.60,.62,.64,.66,.68,.70},",
        r"  grid=major,grid style={black!12},",
        r"  legend style={at={(.5,-.24)},anchor=north,draw=none,font=\footnotesize},",
        r"  legend columns=2,",
        r"]",
    ]
    for key in labels:
        coords = " ".join(
            f"({float(row['budget']):g},{float(row['macro_f1']):.8f})"
            for row in summary["curves"][key]
        )
        lines.extend(
            [
                f"\\addplot+[{styles[key]},thick] coordinates {{{coords}}};",
                "\\addlegendentry{" + labels[key] + "}",
            ]
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    return "\n".join(lines)


def main() -> None:
    budget_rel = "runs/benchmarks/paper_cja_mech_v1/budget_allocation.json"
    items_rel = "runs/benchmarks/paper_cja_mech_v1/final_fov/fov_ladder_eval_items.jsonl"
    budget, budget_hash = audited_read(budget_rel)
    rows, items_hash = audited_read(items_rel, jsonl=True)

    view_tex, counts = build_view_change_figure(rows)
    if counts != {
        "incorrect_to_correct": 191,
        "correct_to_incorrect": 126,
        "incorrect_to_different_incorrect": 66,
        "changed_total": 383,
    }:
        raise RuntimeError(f"Unexpected transition counts: {counts}")

    outputs = {
        "figures/view_change_outcomes.tex": write("figures/view_change_outcomes.tex", view_tex),
        "figures/budget_curve.tex": write("figures/budget_curve.tex", build_budget_figure(budget)),
    }
    manifest = {
        "schema": "cja-section6-figure-manifest/1.0",
        "scope": "Audited offline records only; no model execution",
        "inputs": {
            budget_rel: budget_hash,
            items_rel: items_hash,
        },
        "derived_values": counts,
        "outputs": outputs,
        "limitations": [
            "The perception backbone recorded for these inputs has prior exposure to evaluation events.",
            "The random allocation curve represents the recorded seeded allocation, not uncertainty over random-policy seeds.",
            "The budget curve is an offline building-allocation analysis and does not represent a UAV route cost.",
        ],
    }
    write("review/section6_figure_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"derived_values": counts, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
