# P3 ablation implementation notes

Status: design audit only. Implementation starts after P2 development freeze.

## Problems in the current code

1. `report_agent_vqa.py` hard-codes `A5_EXPECTED` as the matched-budget reference. The new main policy is `T1_TASK`, and the old A5 table belongs to the incompatible four-class perception chain.
2. Existing `AB_CENTER`, `AB_DESCEND`, and `AB_FULL` use a fixed trigger plus per-question action counts. They do not replay the task policy's action schedule and do not centre a visible-but-unlocalized damage marker.
3. In the fixed fallback, `center_only` can emit `(north,east)=(0,0)` when the detector misses the target. That is a no-op rather than a centre-only action.
4. The unchanged-view reask exists as a separate diagnostic script but is not represented in the unified ablation report.
5. `D0_RULE`, `V2_STATE_VLM`, and `A0_VLM` exist and their switch identities are tested, but they have not been run on the new frozen final set.
6. The question-only and fitted per-question-type majority baselines have no reproducible fit/evaluation artifact.

## Required paired design

First run and freeze `T1_TASK`. Export its per-`qid`, per-step action schedule. All motion ablations consume the same schedule and generation-call keys:

- `AB_NOOP`: acquire another observation at the identical pose.
- `AB_CENTER`: move toward the public damage marker but retain altitude.
- `AB_DESCEND`: descend at the current geographic centre.
- `AB_FULL`: centre on the public damage marker and descend.
- `AB_WIDE`: retain the full task ROI and reject any scheduled descent that would crop it.

For count/spatial questions, centre-only is expected to be a no-op because the task action preserves the ROI centre. Results must therefore be reported by question type; pooling can hide this structural equivalence.

The report must verify equal attempted observation count and equal generation-call keys, in addition to equal executed motion count. A failed or geometrically zero action must not be counted as equivalent to a successful action without being labelled.

## Answer-path controls

- `D0_RULE`: frozen ChangeOS structured evidence plus deterministic answer mapping.
- `V2_STATE_VLM`: static image/state plus pure VLM.
- `A0_VLM`: search-capable, no-reobservation pure VLM.
- Hybrid counterparts retain the same perception and navigation switches.

These rows belong in a diagnostic table because changing answer mode and policy together prevents clean attribution.

## Language-only baselines

- Fit the majority answer independently for each question type using development data only; save counts, selected answer, ties, source hash, and fit split.
- Train/evaluate a question-only baseline without image, coordinates, target subtype, or detector evidence. Exact UUIDs must be normalized so the model cannot memorize target identifiers.
- Apply both frozen baselines to the untouched final set. Report per-type and aggregate accuracy alongside class frequencies.

