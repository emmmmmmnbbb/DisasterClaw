# Strict evaluation protocol

This protocol is fixed before running the replacement paper experiments.

## Data split

- Split by disaster event, never by tile.
- Train, validation, test, and holdout event sets must be pairwise disjoint.
- The split generator must emit a manifest containing event names and hashes;
  training aborts when disjointness validation fails.
- Temperature is fitted on validation only. Test and holdout are evaluated
  once after model and policy choices are frozen.

## Evaluation sets

- Navigation benchmark: the existing 40-task set is retained only for
  historical comparability.
- Active-reinspection benchmark:
  `backend/data/benchmarks/vln_recheck_testset.json`, 40 approved
  evidence-rich tasks spanning ten xBD events.
- The evidence-rich set measures reinspection decisions; its all-easy,
  near-target construction is disclosed and it is not used to claim
  long-range navigation performance.

## Configurations

- Grounding is fixed to `hybrid`.
- Navigation: B0, B1, B2, B3.
- Reinspection: NONE, RANDOM, FIXED, HEURISTIC, ENTROPY, INFOGAIN.
- Seeds: 41, 42, 43. Each item is paired across configurations and seeds.
- No configuration-specific task filtering or early result selection.

## Primary outcomes

1. Evidence-bearing macro damage F1 after the final observation.
2. Mean observed uncertainty reduction per completed or episode-finalized
   reinspection.
3. Reinspection decision utility: damage gain per extra action and meter.

## Secondary outcomes

- SR, semSR (reported separately), NE, semNE, SPL, steps, and path length.
- Trigger, completion, pending, confirmation, dismissal, and inconclusive
  rates.
- Accuracy, ECE, Brier score, and NLL for bitemporal damage prediction.

## Statistical analysis

- Aggregate repeated seeds at the item level before task-level inference.
- Report bootstrap 95% confidence intervals for all primary outcomes.
- Use paired permutation or Wilcoxon signed-rank tests as appropriate.
- Correct the six-policy family with Holm's method.
- Report effect sizes and exact sample counts even when non-significant.
- Treat a zero-trigger policy/run as missing for conditional delta-U, while
  reporting its trigger rate as zero; never impute a successful reduction.

## Robustness and failure analysis

- GPS perturbation: 0, 2, 5, and 10 m.
- Degraded observation: valid crop versus forced degraded geometry.
- Held-out disaster events: report separately from same-domain test.
- Categorize failures as grounding, planning, perception, reinspection, or
  boundary/coverage.

## Reporting guardrails

- Historical repeat-1 results remain labeled “historical diagnostic.”
- Strict results replace historical numbers only when all seeds finish.
- Interrupted or crashed episodes remain in the denominator and are listed.
- Hardware, wall time, model hashes, environment, and source run directories
  are included in each artifact.
