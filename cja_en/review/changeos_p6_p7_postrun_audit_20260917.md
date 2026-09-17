# ChangeOS P6/P7 post-run evidence audit (2026-09-17)

## Material Passport

- Material: frozen P6 three-event online report and three P7 final cost audits.
- Verification status: `ANALYZED` (all 12 original shards passed the strict local audit; no independent reproduction run was made).
- P6 protocol SHA-256: `41ccbb5fed5e8ec95be36b20e7dd43693cc237fb2d1e63246815638a4e83ed2c`.
- P6 aggregate SHA-256: `9e4370325bc9e927ab2f4e03847c4ecae41719f49ccbb293ecfc8baa2545a9f4`.
- Source: `runs/benchmarks/changeos_binary/p6_final_three_event_v1_20260915_report.json`; 12 per-shard `p6_shard_audit.json` files. Cost sources: `runs/benchmarks/changeos_binary/p7_cost_observability_final_repeat{0,1,2}_20260917/report.json`.

## Identity and completeness

All 12 planned shard/repeat combinations are present and individually marked valid. The aggregate reports `VALIDATED_COMPLETE`: 3,360 online episodes, comprising 160 distinct questions, three generation repeats and seven policies. Each shard has 280 rows; the 12 `results.json` files report zero execution errors. The final question set has 44 ROIs from three events. Repeats are repeated measurements of the same questions, **not** 480 independent questions. The report's four primary intervals use 5,000 ROI-cluster bootstrap draws and Bonferroni 98.75% limits; the cluster resampling does not establish generalization to unseen disasters.

## Confirmatory results

| Policy | Correct / 480 | Accuracy | Executed reobservations | Mean episode wall time |
|---|---:|---:|---:|---:|
| A0_HOLD | 347 | 72.29% | 0 | 17.03 s |
| A1_RANDOM | 326 | 67.92% | 419 | 33.65 s |
| A2_ALWAYS | 307 | 63.96% | 923 | 51.07 s |
| A3U_RAW_ENTROPY | 339 | 70.63% | 626 | 39.61 s |
| A3_ENTROPY (supplemental) | 341 | 71.04% | 628 | 38.94 s |
| A4_CONFORMAL (supplemental) | 338 | 70.42% | 628 | 38.95 s |
| T1_TASK | 360 | 75.00% | 161 | 17.92 s |

| Prespecified contrast | Paired difference | 98.75% ROI-cluster interval | Interpretation |
|---|---:|---:|---|
| T1_TASK − A0_HOLD | +2.71 pp | [−2.04, +7.50] pp | Includes zero; no stable gain established over holding. |
| T1_TASK − A1_RANDOM | +7.08 pp | [0.00, +14.70] pp | Touches zero; do not claim a strict positive effect. |
| T1_TASK − A2_ALWAYS | +11.04 pp | [+0.86, +21.38] pp | Positive within the prespecified ROI-cluster analysis. |
| T1_TASK − A3U_RAW_ENTROPY | +4.38 pp | [−4.33, +12.96] pp | Includes zero; no stable gain established over raw entropy. |

T1_TASK's accuracy by repeat is 75.00%, 74.38%, and 75.63%; A0_HOLD's is 72.50%, 72.50%, and 71.88%. This repeat consistency does not remove uncertainty from only 44 ROI clusters and three events. T1_TASK is lower than A0_HOLD in Hurricane Michael (54.86% vs 58.33%) but higher in Moore (84.31% vs 77.12%) and Nepal (83.06% vs 79.23%). These event results are descriptive. The policy comparison measures whole-policy outcomes, not an action-only causal effect, because later VLM call counts may differ.

Within-policy episode fields record 36 corrected and 21 harmed answers for T1_TASK, versus 74 corrected and 132 harmed for A2_ALWAYS. These are transition counts within each policy's own episodes, not paired counterfactual differences against A0_HOLD. Realized policies form accuracy–cost operating points; no randomized reobservation-cap sweep was performed. The predeclared decision rule therefore selects the platform/mechanism-measurement framing, **not** a claimed stable T1_TASK advantage over holding or entropy.

## P7 final cost observability

The existing P7 auditor was applied separately to each generation repeat (four shards per audit); it is not repeat-aware when all 12 shards are passed together. The three reports cover all 3,360 episodes. For every episode, logged recheck requests, budget-decrement-inferred allocations, executed rechecks and reported `n_reobservations` agree; no recheck request/execution discrepancy was found. Search-action requested/allocated detail and failure reasons are not comprehensively logged, so P7's first checklist item remains incomplete.

State-to-state geometric motion is complete for 3,318/3,360 episodes. The remaining 42 end in `fly_relative` with no next recorded position; they are excluded from complete-trajectory claims. Per-policy complete counts over 480 episodes are A0 474, A1 477, A2 464, A3U 475, A3 476, A4 476 and T1 476. T1_TASK's observed mean is 0.335 executed reobservations and 17.92 s wall time per question-repeat; A2_ALWAYS's is 1.923 and 51.07 s. End-to-end wall time is observed software runtime on this execution, **not** flight duration. Horizontal/vertical distances are state-derived geometric estimates; no render, ChangeOS, controller or Qwen component timing, airframe dynamics or PX4/Gazebo action mapping is available. `d/v` flight-time estimates must not be described as measured flight time.

## Statistical fallacy scan (11/11 checked)

1. Simpson's paradox: checked overall versus event directions; T1–A0 is positive overall but negative in Hurricane Michael, so event heterogeneity must be shown. This is not a complete all-strata reversal.
2. Ecological fallacy: ROI-cluster intervals are not evidence about unseen events or individual aircraft operations.
3. Berkson's paradox: selected, coverable xBD ROI population limits external validity; no specific induced negative correlation established.
4. Collider bias: no covariate-adjusted causal model is reported, so no adjustment-induced collider claim is made.
5. Base-rate neglect: task accuracy is reported with question-type/event breakdown; it is not a sensitivity/specificity endpoint. Development-set majority-answer baseline is not a final-set baseline.
6. Regression to the mean: policies were not selected by extreme final outcomes; development-guided rule selection is separately disclosed.
7. Survivorship bias: all 3,360 validated episodes remain in the accuracy denominator, including abstentions; 42 incomplete terminal-motion traces affect only complete-motion summaries.
8. Look-elsewhere effect: four primary contrasts were frozen and Bonferroni-corrected; subgroup findings and supplemental policies remain descriptive.
9. Garden of forking paths: protocol and code/input hashes were frozen before P6; post-run route selection follows the stated decision rule. The prior development work remains a design source, not independent confirmation.
10. Correlation versus causation: policy-level experimental contrasts support statements about these frozen policy executions, not isolated action-only or physical-flight effects.
11. Reverse causality: temporal order is observed in episodes, but within-policy correction/harm counts cannot identify counterfactual effects of a single action.

## Outstanding gates

- P4: no randomized common-cap budget curve; only policy operating points and earlier offline diagnostic sweeps.
- P5: current-visible GT + Qwen, full-task GT + Qwen, and always-floor/full-coverage reference remain absent.
- P7: module-level timings, comprehensive search-action request/allocation/failure logs, and physical action mapping remain absent. Conservative imagery-grounded active-observation wording avoids a flight-validation claim.
- P8: replace the old four-class/offline-first main paper with this frozen binary closed-loop result chain; move historical four-class/VLN diagnostics to an appendix and preserve the negative/uncertain comparisons.
