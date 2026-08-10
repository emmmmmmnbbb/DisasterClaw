# DisasterClaw claim contract

This file freezes the claims that the paper may make. Any stronger wording
requires a new, traceable experiment and statistical support.

## Working title

**DisasterClaw: Calibrated Uncertainty-Driven Active Reinspection for
Language-Guided UAV Disaster Response**

## Problem statement

Language-guided UAV disaster response couples navigation with high-stakes
damage assessment. A fixed-altitude observation can be ambiguous because of
ground sampling distance, scene change, and domain shift, while unconditional
reinspection wastes flight time. DisasterClaw studies whether calibrated
predictive uncertainty can be converted into an explicit flight action:
descend and recenter for another observation only when the expected
information gain justifies its cost.

## Contributions

1. **Task and benchmark.** We formulate bird's-eye disaster VLN with active
   reinspection on georeferenced pre/post xBD imagery and evaluate navigation,
   damage judgment, calibration, and flight cost jointly.
2. **Method.** We introduce a calibrated uncertainty-driven reinspection
   controller that combines bitemporal damage probabilities, temperature
   scaling, and expected information gain with descend-and-recenter actions.
3. **Evaluation protocol.** We provide event-disjoint split checks, an
   evidence-rich manually reviewed task subset, paired multi-seed evaluation,
   confidence intervals, and explicit failure analysis.

HSPM, STMR, the geographic semantic map, hybrid grounding, and topological
memory are system components, not standalone contribution claims.

## Research questions

- **RQ1:** Are the damage probabilities calibrated well enough to support
  action selection?
- **RQ2:** Does the policy identify when reinspection is useful better than
  no, random, fixed, and heuristic reinspection?
- **RQ3:** Does reinspection improve damage judgment on evidence-bearing
  episodes?
- **RQ4:** What uncertainty or judgment gain is obtained per additional
  action and meter flown?
- **RQ5:** How robust are calibration and decisions on held-out disaster
  events and degraded observations?
- **RQ6:** How does active reinspection affect navigation SR, NE, SPL, and
  path length?

## Evidence status

- Strict event-disjoint baseline calibration: test ECE 0.095→0.037 with
  unchanged accuracy 0.765; holdout ECE 0.046→0.084 under the same
  validation-fitted temperature. In-domain calibration improves; holdout
  calibration does not uniformly improve.
- Strict difference attention does not repeat the old holdout accuracy lift:
  test/holdout accuracy stay ≈0.765/0.803 for both fusions; calibrated test
  ECE is slightly better for difference attention, calibrated holdout ECE is
  better for concatenation.
- Strict YOLO detector (event-disjoint) reaches test mAP@0.5≈0.144 and
  holdout mAP@0.5≈0.175; low recall remains a grounding bottleneck.
- The historical 40-task navigation run has low SR and no statistically
  significant B0--B3 improvement. It supports feasibility and failure
  analysis only until the strict evidence-rich rerun finishes.
- The historical six-policy reinspection evaluation contains only one
  evidence-triggering episode. It is underpowered and cannot support a
  superiority claim.
- Historical (non-strict) E10/E15 numbers are superseded for claim-making by
  the event-disjoint rerun above.

## Prohibited claims

- Do not claim that HSPM, memory, or active reinspection significantly
  improves SR from the historical runs.
- Do not describe `semSR` as strict target success.
- Do not describe oracle-seeded memory as online learning performance.
- Do not claim unseen-disaster generalization from a model whose training
  events overlap the evaluation events.
- Do not equate improved ECE, Brier score, or NLL with improved accuracy.
- Do not imply real-flight safety, multi-UAV coordination, or first-person
  UAV validation; the current platform is a single-UAV bird's-eye simulator.
- Do not report pending strict experiments as completed results.
