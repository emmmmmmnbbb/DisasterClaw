# Triggered MESSI trajectory examples

These two examples are taken from the 20-sample same-trigger control. Hold and
both second-observation arms share the first high-view model output. The
repeated-high and real-low arms each use one controller action and two
deterministic answer calls. Entropy is normalized binary entropy from the
SegFormer-derived task score. The score is not calibrated.

## Correction: `100_0041_r11`

Reference answer: YES.

| Step | Observation / action | SegFormer task score | Candidate answer | Controller decision / budget |
|---:|---|---:|---|---|
| 1 | High ROI at 99.9 m | 0.218 | NO (confidence 0.782) | `reobserve`, entropy 0.757, one action available |
| 2a | Repeat same high ROI at 99.9 m | 0.218 | NO | `answer`, `sufficient_evidence`, budget 0 |
| 2b | Registered low ROI at 29.9 m | 0.809 | YES (confidence 0.809) | `answer`, `sufficient_evidence`, budget 0 |

The gate requested `north_m=0`, `east_m=0`, `up_m=-70.0`, speed 10 m/s. The
repeated-high arm retained the high image; the real-low arm received the
pre-registered low crop only after the gate action. Repeating the high view
preserved the wrong answer, while the low view corrected it.

## Harm: `100_0004_r08`

Reference answer: NO.

| Step | Observation / action | SegFormer task score | Candidate answer | Controller decision / budget |
|---:|---|---:|---|---|
| 1 | High ROI at 100.1 m | 0.477 | NO (confidence 0.523) | `reobserve`, entropy 0.998, one action available |
| 2a | Repeat same high ROI at 100.1 m | 0.477 | NO | `answer`, `sufficient_evidence`, budget 0 |
| 2b | Registered low ROI at 30.0 m | 0.500 | YES (confidence 0.500) | `answer`, `sufficient_evidence`, budget 0 |

The gate requested `north_m=0`, `east_m=0`, `up_m=-70.1`, speed 10 m/s. The
low-view score lies at the answer boundary, where the fixed 0.5 rule changes
the answer to YES and harms the initially correct result. This is an example of
an incorrect change, not evidence for tuning the threshold on this sample.

The complete raw pair records, including YOLO context detections, both
observation events, action parameters and full `AgentVqaController` trajectory
objects, are in `single_temporal_triggered_paired_records.jsonl`.
