# MESSI single-temporal DisasterClaw replay

This is the MESSI experiment using the repository's actual `AgentVqaController`
and `RecheckController`, adapted for a current-frame vegetation-presence task.
It assesses autonomous re-observation decisions on registered real UAV frames;
it is an offline replay, not a flight deployment or an unchanged end-to-end
ChangeOS run.

## Models and task adapter

- YOLO uses the user's RescueNet weights at
  `/home/lc/Langchain-Chatchat/tools/mars/results/yolo/runs/train/mars_det_yolov8n4/weights/best.pt`,
  confidence 0.25, IoU 0.45 and 640-pixel inference size. Its six classes cover
  building damage, vehicles and water. It has no vegetation class, so its
  detections are retained as scene context and are not used to claim vegetation.
- SegFormer uses the user's cached `nvidia/segformer-b2-finetuned-ade-512-512`
  checkpoint. Vegetation is the frozen ADE20K union `{4, 9, 17, 72}` (tree,
  grass, plant, palm). Field and flower are excluded from the task definition.
- The normalized vegetation soft-occupancy score is mapped to a task score
  `p = clip((fraction - 0.005) / (0.05 - 0.005), 0, 1)`. The YES/NO decision
  threshold is 0.5. This interpolation uses the already-defined MESSI mask
  selection boundaries; it is not a probability calibration claim.
- `AgentVqaController` keeps its evidence memory, answer/stop path, bounded
  re-observation loop and trajectory. Its task answer branch is adapted to use
  the SegFormer binary evidence. The question remains a presence task; no
  vegetation label is presented as a damage class. No generative VLM is used.
- The original `RecheckController` state machine is used with normalized
  two-class entropy, threshold 0.50 inherited from its existing default, one action maximum, and the actual relative
  altitude difference to the registered low frame. Motion is descend-only:
  there is no free horizontal recenter. The only task adaptation in its shared
  evidence filter is accepting an explicitly named `vegetation` task label.
- Model provenance is frozen in `results/single_temporal_model_provenance.json`.
  The selected SegFormer checkpoint was already cached before the MESSI replay;
  the IrYamim threshold or answers were not used to select models or tune this
  adapter. The same 34 source regions had appeared in earlier Qwen-only MESSI
  explorations, so this stratum is not a dataset-naive independent holdout.

## Information boundary and paired arms

The model receives the unmarked 800-by-800 registered high ROI as its first
observation. Hold and agent arms share that exact high-frame model output. The
hold arm disallows re-observation. The agent arm may receive the registered
low-altitude ROI only after the real controller requests re-observation and
the gate returns a descent. Each episode permits one action and no search.
The episode log records each image access, current-frame predictions, gate
outcome, action, controller trajectory, stop reason and latency.

Annotation masks and their fractions remain in separate ground-truth files.
The runtime does not load those files. The evaluated 409 regions are the
preselected label-stable subset from the existing protocol: both heights have
at least 5% vegetation for YES or at most 0.5% for NO. Ambiguous/disagreeing
regions and pilot sequence `100_0001` are excluded. This permits paired scoring
but narrows the finding to stable visible-presence cases. Descend, training path
and test path remain separate strata; adjacent regions and repeated frames are
correlated.

## Run and score

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python experiments/messi_case/scripts/run_single_temporal_agent.py \
  --split full_descend --device cuda:3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python experiments/messi_case/scripts/run_single_temporal_agent.py \
  --split full_train_paths --device cuda:3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python experiments/messi_case/scripts/run_single_temporal_agent.py \
  --split full_test_paths --device cuda:3
python experiments/messi_case/scripts/score_single_temporal_agent.py
```

Runs are written to `runs/single_temporal_<split>/episodes.jsonl`; the combined
paired scores go to `results/single_temporal_agent_summary.json`.

After the agent runs finish, the same action-triggered IDs can be run through a
matched repeated-high control. It shares the original high-view output, invokes
the same controller and deterministic answer adapter, spends the same one
re-observation action and makes a second answer call, but reacquires the same
high ROI instead of opening the low ROI:

```bash
for split in full_descend full_train_paths full_test_paths; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python experiments/messi_case/scripts/run_single_temporal_agent.py \
    --split "$split" --device cuda:3 --triggered-repeat-high
done
python experiments/messi_case/scripts/score_trigger_repeat_control.py
```

## Full replay result (2026-09-23)

| Stratum | n | Hold correct | Agent correct | Rechecks | Corrected / harmed |
|---|---:|---:|---:|---:|---:|
| Descend grid | 280 | 259 | 260 | 12 | 3 / 2 |
| Agamim paths | 95 | 85 | 87 | 7 | 3 / 1 |
| IrYamim test path | 34 | 25 | 25 | 1 | 0 / 0 |
| Combined | 409 | 369 | 372 | 20 | 6 / 3 |

All 818 episodes ended with `sufficient_evidence`; no invalid outputs or
execution errors occurred. The net paired change is +3 correct answers (+0.7
percentage points). These descriptive item-level differences do not establish
a reliable gain, and the IrYamim Test-path stratum did not improve. Its regions
had appeared in earlier Qwen-only exploratory runs, so it is not an untouched
holdout for the complete MESSI project.

## Same-trigger, equal-budget observation control

The original paired hold arm answered once; therefore its contrast with the
agent also included a second answer opportunity. We addressed this on exactly
the 20 samples whose first-view policy requested a re-observation. Both the
real-low and repeated-high arms share the same first output, make two answer
calls, follow the same `AgentVqaController` path, and spend one action. Only the
second image differs. The repeated-high arm opens the same high ROI again; the
real-low arm opens the registered 30-m ROI. No annotation is available to the
controller in either arm.

| Triggered cohort, n=20 | Correct |
|---|---:|
| Initial high answer (shared first call) | 12 |
| Repeat the same high ROI (second call) | 12 |
| Acquire the registered low ROI (second call) | 15 |

Relative to repeat-high, the real low view corrected six answers and harmed
three. Nine items remained correct in both arms; two remained wrong in both.
The full paired events and controller trajectories are stored in
`results/single_temporal_triggered_paired_records.jsonl`; a compact summary and
two representative traces are in `results/single_temporal_trigger_control.json`
and `results/single_temporal_trigger_examples.md`.

The examples are `100_0041_r11` (correction) and `100_0004_r08` (harm). For the
correction, SegFormer's high-view task score was 0.218 (NO; entropy 0.757), the
repeated high view stayed at 0.218/NO, and the low view scored 0.809/YES. For the
harm, the high and repeated-high score was 0.477/NO (entropy 0.998), while the
low view scored 0.500/YES against a NO reference. The controller requested a
70-m descent with zero north/east displacement in both cases; the repeat-high
control kept the observation at its original relative altitude. These paired
examples illustrate both the added detail and the errors from threshold-near
low-view predictions.

This test-path stratum was not used to tune the current model or its inherited
0.50 trigger after inference. Because those same IrYamim regions were present
in prior Qwen-only MESSI experiments, however, the results are reported as a
predefined test-path stratum, not as an independent untouched holdout. No
further model or threshold tuning is performed on those 34 items.
