# Exhaustive public MESSI descend-pair experiment

## Scope and freeze

This extension covers **all eligible cells of the fixed 24-cell grid** in
each of the six annotated descend sequences of the [public MESSI sample
folder](https://drive.google.com/drive/folders/1KKKE3QRbXDS-oTKi5N_MKlK2Eid_zKFd?usp=sharing).
It is not the full 2,525-image MESSI release. The `100_0001` sequence remains
pilot-only because it was used for prompt development in the preceding case.
The five evaluation sequences are `100_0002`, `100_0003`, `100_0004`,
`100_0042`, and `100_0043`. The six sequences and all 73 files listed in the
public folder are inventoried in `manifests/public_sample_files.json`.

Each sequence uses the high frame nearest 100 m and low frame nearest 30 m
relative to takeoff. `inspect_pairs.py` registers these frames and evaluates
the same 24 nonoverlapping 760-by-760 low-frame grid cells. A cell is included
if both MESSI vegetation-mask fractions are at least 0.05 (positive) or both
at most 0.005 (negative). Ambiguous or discordant cells are excluded before
inference and recorded in `manifests/exhaustive_descend_exclusions.csv`. No
class balancing, manual cherry-picking, or selection on model output is done.
The initial selection contains 107 cells across five evaluation sequences;
13 evaluation-sequence cells are excluded by mask agreement, and all 24 pilot
cells are excluded. The selection SHA-256 is
`d2e494ea8db079ed43b13053f7df573cc4372e99c8f4b6487ebac4eb0bb80d63`.
The generated sample and separate ground-truth manifests have SHA-256 values
`56b310c89e957402ea71b070b21d42a5994f2b6e65947eebfc6ca1bf2b85d75a`
and `3145034c83ce1c15033a7d838b46b72d77329b40ec0f55738a13e1bc81e9c318`.

The original 12-item task asked specifically about tree/shrub canopy. The
MESSI vegetation class also covers grass; therefore this extension asks
whether **visible vegetation, including grass, trees, or shrubs**, is present.
It is a separate experiment and its accuracy must not be pooled with the
original canopy task. Reference answers are annotation-derived at both
heights, without per-item human confirmation. The MESSI mask is an imperfect
proxy for visual answerability, so failure analysis must retain potential
annotation and registration issues rather than silently deleting cases.

## Calls and decision rule

The fixed-observation diagnostic retains the A/B/C design: first high image;
then an isolated repeat-high, high-source crop, or real low crop. The same
first model output is used for all branches. Qwen2.5-VL-7B-Instruct runs at
the cached revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, greedy
decoding, 16-token cap and seed 0. All normal responses, `UNCERTAIN`, invalid
outputs, and runtime failures remain in the log.

The action-gated replay uses the same implementation and threshold as the
12-item follow-up, with only the question and first-image prompt changed to
vegetation. It compares zero allowed new observations with one autonomous
recheck; the first Qwen response is shared exactly across arms. The low image
is opened only after an authorized action. The task adapter invokes the
original gate's damage-target branch because its presence branch always
skips local recheck; binary entropy comes from uncalibrated Qwen self-reported
confidence rather than ChangeOS. See `agent_replay_protocol.md` for details.
All items retain their original denominator; no result-driven exclusions.

Report correct counts, accuracies, corrections, harms, rechecks, stop reasons,
invalid outputs, full trajectories, and sequence-level results. These grid
cells share frames and scenes, so 107 is the number of decisions rather than
107 independent environments. No significance claim will be made from an
unclustered per-cell test.

## Reproduction

The generated manifests and images are created from the already downloaded
six descend sequences. The inference scripts support `--resume` and refuse
to mix configurations. Logs and scores are kept separate from the original
12-item case.

```sh
python experiments/messi_case/scripts/build_exhaustive_descend.py
python experiments/messi_case/scripts/make_inputs.py experiments/messi_case/manifests/exhaustive_descend_selection.json --output-subdir exhaustive_descend_inputs
python experiments/messi_case/scripts/run_case.py --split exhaustive_descend --device cuda:1
python experiments/messi_case/scripts/score_case.py --split exhaustive_descend
python experiments/messi_case/scripts/run_agent_replay.py --split exhaustive_descend --device cuda:1
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_descend
```
