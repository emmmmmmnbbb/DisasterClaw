# Public MESSI path-frame experiment

This additional exploratory stratum covers every public Agamim Path A/B/C
30-m frame paired to the nearest 100-m frame by GPS. The public folder lists
40, 51, and 45 low frames in A, B, and C (136 total). For each low frame the
predeclared ROI is the central 760-by-760 pixels. Nearest-frame GPS distance
must be at most 12 m; SIFT/RANSAC registration must meet the implementation's
40-inlier minimum; the mapped ROI must fit inside the high frame. The existing
MESSI vegetation masks at both heights define a positive when both ROI mask
fractions are at least 0.05 and a negative when both are at most 0.005.
Others are excluded and logged in `derived/public_paths/exclusions.jsonl`.

Of 136 low frames, 120 yielded registered in-frame pairs. Twenty-five of
those pairs were ambiguous or discordant by the mask rule, leaving 95
evaluation pairs: 83 positive and 12 negative. The severe class imbalance is
an outcome of the systematic sampling rule; no class balancing is performed.
The selection is frozen in `manifests/exhaustive_paths_selection.json`, SHA-256
`c4803b4d82b287f0d8c5de062dcaa15b4c44c1723e29965b185ef2269dbd26d1`.
The generated sample and separate ground-truth manifests have SHA-256 values
`b4d9767ed92762807633d06457ea984d30dc2dfba17fcad1fe5ab5a5b03a3a11`
and `b6d06d0399d6441b9cccba0e2f9c4cf3166fbea638fa85d1570cacb6f986c027`.
Reference answers are annotation-derived, without per-item visual review.
The question asks about visible vegetation including grass, trees, and shrubs.

The same Qwen fixed-observation A/B/C branches and adapted DisasterClaw
action-gated replay as the exhaustive descend stratum are run, with separate
logs and scores. All observed outputs, corrections, harms, rechecks, stop
reasons and full action trajectories are retained. The geographic paths
overlap and adjacent low frames share content; 95 paired ROIs do not
represent 95 independent environments. The public Test paths cannot enter
the accuracy denominator because the public sample folder exposes their
images and poses without corresponding annotations. Intermediate 50/70-m
frames are not actions in this fixed 100-to-30-m protocol.

Reproduction after downloading the public path archives:

```sh
python experiments/messi_case/scripts/inspect_public_paths.py
python experiments/messi_case/scripts/build_exhaustive_paths.py
python experiments/messi_case/scripts/make_path_inputs.py
python experiments/messi_case/scripts/run_case.py --split exhaustive_paths --device cuda:1
python experiments/messi_case/scripts/score_case.py --split exhaustive_paths
python experiments/messi_case/scripts/run_agent_replay.py --split exhaustive_paths --device cuda:3
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_paths
```
