# MESSI real-imagery case and public-sample extension

This directory implements the separate exploratory case in
`shuanglan/polished/sections/06_results_discussion.tex`. It follows
`shuanglan/MESSI航空应用补充案例执行计划.md`, with the one allowed pilot-stage task
change documented in `protocol.md`. The case uses MESSI images under the
[authors' CC BY-NC-SA 3.0 license](https://github.com/messi-dataset/messi-dataset),
and cites Pinkovich et al. (TMLR 2025). Raw images are not committed. The
paper figure credits MESSI and retains the license notice.

## Contents and result

The complete 2,525-image release is now available locally at
`/home/lc/datasets/MESSI`. The **full-release 100-to-30-m protocol** uses
every eligible region under the same fixed pairing, registration and
mask-agreement rules: 280 descend-grid regions, 95 Agamim path regions,
and 34 IrYamim test-path regions, 409 total. See `full_dataset_protocol.md`,
`manifests/full_dataset_inventory.json`, the `full_100_30_*` manifests and
`scripts/audit_full_messi.py`. The 50/70-m frames and 60-m-only
Ha-Medinah Square path are inventoried but cannot enter this single-action
100-to-30-m comparison. The `100_0001` pilot and all pairing failures are
listed among exclusions. The current controller experiment uses the existing
RescueNet YOLO and ADE20K SegFormer models with the repository's actual
`AgentVqaController` and `RecheckController`; see
`single_temporal_agent_protocol.md`.

The full-release fixed-observation Qwen A/B/C scores are A=287/409, B=341/409,
and C=341/409. C corrects 13 B errors and harms 13 B-correct cases. An earlier
Qwen token-entropy adapter rechecked 198/409 and changed its paired hold score
from 309/409 to 329/409 (44 corrections, 24 harms); that exploratory replay is
reported separately from the current legacy-model experiment.

In the current single-temporal replay, the controller uses 20 lower views and
changes accuracy from 369/409 to 372/409 (six corrections, three harms). By
stratum, it changes 259/280 to 260/280 on descends, 85/95 to 87/95 on Agamim
paths, and **25/34 to 25/34** on the IrYamim Test-path stratum. The 409 regions
remain a preselected, label-stable subset with correlated observations; see
the full protocol and outputs in `single_temporal_agent_protocol.md`,
`results/single_temporal_agent_summary.json`, and
`runs/single_temporal_full_*/episodes.jsonl`.

For the 20 samples that triggered, an equal-budget repeat-high control answered
12/20 correctly, matching the shared first high-view score; the real-low arm
answered 15/20. Relative to the repeat-high arm, the low view yielded six
corrections and three harms; nine items stayed correct and two stayed wrong in
both. IrYamim was included in earlier Qwen-only MESSI explorations, so its
SegFormer result is not described as an untouched holdout. Models and the
0.50 gate were fixed before this replay and are not tuned against that path.
See `results/single_temporal_trigger_control.json` and
`results/single_temporal_trigger_examples.md` for the matched-control summary
and full example traces.

To prepare and audit inputs from the unpacked official release:

```bash
python experiments/messi_case/scripts/prepare_full_messi.py --data /home/lc/datasets/MESSI
python experiments/messi_case/scripts/audit_full_messi.py
```

Each of `full_descend`, `full_train_paths`, and `full_test_paths` then runs
through the frozen A/B/C fixed-observation comparison and the exploratory
token-entropy action replay. The scripts require a local Qwen cache and CUDA;
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` avoids network checks.

```bash
for split in full_descend full_train_paths full_test_paths; do
  python experiments/messi_case/scripts/run_case.py --split "$split" --device cuda:0
  python experiments/messi_case/scripts/score_case.py --split "$split"
  python experiments/messi_case/scripts/run_agent_replay_logits.py --split "$split" --device cuda:2
  python experiments/messi_case/scripts/score_agent_replay.py --split "$split" --variant logits
done
python experiments/messi_case/scripts/audit_native_presence_gate.py
python experiments/messi_case/scripts/score_native_presence_gate.py
python experiments/messi_case/scripts/aggregate_full_messi.py
```

The original 12-region canopy case below is a separate exploratory cohort.
The later extension systematically covers the eligible annotated 100-to-30-m
pairs in the accessible public sample under fixed ROI and registration rules.
It contains 107 grid cells from five descend sequences and 95 central regions
from Agamim Paths A--C. Its question is visible vegetation, including grass,
trees, and shrubs. These cohorts have different labels and are not pooled.

On the 202 extension items, high-source crop B and real low-image C each answer
166/202 correctly; C helps eight B-wrong items and harms eight B-correct items.
The original self-reported-confidence replay requests all 202 low images and
changes its paired hold score from 138/202 to 158/202. A separate exploratory
token-entropy adapter, designed after that result, requests 102/202 low images,
stops on 100/202 high views, and changes its paired hold score from 155/202 to
165/202, with 26 corrections and 16 harms. The two replay variants use
different prompts and first-answer methods; compare each only to its own hold
arm. These are offline replays with predetermined candidate views, not real
flight or a test of the unchanged disaster-damage policy.

See `exhaustive_descend_protocol.md`, `exhaustive_paths_protocol.md`, and
`logit_replay_protocol.md` for selection and adapter details;
`results/public_sample_summary.json` has the combined counts and the
per-split result files retain per-item outcomes. These 202 items were the
earlier public-sample extension; the subsequent full-release experiment
above is a distinct run. Items within a sequence or path are correlated.

- `raw/`: downloaded original ZIPs and CSVs; excluded from Git.
- `manifests/data_inventory.csv`: source file IDs, SHA-256, counts, and CRC check.
- `manifests/pilot_*`: four `100_0001` items excluded from formal scoring.
- `manifests/formal_selection.json`: human-reviewed 12-region selection,
  recorded before formal model calls.
- `manifests/formal_samples.jsonl`: model input manifest, with geometry and
  source/input hashes but no reference answers.
- `manifests/formal_ground_truth.jsonl`: answers withheld from inference.
- `manifests/exclusions.csv`: the 60 other inspected grid regions and why each
  was left out before formal inference.
- `derived/`: generated crops and candidate previews; excluded from Git and
  rebuildable from raw files.
- `runs/formal/`: exact prompts, model configuration, raw replies, parsed
  answers, latencies, and status for all 48 calls.
- `results/`: per-sample, overall, sequence, paired, and error-analysis output.

The formal result is C=11/12 versus B=10/12 correct, a descriptive +8.33-point
difference. C helps two B-wrong regions and harms one B-correct region. The
first high observation is 7/12 and repeated-high A is 5/12. All 12 triplets
completed. This is a canopy-presence task on three urban sequences and does
not evaluate disaster-damage recognition or autonomous UAV movement.

## Rebuild

The run used Python 3.11.15 in `/home/lc/miniconda3/envs/disasterclaw`, with
PyTorch 2.11.0+cu130, Transformers 5.5.4, Pillow 12.2.0, OpenCV 4.13.0,
NumPy 2.4.4, Matplotlib 3.10.8, and gdown 6.4.0. Exact run details are in
`runs/formal/environment.json`.
The local cached model is `Qwen/Qwen2.5-VL-7B-Instruct` revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, run on CUDA device 1.
Use an environment with a CUDA GPU and the same model revision for a fresh
inference run. Model outputs may differ across hardware and library versions.

From the repository root:

```bash
conda activate disasterclaw
python -m pip install gdown==6.4.0
python experiments/messi_case/scripts/download_data.py
python experiments/messi_case/scripts/audit_data.py
for seq in 100_0001 100_0002 100_0003 100_0042; do
  python experiments/messi_case/scripts/inspect_pairs.py "$seq" --class-name vegetation
done
python experiments/messi_case/scripts/record_exclusions.py
python experiments/messi_case/scripts/make_inputs.py experiments/messi_case/manifests/pilot_selection.json
python experiments/messi_case/scripts/make_inputs.py experiments/messi_case/manifests/formal_selection.json
python experiments/messi_case/scripts/run_case.py --split formal --device cuda:1
python experiments/messi_case/scripts/score_case.py
python experiments/messi_case/scripts/make_figure.py
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error shuanglan/polished/main.tex
```

For the systematic public-sample extension, first obtain the remaining public
sample archives and verify them using the recorded Drive IDs and SHA-256 hashes
in `manifests/public_sample_files.json`, `manifests/data_inventory.csv`, and
`manifests/public_path_inventory.csv`. Then run:

```bash
python experiments/messi_case/scripts/audit_data.py
python experiments/messi_case/scripts/audit_public_paths.py
python experiments/messi_case/scripts/build_exhaustive_descend.py
python experiments/messi_case/scripts/inspect_public_paths.py
python experiments/messi_case/scripts/build_exhaustive_paths.py
python experiments/messi_case/scripts/make_inputs.py experiments/messi_case/manifests/exhaustive_descend_selection.json
python experiments/messi_case/scripts/make_path_inputs.py
python experiments/messi_case/scripts/run_case.py --split exhaustive_descend --device cuda:3
python experiments/messi_case/scripts/run_case.py --split exhaustive_paths --device cuda:3
python experiments/messi_case/scripts/score_case.py --split exhaustive_descend
python experiments/messi_case/scripts/score_case.py --split exhaustive_paths
python experiments/messi_case/scripts/run_agent_replay.py --split exhaustive_descend --device cuda:3
python experiments/messi_case/scripts/run_agent_replay.py --split exhaustive_paths --device cuda:3
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_descend
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_paths
python experiments/messi_case/scripts/run_agent_replay_logits.py --split exhaustive_descend --device cuda:3
python experiments/messi_case/scripts/run_agent_replay_logits.py --split exhaustive_paths --device cuda:3
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_descend --variant logits
python experiments/messi_case/scripts/score_agent_replay.py --split exhaustive_paths --variant logits
python experiments/messi_case/scripts/aggregate_public_sample.py
```

`run_case.py` intentionally refuses to overwrite an existing response log;
archive or move `runs/formal/responses.jsonl` before a new run. The formal
manifest is frozen by SHA-256 in `protocol.md`; any regenerated inputs should
be checked against the stored hashes before comparing outputs. All model
requests are local and no data are uploaded to a model service.

## Source

- [MESSI article](https://openreview.net/forum?id=ayWqZ1wyIv)
- [MESSI author's repository and data license](https://github.com/messi-dataset/messi-dataset)
- [Public sample dataset](https://drive.google.com/drive/folders/1KKKE3QRbXDS-oTKi5N_MKlK2Eid_zKFd?usp=sharing)
