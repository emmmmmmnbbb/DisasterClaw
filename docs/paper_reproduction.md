# DisasterClaw paper reproduction

## 1. Strict event-disjoint data

The fixed event groups are:

- validation: `hurricane-harvey,mexico-earthquake`
- test: `hurricane-michael,palu-tsunami`
- holdout: `nepal-flooding,moore-tornado,pinery-bushfire`
- training: every remaining event from xBD `train,tier3,test`

Generate both datasets:

```bash
python scripts/training/gen_xbd_yolo_dataset.py \
  --xbd-root /home/lc/datasets/xbd \
  --out /home/lc/datasets/xbd_yolo_strict_v1 \
  --splits train,tier3 --test-split test --strict-event-split \
  --val-disasters hurricane-harvey,mexico-earthquake \
  --test-disasters hurricane-michael,palu-tsunami \
  --holdout-disasters nepal-flooding,moore-tornado,pinery-bushfire \
  --seed 42

python scripts/training/gen_xbd_change_dataset.py \
  --xbd-root /home/lc/datasets/xbd \
  --out /home/lc/datasets/xbd_change_strict_v1 \
  --splits train,tier3 --test-split test --strict-event-split \
  --val-disasters hurricane-harvey,mexico-earthquake \
  --test-disasters hurricane-michael,palu-tsunami \
  --holdout-disasters nepal-flooding,moore-tornado,pinery-bushfire \
  --seed 42
```

Both training entry points reject a non-disjoint manifest when
`--require-event-disjoint` is supplied.

## 2. Perception training

```bash
python scripts/training/train_xbd_yolo.py \
  --data /home/lc/datasets/xbd_yolo_strict_v1/data.yaml \
  --model yolov8s.pt --imgsz 1024 --epochs 60 --batch 8 --device 0 \
  --seed 42 --name xbd_yolov8s_strict_v1 --require-event-disjoint

python backend/change_perception.py train \
  --data-dir /home/lc/datasets/xbd_change_strict_v1 \
  --epochs 8 --batch-size 32 --workers 4 --device cuda:0 --seed 0 \
  --out backend/outputs/change_perception/strict_baseline_seed0.pt \
  --require-event-disjoint

python backend/change_perception.py train \
  --data-dir /home/lc/datasets/xbd_change_strict_v1 \
  --epochs 8 --batch-size 32 --workers 4 --device cuda:0 --seed 0 \
  --diff-attention \
  --out backend/outputs/change_perception/strict_diff_attention_seed0.pt \
  --require-event-disjoint
```

## 3. Evidence-rich reinspection benchmark

```bash
python scripts/benchmarks/gen_vln_testset.py \
  --profile evidence-rich --n 60 --seed 17 --disasters "" \
  --out backend/data/benchmarks/vln_recheck_testset_draft.json

python scripts/benchmarks/render_vln_review_sheet.py \
  backend/data/benchmarks/vln_recheck_testset_draft.json \
  --out-dir runs/benchmarks/vln_recheck_review

# Run only after every contact-sheet panel has been inspected.
python scripts/benchmarks/review_vln_testset.py \
  backend/data/benchmarks/vln_recheck_testset_draft.json \
  --out backend/data/benchmarks/vln_recheck_testset.json \
  --reviewer "<reviewer>" --contact-sheets-reviewed
```

The approved v1 set has 40 single-target tasks across ten events. Starts are
outside the 25 m success radius but inside the default 60 m observation radius.

## 4. Navigation and reinspection

After the strict checkpoints exist, run the frozen suite:

```bash
bash scripts/benchmarks/run_paper_experiments.sh
```

If perception/calibration and E11 are already finished, run only the
remaining navigation suites:

```bash
bash scripts/benchmarks/run_remaining_paper_nav.sh
```

The full runner evaluates YOLO metrics, baseline/diff-attention calibration on
test and holdout, B0--B3 navigation, the six reinspection policies, GPS
noise (0/2/5/10\,m), and forced degraded observation. Navigation uses
`--repeat 3 --seed 41`, which expands to paired seeds 41/42/43.

Manual equivalents (from `backend/` after sourcing `.env`):

```bash
export YOLO_WEIGHTS=../runs/train/xbd_yolov8s_strict_v1/weights/best.pt
export CHANGE_PERCEPTION_CKPT=outputs/change_perception/strict_diff_attention_seed0.pt
export PERCEPTION_DEVICE=cuda:0

python ../scripts/benchmarks/bench_vln_navigation.py \
  --testset data/benchmarks/vln_recheck_testset.json \
  --configs B0,B1,B2,B3 \
  --grounder hybrid --repeat 3 --seed 41 --tag strict_e1

python ../scripts/benchmarks/bench_vln_navigation.py \
  --testset data/benchmarks/vln_recheck_testset.json \
  --configs E11_NONE,E11_RANDOM,E11_FIXED,E11_HEURISTIC,E11_ENTROPY,E11_INFOGAIN \
  --grounder hybrid --repeat 3 --seed 41 --tag strict_e11
```

Use `--gps-noise-sigma-m 2`, `5`, or `10` for E5 and
`--force-degraded` for E7. The report aggregates repeats at item level,
separates evidence from no-evidence episodes, and applies Holm correction.

## 5. Paper

```bash
cd /home/lc/disasterclaw
python scripts/benchmarks/export_paper_assets.py
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

Replace `--navigation` / `--calibration-dir` with the strict run outputs
before treating table numbers as final.
