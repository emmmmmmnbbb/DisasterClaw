#!/usr/bin/env bash
set -euo pipefail
cd /home/lc/disasterclaw
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
DEVICE=${1:-cuda:0}
OUT=${2:-experiments/messi_case/supplement_random_budget_gpu}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/disasterclaw_mplconfig OMP_NUM_THREADS=4
"$PY" -u experiments/messi_case/scripts/run_random_budget_supplement.py --device "$DEVICE" --out-dir "$OUT"
"$PY" -u experiments/messi_case/scripts/score_random_budget_supplement.py --out-dir "$OUT"
printf '\nCompleted: %s/summary.json\n' "$OUT"
