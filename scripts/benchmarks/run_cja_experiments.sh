#!/usr/bin/env bash
# CJA experiment runner. Long jobs can be limited with LIMIT=...
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/backend:${PYTHONPATH:-}"
export BUILDING_PROPOSER="${BUILDING_PROPOSER:-unet}"
export BUILDING_LOC_CKPT="${BUILDING_LOC_CKPT:-$REPO_ROOT/backend/outputs/building_localization/resnet34_strict_v1.pt}"
export CHANGE_PERCEPTION_CKPT="${CHANGE_PERCEPTION_CKPT:-$REPO_ROOT/backend/outputs/change_perception/strict_diff_attention_seed0_v2.pt}"
export VLN_CHANGE_PERCEPTION=1
export GSD_LADDER=1
export VLN_ENTROPY_TABLE="${VLN_ENTROPY_TABLE:-$REPO_ROOT/backend/data/gsd_entropy_table.json}"
export PAPER_RUN_ROOT="${PAPER_RUN_ROOT:-$REPO_ROOT/runs/benchmarks/paper_cja_v1}"
LIMIT="${LIMIT:-400}"
PYTHON_BIN="${PYTHON_BIN:-${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}}"
mkdir -p "$PAPER_RUN_ROOT"

echo "[1] power analysis"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/power_analysis.py" \
  --out "$PAPER_RUN_ROOT/power_analysis.json"

echo "[2] event-isolated benchmark (test+holdout only)"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/gen_vln_testset.py" \
  --profile evidence-rich --n "${BENCH_N:-200}" --seed 17 \
  --disasters hurricane-michael,palu-tsunami,moore-tornado,nepal-flooding,pinery-bushfire \
  --require-eval-events \
  --out "$REPO_ROOT/backend/data/benchmarks/vln_recheck_eval_v2.json"

echo "[3] GSD ladder + entropy table (val then test subset)"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/eval_gsd_ladder.py" \
  --ckpt "$CHANGE_PERCEPTION_CKPT" \
  --split val --limit "$LIMIT" --device "${PERCEPTION_DEVICE:-cuda:0}" \
  --out "$PAPER_RUN_ROOT/gsd_ladder_val.json" \
  --table-out "$REPO_ROOT/backend/data/gsd_entropy_table.json" \
  --items-out "$PAPER_RUN_ROOT/gsd_ladder_val_items.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/eval_gsd_ladder.py" \
  --ckpt "$CHANGE_PERCEPTION_CKPT" \
  --split test --limit "$LIMIT" --device "${PERCEPTION_DEVICE:-cuda:0}" \
  --out "$PAPER_RUN_ROOT/gsd_ladder_test.json" \
  --table-out "$PAPER_RUN_ROOT/gsd_entropy_table_test.json" \
  --items-out "$PAPER_RUN_ROOT/gsd_ladder_test_items.jsonl"

echo "[4] budget allocation (X2 subject)"
QHAT=$("$PYTHON_BIN" -c "import json; print(json.load(open('$REPO_ROOT/backend/data/gsd_entropy_table.json')).get('conformal_qhat', 0.9))")
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/eval_budget_allocation.py" \
  --items "$PAPER_RUN_ROOT/gsd_ladder_test_items.jsonl" --qhat "$QHAT" \
  --out "$PAPER_RUN_ROOT/budget_allocation.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/sweep_recheck_hparams.py" \
  --items "$PAPER_RUN_ROOT/gsd_ladder_test_items.jsonl" --qhat "$QHAT" \
  --out "$PAPER_RUN_ROOT/recheck_hparam_sweep.json"

echo "[5] RescueNet shift (eval only)"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/eval_rescuenet_shift.py" \
  --ckpt "$CHANGE_PERCEPTION_CKPT" \
  --out "$PAPER_RUN_ROOT/rescuenet_shift.json"

TS="$REPO_ROOT/backend/data/benchmarks/vln_recheck_eval_v2.json"
QHAT_VAL=$("$PYTHON_BIN" -c "import json; print(json.load(open('$REPO_ROOT/backend/data/gsd_entropy_table.json')).get('conformal_qhat', 0.9))")
export VLN_CONFORMAL_QHAT="$QHAT_VAL"
NAV_LIMIT="${NAV_LIMIT:-0}"   # 0 = 全部 200 题；冒烟可设 NAV_LIMIT=4

echo "[6] X3 oracle ladder L0-L3"
cd "$REPO_ROOT/backend"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$TS" \
  --configs L0,L1,L2,L3 --grounder hybrid --limit "$NAV_LIMIT" --repeat 1 --seed 42 \
  --out-dir "$PAPER_RUN_ROOT/x3_oracle" --tag x3_oracle

echo "[7] X0 single-item trace"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/diag_x0_nav.py" --trace \
  --testset "$TS" \
  --out "$PAPER_RUN_ROOT/x0_diag.json"

echo "[8] X6 end-to-end: main ablation B0-B3"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$TS" --configs B0,B1,B2,B3 --grounder hybrid \
  --limit "$NAV_LIMIT" --repeat 1 --seed 42 \
  --out-dir "$PAPER_RUN_ROOT/x6_main" --tag x6_main

echo "[9] X6 recheck-policy family (E11)"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$TS" \
  --configs E11_RANDOM,E11_FIXED,E11_ENTROPY,E11_INFOGAIN,E11_CONFORMAL --grounder hybrid \
  --limit "$NAV_LIMIT" --repeat 1 --seed 42 \
  --out-dir "$PAPER_RUN_ROOT/x6_e11" --tag x6_e11

echo "[10] X6 GPS noise + forced-degraded suites (B2)"
for SIGMA in 2 5 10; do
  "$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
    --testset "$TS" --configs B2 --grounder hybrid --gps-noise-sigma-m "$SIGMA" \
    --limit "$NAV_LIMIT" --repeat 1 --seed 42 \
    --out-dir "$PAPER_RUN_ROOT/x6_gps$SIGMA" --tag "x6_gps$SIGMA"
done
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$TS" --configs B2 --grounder hybrid --force-degraded \
  --limit "$NAV_LIMIT" --repeat 1 --seed 42 \
  --out-dir "$PAPER_RUN_ROOT/x6_degraded" --tag x6_degraded

echo "[11] export paper assets"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/export_cja_assets.py" \
  --run-root "$PAPER_RUN_ROOT" --out-dir "$REPO_ROOT/paper_cja/generated"

echo "DONE CJA pipeline → $PAPER_RUN_ROOT"
