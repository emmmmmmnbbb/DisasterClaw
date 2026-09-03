#!/usr/bin/env bash
# Reproducible priority pipeline for the active-reobservation paper experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/benchmarks/paper_cja_mech_v1}"
REGISTRY="${REGISTRY:-$ROOT/backend/data/benchmarks/tile_consumption_registry.json}"
SELECTION_SET="${SELECTION_SET:-$ROOT/backend/data/benchmarks/agent_vqa_selection_v2.json}"
FINAL_SET="${FINAL_SET:-$ROOT/backend/data/benchmarks/agent_vqa_final_v2.json}"
ENTROPY_TABLE="${ENTROPY_TABLE:-$ROOT/backend/data/fov_entropy_table.json}"
N_GPU="${N_GPU:-1}"
SELECTION_N="${SELECTION_N:-100}"
FINAL_N="${FINAL_N:-160}"
DEVICE="${PERCEPTION_DEVICE:-cuda:0}"
FOV_GPU_IDS="${FOV_GPU_IDS:-0}"
PY="${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}"
CALIB_DATA_DIR="${CALIB_DATA_DIR:-/home/lc/datasets/xbd_change_strict_v1}"
CALIB_CKPT="${CALIB_CKPT:-$ROOT/backend/outputs/change_perception/strict_diff_attention_seed0_v2.pt}"

mkdir -p "$RUN_ROOT"
cd "$ROOT"

"$PY" scripts/benchmarks/build_tile_consumption_registry.py --out "$REGISTRY" --reset

"$PY" scripts/benchmarks/eval_fov_ladder.py \
  --split val --device "$DEVICE" --out-dir "$RUN_ROOT" \
  --table-out "$ENTROPY_TABLE" --registry "$REGISTRY" --register-fit
"$PY" scripts/benchmarks/calibration_bench.py \
  --data-dir "$CALIB_DATA_DIR" --ckpt "$CALIB_CKPT" --subset val \
  --device "$DEVICE" --require-event-disjoint --out-dir "$RUN_ROOT/calibration_val"

"$PY" scripts/benchmarks/gen_agent_vqa_testset_v2.py \
  --centered-start --n "$SELECTION_N" --seed 1701 \
  --exclude-registry "$REGISTRY" --eval-role selection --update-registry \
  --out "$SELECTION_SET"
"$PY" scripts/benchmarks/review_agent_vqa_testset.py \
  --in "$SELECTION_SET" --out "$RUN_ROOT/selection_testset_review.json"

TEMPERATURE="$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/fov_ladder_val.json'))['temperature'])")"
IFS=',' read -r -a FOV_GPUS <<< "$FOV_GPU_IDS"
FOV_PIDS=()
FOV_INPUTS=()
for i in "${!FOV_GPUS[@]}"; do
  n="${#FOV_GPUS[@]}"
  "$PY" scripts/benchmarks/eval_fov_ladder.py \
    --split eval --device "cuda:${FOV_GPUS[$i]}" --out-dir "$RUN_ROOT/selection" \
    --tiles-from "$SELECTION_SET" --temperature "$TEMPERATURE" --shard "$i/$n" &
  FOV_PIDS+=($!)
  FOV_INPUTS+=("$RUN_ROOT/selection/fov_ladder_eval_shard${i}of${n}_items.jsonl")
done
for pid in "${FOV_PIDS[@]}"; do wait "$pid"; done
"$PY" scripts/benchmarks/merge_fov_ladder_shards.py \
  --inputs "${FOV_INPUTS[@]}" --split eval --out-dir "$RUN_ROOT/selection" \
  --temperature "$TEMPERATURE"

"$PY" scripts/benchmarks/eval_budget_allocation.py \
  --items "$RUN_ROOT/selection/fov_ladder_eval_items.jsonl" \
  --fit-items "$RUN_ROOT/fov_ladder_val_items.jsonl" \
  --qhat "$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/fov_ladder_val.json'))['conformal_qhat_alpha01'])")" \
  --temperature "$TEMPERATURE" \
  --out "$RUN_ROOT/selection/budget_allocation.json"

"$PY" scripts/benchmarks/sweep_recheck_hparams.py \
  --items "$RUN_ROOT/selection/fov_ladder_eval_items.jsonl" \
  --fit-items "$RUN_ROOT/fov_ladder_val_items.jsonl" \
  --qhat "$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/fov_ladder_val.json'))['conformal_qhat_alpha01'])")" \
  --temperature "$TEMPERATURE" \
  --out "$RUN_ROOT/recheck_hparam_sweep.json"

# Smoke the online A5 path before freezing; any missing/stale entropy table fails closed.
DETECTOR_BACKEND=xview2_first VLN_ENTROPY_TABLE="$ENTROPY_TABLE" \
  "$PY" scripts/benchmarks/bench_agent_vqa.py \
  --testset "$SELECTION_SET" --configs A5_EXPECTED,A3_ENTROPY \
  --limit 8 --out-dir "$RUN_ROOT/selection_smoke"

"$PY" scripts/benchmarks/gen_agent_vqa_testset_v2.py \
  --centered-start --n "$FINAL_N" --seed 2909 \
  --exclude-registry "$REGISTRY" --eval-role final --update-registry \
  --out "$FINAL_SET"
"$PY" scripts/benchmarks/review_agent_vqa_testset.py \
  --in "$FINAL_SET" --out "$RUN_ROOT/final_testset_review.json"

"$PY" scripts/benchmarks/freeze_recheck_manifest.py \
  --selection-sweep "$RUN_ROOT/recheck_hparam_sweep.json" \
  --selection-testset "$SELECTION_SET" --final-testset "$FINAL_SET" \
  --fit-report "$RUN_ROOT/fov_ladder_val.json" \
  --entropy-table "$ENTROPY_TABLE" --registry "$REGISTRY" \
  --backend xview2_first --leaky --out "$RUN_ROOT/frozen_manifest.json"

"$PY" scripts/benchmarks/eval_fov_ladder.py \
  --split eval --device "$DEVICE" --out-dir "$RUN_ROOT/final_fov" \
  --tiles-from "$FINAL_SET" --temperature "$TEMPERATURE"
"$PY" scripts/benchmarks/eval_budget_allocation.py \
  --items "$RUN_ROOT/final_fov/fov_ladder_eval_items.jsonl" \
  --fit-items "$RUN_ROOT/fov_ladder_val_items.jsonl" \
  --qhat "$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/fov_ladder_val.json'))['conformal_qhat_alpha01'])")" \
  --temperature "$TEMPERATURE" \
  --out "$RUN_ROOT/budget_allocation.json"

CONFIGS="A5_EXPECTED,A1_RANDOM_MATCHED,A2_FIXED_MATCHED,A0_HOLD,A3U_RAW_ENTROPY,A3_ENTROPY,A4_CONFORMAL,AB_CENTER,AB_DESCEND,AB_FULL"
DETECTOR_BACKEND=xview2_first NO_RESUME=1 bash scripts/benchmarks/run_agent_vqa_parallel.sh \
  "$CONFIGS" all "$N_GPU" paper_cja_mech_final \
  --testset "$FINAL_SET" --frozen-manifest "$RUN_ROOT/frozen_manifest.json" \
  --matched-reference A5_EXPECTED

cp -a "$ROOT/runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_reports" "$RUN_ROOT/final_reports"

"$PY" scripts/benchmarks/eval_identifiability.py \
  --split eval --backend xview2_eventdisjoint --device "$DEVICE" --limit 0 \
  --tiles-from "$FINAL_SET" \
  --out "$RUN_ROOT/boundary_identifiability_eventdisjoint.json"

DETECTOR_BACKEND=xview2_eventdisjoint NO_RESUME=1 bash scripts/benchmarks/run_agent_vqa_parallel.sh \
  "A0_HOLD,A5_EXPECTED" all "$N_GPU" paper_cja_mech_boundary \
  --testset "$FINAL_SET" --frozen-manifest "$RUN_ROOT/frozen_manifest.json"

"$PY" scripts/benchmarks/export_cja_assets.py --run-root "$RUN_ROOT"
echo "[ok] priority experiment pipeline complete: $RUN_ROOT"
