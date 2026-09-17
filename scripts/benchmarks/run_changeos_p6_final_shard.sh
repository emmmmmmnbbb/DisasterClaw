#!/usr/bin/env bash
# One frozen P6 final generation repeat and shard. No scheduler or auto-retry.
set -euo pipefail

ROOT=/home/lc/disasterclaw
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
PROTOCOL=cja_en/review/changeos_p6_final_protocol_20260915.json
TEST=backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json
REVIEW=runs/benchmarks/changeos_binary/final_candidate_20260915_human_review_validated.json
FROZEN=runs/benchmarks/changeos_binary/final_three_event_20260915_frozen_manifest.json
CONFIGS=A0_HOLD,A1_RANDOM,A2_ALWAYS,A3U_RAW_ENTROPY,A3_ENTROPY,A4_CONFORMAL,T1_TASK
REPEAT=${REPEAT:-0}
SHARD=${SHARD:-0/4}
GPU_ID=${GPU_ID:-0}
DRY_RUN=${DRY_RUN:-1}
RUN_SUFFIX=${RUN_SUFFIX:-}

if [[ ! $REPEAT =~ ^[012]$ || ! $SHARD =~ ^[0-3]/4$ || ! $GPU_ID =~ ^[0-9]+$ ]]; then
  echo "[P6] REPEAT must be 0/1/2, SHARD must be 0/4..3/4, GPU_ID must be numeric" >&2
  exit 2
fi
if [[ $DRY_RUN != 0 && $DRY_RUN != 1 ]]; then
  echo "[P6] DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ -n $RUN_SUFFIX && ! $RUN_SUFFIX =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "[P6] RUN_SUFFIX must contain only letters, digits, and underscores" >&2
  exit 2
fi

SHARD_I=${SHARD%%/*}
OUT=runs/benchmarks/changeos_binary/p6_final_three_event_v1_20260915_repeat${REPEAT}_shard${SHARD_I}of4${RUN_SUFFIX:+_$RUN_SUFFIX}
LOG=/tmp/bench_p6_final_three_event_v1_20260915_repeat${REPEAT}_shard${SHARD_I}of4${RUN_SUFFIX:+_$RUN_SUFFIX}.log

cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DATASET_MODE=xbd DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary
export CHANGEOS_WEIGHTS="$ROOT/backend/outputs/changeos/changeos_r34.pt"
export VLM_PROVIDER=qwen_vl_local VLM_LOCAL_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export VLM_LOCAL_TOP_P=0.9 VLM_LOCAL_REPETITION_PENALTY=1.1
export VLM_LOCAL_CHECKPOINT= CHECKPOINT_PATH=
export BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export AGENT_VQA_CONFIDENCE_THRESHOLD=0.5
export PERCEPTION_DEVICE="cuda:$GPU_ID" VLM_LOCAL_DEVICE="cuda:$GPU_ID"
export PERCEPTION_OUTPUT_DIR="backend/outputs/uav_view_p6_final_repeat${REPEAT}_shard${SHARD_I}"
export MPLCONFIGDIR=/tmp/disasterclaw_mplconfig

BENCH=("$PY" -u scripts/benchmarks/bench_agent_vqa.py
  --testset "$TEST" --review-report "$REVIEW" --frozen-manifest "$FROZEN"
  --configs "$CONFIGS" --seed 42 --generation-seed 42000
  --generation-repeat "$REPEAT" --shard "$SHARD"
  --tag changeos_binary_p6_final --out-dir "$OUT")
PREFLIGHT=("$PY" scripts/benchmarks/preflight_changeos_p6_final.py
  --protocol "$PROTOCOL" --repeat "$REPEAT" --shard "$SHARD_I" --out-dir "$OUT")

if [[ $DRY_RUN == 1 ]]; then
  "${PREFLIGHT[@]}"
  echo "[P6] DRY_RUN=1: no GPU episode started. To launch, set DRY_RUN=0."
  printf '[P6] command: '
  printf '%q ' "${BENCH[@]}"
  printf '\n'
  echo "[P6] planned log: $LOG"
  exit 0
fi

mkdir -p "$MPLCONFIGDIR"
LOCK=/tmp/disasterclaw_changeos_p6_final_gpu${GPU_ID}.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[P6] another final shard has the GPU $GPU_ID lock" >&2
  exit 2
fi
"${PREFLIGHT[@]}" --receipt "$OUT/preflight.json"
echo "[P6] launching $SHARD repeat=$REPEAT GPU=$GPU_ID; no auto-retry or resume"
"${BENCH[@]}" 2>&1 | tee "$LOG"
"$PY" scripts/benchmarks/report_changeos_p6_final.py \
  --protocol "$PROTOCOL" --runs "$OUT" --partial \
  --out "$OUT/p6_shard_audit.json"
echo "[P6] process and strict shard audit both passed; no auto-retry"
