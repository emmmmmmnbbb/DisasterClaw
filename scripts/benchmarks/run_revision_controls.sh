#!/usr/bin/env bash
# One immutable shard of the two post-review controls. No retry or resume.
set -euo pipefail

ROOT=/home/lc/disasterclaw
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
KIND=${KIND:-matched}
GPU_ID=${GPU_ID:-2}
SHARD=${SHARD:-0}
REPEAT=${REPEAT:-0}
LIMIT=${LIMIT:-0}
TAG=${TAG:-}
if [[ ! $KIND =~ ^(matched|shortcuts)$ || ! $GPU_ID =~ ^[0-3]$ \
      || ! $SHARD =~ ^[0-3]$ || ! $REPEAT =~ ^[0-2]$ \
      || ! $LIMIT =~ ^[0-9]+$ || ( -n $TAG && ! $TAG =~ ^[A-Za-z0-9_]+$ ) ]]; then
  echo "Invalid KIND/GPU_ID/SHARD/REPEAT/LIMIT/TAG" >&2
  exit 2
fi
cd "$ROOT"
TEST=backend/data/benchmarks/agent_vqa_final_candidate_20260915_binary_human_reviewed_validated.json
DEV=backend/data/benchmarks/agent_vqa_v2_binary.json
PROTOCOL=cja_en/review/changeos_p6_final_protocol_20260915.json
REF=runs/benchmarks/changeos_binary
SUFFIX=${TAG:+_$TAG}
if [[ $KIND == matched ]]; then
  OUT="$REF/revision_matched_view_repeat${REPEAT}_shard${SHARD}of4${SUFFIX}"
else
  OUT="$REF/revision_shortcuts_shard${SHARD}of4${SUFFIX}"
fi
if [[ -e $OUT ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DATASET_MODE=xbd DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary
export CHANGEOS_WEIGHTS="$ROOT/backend/outputs/changeos/changeos_r34.pt"
export VLM_PROVIDER=qwen_vl_local VLM_LOCAL_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export VLM_LOCAL_TOP_P=0.9 VLM_LOCAL_REPETITION_PENALTY=1.1
export VLM_LOCAL_CHECKPOINT= CHECKPOINT_PATH=
export BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export AGENT_VQA_CONFIDENCE_THRESHOLD=0.5
export PERCEPTION_DEVICE="cuda:$GPU_ID" VLM_LOCAL_DEVICE="cuda:$GPU_ID"
export PERCEPTION_OUTPUT_DIR="backend/outputs/uav_view_revision_matched_repeat${REPEAT}_shard${SHARD}${SUFFIX}"
export MPLCONFIGDIR=/tmp/disasterclaw_mplconfig
mkdir -p "$MPLCONFIGDIR"
LOCK=/tmp/disasterclaw_revision_gpu${GPU_ID}.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Revision control runner already uses GPU $GPU_ID" >&2
  exit 2
fi

echo "[revision] kind=$KIND gpu=$GPU_ID shard=$SHARD repeat=$REPEAT limit=$LIMIT out=$OUT"
if [[ $KIND == matched ]]; then
  CMD=("$PY" -u scripts/benchmarks/eval_revision_matched_view.py
    --testset "$TEST" --protocol "$PROTOCOL" --reference-runs "$REF"
    --out-dir "$OUT" --repeat "$REPEAT" --shard "$SHARD")
else
  CMD=("$PY" -u scripts/benchmarks/eval_revision_shortcuts.py
    --testset "$TEST" --dev-testset "$DEV" --protocol "$PROTOCOL"
    --out-dir "$OUT" --shard "$SHARD")
fi
if [[ $LIMIT != 0 ]]; then CMD+=(--limit "$LIMIT"); fi
timeout --signal=TERM --kill-after=60s 4h "${CMD[@]}"
