#!/usr/bin/env bash
# P4 binary ChangeOS policy curves on the development set.
set -uo pipefail

ROOT=/home/lc/disasterclaw
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
TEST=backend/data/benchmarks/agent_vqa_v2_binary.json
FROZEN=runs/benchmarks/changeos_binary/frozen_manifest.json
CONFIGS=${CONFIGS:-A0_HOLD,A1_RANDOM,A2_ALWAYS,A3U_RAW_ENTROPY,A3_ENTROPY,A4_CONFORMAL,T1_TASK}
RUN_PREFIX=${RUN_PREFIX:-runs/benchmarks/changeos_binary/p4_policy_dev_v1_20260914}
LOG_PREFIX=${LOG_PREFIX:-/tmp/bench_p4_policy_dev_v1_20260914}
MIN_FREE_MIB=${MIN_FREE_MIB:-22000}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-20}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
ROWS_PER_SHARD=40
GLOBAL_LOCK=/tmp/disasterclaw_changeos_p4_policy_scheduler.lock

cd "$ROOT" || exit 1
exec 9>"$GLOBAL_LOCK"
if ! flock -n 9; then
  echo "[scheduler] another P4 policy scheduler is already running"
  exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary
export MPLCONFIGDIR=/tmp/disasterclaw_mplconfig
mkdir -p "$MPLCONFIGDIR"

fingerprint() {
  "$PY" -c 'import importlib.util; from pathlib import Path; p=Path("scripts/benchmarks/bench_agent_vqa.py"); s=importlib.util.spec_from_file_location("bench_fingerprint",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.source_fingerprint())'
}
SOURCE_FINGERPRINT=$(fingerprint) || exit 1

declare -a pending=(0 1 2 3)
declare -A pid_by_shard gpu_by_shard attempts_by_shard
failed=0
running_count=0

validate_shard() {
  local shard=$1 out=${RUN_PREFIX}_shard${shard}of4
  "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); expected=int(sys.argv[2]); r=p/"results.json"; e=p/"episodes.jsonl"; d=json.loads(r.read_text()) if r.is_file() else {}; rows=[json.loads(x) for x in e.read_text().splitlines() if x.strip()] if e.is_file() else []; keys={(x.get("config"),x.get("qid")) for x in rows}; fp=((d.get("env") or {}).get("source_fingerprint")); raise SystemExit(0 if d.get("valid_for_analysis") is True and len(rows)==expected and len(keys)==expected and fp==sys.argv[3] else 1)' \
    "$out" "$((ROWS_PER_SHARD * $(awk -F, '{print NF}' <<<"$CONFIGS")))" "$SOURCE_FINGERPRINT"
}

launch_shard() {
  local shard=$1 gpu=$2 out=${RUN_PREFIX}_shard${shard}of4 log=${LOG_PREFIX}_shard${shard}.log
  local lock=/tmp/disasterclaw_changeos_p4_policy_shard${shard}.lock
  [[ "$(fingerprint)" == "$SOURCE_FINGERPRINT" ]] || return 2
  attempts_by_shard[$shard]=$(( ${attempts_by_shard[$shard]:-0} + 1 ))
  echo "[scheduler] $(date '+%F %T') launch shard $shard/4 on GPU $gpu attempt ${attempts_by_shard[$shard]}"
  PERCEPTION_DEVICE="cuda:$gpu" VLM_LOCAL_DEVICE="cuda:$gpu" \
    PERCEPTION_OUTPUT_DIR="backend/outputs/uav_view_p4_policy_shard${shard}" \
    flock -n "$lock" "$PY" -u scripts/benchmarks/bench_agent_vqa.py \
      --testset "$TEST" --frozen-manifest "$FROZEN" --configs "$CONFIGS" \
      --seed 42 --generation-seed 42000 --generation-repeat 0 \
      --tag changeos_binary_p4_policy --shard "$shard/4" \
      --out-dir "$out" --resume --allow-crash-resume >>"$log" 2>&1 &
  pid_by_shard[$shard]=$!; gpu_by_shard[$shard]=$gpu
}

while (( ${#pending[@]} > 0 || running_count > 0 )); do
  for shard in "${!pid_by_shard[@]}"; do
    pid=${pid_by_shard[$shard]}; kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?; unset 'pid_by_shard[$shard]' 'gpu_by_shard[$shard]'; running_count=$((running_count-1))
    if (( rc==0 )) && validate_shard "$shard"; then
      echo "[scheduler] shard $shard complete and validated"
    elif (( ${attempts_by_shard[$shard]:-0} < MAX_ATTEMPTS )); then
      pending+=("$shard")
    else
      echo "[scheduler] shard $shard failed after $MAX_ATTEMPTS attempts"
      failed=1
    fi
  done
  declare -A busy_gpu=(); for shard in "${!gpu_by_shard[@]}"; do busy_gpu[${gpu_by_shard[$shard]}]=1; done
  remaining=()
  for shard in "${pending[@]}"; do
    selected=""
    while IFS=, read -r gpu free util; do
      gpu=${gpu// /}; free=${free// /}; util=${util// /}
      if [[ -z ${busy_gpu[$gpu]:-} ]] && (( free >= MIN_FREE_MIB )) && (( util <= MAX_GPU_UTIL )); then selected=$gpu; break; fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [[ -n "$selected" ]] && launch_shard "$shard" "$selected"; then
      busy_gpu[$selected]=1; running_count=$((running_count+1))
    else
      remaining+=("$shard")
    fi
  done
  pending=("${remaining[@]}")
  (( ${#pending[@]} > 0 || running_count > 0 )) && sleep "$POLL_SECONDS"
done
echo "[scheduler] finished failed=$failed"; exit "$failed"
