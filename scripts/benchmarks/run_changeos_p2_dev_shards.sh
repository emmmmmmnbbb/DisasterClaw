#!/usr/bin/env bash
# Schedule the P2 task-conditioned development run without competing with busy GPUs.
set -uo pipefail

ROOT=/home/lc/disasterclaw
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
TEST=backend/data/benchmarks/agent_vqa_v2_binary.json
FROZEN=runs/benchmarks/changeos_binary/frozen_manifest.json
CONFIGS=T1_TASK
RUN_PREFIX=${RUN_PREFIX:-runs/benchmarks/changeos_binary/p2_task_dev_v2_20260913}
LOG_PREFIX=${LOG_PREFIX:-/tmp/bench_p2_task_dev_v2_20260913}
MIN_FREE_MIB=${MIN_FREE_MIB:-22000}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-20}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

cd "$ROOT" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary
export MPLCONFIGDIR=/tmp/disasterclaw_mplconfig
mkdir -p "$MPLCONFIGDIR"

fingerprint() {
  "$PY" -c 'import importlib.util; from pathlib import Path; p=Path("scripts/benchmarks/bench_agent_vqa.py"); s=importlib.util.spec_from_file_location("bench_fingerprint",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.source_fingerprint())'
}

SOURCE_FINGERPRINT=$(fingerprint) || exit 1
echo "[scheduler] $(date '+%F %T') source_fingerprint=$SOURCE_FINGERPRINT"
echo "[scheduler] minimum free memory: ${MIN_FREE_MIB} MiB"

declare -a pending=(0 1 2 3)
declare -A pid_by_shard gpu_by_shard attempts_by_shard
failed=0
running_count=0

validate_shard() {
  local shard=$1
  "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); r=p/"results.json"; e=p/"episodes.jsonl"; d=json.loads(r.read_text()) if r.is_file() else {}; lines=sum(1 for _ in e.open()) if e.is_file() else 0; raise SystemExit(0 if d.get("valid_for_analysis") is True and lines==40 else 1)' \
    "${RUN_PREFIX}_shard${shard}of4"
}

launch_shard() {
  local shard=$1 gpu=$2
  local out=${RUN_PREFIX}_shard${shard}of4
  local log=${LOG_PREFIX}_shard${shard}.log
  local current
  current=$(fingerprint) || return 1
  if [[ "$current" != "$SOURCE_FINGERPRINT" ]]; then
    echo "[scheduler] source changed before shard $shard; refusing mixed-code run" | tee -a "$log"
    return 2
  fi
  attempts_by_shard[$shard]=$(( ${attempts_by_shard[$shard]:-0} + 1 ))
  echo "[scheduler] $(date '+%F %T') launch shard $shard/4 on GPU $gpu attempt ${attempts_by_shard[$shard]}"
  PERCEPTION_DEVICE="cuda:$gpu" VLM_LOCAL_DEVICE="cuda:$gpu" \
    PERCEPTION_OUTPUT_DIR="backend/outputs/uav_view_fixed_shard${shard}" \
    "$PY" -u scripts/benchmarks/bench_agent_vqa.py \
      --testset "$TEST" --frozen-manifest "$FROZEN" --configs "$CONFIGS" \
      --seed 42 --generation-seed 42000 --generation-repeat 0 \
      --tag changeos_binary_fixed --shard "$shard/4" \
      --out-dir "$out" --resume --allow-crash-resume \
      >> "$log" 2>&1 &
  pid_by_shard[$shard]=$!
  gpu_by_shard[$shard]=$gpu
}

while (( ${#pending[@]} > 0 || running_count > 0 )); do
  # Reap completed workers and validate content, not merely process disappearance.
  for shard in "${!pid_by_shard[@]}"; do
    pid=${pid_by_shard[$shard]}
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    wait "$pid"; rc=$?
    gpu=${gpu_by_shard[$shard]}
    unset 'pid_by_shard[$shard]' 'gpu_by_shard[$shard]'
    running_count=$((running_count - 1))
    if (( rc == 0 )) && validate_shard "$shard"; then
    echo "[scheduler] $(date '+%F %T') shard $shard complete and validated (40 rows)"
    elif (( ${attempts_by_shard[$shard]:-0} < MAX_ATTEMPTS )); then
      echo "[scheduler] $(date '+%F %T') shard $shard failed rc=$rc; queued for resume"
      pending+=("$shard")
    else
      echo "[scheduler] $(date '+%F %T') shard $shard failed after $MAX_ATTEMPTS attempts"
      failed=1
    fi
  done

  # GPUs already assigned by this scheduler cannot receive another shard.
  declare -A busy_gpu=()
  for shard in "${!gpu_by_shard[@]}"; do busy_gpu[${gpu_by_shard[$shard]}]=1; done

  if (( ${#pending[@]} > 0 )); then
    mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' %' | awk -F, -v max_util="$MAX_GPU_UTIL" '$3+0 <= max_util' | sort -t, -k2,2nr)
    remaining=()
    for shard in "${pending[@]}"; do
      selected=""
      for row in "${gpu_rows[@]}"; do
        gpu=${row%%,*}; rest=${row#*,}; free=${rest%%,*}
        if [[ -z ${busy_gpu[$gpu]:-} ]] && (( free >= MIN_FREE_MIB )); then
          selected=$gpu
          break
        fi
      done
      if [[ -n "$selected" ]]; then
        if launch_shard "$shard" "$selected"; then
          busy_gpu[$selected]=1
          running_count=$((running_count + 1))
        else
          failed=1
          break 2
        fi
      else
        remaining+=("$shard")
      fi
    done
    pending=("${remaining[@]}")
  fi

  if (( ${#pending[@]} > 0 || running_count > 0 )); then
    sleep "$POLL_SECONDS"
  fi
done

echo "[scheduler] $(date '+%F %T') finished failed=$failed"
exit "$failed"
