#!/usr/bin/env bash
# Foreground 12-shard queue intended to be launched once with nohup.
# Progresses to the next new shard only after the previous strict audit passes.
# Never retries, resumes, or overwrites an episode directory.
set -euo pipefail

ROOT=/home/lc/disasterclaw
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
GPU_ID=${GPU_ID:-1}
PROTOCOL=cja_en/review/changeos_p6_final_protocol_20260915.json
RUN_PREFIX=runs/benchmarks/changeos_binary/p6_final_three_event_v1_20260915
REPORT=${RUN_PREFIX}_report.json

if [[ ! $GPU_ID =~ ^[0-9]+$ ]]; then
  echo "[P6 queue] GPU_ID must be numeric" >&2
  exit 2
fi
cd "$ROOT"
echo "[P6 queue] $(date '+%F %T') start on GPU $GPU_ID: 3 repeats x 4 shards; no retries"

declare -a run_dirs=()
for repeat in 0 1 2; do
  for shard in 0 1 2 3; do
    out=${RUN_PREFIX}_repeat${repeat}_shard${shard}of4
    if [[ -e $out ]]; then
      echo "[P6 queue] refusing existing target: $out" >&2
      exit 2
    fi
    echo "[P6 queue] $(date '+%F %T') launch repeat=$repeat shard=$shard/4"
    DRY_RUN=0 GPU_ID="$GPU_ID" REPEAT="$repeat" SHARD="$shard/4" \
      bash scripts/benchmarks/run_changeos_p6_final_shard.sh
    run_dirs+=("$out")
    echo "[P6 queue] $(date '+%F %T') audited repeat=$repeat shard=$shard/4"
  done
done

if [[ -e $REPORT ]]; then
  echo "[P6 queue] refusing existing final report: $REPORT" >&2
  exit 2
fi
"$PY" scripts/benchmarks/report_changeos_p6_final.py \
  --protocol "$PROTOCOL" --runs "${run_dirs[@]}" --out "$REPORT"
echo "[P6 queue] $(date '+%F %T') all 12 shards validated; report=$REPORT"
