#!/usr/bin/env bash
# Parallel half: repeat 1 shards 2-3 and repeat 2 shards 0-3.
# The first queue covers repeat 0 all shards and repeat 1 shards 0-1.
set -euo pipefail
ROOT=/home/lc/disasterclaw
GPU_ID=${GPU_ID:-3}
TAG=${TAG:-full_20260920}
if [[ ! $GPU_ID =~ ^[0-3]$ || ! $TAG =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "Invalid GPU_ID or TAG" >&2
  exit 2
fi
cd "$ROOT"
LOG_DIR="runs/benchmarks/changeos_binary/revision_matched_tail_logs_$TAG"
if [[ -e $LOG_DIR ]]; then
  echo "Refusing existing log directory $LOG_DIR" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
for repeat in 1 2; do
  for shard in 0 1 2 3; do
    if [[ $repeat == 1 && $shard -lt 2 ]]; then continue; fi
    LOG="$LOG_DIR/repeat${repeat}_shard${shard}.log"
    echo "[revision matched tail] start repeat=$repeat shard=$shard GPU=$GPU_ID"
    KIND=matched GPU_ID="$GPU_ID" REPEAT="$repeat" SHARD="$shard" TAG="$TAG" \
      bash scripts/benchmarks/run_revision_controls.sh >"$LOG" 2>&1
    tail -n 2 "$LOG"
    echo "[revision matched tail] passed repeat=$repeat shard=$shard"
  done
done
echo "[revision matched tail] all 6 strict shards passed"
