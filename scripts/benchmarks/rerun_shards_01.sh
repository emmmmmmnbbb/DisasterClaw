#!/usr/bin/env bash
# 等 shard 2/3 跑完后，在空闲显存最多的两张卡上补跑 shard 0/1。
set -u
PY=/home/lc/miniconda3/envs/disasterclaw/bin/python
TEST=backend/data/benchmarks/agent_vqa_v2_binary.json
FROZEN=runs/benchmarks/changeos_binary/frozen_manifest.json
CONFIGS=V1_STRUCT,V2_STATE,A0_HOLD,A1_RANDOM,A2_ALWAYS,A3_ENTROPY,A3U_RAW_ENTROPY,A4_CONFORMAL
cd /home/lc/disasterclaw || exit 1
echo "[rerun01] $(date '+%F %T') 等待 shard 2/3 结束..."
while pgrep -f "bench_agent_vqa.py.*shard[23]of4" >/dev/null 2>&1; do sleep 60; done
echo "[rerun01] $(date '+%F %T') shard 2/3 已结束，选取空闲 GPU..."
G0=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
G1=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep -v "^$G0," | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "[rerun01] 使用 GPU $G0 跑 shard 0, GPU $G1 跑 shard 1"
PERCEPTION_DEVICE=cuda:$G0 VLM_LOCAL_DEVICE=cuda:$G0 PERCEPTION_OUTPUT_DIR=backend/outputs/uav_view_shard0 DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary $PY -u scripts/benchmarks/bench_agent_vqa.py --testset $TEST --frozen-manifest $FROZEN --configs $CONFIGS --seed 42 --tag changeos_binary_full --shard 0/4 --out-dir runs/benchmarks/changeos_binary/full_20260913_shard0of4 > /tmp/bench_shard0.log 2>&1 &
PERCEPTION_DEVICE=cuda:$G1 VLM_LOCAL_DEVICE=cuda:$G1 PERCEPTION_OUTPUT_DIR=backend/outputs/uav_view_shard1 DETECTOR_BACKEND=changeos DAMAGE_LABEL_MODE=binary $PY -u scripts/benchmarks/bench_agent_vqa.py --testset $TEST --frozen-manifest $FROZEN --configs $CONFIGS --seed 42 --tag changeos_binary_full --shard 1/4 --out-dir runs/benchmarks/changeos_binary/full_20260913_shard1of4 > /tmp/bench_shard1.log 2>&1 &
wait
echo "[rerun01] $(date '+%F %T') shard 0/1 全部完成。"
