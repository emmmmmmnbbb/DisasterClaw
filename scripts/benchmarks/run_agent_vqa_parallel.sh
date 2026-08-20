#!/usr/bin/env bash
# scripts/benchmarks/run_agent_vqa_parallel.sh — 多 GPU 按题分片并行跑 Agent-VQA 评测
#
# 用法:
#   bash scripts/benchmarks/run_agent_vqa_parallel.sh <configs> <split> <n_gpu> [tag] [extra args...]
# 例:
#   bash scripts/benchmarks/run_agent_vqa_parallel.sh V0_RAW,A0_HOLD,A3_ENTROPY test 4 pilot
#
# 每个 GPU 跑全配置的一个题分片 (shard i/N)，保持每片内 (config,qid) 配对完整，
# 便于 report_agent_vqa.py 合并后做配对 bootstrap。每片独立 out-dir，续跑友好。
set -euo pipefail

CONFIGS="${1:?usage: $0 <configs> <split> <n_gpu> [tag] [extra]}"
SPLIT="${2:?}"
N_GPU="${3:?}"
TAG="${4:-parallel}"
shift 4 || true
EXTRA="$*"

REPO=/home/lc/disasterclaw
cd "$REPO/backend"
set -a; source ../.env; set +a

PIDS=()
for i in $(seq 0 $((N_GPU - 1))); do
  export PERCEPTION_DEVICE="cuda:$i"
  export VLM_LOCAL_DEVICE="cuda:$i"
  outdir="../runs/benchmarks/cja_agent_vqa/${TAG}_shard${i}of${N_GPU}"
  echo "[launch] GPU $i -> $outdir  configs=$CONFIGS split=$SPLIT shard $i/$N_GPU"
  python ../scripts/benchmarks/bench_agent_vqa.py \
    --configs "$CONFIGS" --split "$SPLIT" --shard "$i/$N_GPU" \
    --out-dir "$outdir" --tag "${TAG}_s${i}" --resume $EXTRA \
    > "$outdir.log" 2>&1 &
  PIDS+=($!)
done

echo "[launch] waiting for $N_GPU shards..."
FAIL=0
for p in "${PIDS[@]}"; do
  if wait "$p"; then
    echo "[done] pid $p OK"
  else
    echo "[done] pid $p FAILED (exit $?)"; FAIL=1
  fi
done

echo "[launch] merging shards with report_agent_vqa.py..."
RUNS=""
for i in $(seq 0 $((N_GPU - 1))); do
  RUNS="$RUNS ../runs/benchmarks/cja_agent_vqa/${TAG}_shard${i}of${N_GPU}"
done
python ../scripts/benchmarks/report_agent_vqa.py \
  --runs $RUNS --out "../runs/benchmarks/cja_agent_vqa/${TAG}_reports" || true

echo "[launch] all done (fail=$FAIL). merged reports in ${TAG}_reports/"
exit $FAIL
