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
EXTRA_ARGS=("$@")
SPLIT_ARGS=()
if [[ "$SPLIT" != "all" && "$SPLIT" != "-" ]]; then
  SPLIT_ARGS=(--split "$SPLIT")
fi
RESUME_ARGS=()
APPEND_LOGS=0
if [[ "${NO_RESUME:-0}" != "1" ]]; then
  RESUME_ARGS=(--resume)
  APPEND_LOGS=1
fi

REPO=/home/lc/disasterclaw
PYTHON_BIN="${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}"
cd "$REPO/backend"
REQUESTED_DETECTOR_BACKEND="${DETECTOR_BACKEND:-}"
set -a; source ../.env; set +a
if [[ -n "$REQUESTED_DETECTOR_BACKEND" ]]; then
  export DETECTOR_BACKEND="$REQUESTED_DETECTOR_BACKEND"
fi

PIDS=()
# GPU_IDS: 逗号分隔的物理 GPU 编号，默认 0..N_GPU-1。
# 某张卡故障时例如 GPU_IDS=0,1,2 N_GPU=3，避免新进程枚举坏卡导致 CUDA init 失败。
IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS:-$(seq -s, 0 $((N_GPU - 1)))}"
if [[ ${#GPU_IDS_ARR[@]} -ne $N_GPU ]]; then
  echo "[ERROR] GPU_IDS 数量 ${#GPU_IDS_ARR[@]} 与 N_GPU=$N_GPU 不一致" >&2
  exit 2
fi
for i in $(seq 0 $((N_GPU - 1))); do
  phys="${GPU_IDS_ARR[$i]}"
  outdir="../runs/benchmarks/cja_agent_vqa/${TAG}_shard${i}of${N_GPU}"
  mkdir -p "$outdir"
  echo "[launch] shard $i/$N_GPU physical GPU $phys -> $outdir  configs=$CONFIGS split=$SPLIT"
  # 错开加载，避免 4 份 Qwen2.5-VL-7B 同时进 CPU 内存。
  if [[ "$i" -gt 0 ]]; then sleep 25; fi
  shard_log="${outdir}.log"
  if [[ "$APPEND_LOGS" -eq 1 ]]; then
    {
      echo
      echo "===== RESUME $(date -Iseconds) shard=$i/$N_GPU gpu=$phys ====="
    } >> "$shard_log"
  else
    : > "$shard_log"
  fi
  PYTHONUNBUFFERED=1 PERCEPTION_DEVICE="cuda:$phys" VLM_LOCAL_DEVICE="cuda:$phys" \
    PERCEPTION_OUTPUT_DIR="$REPO/backend/outputs/uav_view_shard${i}" \
    HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/disasterclaw-mpl \
    "$PYTHON_BIN" ../scripts/benchmarks/bench_agent_vqa.py \
      --configs "$CONFIGS" "${SPLIT_ARGS[@]}" --shard "$i/$N_GPU" \
      --out-dir "$outdir" --tag "${TAG}_s${i}" "${RESUME_ARGS[@]}" "${EXTRA_ARGS[@]}" \
      >> "$shard_log" 2>&1 &
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
RUNS=()
for i in $(seq 0 $((N_GPU - 1))); do
  RUNS+=("../runs/benchmarks/cja_agent_vqa/${TAG}_shard${i}of${N_GPU}")
done
if [[ $FAIL -eq 0 ]]; then
  if ! "$PYTHON_BIN" ../scripts/benchmarks/report_agent_vqa.py \
    --runs "${RUNS[@]}" --out "../runs/benchmarks/cja_agent_vqa/${TAG}_reports"; then
    echo "[ERROR] shard report merge failed" >&2
    FAIL=1
  fi
else
  echo "[ERROR] at least one shard failed; skipping merged report" >&2
fi

echo "[launch] all done (fail=$FAIL). merged reports in ${TAG}_reports/"
exit $FAIL
