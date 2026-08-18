#!/usr/bin/env bash
# Run navigation suites that remain after perception/calibration and E11.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}"
DEVICE="${PAPER_DEVICE:-cuda:0}"
VLM_DEVICE="${PAPER_VLM_DEVICE:-cuda:2}"
YOLO_CKPT="${PAPER_YOLO_CKPT:-$REPO_ROOT/runs/train/xbd_yolov8s_strict_v1/weights/best.pt}"
LOC_CKPT="${PAPER_LOC_CKPT:-$REPO_ROOT/backend/outputs/building_localization/resnet34_strict_v1.pt}"
DIFF_CKPT="${PAPER_DIFF_CKPT:-$REPO_ROOT/backend/outputs/change_perception/strict_diff_attention_seed0.pt}"
RECHECK_SET="${PAPER_RECHECK_SET:-$REPO_ROOT/backend/data/benchmarks/vln_recheck_testset.json}"
RUN_ROOT="${PAPER_RUN_ROOT:-$REPO_ROOT/runs/benchmarks/paper_strict_unet_v1}"

mkdir -p "$RUN_ROOT"
test -f "$YOLO_CKPT"
test -f "$LOC_CKPT"
test -f "$DIFF_CKPT"
test -f "$RECHECK_SET"

cd "$REPO_ROOT/backend"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
export YOLO_WEIGHTS="$YOLO_CKPT"
export BUILDING_PROPOSER=unet
export BUILDING_LOC_CKPT="$LOC_CKPT"
export CHANGE_PERCEPTION_CKPT="$DIFF_CKPT"
export VLN_CHANGE_PERCEPTION=1
export PERCEPTION_DEVICE="$DEVICE"
export VLM_LOCAL_DEVICE="$VLM_DEVICE"
export VLM_LOCAL_MIN_FREE_GPU_GB="${PAPER_VLM_MIN_FREE_GB:-12}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$RECHECK_SET" \
  --configs B0,B1,B2,B3 \
  --grounder hybrid --repeat 3 --seed 41 \
  --out-dir "$RUN_ROOT/e1_nav" --tag strict_e1

for sigma in 0 2 5 10; do
  "$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
    --testset "$RECHECK_SET" --configs B2 --grounder hybrid \
    --repeat 3 --seed 41 --gps-noise-sigma-m "$sigma" \
    --out-dir "$RUN_ROOT/e5_gps_${sigma}m" --tag "strict_e5_${sigma}m"
done

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$RECHECK_SET" --configs B2 --grounder hybrid \
  --repeat 3 --seed 41 --force-degraded \
  --out-dir "$RUN_ROOT/e7_degraded" --tag strict_e7_degraded

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_report.py" "$RUN_ROOT/e1_nav"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/export_paper_assets.py" \
  --navigation "$RUN_ROOT/e1_nav/results.json" \
  --calibration-dir "$RUN_ROOT/calibration_baseline" \
  --diff-calibration-dir "$RUN_ROOT/calibration_diff_attention" \
  --yolo-metrics "$RUN_ROOT/yolo_metrics.json" \
  --loc-metrics "$RUN_ROOT/loc_metrics.json" \
  --out "$REPO_ROOT/paper/generated"

echo "Remaining navigation suites completed under $RUN_ROOT"
