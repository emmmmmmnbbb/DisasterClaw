#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}"
DEVICE="${PAPER_DEVICE:-cuda:0}"
VLM_DEVICE="${PAPER_VLM_DEVICE:-cuda:2}"
CHANGE_DATA="${PAPER_CHANGE_DATA:-/home/lc/datasets/xbd_change_strict_v1}"
BASELINE_CKPT="${PAPER_BASELINE_CKPT:-$REPO_ROOT/backend/outputs/change_perception/strict_baseline_seed0.pt}"
DIFF_CKPT="${PAPER_DIFF_CKPT:-$REPO_ROOT/backend/outputs/change_perception/strict_diff_attention_seed0.pt}"
YOLO_CKPT="${PAPER_YOLO_CKPT:-$REPO_ROOT/runs/train/xbd_yolov8s_strict_v1/weights/best.pt}"
LOC_CKPT="${PAPER_LOC_CKPT:-$REPO_ROOT/backend/outputs/building_localization/resnet34_strict_v1.pt}"
LOC_DATA="${PAPER_LOC_DATA:-/home/lc/datasets/xbd_loc_strict_v1}"
RECHECK_SET="${PAPER_RECHECK_SET:-$REPO_ROOT/backend/data/benchmarks/vln_recheck_testset.json}"
RUN_ROOT="${PAPER_RUN_ROOT:-$REPO_ROOT/runs/benchmarks/paper_strict_unet_v1}"

mkdir -p "$RUN_ROOT"
test -f "$BASELINE_CKPT"
test -f "$DIFF_CKPT"
test -f "$YOLO_CKPT"
test -f "$LOC_CKPT"
test -f "$RECHECK_SET"
export VLM_LOCAL_DEVICE="$VLM_DEVICE"
export VLM_LOCAL_MIN_FREE_GPU_GB="${PAPER_VLM_MIN_FREE_GB:-12}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/eval_xbd_yolo.py" \
  --data /home/lc/datasets/xbd_yolo_strict_v1/data.yaml \
  --weights "$YOLO_CKPT" --device "${DEVICE#cuda:}" --imgsz 1024 --batch 8 \
  --require-event-disjoint --out "$RUN_ROOT/yolo_metrics.json"

if [[ -f "$LOC_CKPT" && -d "$LOC_DATA" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/eval_xbd_loc.py" \
    --data-dir "$LOC_DATA" \
    --weights "$LOC_CKPT" --device "$DEVICE" \
    --require-event-disjoint --out "$RUN_ROOT/loc_metrics.json"
fi

for model_name in baseline diff_attention; do
  if [[ "$model_name" == "baseline" ]]; then
    ckpt="$BASELINE_CKPT"
  else
    ckpt="$DIFF_CKPT"
  fi
  for subset in test holdout; do
    "$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/calibration_bench.py" \
      --data-dir "$CHANGE_DATA" --ckpt "$ckpt" --subset "$subset" \
      --device "$DEVICE" --batch-size 128 --workers 4 \
      --require-event-disjoint \
      --out-dir "$RUN_ROOT/calibration_${model_name}"
  done
done

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

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$RECHECK_SET" \
  --configs B0,B1,B2,B3 \
  --grounder hybrid --repeat 3 --seed 41 \
  --out-dir "$RUN_ROOT/e1_nav" --tag strict_e1

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_vln_navigation.py" \
  --testset "$RECHECK_SET" \
  --configs E11_NONE,E11_RANDOM,E11_FIXED,E11_HEURISTIC,E11_ENTROPY,E11_INFOGAIN \
  --grounder hybrid --repeat 3 --seed 41 \
  --out-dir "$RUN_ROOT/e11" --tag strict_e11

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
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/bench_report.py" "$RUN_ROOT/e11"
"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmarks/export_paper_assets.py" \
  --navigation "$RUN_ROOT/e1_nav/results.json" \
  --reinspection "$RUN_ROOT/e11/results.json" \
  --calibration-dir "$RUN_ROOT/calibration_baseline" \
  --diff-calibration-dir "$RUN_ROOT/calibration_diff_attention" \
  --yolo-metrics "$RUN_ROOT/yolo_metrics.json" \
  --loc-metrics "$RUN_ROOT/loc_metrics.json" \
  --out "$REPO_ROOT/paper/generated"
echo "Strict paper experiments completed under $RUN_ROOT"
