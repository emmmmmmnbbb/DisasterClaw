#!/usr/bin/env bash
# Finalize paper tables once paper_strict_unet_v1 E1/E11 finish.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${PAPER_RUN_ROOT:-$REPO_ROOT/runs/benchmarks/paper_strict_unet_v1}"
PY="${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}"

test -f "$ROOT/e11/results.json"
test -f "$ROOT/e1_nav/results.json"
test -f "$ROOT/loc_metrics.json"

# Prefer existing YOLO metrics from prior strict run if not regenerated here.
YOLO_METRICS="${ROOT}/yolo_metrics.json"
if [[ ! -f "$YOLO_METRICS" ]]; then
  YOLO_METRICS="$REPO_ROOT/runs/benchmarks/paper_strict_v1/yolo_metrics.json"
fi
test -f "$YOLO_METRICS"

CAL_BASE="${ROOT}/calibration_baseline"
CAL_DIFF="${ROOT}/calibration_diff_attention"
if [[ ! -d "$CAL_BASE" ]]; then
  CAL_BASE="$REPO_ROOT/runs/benchmarks/paper_strict_v1/calibration_baseline"
fi
if [[ ! -d "$CAL_DIFF" ]]; then
  CAL_DIFF="$REPO_ROOT/runs/benchmarks/paper_strict_v1/calibration_diff_attention"
fi

$PY "$REPO_ROOT/scripts/benchmarks/export_paper_assets.py" \
  --navigation "$ROOT/e1_nav/results.json" \
  --reinspection "$ROOT/e11/results.json" \
  --calibration-dir "$CAL_BASE" \
  --diff-calibration-dir "$CAL_DIFF" \
  --yolo-metrics "$YOLO_METRICS" \
  --loc-metrics "$ROOT/loc_metrics.json" \
  --out "$REPO_ROOT/paper/generated"

echo "Exported paper assets from $ROOT"
