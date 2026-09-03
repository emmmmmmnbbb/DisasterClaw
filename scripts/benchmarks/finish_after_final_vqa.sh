#!/usr/bin/env bash
# Post-process after paper_cja_mech_final Agent-VQA shards finish.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1
PY="${DISASTERCLAW_PYTHON_BIN:-/home/lc/miniconda3/envs/disasterclaw/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/benchmarks/paper_cja_mech_v1}"
REPORTS="$ROOT/runs/benchmarks/cja_agent_vqa/paper_cja_mech_final_reports"
if [[ ! -d "$REPORTS" ]]; then
  echo "[ERROR] merged reports not found: $REPORTS" >&2
  echo "Wait for run_agent_vqa_parallel.sh to finish both shards." >&2
  exit 2
fi
rm -rf "$RUN_ROOT/final_reports"
cp -a "$REPORTS" "$RUN_ROOT/final_reports"
"$PY" "$ROOT/scripts/benchmarks/export_cja_assets.py" --run-root "$RUN_ROOT"
echo "[ok] copied reports and exported assets → $RUN_ROOT and paper_cja/generated"
