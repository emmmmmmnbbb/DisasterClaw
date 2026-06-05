#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "${PROJECT_DIR}/.env" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
fi

has_backend_runtime() {
  local py_bin="$1"
  [ -x "$py_bin" ] || return 1
  "$py_bin" -c "import flask, flask_socketio, flask_cors" >/dev/null 2>&1
}

choose_python_bin() {
  local candidates=()

  if [ -n "${DISASTERCLAW_PYTHON_BIN:-}" ]; then
    candidates+=("${DISASTERCLAW_PYTHON_BIN}")
  fi
  if [ -n "${CONDA_PREFIX:-}" ]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=("/home/lc/miniconda3/envs/disasterclaw/bin/python")
  candidates+=("${PROJECT_DIR}/../AerialClaw/.venv/bin/python")
  candidates+=("python3")

  for candidate in "${candidates[@]}"; do
    if has_backend_runtime "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

if ! PYTHON_BIN="$(choose_python_bin)"; then
  echo "No usable Python runtime found for backend/app.py" >&2
  echo "Set DISASTERCLAW_PYTHON_BIN or install flask/flask_socketio/flask_cors into an available env." >&2
  exit 1
fi

cd "${PROJECT_DIR}/backend"
exec "${PYTHON_BIN}" app.py
