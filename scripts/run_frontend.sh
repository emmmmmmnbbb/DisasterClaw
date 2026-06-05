#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="${PROJECT_DIR}/frontend"
SHARED_NODE_MODULES="${PROJECT_DIR}/../AerialClaw/ui/node_modules"

if [ -f "${PROJECT_DIR}/.env" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
fi

if [ ! -d "${FRONTEND_DIR}/node_modules" ] && [ -d "${SHARED_NODE_MODULES}" ]; then
  ln -sfn "${SHARED_NODE_MODULES}" "${FRONTEND_DIR}/node_modules"
fi

cd "${FRONTEND_DIR}"
exec ./node_modules/.bin/vite --host "${FRONTEND_HOST:-127.0.0.1}" --port "${FRONTEND_PORT:-5173}"
