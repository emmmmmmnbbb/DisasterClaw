#!/usr/bin/env bash
set -euo pipefail

# This script is intended to be run on your LOCAL workstation.
# It forwards local port -> remote DisasterClaw backend port via SSH.

LOCAL_PORT="${DISASTERCLAW_LOCAL_WEB_PORT:-5011}"
REMOTE_PORT="${DISASTERCLAW_REMOTE_WEB_PORT:-5011}"
REMOTE_HOST="${DISASTERCLAW_REMOTE_HOST:-}"
REMOTE_USER="${DISASTERCLAW_REMOTE_USER:-}"
SSH_PORT="${DISASTERCLAW_REMOTE_PORT:-22}"

if [ -z "${REMOTE_HOST}" ] || [ -z "${REMOTE_USER}" ]; then
  echo "ERROR: Please set DISASTERCLAW_REMOTE_HOST and DISASTERCLAW_REMOTE_USER first."
  echo "Optional: DISASTERCLAW_REMOTE_PORT, DISASTERCLAW_REMOTE_WEB_PORT, DISASTERCLAW_LOCAL_WEB_PORT"
  exit 1
fi

ssh -fNT \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  -p "${SSH_PORT}" \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=5 \
  -o ExitOnForwardFailure=yes \
  "${REMOTE_USER}@${REMOTE_HOST}"

echo "DisasterClaw SSH tunnel started: http://127.0.0.1:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"

