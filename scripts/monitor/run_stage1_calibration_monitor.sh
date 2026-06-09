#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-stage1_monitor}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-16006}
PYTHON=${PYTHON:-/workspace/qwen3vl-env/bin/python}
SERVER=/workspace/fundus-qwen3vl-project/scripts/monitor/serve_stage1_arm_b_monitor.py

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "$PYTHON -u $SERVER --host $HOST --port $PORT"

echo "Stage1 calibration monitor upstream: http://$HOST:$PORT/ (public Caddy entry: :6006, tmux: $SESSION)"
