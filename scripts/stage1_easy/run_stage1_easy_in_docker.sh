#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-gb10_pytorch}"
PORT="${2:-8778}"

if ! docker inspect "$CONTAINER" &>/dev/null; then
  echo "Container not found: $CONTAINER"
  exit 1
fi

# Start monitor in container (built-in pidfile replace keeps a single instance on PORT)
docker exec -d "$CONTAINER" bash -lc \
  "cd /workspace/LLaMA-Factory && exec python3 -u scripts/stage1_easy/serve_stage1_easy_monitor.py --host 0.0.0.0 --port ${PORT} --replace"

sleep 1
IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER" 2>/dev/null | head -1)

echo "Stage1 Easy monitor started in container: $CONTAINER (port $PORT)"
echo ""
echo "  A) If docker run uses  -p ${PORT}:${PORT}  →  http://127.0.0.1:${PORT}/"
echo "  B) Linux bridge IP (may not work on Docker Desktop):     http://${IP}:${PORT}/"
echo ""
echo "Now starting the pipeline inside container..."
echo ""

docker exec -d "$CONTAINER" bash -lc "cd /workspace/LLaMA-Factory && python3 scripts/stage1_easy/run_stage1_easy.py"

