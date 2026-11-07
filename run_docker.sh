#!/usr/bin/env bash
# Start the dashboard under Docker, with candle history on a named volume.
#
#   ./run_docker.sh                 # build + start, port 12000
#   ./run_docker.sh 8080            # use a different host port
#   docker logs -f qx-dashboard     # follow the feed
#   docker rm -f qx-dashboard       # stop
#
# The URL is only as permanent as the machine this runs on. To get a stable
# public https:// link, deploy the same image to Fly.io / Render (see DEPLOY.md).
set -u
cd "$(dirname "$0")"

DOCKER="${DOCKER:-docker}"   # use DOCKER="sudo docker" if your user needs sudo
PORT="${1:-12000}"
NAME="${2:-qx-dashboard}"

$DOCKER rm -f "$NAME" >/dev/null 2>&1

$DOCKER run -d --name "$NAME" \
  -p "${PORT}:12000" \
  -e CANDLE_DB=/data/candles.db \
  -v qx-candles:/data \
  --restart unless-stopped \
  "$($DOCKER build -q .)"

echo "dashboard container '$NAME' on http://localhost:${PORT}"
echo "logs: $DOCKER logs -f $NAME"
