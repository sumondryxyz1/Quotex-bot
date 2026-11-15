#!/usr/bin/env bash
# Keep the signal dashboard alive on the public work host.
#
#   ./watchdog.sh              # supervise all pairs on port 12000
#   ./watchdog.sh EURUSD_otc 60 12001
#
# Restarts the dashboard if it exits (container restart, feed crash, OOM).
# Only one instance is ever allowed: the dashboard serves one feed session
# per source IP, so a second one silently gets assets=0.
set -u
cd "$(dirname "$0")"

ASSETS="${1:-}"
PERIODS="${2:-}"
PORT="${3:-12000}"
INTERVAL="${WATCHDOG_INTERVAL:-15}"
LOG=/tmp/dashboard.log

if [ ! -x .venv/bin/python ]; then
  echo "creating venv..."
  python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

mkdir -p out data

ARGS=""
[ -n "$ASSETS" ] && ARGS="$ARGS --assets $ASSETS"
[ -n "$PERIODS" ] && ARGS="$ARGS --periods $PERIODS"

running() {
  pgrep -f "dashboard.py --port ${PORT}" >/dev/null 2>&1
}

echo "watchdog up (port $PORT, every ${INTERVAL}s) — log: $LOG"
while true; do
  if ! running; then
    echo "$(date '+%F %T') dashboard down — starting"
    nohup .venv/bin/python -u dashboard.py --port "$PORT" $ARGS >> "$LOG" 2>&1 &
  fi
  sleep "$INTERVAL"
done
