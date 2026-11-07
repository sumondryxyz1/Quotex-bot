#!/usr/bin/env bash
# Start the multi-pair signal dashboard on the public work host (port 12000).
#
#   ./run_dashboard.sh                      # all pairs, all timeframes
#   ./run_dashboard.sh EURUSD_otc,BTCUSD_otc
#   ./run_dashboard.sh EURUSD_otc,BTCUSD_otc 60,300
#
# The dashboard is then reachable at the work-1 URL shown in the environment.
set -u
cd "$(dirname "$0")"

ASSETS="${1:-}"
PERIODS="${2:-}"
PORT="${3:-12000}"

if [ ! -x .venv/bin/python ]; then
  echo "creating venv..."
  python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

pkill -f "dashboard.py" 2>/dev/null
sleep 1
mkdir -p out data

ARGS=""
[ -n "$ASSETS" ] && ARGS="$ARGS --assets $ASSETS"
[ -n "$PERIODS" ] && ARGS="$ARGS --periods $PERIODS"

nohup .venv/bin/python -u dashboard.py --port "$PORT" $ARGS \
  > /tmp/dashboard.log 2>&1 &
echo "dashboard started (pid $!) on port $PORT — log: /tmp/dashboard.log"
echo "open: https://work-1-ajdlpljacpvdqsuz.prod-runtime.all-hands.dev/"

