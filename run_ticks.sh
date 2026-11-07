#!/usr/bin/env bash
# Start the raw tick collector (order-flow data layer).
#
# Collects every pair's ticks (price + broker direction flag) into the ticks
# table of $CANDLE_DB. Runs for days; reconnects on drop.
#
# Usage:
#   ./run_ticks.sh                      # all pairs, forever
#   ./run_ticks.sh --pairs EURUSD_otc,BTCUSD_otc
#   ./run_ticks.sh --minutes 60         # stop after an hour
#   ./run_ticks.sh --stats              # show what is stored, exit
set -u
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "creating venv..."
  python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

mkdir -p data out
nohup .venv/bin/python -u collect_ticks.py "$@" > /tmp/ticks.log 2>&1 &
echo "tick collector started (pid $!) — log: /tmp/ticks.log"
echo "tail it with:  tail -f /tmp/ticks.log"
