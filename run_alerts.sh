#!/usr/bin/env bash
# Start Telegram signal alerts.
#
# One-time setup:
#   1. @BotFather -> /newbot -> copy the token
#   2. message your bot once, then:  ./run_alerts.sh --whoami --token <TOKEN>
#   3. put the token + chat id in telegram.json:
#        {"bot_token": "123456:ABC...", "chat_id": "987654321"}
#      (or export TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
#
# Usage:
#   ./run_alerts.sh --test                    # verify the connection
#   ./run_alerts.sh                           # watch 5s+15s (default)
#   ./run_alerts.sh --periods all             # every pair x every timeframe
#   ./run_alerts.sh --engine smart            # regime-aware engine
#   ./run_alerts.sh --periods 5,15,60 --min-score 6
set -u
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "creating venv..."
  python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

pkill -f "alerts.py" 2>/dev/null
sleep 1
mkdir -p out data

nohup .venv/bin/python -u alerts.py "$@" > /tmp/alerts.log 2>&1 &
echo "alerts started (pid $!) — log: /tmp/alerts.log"
echo "tail it with:  tail -f /tmp/alerts.log"
