# Quotex-bot — live candle dashboard + signal scanner

A live market-data dashboard and signal scanner built on the public
`market-qx.info` WebSocket feed (a Quotex white-label).

It streams live candles for ~92 pairs across 8 timeframes into a local SQLite
store, runs EMA/RSI/MACD/Bollinger/Stochastic analysis on them, and shows
CALL / PUT signals with an entry time and expiry time.

## What it does

- **Live candles** — connects to the feed and builds OHLC candles in real time.
- **All-pair scan** — every pair × every timeframe in one pass (~0.6 s).
- **Two engines** — `rule` (indicator votes) and `smart` (regime + volatility).
- **Entry / expiry time** — each signal says *when* to enter, not just the
  direction. Entry is snapped to the next candle boundary and is never in the
  past.
- **History** — candles persist in `data/candles.db` (the only durable asset).
- **Telegram alerts** — optional, via `alerts.py`.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_dashboard.sh                 # http://localhost:12000
```

Or with Docker:

```bash
./run_docker.sh                    # http://localhost:12000
```

For a permanent public `https://` link, see **[DEPLOY.md](DEPLOY.md)**.

## CLI

```bash
python scan.py 5                   # scan 5s pairs
python scan.py --all-tf --top 12   # every timeframe
python scan.py 60 --expiry 3       # 3-candle expiry
python alerts.py --periods all --engine smart   # Telegram loop
python journal.py --report         # your live track record
python golive.py --report          # demo -> real money readiness gate
```

## Tests

```bash
for t in test_*.py; do .venv/bin/python "$t"; done
```

## Notes

- The feed allows **one live connection per IP**. Do not run two dashboards on
  the same host — the second one gets no data.
- Signals are statistical hints, not advice. Nothing here places trades.

## Going from demo to real money

Run it on demo, let `journal.py` settle signals, then ask `golive.py` whether the
measured edge justifies real money. The dashboard shows the verdict live. It
stays **NOT READY** until the demo record is statistically real — treat that as a
feature, not a bug. See [AGENTS.md](AGENTS.md) for the exact criteria and the
full architecture.
