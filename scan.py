#!/usr/bin/env python3
"""Scan every pair — one timeframe, several, or all of them — in one command.

This is the batch companion to the dashboard's "Analyze" button: instead of
picking one pair at a time, it runs the engine over the whole candle store and
ranks the strongest CALL/PUT setups. The same `scan_all()` is what the
dashboard's "Scan all pairs" button and the Telegram alerts use, so every
surface agrees.

Usage
-----
    python scan.py                     # 5s, all pairs
    python scan.py 15                  # 15s
    python scan.py --all-tf            # every timeframe that has data
    python scan.py 5,15,60             # several timeframes
    python scan.py --engine smart      # regime-aware engine
    python scan.py --all-tf --top 30   # top 30 across all timeframes
    python scan.py 5 --watch           # re-scan every 30s (live)
    python scan.py --json              # machine-readable output

Signals are alerts only — nothing is ever traded.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from analysis import analyze, analyze_smart, analyze_ensemble
from candle_store import CandleStore, PERIODS

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


def parse_periods(spec: str) -> list[int]:
    """'5' -> [5]; '5,15,60' -> [5,15,60]; 'all' -> every offered timeframe."""
    if not spec or spec.strip().lower() == "all":
        return list(PERIODS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or list(PERIODS)


def entry_plan(period: int, last_time: int, now: float,
               expiry_candles: int = 1) -> dict:
    """When to enter and when the trade expires, for a signal read off candles.

    The engine only sees candles that have already closed, so the signal is
    known at `last_time + period` — the open of the next candle. Binary options
    settle on candle boundaries, so entry is snapped to the next boundary at or
    after `now` (never a moment already in the past, which would be a look-ahead
    lie). Expiry is that entry plus `expiry_candles` candles.
    """
    nxt = last_time + period                    # boundary after the last closed candle
    boundary = int(now // period) * period      # boundary of the candle forming now
    if boundary < now:
        boundary += period                      # first boundary still in the future
    entry = max(nxt, boundary)
    n = max(1, expiry_candles)
    expiry = entry + period * n
    return {
        "entry_at": entry,
        "expiry_at": expiry,
        "entry_in": max(0, int(round(entry - now))),
        "expires_in": max(0, int(round(expiry - now))),
        "expiry_candles": n,
    }


def _is_dead(candles: list[list], window: int = 20) -> bool:
    """True if the last `window` candles never moved — a closed/illiquid pair.

    Real-market pairs (non-OTC) stop ticking when the market is shut, so their
    candles have identical open/close. Scanning them yields only flat signals.
    """
    if len(candles) < window:
        return False
    tail = candles[-window:]
    return all(c[1] == c[2] for c in tail)


def scan_all(store: CandleStore, periods: list[int], engine: str = "rule",
             top: int = 0, include_neutral: bool = False,
             min_candles: int = 0, expiry_candles: int = 1,
             skip_dead: bool = True) -> dict:
    """Run the engine over every (pair, timeframe) and return a ranked report.

    Returns a plain dict so the dashboard, the CLI, and the alert loop can all
    consume it. `signals` holds fired CALL/PUT results sorted by strength;
    `ready` / `waiting` count series with and without enough candles. Every row
    carries an entry/expiry plan (see `entry_plan`) so the caller knows *when*
    the signal is meant to be taken, not just its direction.

    `skip_dead` drops pairs whose last N candles never moved (real-market pairs
    while the market is closed). They produce only flat, meaningless signals.
    """
    fn = analyze_smart if engine == "smart" else (analyze_ensemble if engine == "ensemble" else analyze)
    pairs = store.series(min_candles=min_candles, periods=periods)
    now = time.time()
    fired, waiting, dead = [], 0, 0
    for asset, period in pairs:
        cs = store.load(asset, period, limit=5000)
        if skip_dead and _is_dead(cs):
            dead += 1
            continue
        res = fn(asset, period, cs)
        if not res.enough:
            waiting += 1
            continue
        fired.append((res, cs[-1][0] if cs else 0))

    signals = [(r, t) for r, t in fired if r.action != "NEUTRAL"]
    signals.sort(key=lambda rt: (-abs(rt[0].score), rt[0].asset, rt[0].period))
    rows = signals if not include_neutral else signals + [
        (r, t) for r, t in fired if r.action == "NEUTRAL"]

    def row(r, last_time):
        plan = entry_plan(r.period, last_time, now, expiry_candles)
        return {
            "asset": r.asset, "period": r.period,
            "label": PERIOD_LABEL.get(r.period, f"{r.period}s"),
            "action": r.action, "score": r.score,
            "bull": r.bull, "bear": r.bear,
            "regime": r.regime, "confidence": r.confidence,
            "candles": r.candles, "reasons": r.reasons,
            **plan,
        }

    return {
        "engine": engine,
        "periods": periods,
        "now": int(now),
        "scanned": len(pairs),
        "ready": len(fired),
        "waiting": waiting,
        "dead": dead,
        "signal_count": len(signals),
        "signals": [row(r, t) for r, t in (rows[:top] if top else rows)],
    }


def _clock(ts: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _countdown(secs: int) -> str:
    if secs <= 0:
        return "now"
    if secs < 60:
        return f"in {secs}s"
    return f"in {secs // 60}m{secs % 60:02d}s"


def _print_report(rep: dict, top: int):
    labels = ",".join(PERIOD_LABEL.get(p, str(p)) for p in rep["periods"])
    multi = len(rep["periods"]) > 1
    scope = "all timeframes" if len(rep["periods"]) == len(PERIODS) else labels
    print(f"\n=== scan {scope} | engine {rep['engine']} | "
          f"{rep['scanned']} series ({rep['ready']} ready, {rep['waiting']} collecting"
          f"{', ' + str(rep.get('dead', 0)) + ' dead' if rep.get('dead') else ''})"
          f" | {rep['signal_count']} signals | now {_clock(rep['now'])} ===")
    if not rep["signals"]:
        print("  no signal above threshold right now "
              "(that is normal — most candles are noise)")
        return
    for r in rep["signals"][:top]:
        arrow = {"CALL": "▲ CALL", "PUT": "▼ PUT"}.get(r["action"], "· NEUTRAL")
        tf = f"{r['label']:>4}" if multi else ""
        if rep["engine"] == "smart":
            detail = f"{r['regime'] or '-':5} conf {r['confidence']:.2f}"
        else:
            detail = f"score {r['score']:>2}  (bull {r['bull']}/bear {r['bear']})"
        print(f"  {arrow:9} {r['asset']:14} {tf} {detail}  candles {r['candles']}")
        print(f"             entry {_clock(r['entry_at'])} ({_countdown(r['entry_in'])})"
              f"  →  expiry {_clock(r['expiry_at'])} ({_countdown(r['expires_in'])})")
        for why in r["reasons"][:3]:
            print(f"             - {why}")


def main():
    ap = argparse.ArgumentParser(description="Scan all pairs for signals")
    ap.add_argument("periods", nargs="?", default="5",
                    help="5 | 5,15,60 | all (default 5)")
    ap.add_argument("--all-tf", action="store_true",
                    help="scan every timeframe that has data")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--all", action="store_true", help="also list NEUTRAL rows")
    ap.add_argument("--engine", choices=["rule", "smart", "ensemble"], default="rule")
    ap.add_argument("--min-candles", type=int, default=0,
                    help="only scan series with at least this many candles")
    ap.add_argument("--expiry", type=int, default=1, metavar="N",
                    help="expiry in candles after entry (default 1)")
    ap.add_argument("--json", action="store_true", help="print JSON, not a table")
    ap.add_argument("--watch", action="store_true",
                    help="re-scan every 30s until Ctrl-C")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"no candle store at {args.db} — start the dashboard "
                         f"first so it can collect candles")

    periods = list(PERIODS) if args.all_tf else parse_periods(args.periods)
    store = CandleStore(args.db)

    def once():
        rep = scan_all(store, periods, args.engine,
                       top=args.top, include_neutral=args.all,
                       min_candles=args.min_candles,
                       expiry_candles=args.expiry)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            _print_report(rep, args.top)
        return rep

    if not args.watch:
        once()
        return
    try:
        while True:
            once()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
