#!/usr/bin/env python3
"""Edge finder — which pairs/timeframes actually beat a coin flip?

This is the honest companion to the dashboard. The dashboard shows signals that
*look* confident; this script measures whether those signals were ever right,
pair by pair, on data the engine has not seen.

For each (pair, timeframe) it walks the stored candles forward: the engine sees
candles [0..i] and we score its CALL/PUT against candle i+1. It then reports the
hit-rate with a 95% interval and a binomial p-value against 50%.

A binary-option trade only breaks even when

    hit-rate  >  1 / (1 + payout)

At an 85% payout that is ~54.1%; at 90% it is ~52.6%. Anything below that line
loses money no matter how good the signal looks.

Usage
-----
    python edge.py                      # 1m, all pairs, smart engine
    python edge.py --period 60 --engine rule
    python edge.py --period 300 --min-signals 30
"""
from __future__ import annotations

import argparse
import math

from candle_store import CandleStore, PERIODS

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


def wilson(wins: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z, p = 1.96, wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def p_value(wins: int, n: int) -> float:
    """Two-sided binomial p-value vs 50%."""
    if not n:
        return 1.0
    k = min(wins, n - wins)
    # normal approximation is fine for n >= 30; exact-ish below via log-gamma
    if n >= 30:
        z = (wins - n / 2) / math.sqrt(n / 4)
        return math.erfc(abs(z) / math.sqrt(2))
    total = 0.0
    for j in range(k + 1):
        total += math.comb(n, j)
    return min(1.0, 2 * total / (2 ** n))


def make_engine(name: str, asset: str):
    if name == "rule":
        from signal_bot import SignalEngine, Strategy
        eng = SignalEngine(Strategy(), threshold=4)
        return lambda cs: eng.evaluate(asset, cs)
    from smart_engine import SmartEngine, SmartStrategy
    eng = SmartEngine(SmartStrategy())
    return eng.evaluate


def run(store: CandleStore, period: int, engine_name: str,
        min_signals: int, payout: float, warmup: int = 60,
        kind: str = "all", horizon: int = 1):
    breakeven = 1.0 / (1.0 + payout)
    assets = [r[0] for r in store._db.execute(
        "SELECT DISTINCT asset FROM candles WHERE period=?", (period,)).fetchall()]
    if kind == "otc":
        assets = [a for a in assets if a.endswith("_otc")]
    elif kind == "real":
        assets = [a for a in assets if not a.endswith("_otc")]

    rows = []
    for a in assets:
        cs = store.load(a, period, limit=20000)
        if len(cs) < warmup + 20:
            continue
        ev = make_engine(engine_name, a)
        n = w = 0
        for i in range(warmup, len(cs) - horizon):
            sig = ev(cs[:i + 1])
            if sig is None:
                continue
            move = cs[i + horizon][2] - cs[i][2]   # close-to-close, like the journal
            if move == 0:
                continue
            n += 1
            if (move > 0) if sig.action == "CALL" else (move < 0):
                w += 1
        if n >= min_signals:
            lo, hi = wilson(w, n)
            rows.append((a, n, w / n, lo, hi, p_value(w, n)))

    label = PERIOD_LABEL.get(period, f"{period}s")
    print(f"\nedge finder | {label} | {engine_name} engine | kind={kind} | "
          f"horizon {horizon} | breakeven at {payout * 100:.0f}% payout = "
          f"{breakeven * 100:.1f}%\n")
    if not rows:
        print(f"  no pair had >= {min_signals} signals on {label} yet.")
        print("  collect more candles, or lower --min-signals.\n")
        return

    rows.sort(key=lambda r: -r[2])
    print(f"  {'pair':<16} {'n':>6} {'hit%':>7} {'95% CI':>16} {'p':>7}  verdict")
    good = 0
    for a, n, hit, lo, hi, p in rows:
        # a pair is worth trading only if the CI's lower bound clears breakeven
        ok = lo > breakeven
        verdict = "EDGE" if ok else ("lean+" if hit > breakeven else "-")
        good += ok
        print(f"  {a:<16} {n:>6} {hit * 100:>6.1f}% "
              f"[{lo * 100:>5.1f},{hi * 100:>5.1f}] {p:>7.3f}  {verdict}")

    print(f"\n  {good} of {len(rows)} pairs have a statistically real edge "
          f"(CI lower bound > breakeven).")
    if good == 0:
        print("  That means: on this data, no pair beats the payout. Trading any")
        print("  of them is a coin flip minus the spread.\n")
    else:
        print("  Trade only the EDGE pairs, and re-check often — edges decay.\n")


def main():
    ap = argparse.ArgumentParser(description="Find pairs that beat a coin flip")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--engine", choices=["rule", "smart"], default="smart")
    ap.add_argument("--min-signals", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=1,
                    help="expiry in candles (close-to-close)")
    ap.add_argument("--kind", choices=["all", "otc", "real"], default="all")
    ap.add_argument("--payout", type=float, default=0.85,
                    help="broker payout, e.g. 0.85 for 85%%")
    ap.add_argument("--db", default="data/candles.db")
    args = ap.parse_args()
    store = CandleStore(args.db)
    run(store, args.period, args.engine, args.min_signals, args.payout,
        kind=args.kind, horizon=args.horizon)


if __name__ == "__main__":
    main()
