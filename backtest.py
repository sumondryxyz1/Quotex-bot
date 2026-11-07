#!/usr/bin/env python3
"""Honest backtest harness for signal engines.

The whole point of this file is to answer one question: *does the engine beat a
coin flip on data it has never seen?* Anything else (more indicators, nicer
rules) is decoration until that is measured.

Method
------
For a given (asset, period) we walk the stored candles forward. At each closed
candle we ask the engine for a signal, then look at the NEXT candle and record
whether the direction was right. No peeking: the engine only ever sees candles
[0..i], and we score it against candle i+1.

Reported per engine:
  - signals fired, CALL/PUT split
  - hit-rate (win rate), with a 95% confidence interval
  - expectancy in "pips" of ATR (avg move captured, net of a spread proxy)
  - a binomial p-value vs 50% (is the edge real or luck?)

Usage
-----
    python backtest.py                        # 5s, all pairs, current engine
    python backtest.py --period 15
    python backtest.py --compare              # old vs smart engine, side by side
    python backtest.py --pairs EURUSD_otc,GBPUSD_otc
"""
from __future__ import annotations

import argparse
import random
import math

from candle_store import CandleStore

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


class Result:
    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.wins = 0
        self.calls = 0
        self.puts = 0
        self.atr_sum = 0.0     # net move in ATR units, signed

    @property
    def hit(self) -> float:
        return self.wins / self.n if self.n else 0.0

    def ci(self) -> tuple[float, float]:
        """Wilson 95% interval for the hit-rate."""
        if not self.n:
            return 0.0, 0.0
        z = 1.96
        p = self.hit
        d = 1 + z * z / self.n
        c = p + z * z / (2 * self.n)
        m = z * math.sqrt(p * (1 - p) / self.n + z * z / (4 * self.n * self.n))
        return (c - m) / d, (c + m) / d

    def p_value(self) -> float:
        """Two-sided binomial p-value for hit-rate vs 50%."""
        if not self.n:
            return 1.0
        k = self.wins
        n = self.n
        # normal approximation is fine for n>=30; exact sum otherwise
        if n >= 30:
            z = (k - n / 2) / math.sqrt(n / 4)
            return math.erfc(abs(z) / math.sqrt(2))
        # exact two-sided
        total = 0.0
        for i in range(n + 1):
            pr = math.comb(n, i) * 0.5 ** n
            if abs(i - n / 2) >= abs(k - n / 2) - 1e-9:
                total += pr
        return min(1.0, total)


class RandomEngine:
    """Control: fires with a realistic rate, direction is a coin flip.

    If the real engines do not beat this, they have no edge — they are just
    generating signals.
    """

    def __init__(self, rate: float = 0.15, seed: int = 0):
        self.rate = rate
        self.rng = random.Random(seed)

    def evaluate(self, candles):
        if len(candles) < 60:
            return None
        if self.rng.random() > self.rate:
            return None

        class S:
            action = "CALL" if self.rng.random() < 0.5 else "PUT"

        return S()


def backtest_pair(engine, candles: list[list], warmup: int,
                  name: str, spread_atr: float = 0.15) -> Result:
    """Walk forward; engine sees [0..i], outcome measured on candle i+1.

    `engine` is any object with `.evaluate(candles) -> signal | None`; the signal
    only needs an `.action` of "CALL" or "PUT".

    `spread_atr` is a rough cost proxy in ATR units (broker spread + slippage),
    subtracted from every trade so the expectancy is not fantasy.
    """
    res = Result(name)
    for i in range(warmup, len(candles) - 1):
        sig = engine.evaluate(candles[:i + 1])
        if sig is None:
            continue
        nxt = candles[i + 1]
        move = nxt[2] - nxt[1]                    # next candle: close - open
        if move == 0:
            continue
        won = (move > 0) if sig.action == "CALL" else (move < 0)
        res.n += 1
        res.wins += 1 if won else 0
        if sig.action == "CALL":
            res.calls += 1
        else:
            res.puts += 1
        atr = _atr_at(candles, i)
        if atr:
            signed = (move / atr) if sig.action == "CALL" else (-move / atr)
            res.atr_sum += signed - spread_atr
    return res


def _atr_at(candles: list[list], i: int, period: int = 14) -> float:
    import indicators as ta
    highs = [c[3] for c in candles[:i + 1]]
    lows = [c[4] for c in candles[:i + 1]]
    closes = [c[2] for c in candles[:i + 1]]
    a = ta.atr(highs, lows, closes, period)
    return a[i] if a and a[i] else 0.0


def main():
    ap = argparse.ArgumentParser(description="Backtest signal engines honestly")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--pairs", default="", help="comma list (default: all with data)")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--compare", action="store_true", help="old vs smart engine")
    ap.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    ap.add_argument("--control", action="store_true", help="add a random baseline")
    ap.add_argument("--ensemble", action="store_true", help="add the 3-model ensemble")
    args = ap.parse_args()

    store = CandleStore(args.db)
    if args.pairs:
        assets = [a.strip() for a in args.pairs.split(",") if a.strip()]
    else:
        assets = [r[0] for r in store._db.execute(
            "SELECT DISTINCT asset FROM candles WHERE period=?", (args.period,)).fetchall()]
    if args.limit:
        assets = assets[:args.limit]

    from signal_bot import SignalEngine, Strategy

    class OldAdapter:
        """Wraps the old engine (evaluate(asset, candles)) to the common API."""

        def __init__(self, asset, threshold=4):
            self.engine = SignalEngine(Strategy(), threshold=threshold)
            self.asset = asset

        def evaluate(self, candles):
            return self.engine.evaluate(self.asset, candles)

    engines = [("current", lambda a: OldAdapter(a))]
    if args.compare:
        from smart_engine import SmartEngine, SmartStrategy
        engines.append(("smart", lambda a: SmartEngine(SmartStrategy())))
    if args.ensemble:
        from ensemble import EnsembleEngine, EnsembleStrategy
        engines.append(("ensemble", lambda a: EnsembleEngine(EnsembleStrategy())))
    if args.control:
        engines.append(("random", lambda a: RandomEngine()))

    label = PERIOD_LABEL.get(args.period, f"{args.period}s")
    print(f"\nbacktest {label} | {len(assets)} pairs | measure: next-candle direction")
    print("(random = same signal count/rate, coin-flip direction: the null)\n")

    for name, factory in engines:
        total = Result(name)
        per_pair = []
        for a in assets:
            cs = store.load(a, args.period, limit=20000)
            if len(cs) < 80:
                continue
            r = backtest_pair(factory(a), cs, warmup=60, name=a)
            if r.n:
                per_pair.append(r)
                total.n += r.n
                total.wins += r.wins
                total.calls += r.calls
                total.puts += r.puts
                total.atr_sum += r.atr_sum
        lo, hi = total.ci()
        exp = total.atr_sum / total.n if total.n else 0.0
        print(f"  {name:8} signals {total.n:>6}  (C {total.calls}/P {total.puts})  "
              f"hit {total.hit * 100:5.1f}%  95% CI [{lo * 100:.1f}, {hi * 100:.1f}]  "
              f"exp {exp:+.3f} ATR  p={total.p_value():.3f}")
        if per_pair:
            per_pair.sort(key=lambda r: -r.hit)
            best = per_pair[0]
            worst = per_pair[-1]
            print(f"           best pair {best.name} {best.hit * 100:.0f}% (n={best.n})"
                  f" | worst {worst.name} {worst.hit * 100:.0f}% (n={worst.n})")
    print()


if __name__ == "__main__":
    main()
