#!/usr/bin/env python3
"""Reduce wrong signals: sweep selectivity settings and measure precision.

The rule engine fires whenever its net score clears a threshold. Lower threshold
= more signals (and more wrong ones). This tool sweeps the knobs that make the
engine *quieter*, and reports, on held-out data, whether being pickier actually
lowers the wrong-signal rate or just shrinks the sample.

Knobs swept:
  - threshold      : minimum |net score| to fire
  - agree          : require the timeframe's own trend regime to match
  - min_vol        : skip candles whose ATR is in the bottom X% (dead/flat)

For each setting we report coverage (how many signals survive) and precision
(hit-rate) with a Wilson CI, on a train/test split so the choice is honest.

Usage
-----
    python selective.py                 # 5s, OTC pairs
    python selective.py --period 60
    python selective.py --kind real
"""
from __future__ import annotations

import argparse
import math

from candle_store import CandleStore
import indicators as ta
from signal_bot import SignalEngine, Strategy

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


def wilson(w: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def atr_series(candles: list[list], period: int = 14) -> list:
    highs = [c[3] for c in candles]
    lows = [c[4] for c in candles]
    closes = [c[2] for c in candles]
    return ta.atr(highs, lows, closes, period)


def regime_up(candles: list[list], i: int, period: int = 21) -> bool:
    """Simple trend regime: EMA21 rising over the last few candles."""
    closes = [c[2] for c in candles[:i + 1]]
    e = ta.ema(closes, period)
    if e[i] is None or i < 3 or e[i - 3] is None:
        return False
    return e[i] > e[i - 3]


def evaluate(engine: SignalEngine, asset: str, candles: list[list], i: int,
             window: int = 0):
    """Run the engine on candles[0..i] (closed) and return the signal or None.

    `window` > 0 truncates the history to the last `window` candles. EMA/RSI are
    recursive, so this is a close approximation of the full-history signal — fine
    for a relative sweep, and it makes the tool O(n) instead of O(n^2).
    """
    lo = max(0, i + 1 - window) if window else 0
    return engine.evaluate(asset, candles[lo:i + 1])


def mtf_trend(candles_htf: list[list], t: int, fast: int = 9, slow: int = 21):
    """Trend of the higher timeframe at time t: +1 up, -1 down, 0 unknown.

    Uses the last higher-timeframe candle that closed at or before `t`.
    """
    idx = None
    for k in range(len(candles_htf) - 1, -1, -1):
        if candles_htf[k][0] <= t:
            idx = k
            break
    if idx is None or idx < slow + 1:
        return 0
    closes = [c[2] for c in candles_htf[:idx + 1]]
    ef = ta.ema(closes, fast)
    es = ta.ema(closes, slow)
    if ef[idx] is None or es[idx] is None:
        return 0
    return 1 if ef[idx] > es[idx] else -1


def sweep(store: CandleStore, period: int, kind: str, payout: float,
          thresholds: list[int], test_frac: float = 0.4, window: int = 0,
          limit: int = 0, mtf_period: int = 0):
    breakeven = 1.0 / (1.0 + payout)
    assets = [r[0] for r in store._db.execute(
        "SELECT DISTINCT asset FROM candles WHERE period=?", (period,)).fetchall()]
    if kind == "otc":
        assets = [a for a in assets if a.endswith("_otc")]
    elif kind == "real":
        assets = [a for a in assets if not a.endswith("_otc")]
    if limit:
        assets = assets[:limit]

    panels = []
    for a in assets:
        cs = store.load(a, period, limit=20000)
        if len(cs) >= 200:
            panels.append((a, cs))
    if not panels:
        print(f"no data for {PERIOD_LABEL.get(period, period)} kind={kind}")
        return

    n_min = min(len(cs) for _, cs in panels)
    cut = int(n_min * (1 - test_frac))
    label = PERIOD_LABEL.get(period, f"{period}s")

    print(f"\nselective sweep | {label} | {len(panels)} pairs (kind={kind}) | "
          f"test = last {test_frac:.0%}")
    print(f"breakeven at {payout * 100:.0f}% payout = {breakeven * 100:.1f}%\n")
    print(f"  {'setting':<28} {'signals':>8} {'coverage':>9} "
          f"{'hit%':>6} {'95% CI':>16}  verdict")

    # baseline coverage = total evaluable candles on the test window
    variants = [(False, 0.0), (True, 0.0), (False, 0.5), (True, 0.5)]
    for t in thresholds:
        for agree, minvol in variants:
            for use_mtf in ([False, True] if mtf_period else [False]):
                n = w = 0
                evals = 0
                for a, cs in panels:
                    eng = SignalEngine(Strategy(), threshold=t)
                    atr = atr_series(cs)
                    htf = store.load(a, mtf_period, limit=20000) if mtf_period else None
                    for i in range(cut, len(cs) - 1):
                        if len(cs[:i + 1]) < 60:
                            continue
                        evals += 1
                        # volatility gate
                        if minvol > 0:
                            window_atr = [x for x in atr[max(0, i - 100):i + 1] if x]
                            if not window_atr:
                                continue
                            thr = sorted(window_atr)[int(len(window_atr) * minvol)]
                            if atr[i] is None or atr[i] < thr:
                                continue
                        sig = evaluate(eng, a, cs, i, window)
                        if sig is None:
                            continue
                        if agree:
                            up = regime_up(cs, i)
                            if (sig.action == "CALL") != up:
                                continue
                        if use_mtf and htf:
                            tr = mtf_trend(htf, cs[i][0])
                            if tr == 0 or (sig.action == "CALL") != (tr > 0):
                                continue
                        move = cs[i + 1][2] - cs[i + 1][1]
                        if move == 0:
                            continue
                        n += 1
                        if (move > 0) if sig.action == "CALL" else (move < 0):
                            w += 1
                cov = n / evals if evals else 0.0
                hit = w / n if n else 0.0
                lo, hi = wilson(w, n)
                tag = f"thr={t} agree={int(agree)} vol={minvol:.1f}"
                if use_mtf:
                    tag += f" mtf={mtf_period}s"
                if n < 30:
                    verdict = "too few"
                elif lo > breakeven:
                    verdict = "EDGE"
                elif hi < breakeven:
                    verdict = "worse than random"
                else:
                    verdict = "~noise"
                print(f"  {tag:<28} {n:>8} {cov * 100:>8.1f}% "
                      f"{hit * 100:>5.1f}% [{lo * 100:>5.1f},{hi * 100:>5.1f}]  {verdict}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Sweep selectivity to cut wrong signals")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--kind", choices=["all", "otc", "real"], default="otc")
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--thresholds", default="2,3,4,5,6,7,8")
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--window", type=int, default=300,
                    help="truncate engine history to this many candles (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    ap.add_argument("--mtf", type=int, default=0,
                    help="higher timeframe (s) the signal must agree with (0 = off)")
    ap.add_argument("--db", default="data/candles.db")
    args = ap.parse_args()
    th = [int(x) for x in args.thresholds.split(",") if x.strip()]
    store = CandleStore(args.db)
    sweep(store, args.period, args.kind, args.payout, th, args.test_frac,
          args.window, args.limit, args.mtf)


if __name__ == "__main__":
    main()
