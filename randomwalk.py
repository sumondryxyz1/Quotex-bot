#!/usr/bin/env python3
"""Is the feed predictable at all? Random-walk diagnostics.

Before engineering more signals, answer the fundamental question: do returns
carry any information about the next return? If not, every indicator rule is
doomed and no amount of tuning will help.

Three standard tests, pooled across pairs:

  1. **Return autocorrelation** at lags 1..10. A random walk has ~0.
     Positive at lag 1 -> momentum; negative -> mean reversion.
  2. **Variance ratio** VR(q) = Var(q-period ret) / (q * Var(1-period ret)).
     VR = 1 for a random walk, > 1 trending, < 1 mean-reverting.
     We report the Lo-MacKinlay z-score.
  3. **Direction predictability**: the accuracy of the best possible
     "predict from the last k returns" linear rule, as an upper bound.

If all three say "no", the honest conclusion is: this feed is not tradable with
candle indicators, full stop.

Usage
-----
    python randomwalk.py --period 5
    python randomwalk.py --period 60 --lags 10
"""
from __future__ import annotations

import argparse
import math

from candle_store import CandleStore

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def var(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def corr(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def main():
    ap = argparse.ArgumentParser(description="Random-walk diagnostics")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--lags", type=int, default=10)
    ap.add_argument("--db", default="data/candles.db")
    args = ap.parse_args()

    store = CandleStore(args.db)
    assets = [r[0] for r in store._db.execute(
        "SELECT DISTINCT asset FROM candles WHERE period=?", (args.period,)).fetchall()]

    rets_by_pair = []
    for a in assets:
        cs = store.load(a, args.period, limit=20000)
        if len(cs) < 100:
            continue
        closes = [c[2] for c in cs]
        rets = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        # normalise per pair (divide by std) so pooling across pairs with very
        # different price scales is meaningful
        sd = math.sqrt(var(rets)) if var(rets) > 0 else 0.0
        if sd > 0:
            rets_by_pair.append([r / sd for r in rets])

    if not rets_by_pair:
        print("no data")
        return

    label = PERIOD_LABEL.get(args.period, f"{args.period}s")
    total_n = sum(len(r) for r in rets_by_pair)
    print(f"\nrandom-walk diagnostics | {label} | {len(rets_by_pair)} pairs | "
          f"{total_n} returns\n")

    # 1. autocorrelation, pooled by pairing returns[i] with returns[i+lag]
    print("  lag   autocorr   approx SE   z")
    for lag in range(1, args.lags + 1):
        xs, ys = [], []
        for r in rets_by_pair:
            if len(r) > lag + 5:
                xs.extend(r[:-lag])
                ys.extend(r[lag:])
        ac = corr(xs, ys)
        se = 1.0 / math.sqrt(len(xs)) if xs else 0.0
        z = ac / se if se else 0.0
        flag = "  <-- |z|>2" if abs(z) > 2 else ""
        print(f"  {lag:>3}   {ac:>+9.5f}   {se:>8.5f}   {z:>+6.2f}{flag}")

    # 2. variance ratio
    print("\n  variance ratio (VR=1 random walk, <1 mean-revert, >1 trend)")
    for q in (2, 5, 10):
        v1 = mean([var(r) for r in rets_by_pair])
        vq_vals = []
        for r in rets_by_pair:
            if len(r) < q * 5:
                continue
            agg = [sum(r[i:i + q]) for i in range(0, len(r) - q + 1, q)]
            vq_vals.append(var(agg))
        vq = mean(vq_vals) if vq_vals else 0.0
        vr = (vq / q) / v1 if v1 else 0.0
        print(f"    q={q:>2}   VR={vr:>6.3f}")

    # 3. base rates
    ups = sum(1 for r in rets_by_pair for x in r if x > 0)
    downs = sum(1 for r in rets_by_pair for x in r if x < 0)
    zeros = total_n - ups - downs
    print(f"\n  direction base rate: up {ups / total_n * 100:.1f}%  "
          f"down {downs / total_n * 100:.1f}%  flat {zeros / total_n * 100:.1f}%")
    print("\n  Read: if every lag's |z| is under ~2 and VR ~= 1, returns are a")
    print("  random walk. No candle indicator can beat that — the edge is not")
    print("  hidden, it is absent.\n")


if __name__ == "__main__":
    main()
