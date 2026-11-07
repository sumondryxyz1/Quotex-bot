#!/usr/bin/env python3
"""Does the tick-level order flow predict the next candle? Tested honestly.

The candle feed discards two things: the broker's per-tick direction flag and
the exact timing of every price change. `collect_ticks.py` stores them. This
module turns a candle's ticks into order-flow features and asks, with a strict
time split, whether they predict the next candle's direction any better than a
coin flip.

Features per candle (all from ticks inside [t, t+period)):
  n_ticks        tick count (activity)
  up_ratio       fraction of ticks that moved the price up
  dir_ratio      fraction of ticks whose direction flag was +1
  volatility     sum of squared tick returns (realised vol)
  range_ticks    (max - min) / first price
  close_pos      where the close sits in the tick range (0..1)
  big_ticks      fraction of ticks larger than the median tick size
  gap            time from candle open to the first tick

Label = did the *next* candle close higher. Same 60/40 time split as everywhere
else; reports hit-rate, Wilson CI and a binomial p-value per feature bucket and
for a small gradient-boosted model on all features.

Usage
-----
    python tickflow.py --period 60
    python tickflow.py --period 60 --asset EURUSD_otc
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Optional

import numpy as np

from candle_store import CandleStore
from edge import p_value


def wilson(w: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def candle_features(ticks: list[tuple], t0: float, period: int) -> Optional[dict]:
    """Order-flow features from the ticks inside one candle window."""
    win = [t for t in ticks if t0 <= t[0] < t0 + period]
    if len(win) < 4:
        return None
    prices = [t[1] for t in win]
    dirs = [t[2] for t in win]
    first, last = prices[0], prices[-1]
    moves = [prices[k] - prices[k - 1] for k in range(1, len(prices))]
    moves = [m for m in moves if m != 0]
    if not moves:
        return None
    ups = sum(1 for m in moves if m > 0)
    sizes = sorted(abs(m) for m in moves)
    med = sizes[len(sizes) // 2] or 1e-12
    rng = (max(prices) - min(prices)) or 1e-12
    return {
        "n_ticks": len(win),
        "up_ratio": ups / len(moves),
        "dir_ratio": sum(1 for d in dirs if d == 1) / len(dirs),
        "volatility": sum(m * m for m in moves) / (first * first),
        "range_ticks": rng / (first or 1e-12),
        "close_pos": (last - min(prices)) / rng,
        "big_ticks": sum(1 for s in sizes if s > med) / len(sizes),
        "gap": (win[0][0] - t0) / period,
    }


FEATURES = ["n_ticks", "up_ratio", "dir_ratio", "volatility", "range_ticks",
            "close_pos", "big_ticks", "gap"]


def build(store: CandleStore, period: int, assets: Optional[list[str]] = None):
    """(features dict of arrays, labels, time-split index) pooled per asset."""
    if assets is None:
        assets = store.tick_assets()
    Xtr, ytr, Xte, yte = defaultdict(list), [], defaultdict(list), []
    for a in assets:
        ticks = store.load_ticks(a, limit=500000)
        if len(ticks) < 200:
            continue
        candles = store.load(a, period, limit=100000)
        rows = []
        for c in candles:
            f = candle_features(ticks, c[0], period)
            if f is None:
                continue
            rows.append((f, 1 if c[2] > c[1] else (0 if c[2] < c[1] else None)))
        rows = [(f, y) for f, y in rows if y is not None]
        if len(rows) < 60:
            continue
        cut = int(len(rows) * 0.6)
        for f, y in rows[:cut]:
            for k in FEATURES:
                Xtr[k].append(f[k])
            ytr.append(y)
        for f, y in rows[cut:]:
            for k in FEATURES:
                Xte[k].append(f[k])
            yte.append(y)
    return (Xtr, np.array(ytr), Xte, np.array(yte))


def run(store: CandleStore, period: int, assets=None, model: bool = True):
    Xtr, ytr, Xte, yte = build(store, period, assets)
    n = len(yte)
    if n == 0:
        print("not enough tick data yet — run collect_ticks.py longer")
        return
    print(f"\n=== tick order-flow, {period}s, {n:,} test candles "
          f"(train {len(ytr):,}) ===")

    # single-feature buckets: split at the median, hit-rate of the "up" side
    print("\n  single features (median split, hit-rate of predicted side):")
    for k in FEATURES:
        tr = np.array(Xtr[k]); te = np.array(Xte[k])
        med = float(np.median(tr))
        pred = (te > med).astype(int)
        if len(set(pred.tolist())) < 2:
            print(f"    {k:12} (no variance)")
            continue
        # is a high value bullish or bearish? learn the sign on TRAIN only
        hit_up = np.mean(ytr[tr > med]) if np.any(tr > med) else 0.5
        sign = 1 if hit_up >= 0.5 else 0
        p = pred if sign == 1 else 1 - pred
        w = int(np.sum(p == yte))
        lo, hi = wilson(w, n)
        print(f"    {k:12} dir={'up' if sign else 'down':4} hit={w/n*100:5.1f}%  "
              f"CI[{lo*100:4.1f},{hi*100:4.1f}]  p={p_value(w, n):.3f}")

    if model:
        from sklearn.ensemble import GradientBoostingClassifier
        Xtr_m = np.array([Xtr[k] for k in FEATURES]).T
        Xte_m = np.array([Xte[k] for k in FEATURES]).T
        m = GradientBoostingClassifier(random_state=0).fit(Xtr_m, ytr)
        p = m.predict_proba(Xte_m)[:, 1]
        pred = (p >= 0.5).astype(int)
        w = int(np.sum(pred == yte))
        lo, hi = wilson(w, n)
        print(f"\n  GBM on all 8 tick features: hit={w/n*100:5.1f}%  "
              f"CI[{lo*100:4.1f},{hi*100:4.1f}]  p={p_value(w, n):.3f}")
        base = max(yte.mean(), 1 - yte.mean())
        print(f"  majority baseline: {base*100:.1f}%   "
              f"(breakeven at 85% payout: 54.1%)")


def main():
    ap = argparse.ArgumentParser(description="Tick order-flow edge test")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--asset", default="")
    args = ap.parse_args()
    store = CandleStore(args.db)
    assets = [a for a in args.asset.split(",") if a] or None
    run(store, args.period, assets)


if __name__ == "__main__":
    main()
