#!/usr/bin/env python3
"""Generate synthetic candle CSVs to validate the training pipeline.

Two regimes:

  random   : pure random walk -> an honest model should show ~0 edge.
  momentum : trending/mean-reverting walk with real autocorrelation -> a model
             should be able to beat the baseline, proving the pipeline learns.

Usage
-----
    python make_synthetic.py
"""
from __future__ import annotations

import csv
import os
import random

random.seed(7)


def write(path, candles):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for c in candles:
            w.writerow(c)
    print(f"wrote {len(candles)} candles -> {path}")


def random_walk(n=3000, start=1.10, vol=0.0004):
    candles = []
    t = 1_700_000_000
    price = start
    for i in range(n):
        o = price
        price += random.gauss(0, vol)
        c = price
        hi = max(o, c) + abs(random.gauss(0, vol / 2))
        lo = min(o, c) - abs(random.gauss(0, vol / 2))
        candles.append([t + i * 60, round(o, 6), round(c, 6), round(hi, 6), round(lo, 6)])
    return candles


def momentum_walk(n=3000, start=1.10, vol=0.0004):
    """Price has momentum: last move partially predicts the next move."""
    candles = []
    t = 1_700_000_000
    price = start
    drift = 0.0
    for i in range(n):
        o = price
        drift = 0.7 * drift + random.gauss(0, vol)
        price += drift
        c = price
        hi = max(o, c) + abs(random.gauss(0, vol / 2))
        lo = min(o, c) - abs(random.gauss(0, vol / 2))
        candles.append([t + i * 60, round(o, 6), round(c, 6), round(hi, 6), round(lo, 6)])
    return candles


if __name__ == "__main__":
    write("data/synthetic_random.csv", random_walk())
    write("data/synthetic_momentum.csv", momentum_walk())
