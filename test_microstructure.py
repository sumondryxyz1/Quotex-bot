#!/usr/bin/env python3
"""Tests for microstructure.py: velocity, imbalance, bursts, autocorrelation."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import microstructure as ms


def test_velocity_sign():
    up = [1.0, 1.001, 1.002, 1.003]
    down = [1.0, 0.999, 0.998, 0.997]
    assert ms.velocity(up) > 0
    assert ms.velocity(down) < 0
    assert ms.velocity([1.0]) == 0.0
    print("PASS test_velocity_sign")


def test_tick_imbalance():
    assert ms.tick_imbalance([1.0, 1.1, 1.2, 1.3]) == 1.0
    assert ms.tick_imbalance([1.3, 1.2, 1.1, 1.0]) == -1.0
    assert ms.tick_imbalance([1.0, 1.1, 1.0]) == 0.0
    assert abs(ms.tick_imbalance([1.0, 1.1, 1.0, 1.1]) - 0.3333) < 1e-3
    print("PASS test_tick_imbalance")


def test_burst_detection():
    counts = [5] * 20 + [50]
    assert ms.burst_detection(counts) is True
    counts2 = [5] * 20 + [6]
    assert ms.burst_detection(counts2) is False
    print("PASS test_burst_detection")


def test_tick_autocorr_bounds():
    rng = random.Random(0)
    prices = [1.0]
    for _ in range(500):
        prices.append(prices[-1] + rng.gauss(0, 1e-4))
    ac = ms.tick_autocorr(prices, 1)
    assert -1.0 <= ac <= 1.0, ac
    # a noisy trending tape has positive lag-1 autocorrelation
    rng2 = random.Random(2)
    drift = 0.0
    trend = [1.0]
    for _ in range(500):
        drift = 0.9 * drift + rng2.gauss(0, 1e-4)
        trend.append(trend[-1] + drift)
    assert ms.tick_autocorr(trend, 1) > 0.3, ms.tick_autocorr(trend, 1)
    print("PASS test_tick_autocorr_bounds")


def test_analyse_shape():
    rng = random.Random(1)
    # 60 one-minute candles
    cs = []
    price = 1.1
    for i in range(60):
        o = price
        price += rng.gauss(0, 0.0005)
        cs.append([i * 60, o, price, max(o, price) + 1e-4, min(o, price) - 1e-4])
    o = [c[1] for c in cs]; c = [c[2] for c in cs]
    h = [c[3] for c in cs]; l = [c[4] for c in cs]
    # ticks for the last candle window
    ticks = [(3540 + k, 1.1 + rng.gauss(0, 1e-4), 1 if rng.random() > 0.5 else -1)
             for k in range(20)]
    st = ms.analyse(ticks, o, c, h, l, 3540, 60, 59, counts_history=[10] * 20)
    assert st.tick_count == 20, st.as_dict()
    assert -1.0 <= st.imbalance <= 1.0
    assert 0.0 <= st.up_ratio <= 1.0
    print("PASS test_analyse_shape")


if __name__ == "__main__":
    test_velocity_sign()
    test_tick_imbalance()
    test_burst_detection()
    test_tick_autocorr_bounds()
    test_analyse_shape()
    print("\nall microstructure tests passed")
