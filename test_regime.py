#!/usr/bin/env python3
"""Tests for regime.py: trend/vol/variance-ratio/hurst classification."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import regime
from regime import (efficiency_ratio, variance_ratio, hurst, classify,
                    trend_following_ok, mean_reversion_ok, STRONG_UP, STRONG_DOWN,
                    SIDEWAYS)


def _arr(cs):
    return ([c[1] for c in cs], [c[2] for c in cs],
            [c[3] for c in cs], [c[4] for c in cs], [c[0] for c in cs])


def _rw(n=300, seed=0, vol=0.0004, start=1.1):
    rng = random.Random(seed)
    cs = []
    price = start
    for i in range(n):
        o = price
        price += rng.gauss(0, vol)
        c = price
        cs.append([i * 60, o, c, max(o, c) + abs(rng.gauss(0, vol / 2)),
                   min(o, c) - abs(rng.gauss(0, vol / 2))])
    return cs


def _trend(n=300, seed=0, slope=0.0008, vol=0.0001):
    rng = random.Random(seed)
    cs = []
    price = 1.0
    for i in range(n):
        o = price
        price += slope + rng.gauss(0, vol)
        c = price
        cs.append([i * 60, o, c, max(o, c) + 1e-5, min(o, c) - 1e-5])
    return cs


def test_efficiency_ratio_trend_vs_chop():
    t = _trend()
    o, c, h, l, tt = _arr(t)
    er_trend = efficiency_ratio(c, len(c) - 1, 20)
    r = _rw(seed=2)
    o2, c2, h2, l2, t2 = _arr(r)
    er_chop = efficiency_ratio(c2, len(c2) - 1, 20)
    assert er_trend > er_chop, (er_trend, er_chop)
    assert 0.0 <= er_chop <= 1.0 and 0.0 <= er_trend <= 1.0
    print("PASS test_efficiency_ratio_trend_vs_chop")


def test_variance_ratio():
    r = _rw(seed=3)
    o, c, h, l, t = _arr(r)
    vr = variance_ratio(c, len(c) - 1, 120)
    assert 0.0 < vr < 3.0, vr    # a random walk sits near 1
    print("PASS test_variance_ratio")


def test_hurst_bounds():
    r = _rw(seed=4)
    o, c, h, l, t = _arr(r)
    hu = hurst(c, len(c) - 1, 100)
    assert 0.0 <= hu <= 1.5, hu
    print("PASS test_hurst_bounds")


def test_classify_trend_and_range():
    t = _trend(slope=0.001)
    o, c, h, l, tt = _arr(t)
    reg = classify(o, c, h, l, tt, len(c) - 1)
    assert reg.trend in (STRONG_UP, "weak_up"), reg.as_dict()
    assert trend_following_ok(reg), reg.as_dict()
    r = _rw(seed=5, vol=0.0004)
    o2, c2, h2, l2, t2 = _arr(r)
    reg2 = classify(o2, c2, h2, l2, t2, len(c2) - 1)
    assert reg2.trend in (SIDEWAYS, "weak_up", "weak_down"), reg2.as_dict()
    print("PASS test_classify_trend_and_range")


def test_classify_short_series_undefined():
    cs = _rw(n=40)
    o, c, h, l, t = _arr(cs)
    reg = classify(o, c, h, l, t, len(cs) - 1)
    assert reg.defined is False
    print("PASS test_classify_short_series_undefined")


def test_regime_causal():
    cs = _rw(n=300, seed=6)
    o, c, h, l, t = _arr(cs)
    i = 200
    a = classify(o[:i + 1], c[:i + 1], h[:i + 1], l[:i + 1], t[:i + 1], i).as_dict()
    b = classify(o, c, h, l, t, i).as_dict()
    assert a == b, (a, b)
    print("PASS test_regime_causal")


if __name__ == "__main__":
    test_efficiency_ratio_trend_vs_chop()
    test_variance_ratio()
    test_hurst_bounds()
    test_classify_trend_and_range()
    test_classify_short_series_undefined()
    test_regime_causal()
    print("\nall regime tests passed")
