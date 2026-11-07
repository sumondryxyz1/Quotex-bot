#!/usr/bin/env python3
"""Tests for liquidity.py: pools, sweeps, displacement, session levels."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import liquidity
from liquidity import (equal_levels, liquidity_pools, displacement,
                       sweep_and_reclaim, session_levels, BULL, BEAR)


def _rw(n=200, seed=0, vol=0.0006, start=1.1):
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


def _arr(cs):
    return ([c[1] for c in cs], [c[2] for c in cs],
            [c[3] for c in cs], [c[4] for c in cs], [c[0] for c in cs])


def test_equal_levels_detects_double_top():
    # two highs at almost the same price -> an equal-high level
    cs = _rw(120)
    o, c, h, l, t = _arr(cs)
    # carve a clean double top: two peaks at ~1.30 with a dip between
    for k, v in {40: 1.298, 41: 1.300, 42: 1.297, 80: 1.297, 81: 1.3005,
                 82: 1.298}.items():
        h[k] = v
        l[k] = v - 0.004
        c[k] = v - 0.001
    eq_h, eq_l = equal_levels(h, l, len(cs) - 1, tol_atr=2.0)
    assert any(abs(x - 1.30) < 0.01 for x in eq_h), eq_h
    print("PASS test_equal_levels_detects_double_top")


def test_liquidity_pools_ranked_by_touches():
    cs = _rw(200, seed=3)
    o, c, h, l, t = _arr(cs)
    for idx in (50, 100, 150):
        h[idx] = 1.40
    pools = liquidity_pools(h, l, len(cs) - 1)
    assert pools, "expected at least one pool"
    assert pools[0].touches >= 2, pools[0]
    print("PASS test_liquidity_pools_ranked_by_touches")


def test_displacement_positive_and_negative():
    cs = _rw(100, seed=1)
    o, c, h, l, t = _arr(cs)
    # inject a big bullish body
    o.append(1.0); c.append(1.02); h.append(1.021); l.append(0.999)
    d = displacement(o, c, h, l, len(c) - 1, atr_mult=0.5)
    assert d["displaced"] is True and d["direction"] == BULL, d
    # a small candle is not displacement
    o.append(1.0); c.append(1.0001); h.append(1.0002); l.append(0.9999)
    d2 = displacement(o, c, h, l, len(c) - 1, atr_mult=2.0)
    assert d2["displaced"] is False, d2
    print("PASS test_displacement_positive_and_negative")


def test_sweep_and_reclaim_bullish():
    cs = _rw(120, seed=5)
    o, c, h, l, t = _arr(cs)
    prior_low = min(l[-40:])
    # candle wicks below the low but closes back above it
    o.append(l[-1]); c.append(prior_low + 0.002)
    h.append(max(o[-1], c[-1]) + 1e-4); l.append(prior_low - 0.002)
    sw = sweep_and_reclaim(o, c, h, l, len(c) - 1, lookback=40)
    assert sw is not None and sw.direction == BULL, sw
    assert sw.penetration > 0 and sw.reclaim > 0, sw
    print("PASS test_sweep_and_reclaim_bullish")


def test_sweep_none_when_no_pierce():
    cs = _rw(120, seed=6)
    o, c, h, l, t = _arr(cs)
    o.append(1.0); c.append(1.0001); h.append(1.0002); l.append(0.9999)
    sw = sweep_and_reclaim(o, c, h, l, len(c) - 1)
    assert sw is None, sw
    print("PASS test_sweep_none_when_no_pierce")


def test_session_levels():
    # a full day of candles, one per minute
    t0 = 1_700_000_000 - (1_700_000_000 % 86400)
    cs = []
    for m in range(0, 24 * 60, 5):
        price = 1.1 + m * 1e-5
        cs.append([t0 + m * 60, price, price, price + 1e-4, price - 1e-4])
    o, c, h, l, t = _arr(cs)
    sess = session_levels(t, h, l, len(cs) - 1)
    assert sess["asian"] is not None, sess
    assert sess["asian"][1] >= sess["asian"][0]
    print("PASS test_session_levels")


if __name__ == "__main__":
    test_equal_levels_detects_double_top()
    test_liquidity_pools_ranked_by_touches()
    test_displacement_positive_and_negative()
    test_sweep_and_reclaim_bullish()
    test_sweep_none_when_no_pierce()
    test_session_levels()
    print("\nall liquidity tests passed")
