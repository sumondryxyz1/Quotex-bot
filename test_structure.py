#!/usr/bin/env python3
"""Tests for structure.py: swings, trend, BOS, CHoCH, MSS (logic, not edge)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import structure
from structure import (swing_series, trend_from_swings, structure_state,
                       internal_external, UP, DOWN)


def _candles_from_prices(prices, step=60, t0=1_700_000_000, seed=0):
    rng = random.Random(seed + 1)
    out = []
    prev = prices[0]
    for k, p in enumerate(prices):
        o = prev
        hi = max(o, p) + abs(rng.gauss(0, 2e-5))
        lo = min(o, p) - abs(rng.gauss(0, 2e-5))
        out.append([t0 + k * step, o, p, hi, lo])
        prev = p
    return out


def _zigzag(waypoints, n_per=12, noise=0.00004, seed=0):
    rng = random.Random(seed)
    prices = []
    for a, b in zip(waypoints, waypoints[1:]):
        for k in range(n_per):
            prices.append(a + (b - a) * k / n_per + rng.gauss(0, noise))
    prices.append(waypoints[-1])
    return _candles_from_prices(prices)


def _arrays(cs):
    return ([c[1] for c in cs], [c[2] for c in cs],
            [c[3] for c in cs], [c[4] for c in cs], [c[0] for c in cs])


def test_swing_series_finds_injected_pivots():
    cs = _zigzag([1.00, 1.10, 1.05, 1.15, 1.08, 1.18], n_per=10)
    o, c, h, l, t = _arrays(cs)
    sw = swing_series(h, l, len(cs) - 1)
    kinds = {s.kind for s in sw}
    assert "high" in kinds and "low" in kinds, sw
    labels = {s.label for s in sw}
    assert labels & {"HH", "HL", "LH", "LL"}, labels
    print("PASS test_swing_series_finds_injected_pivots")


def test_trend_from_swings_uptrend_and_downtrend():
    up = _zigzag([1.00, 1.05, 1.02, 1.08, 1.05, 1.12])
    o, c, h, l, t = _arrays(up)
    sw = swing_series(h, l, len(up) - 1)
    assert trend_from_swings(sw) == UP, [s.label for s in sw]
    down = _zigzag([1.20, 1.15, 1.18, 1.12, 1.15, 1.08])
    o, c, h, l, t = _arrays(down)
    sw = swing_series(h, l, len(down) - 1)
    assert trend_from_swings(sw) == DOWN, [s.label for s in sw]
    print("PASS test_trend_from_swings_uptrend_and_downtrend")


def test_choch_on_trend_flip():
    # up, then a decisive break below the last higher-low -> CHoCH down
    cs = _zigzag([1.00, 1.12, 1.06, 1.16, 0.98], n_per=15)
    o, c, h, l, t = _arrays(cs)
    st = structure_state(o, c, h, l, len(cs) - 1)
    assert st.last_choch is not None, st.as_dict()
    assert st.last_choch.direction == DOWN, st.as_dict()
    print("PASS test_choch_on_trend_flip")


def test_bos_within_trend():
    # clean uptrend that keeps making higher highs -> BOS up, no CHoCH
    cs = _zigzag([1.00, 1.06, 1.03, 1.10, 1.07, 1.16], n_per=15)
    o, c, h, l, t = _arrays(cs)
    st = structure_state(o, c, h, l, len(cs) - 1)
    assert st.last_bos is not None and st.last_bos.direction == UP, st.as_dict()
    print("PASS test_bos_within_trend")


def test_mss_requires_displacement():
    # an uptrend (2 highs, 2 lows), then a violent break below the last low
    cs = _zigzag([1.00, 1.06, 1.03, 1.10, 1.07], n_per=12)
    o, c, h, l, t = _arrays(cs)
    last = cs[-1]
    # a displacement candle: opens at the last close, closes far below the last low
    o.append(last[2]); c.append(1.00); h.append(last[2]); l.append(0.99)
    cs.append([last[0] + 60, last[2], 1.00, last[2], 0.99])
    st = structure_state(o, c, h, l, len(c) - 1, disp_atr=0.3)
    assert st.last_choch is not None, st.as_dict()
    assert st.last_choch.direction == DOWN, st.as_dict()
    assert st.last_choch.kind in ("MSS", "CHoCH"), st.as_dict()
    print("PASS test_mss_requires_displacement")


def test_internal_external():
    cs = _zigzag([1.00, 1.10, 1.02, 1.12, 1.04, 1.14], n_per=20)
    o, c, h, l, t = _arrays(cs)
    ie = internal_external(h, l, len(cs) - 1)
    assert set(ie) == {"internal", "external", "aligned"}, ie
    assert ie["external"] in (UP, DOWN, "range")
    print("PASS test_internal_external")


def test_structure_is_causal():
    cs = _zigzag([1.00, 1.10, 1.05, 1.15], n_per=15)
    o, c, h, l, t = _arrays(cs)
    i = len(cs) - 10
    a = structure_state(o[:i + 1], c[:i + 1], h[:i + 1], l[:i + 1], i).as_dict()
    # recompute with extra (future) candles available but same index
    b = structure_state(o, c, h, l, i).as_dict()
    assert a == b, (a, b)
    print("PASS test_structure_is_causal")


if __name__ == "__main__":
    test_swing_series_finds_injected_pivots()
    test_trend_from_swings_uptrend_and_downtrend()
    test_choch_on_trend_flip()
    test_bos_within_trend()
    test_mss_requires_displacement()
    test_internal_external()
    test_structure_is_causal()
    print("\nall structure tests passed")
