#!/usr/bin/env python3
"""Tests for the research harness: panel building, scoring, filters."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research import (Panel, Score, build_panel, flat_rate, is_otc,
                      score_rule, wilson, p_value, h_always, h_streak_reversal)


def _mk(candles, asset="T_otc"):
    p = build_panel(asset, 5, candles)
    assert p is not None
    return p


def _trend(n, start=1.0, step=0.001):
    """n candles that rise every step."""
    out = []
    for i in range(n):
        c = start + i * step
        out.append([1000 + i * 5, c - step / 2, c, c + step / 2, c - step])
    return out


def test_is_otc():
    assert is_otc("EURUSD_otc") and not is_otc("EURUSD")
    print("PASS test_is_otc")


def test_build_panel_needs_data():
    assert build_panel("X", 5, [[0, 1, 1, 1, 1]] * 10) is None
    print("PASS test_build_panel_needs_data")


def test_always_call_wins_on_uptrend():
    """A pure uptrend: always-CALL should hit ~100%, always-PUT ~0%."""
    p = _mk(_trend(120))
    sc = score_rule(h_always(1)[1], [p], 0, 119, horizon=1)
    assert sc.n > 0
    assert sc.hit() > 0.95, sc.hit()
    sp = score_rule(h_always(-1)[1], [p], 0, 119, horizon=1)
    assert sp.hit() < 0.05, sp.hit()
    print("PASS test_always_call_wins_on_uptrend")


def test_horizon_uses_close_to_close():
    """With horizon=2 the move is close[i+2]-close[i], not the next candle."""
    # first candle up then two flat-ish; the horizon-2 move should still be up
    candles = _trend(80)
    p = _mk(candles)
    s1 = score_rule(h_always(1)[1], [p], 0, 79, horizon=1)
    s2 = score_rule(h_always(1)[1], [p], 0, 79, horizon=2)
    assert s2.n <= s1.n          # fewer i's have i+horizon available
    assert s2.hit() > 0.95
    print("PASS test_horizon_uses_close_to_close")


def test_flat_rate_dead_pair():
    """A constant price has flat rate 1.0."""
    flat = [[1000 + i * 5, 1.0, 1.0, 1.0, 1.0] for i in range(120)]
    p = _mk(flat)
    assert flat_rate(p, 1) == 1.0
    p2 = _mk(_trend(120))
    assert flat_rate(p2, 1) == 0.0
    print("PASS test_flat_rate_dead_pair")


def test_streak_reversal_fires_on_streak():
    candles = []
    c = 1.0
    for i in range(120):
        c += 0.001                     # 120 up closes in a row
        candles.append([1000 + i * 5, c - 0.001, c, c + 0.001, c - 0.002])
    p = _mk(candles)
    sc = score_rule(h_streak_reversal(3)[1], [p], 0, 119, horizon=1)
    assert sc.n > 0
    # fading a relentless uptrend should lose
    assert sc.hit() < 0.1, sc.hit()
    print("PASS test_streak_reversal_fires_on_streak")


def test_wilson_and_pvalue():
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert p_value(50, 100) > 0.9          # exactly coin-flip
    assert p_value(60, 100) < 0.1          # clearly above
    print("PASS test_wilson_and_pvalue")


if __name__ == "__main__":
    test_is_otc()
    test_build_panel_needs_data()
    test_always_call_wins_on_uptrend()
    test_horizon_uses_close_to_close()
    test_flat_rate_dead_pair()
    test_streak_reversal_fires_on_streak()
    test_wilson_and_pvalue()
    print("\nall research tests passed")
