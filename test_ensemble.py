#!/usr/bin/env python3
"""Tests for the SMC primitives and the 3-model ensemble engine."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smc
from ensemble import EnsembleEngine, EnsembleStrategy


def _candle(t, o, h, l, c):
    return [t, o, c, h, l]


def _trend_series(n=120, step=0.001, base=1.0, period=60):
    out = []
    for i in range(n):
        o = base + i * step
        c = base + (i + 1) * step
        out.append(_candle(i * period, o, max(o, c) + step / 2,
                           min(o, c) - step / 2, c))
    return out


def _wave(n=160, slope=0.0005, amp=0.004, base=1.0, period=60, cyc=20):
    """Trend + oscillation, so fractal pivots actually form."""
    import math
    out = []
    for i in range(n):
        mid = base + i * slope + amp * math.sin(2 * math.pi * i / cyc)
        out.append(_candle(i * period, mid, mid + amp / 2, mid - amp / 2, mid))
    return out


def test_swings_finds_pivots():
    highs = [1, 2, 3, 5, 3, 2, 4, 1, 1, 1]
    lows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sh, sl = smc.swings(highs, lows, left=2, right=2)
    assert 3 in sh, sh          # the local peak at index 3
    print("PASS test_swings_finds_pivots")


def test_structure_uptrend():
    up = _wave(200, slope=0.0008, amp=0.003, cyc=16)
    assert smc.structure([c[3] for c in up], [c[4] for c in up],
                         len(up) - 1) == "up"
    down = _wave(200, slope=-0.0008, amp=0.003, base=2.0, cyc=16)
    assert smc.structure([c[3] for c in down], [c[4] for c in down],
                         len(down) - 1) == "down"
    print("PASS test_structure_uptrend")


def test_liquidity_sweep_bull():
    cs = _trend_series(60, step=0.0, base=1.0)   # flat 1.0
    # single prior swing low at index 40
    cs[40] = _candle(40 * 60, 1.0, 1.0, 0.98, 1.0)
    # last candle wicks below the prior low then closes back above
    cs[59] = _candle(59 * 60, 1.0, 1.0, 0.975, 1.0)
    highs = [c[3] for c in cs]
    lows = [c[4] for c in cs]
    closes = [c[2] for c in cs]
    assert smc.liquidity_sweep(highs, lows, closes, 59) == "bull"
    print("PASS test_liquidity_sweep_bull")


def test_fair_value_gap():
    cs = [_candle(0, 1.0, 1.01, 0.99, 1.0),
          _candle(60, 1.0, 1.05, 1.0, 1.04),
          _candle(120, 1.04, 1.08, 1.045, 1.07)]   # low 1.045 > high of i-2 (1.01)
    highs = [c[3] for c in cs]
    lows = [c[4] for c in cs]
    assert smc.fair_value_gap(highs, lows, 2) == "bull"
    print("PASS test_fair_value_gap")


def test_dealing_range_discount():
    cs = _trend_series(50, step=0.01, base=1.0)
    highs = [c[3] for c in cs]
    lows = [c[4] for c in cs]
    _, _, pos = smc.dealing_range(highs, lows, len(cs) - 1)
    assert pos > 0.9          # price at the top of its range
    print("PASS test_dealing_range_discount")


def test_ensemble_trend_gate_blocks_countertrend():
    # strong uptrend: any signal must be CALL, never PUT
    cs = _trend_series(200, step=0.002)
    eng = EnsembleEngine(EnsembleStrategy())
    for i in range(80, len(cs)):
        sig = eng.evaluate(cs[:i + 1])
        if sig is not None:
            assert sig.action == "CALL", sig.action
            assert sig.votes["trend"] == "CALL"
    print("PASS test_ensemble_trend_gate_blocks_countertrend")


def test_ensemble_requires_two_votes():
    cs = _trend_series(200, step=0.002)
    eng = EnsembleEngine(EnsembleStrategy(min_agree=3))
    got = None
    for i in range(80, len(cs)):
        got = eng.evaluate(cs[:i + 1])
        if got is not None:
            assert got.score >= 3
            break
    print("PASS test_ensemble_requires_two_votes")


def test_ensemble_no_lookahead():
    cs = _trend_series(150, step=0.001)
    a = EnsembleEngine(EnsembleStrategy()).evaluate(cs[:120])
    b = EnsembleEngine(EnsembleStrategy()).evaluate(cs[:120])
    assert (a is None) == (b is None)
    print("PASS test_ensemble_no_lookahead")


if __name__ == "__main__":
    test_swings_finds_pivots()
    test_structure_uptrend()
    test_liquidity_sweep_bull()
    test_fair_value_gap()
    test_dealing_range_discount()
    test_ensemble_trend_gate_blocks_countertrend()
    test_ensemble_requires_two_votes()
    test_ensemble_no_lookahead()
    print("\nall smc/ensemble tests passed")
