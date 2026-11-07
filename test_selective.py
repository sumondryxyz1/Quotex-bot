#!/usr/bin/env python3
"""Tests for the selectivity sweep helpers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selective import atr_series, mtf_trend, regime_up, wilson


def _candles(closes, start=1000, step=5):
    return [[start + i * step, c, c, c + 0.001, c - 0.001]
            for i, c in enumerate(closes)]


def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert wilson(0, 0) == (0.0, 0.0)
    print("PASS test_wilson_bounds")


def test_regime_up_on_uptrend():
    closes = [1.0 + i * 0.001 for i in range(80)]
    cs = _candles(closes)
    assert regime_up(cs, len(cs) - 1) is True
    down = [1.0 - i * 0.001 for i in range(80)]
    assert regime_up(_candles(down), 79) is False
    print("PASS test_regime_up_on_uptrend")


def test_mtf_trend_uses_closed_candle():
    # hourly candles, each 3600s apart, rising
    htf = [[3600 * i, 1.0 + i * 0.01, 1.0 + i * 0.01, 1.0 + i * 0.01, 1.0]
           for i in range(60)]
    # at t inside the last candle's window, trend should be up
    assert mtf_trend(htf, 3600 * 59 + 10) == 1
    # before enough history -> unknown
    assert mtf_trend(htf, 3600 * 5) == 0
    falling = [[3600 * i, 2.0 - i * 0.01, 2.0 - i * 0.01, 2.0, 2.0 - i * 0.01]
               for i in range(60)]
    assert mtf_trend(falling, 3600 * 59 + 10) == -1
    print("PASS test_mtf_trend_uses_closed_candle")


def test_atr_series_shape():
    closes = [1.0 + (i % 5) * 0.001 for i in range(80)]
    a = atr_series(_candles(closes), 14)
    assert len(a) == 80
    assert a[-1] is not None and a[-1] > 0
    print("PASS test_atr_series_shape")


if __name__ == "__main__":
    test_wilson_bounds()
    test_regime_up_on_uptrend()
    test_mtf_trend_uses_closed_candle()
    test_atr_series_shape()
    print("\nall selective tests passed")
