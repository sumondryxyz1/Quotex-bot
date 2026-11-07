#!/usr/bin/env python3
"""Tests for the tick store and tick order-flow features."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from candle_store import CandleStore
import tickflow


def test_tick_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = CandleStore(os.path.join(d, "t.db"))
        s.save_ticks([("X_otc", 100.0, 1.10, 1), ("X_otc", 101.0, 1.11, 0),
                      ("Y_otc", 100.0, 2.0, 1)])
        assert s.tick_count() == 3
        assert s.tick_count("X_otc") == 2
        t = s.load_ticks("X_otc")
        assert t == [(100.0, 1.10, 1), (101.0, 1.11, 0)], t
        assert s.tick_assets() == ["X_otc", "Y_otc"]
    print("PASS test_tick_store_roundtrip")


def test_candle_features():
    t0 = 1000.0
    ticks = [(t0 + k, 1.10 + k * 0.0001, 1) for k in range(8)]
    f = tickflow.candle_features(ticks, t0, 60)
    assert f is not None
    assert f["n_ticks"] == 8
    assert abs(f["up_ratio"] - 1.0) < 1e-9          # every tick moved up
    assert abs(f["dir_ratio"] - 1.0) < 1e-9
    assert f["close_pos"] > 0.99                    # closed at the high
    print("PASS test_candle_features")


def test_features_ignore_other_windows():
    t0 = 1000.0
    inwin = [(t0 + k, 1.10 + k * 0.0001, 1) for k in range(6)]
    outwin = [(t0 + 60 + k, 9.99, 1) for k in range(10)]
    f = tickflow.candle_features(inwin + outwin, t0, 60)
    assert f["n_ticks"] == 6, f        # only the in-window ticks
    print("PASS test_features_ignore_other_windows")


def test_flat_candle_returns_none():
    t0 = 1000.0
    flat = [(t0 + k, 1.10, 0) for k in range(6)]
    assert tickflow.candle_features(flat, t0, 60) is None
    print("PASS test_flat_candle_returns_none")


if __name__ == "__main__":
    test_tick_store_roundtrip()
    test_candle_features()
    test_features_ignore_other_windows()
    test_flat_candle_returns_none()
    print("\nall tickflow tests passed")
