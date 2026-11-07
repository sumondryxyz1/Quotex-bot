#!/usr/bin/env python3
"""Tests for P0 payout: breakeven, EV, profit factor, config precedence."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from payout import (DEFAULT_PAYOUT, breakeven, expected_value, profit_factor,
                    PayoutConfig)


def test_breakeven_values():
    # known values from the plan / audit
    assert abs(breakeven(0.85) - 0.5405) < 1e-4
    assert abs(breakeven(0.90) - 0.5263) < 1e-4
    assert abs(breakeven(0.95) - 0.5128) < 1e-4
    assert abs(breakeven(0.70) - 0.5882) < 1e-4
    print("PASS test_breakeven_values")


def test_expected_value_zero_at_breakeven():
    for p in (0.70, 0.85, 0.95):
        be = breakeven(p)
        assert abs(expected_value(be, p)) < 1e-9, (p, expected_value(be, p))
    # above breakeven -> positive EV; below -> negative
    assert expected_value(0.56, 0.85) > 0
    assert expected_value(0.50, 0.85) < 0
    print("PASS test_expected_value_zero_at_breakeven")


def test_profit_factor():
    # 3 wins at 0.85, 1 loss -> 2.55
    assert abs(profit_factor(3, 1, 0.85) - 2.55) < 1e-9
    assert profit_factor(5, 0, 0.85) == float("inf")
    assert profit_factor(0, 5, 0.85) == 0.0
    print("PASS test_profit_factor")


def test_config_precedence():
    cfg = PayoutConfig(default=0.85, pairs={"EURUSD_otc": 0.92})
    assert cfg.get("EURUSD_otc") == 0.92       # per-pair wins
    assert cfg.get("GBPUSD_otc") == 0.85       # falls back to default
    assert cfg.get("") == 0.85
    print("PASS test_config_precedence")


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "payout.json")
        PayoutConfig(default=0.90, pairs={"BTCUSD_otc": 0.80}).save(path)
        cfg = PayoutConfig.load(path)
        assert cfg.default == 0.90
        assert cfg.get("BTCUSD_otc") == 0.80
        # missing file -> safe default
        cfg2 = PayoutConfig.load(os.path.join(d, "nope.json"))
        assert cfg2.default == DEFAULT_PAYOUT
    print("PASS test_config_roundtrip")


if __name__ == "__main__":
    test_breakeven_values()
    test_expected_value_zero_at_breakeven()
    test_profit_factor()
    test_config_precedence()
    test_config_roundtrip()
    print("\nall payout tests passed")
