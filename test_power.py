#!/usr/bin/env python3
"""Tests for P1 statistical power: sample-size and detectable-effect math."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power import required_n, min_detectable_hit, _inv_norm, z_for, zp_for
from payout import breakeven


def test_inverse_normal_known_values():
    assert abs(_inv_norm(0.975) - 1.959964) < 1e-4
    assert abs(_inv_norm(0.80) - 0.841621) < 1e-4
    assert abs(_inv_norm(0.5)) < 1e-9
    print("PASS test_inverse_normal_known_values")


def test_z_helpers():
    assert abs(z_for(0.05) - 1.96) < 1e-9
    assert abs(zp_for(0.80) - 0.8416) < 1e-3
    print("PASS test_z_helpers")


def test_required_n_monotonic_and_positive():
    # a bigger edge needs fewer samples
    n_56 = required_n(0.56, 0.85)
    n_60 = required_n(0.60, 0.85)
    assert n_56 > n_60 > 0, (n_56, n_60)
    # an edge below breakeven is not detectable at any n
    assert required_n(0.53, 0.85) < 0
    print("PASS test_required_n_monotonic_and_positive")


def test_required_n_matches_formula_order_of_magnitude():
    # ~0.9pp edge (0.55 vs 0.5405) needs tens of thousands of samples
    n = required_n(0.55, 0.85)
    assert 15000 < n < 30000, n
    print("PASS test_required_n_matches_formula_order_of_magnitude")


def test_min_detectable_hit():
    # with more data you can detect a smaller edge
    mdh_small = min_detectable_hit(200, 0.85)
    mdh_big = min_detectable_hit(5000, 0.85)
    assert mdh_small > mdh_big > breakeven(0.85)
    print("PASS test_min_detectable_hit")


if __name__ == "__main__":
    test_inverse_normal_known_values()
    test_z_helpers()
    test_required_n_monotonic_and_positive()
    test_required_n_matches_formula_order_of_magnitude()
    test_min_detectable_hit()
    print("\nall power tests passed")
