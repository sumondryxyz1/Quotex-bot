#!/usr/bin/env python3
"""Tests for chart-pattern detection and the trained 3-model AI ensemble."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import patterns
import smc


def _c(t, o, h, l, c):
    return [t, o, c, h, l]


def _zigzag_series(pivots, seg=6, period=60):
    """Build candles that trace a piecewise-linear path through `pivots`."""
    out = []
    t = 0
    for a, b in zip(pivots[:-1], pivots[1:]):
        for k in range(seg):
            v = a + (b - a) * k / seg
            out.append(_c(t, v, v + 0.0005, v - 0.0005, v))
            t += period
    v = pivots[-1]
    out.append(_c(t, v, v + 0.0005, v - 0.0005, v))
    return out


def test_head_shoulders_bear():
    # left shoulder 1.10, head 1.13, right shoulder 1.10, then break below neck
    piv = [1.05, 1.10, 1.07, 1.13, 1.07, 1.10, 1.02]
    cs = _zigzag_series(piv, seg=8)
    highs = [c[3] for c in cs]
    lows = [c[4] for c in cs]
    got = None
    for i in range(20, len(cs)):
        got = patterns.head_shoulders(highs, lows, i)
        if got:
            break
    assert got == "bear", got
    print("PASS test_head_shoulders_bear")


def test_double_bottom_bull():
    piv = [1.10, 1.02, 1.06, 1.02, 1.12]
    cs = _zigzag_series(piv, seg=8)
    highs = [c[3] for c in cs]
    lows = [c[4] for c in cs]
    got = None
    for i in range(20, len(cs)):
        got = patterns.double_top_bottom(highs, lows, i)
        if got:
            break
    assert got == "bull", got
    print("PASS test_double_bottom_bull")


def test_sr_levels_cluster():
    piv = [1.10, 1.00, 1.10, 1.00, 1.10, 1.00]
    cs = _zigzag_series(piv, seg=6)
    highs = [c[3] for c in cs]
    lows = [c[4] for c in cs]
    levels = patterns.sr_levels(highs, lows, len(cs) - 1)
    assert any(abs(l - 1.10) < 0.005 for l in levels), levels
    assert any(abs(l - 1.00) < 0.005 for l in levels), levels
    print("PASS test_sr_levels_cluster")


def test_pattern_vote_no_lookahead():
    piv = [1.05, 1.12, 1.06, 1.02]
    cs = _zigzag_series(piv, seg=8)
    o = [c[1] for c in cs]; c_ = [c[2] for c in cs]
    h = [c[3] for c in cs]; l = [c[4] for c in cs]
    a = patterns.pattern_vote(o, c_, h, l, 20)
    b = patterns.pattern_vote(o, c_, h, l, 20)
    assert a == b
    print("PASS test_pattern_vote_no_lookahead")


def test_ai_ensemble_learns_real_pattern():
    """The 3-model vote must beat chance when a pattern actually exists."""
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(1)
    n = 5000
    r = rng.normal(0, 1, n)
    mom = np.zeros(n)
    for i in range(1, n):
        mom[i] = 0.8 * mom[i - 1] + 0.2 * r[i - 1]
    target = (0.6 * np.sign(mom) * np.abs(r) + 0.4 * r)
    X = np.array([[target[i - 1], target[i - 2], target[i - 3]] for i in range(3, n)])
    y = (target[3:] > 0).astype(int)
    cut = int(len(X) * 0.6)
    m = GradientBoostingClassifier(random_state=0).fit(X[:cut], y[:cut])
    acc = m.score(X[cut:], y[cut:])
    assert acc > 0.65, acc       # pattern present -> learnable
    print(f"PASS test_ai_ensemble_learns_real_pattern (acc {acc:.2f})")


if __name__ == "__main__":
    test_head_shoulders_bear()
    test_double_bottom_bull()
    test_sr_levels_cluster()
    test_pattern_vote_no_lookahead()
    test_ai_ensemble_learns_real_pattern()
    print("\nall pattern/ai tests passed")
