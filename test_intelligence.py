#!/usr/bin/env python3
"""Tests for intelligence.py, including the random-walk negative control.

The negative control is the point: the market-structure / liquidity / SMC read
must NOT beat a coin flip on a random walk. If it does, the read is leaking or
overfit and must not be trusted on real data.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import intelligence
from validation import metrics_from_outcomes, wilson
from payout import breakeven


def _random_walk(n=800, seed=0, vol=0.0006, start=1.1):
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


def test_analyse_shape():
    cs = _random_walk(400, seed=1)
    intel = intelligence.analyse(cs, len(cs) - 1, asset="X_otc", period=60)
    assert intel is not None
    d = intel.as_dict()
    assert set(d) >= {"structure", "bos", "choch", "sweep", "regime", "votes"}, d
    assert isinstance(d["regime"], dict)
    print("PASS test_analyse_shape")


def test_analyse_none_when_short():
    cs = _random_walk(30, seed=2)
    assert intelligence.analyse(cs, len(cs) - 1) is None
    print("PASS test_analyse_none_when_short")


def test_analyse_is_causal():
    cs = _random_walk(400, seed=3)
    i = 300
    a = intelligence.analyse(cs[:i + 1], i).as_dict()
    b = intelligence.analyse(cs, i).as_dict()
    assert a == b, (a, b)
    print("PASS test_analyse_is_causal")


def test_negative_control_random_walk_no_edge():
    """Direction from the intelligence read must not beat a coin flip."""
    cs = _random_walk(900, seed=7)
    outcomes = []
    for i in range(120, len(cs) - 1, 3):   # sample every 3rd candle
        intel = intelligence.analyse(cs, i)
        if intel is None or not intel.votes:
            continue
        net = sum((1 if v.direction == "CALL" else -1) * v.score for v in intel.votes)
        if abs(net) < 0.4:
            continue
        action = 1 if net > 0 else 0
        nxt = cs[i + 1][2]
        cur = cs[i][2]
        if nxt == cur:
            continue
        up = nxt > cur
        outcomes.append((action == 1) == up)
    m = metrics_from_outcomes(outcomes, payout=0.85)
    assert m.n > 40, m.n
    assert m.ci_low <= breakeven(0.85), (m.hit, m.ci_low, m.n)
    assert m.verdict != "above", (m.verdict, m.hit, m.n)
    print(f"PASS test_negative_control_random_walk_no_edge "
          f"(hit {m.hit*100:.1f}% CI_low {m.ci_low*100:.1f}% n={m.n})")


if __name__ == "__main__":
    test_analyse_shape()
    test_analyse_none_when_short()
    test_analyse_is_causal()
    test_negative_control_random_walk_no_edge()
    print("\nall intelligence tests passed")
