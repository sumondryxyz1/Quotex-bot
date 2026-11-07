#!/usr/bin/env python3
"""Tests for P1 validation: stats, folds/embargo, metrics, and the two controls.

The two controls are the heart of the harness:
  * a strategy with a KNOWN injected edge must be detected (positive control),
  * a random / random-walk predictor must NOT show an edge (negative control).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation import (wilson, p_value, max_drawdown, metrics_from_outcomes,
                        make_folds, purged_walk_forward, coin_flip_control)
from payout import breakeven


def test_wilson_known_values():
    lo, hi = wilson(540, 1000)
    assert abs(lo - 0.5090) < 1e-3, lo
    assert abs(hi - 0.5707) < 1e-3, hi
    assert wilson(0, 0) == (0.0, 0.0)
    print("PASS test_wilson_known_values")


def test_p_value_known_values():
    assert p_value(500, 1000) > 0.9
    assert p_value(600, 1000) < 0.001
    print("PASS test_p_value_known_values")


def test_max_drawdown():
    # up to 2.0 then down to 1.0 -> 50% dd
    assert abs(max_drawdown([1.0, 2.0, 1.0]) - 0.5) < 1e-9
    assert max_drawdown([1.0, 2.0, 3.0]) == 0.0
    print("PASS test_max_drawdown")


def test_metrics_verdicts():
    # 60% over 1000 -> above breakeven 54.05%
    good = metrics_from_outcomes([True] * 600 + [False] * 400, payout=0.85)
    assert good.verdict == "above", good.verdict
    assert good.ev > 0
    # 45% -> below
    bad = metrics_from_outcomes([True] * 450 + [False] * 550, payout=0.85)
    assert bad.verdict == "below", bad.verdict
    assert bad.ev < 0
    # 51% over 100 -> indistinguishable (CI straddles breakeven)
    mid = metrics_from_outcomes([True] * 51 + [False] * 49, payout=0.85)
    assert mid.verdict == "indistinguishable", (mid.verdict, mid.ci_low, mid.breakeven)
    print("PASS test_metrics_verdicts")


def test_make_folds_respects_embargo():
    folds = make_folds(1000, 5, 0.4, 5)
    assert len(folds) == 5, folds
    for tr0, tr_end, te_start, te_end in folds:
        assert tr0 == 0
        assert te_start == tr_end + 5, (tr_end, te_start)
        assert te_end <= 1000
    # expanding window: later folds train on more
    assert folds[-1][1] > folds[0][1]
    print("PASS test_make_folds_respects_embargo")


def _random_walk_series(n=1500, seed=1, start=1.1, vol=0.0004):
    rng = random.Random(seed)
    candles = []
    price = start
    t = 1_700_000_000
    for i in range(n):
        o = price
        price += rng.gauss(0, vol)
        c = price
        hi = max(o, c) + abs(rng.gauss(0, vol / 2))
        lo = min(o, c) - abs(rng.gauss(0, vol / 2))
        candles.append([t + i * 60, o, c, hi, lo])
    return candles


def _momentum_series(n=1500, seed=2, start=1.1, vol=0.0004):
    """Price with strong positive autocorrelation (a real, learnable pattern)."""
    rng = random.Random(seed)
    candles = []
    price = start
    drift = 0.0
    t = 1_700_000_000
    for i in range(n):
        o = price
        drift = 0.85 * drift + rng.gauss(0, vol)
        price += drift
        c = price
        hi = max(o, c) + abs(rng.gauss(0, vol / 2))
        lo = min(o, c) - abs(rng.gauss(0, vol / 2))
        candles.append([t + i * 60, o, c, hi, lo])
    return candles


def test_negative_control_random_walk_no_edge():
    """A predictor on a random walk must land near 50%, never clear breakeven."""
    series = [("RW", _random_walk_series())]

    def momentum_pred(asset, candles, train_end, test_end):
        preds = []
        for i in range(train_end + 5, test_end):
            # predict continuation of the last move
            preds.append(1 if candles[i][2] >= candles[i - 1][2] else 0)
        return preds

    r = purged_walk_forward(series, momentum_pred, payout=0.85, embargo=5)
    m = r.pooled
    assert m.n > 100, m.n
    assert m.ci_low <= breakeven(0.85), (m.hit, m.ci_low)
    assert m.verdict != "above", (m.verdict, m.hit)
    print(f"PASS test_negative_control_random_walk_no_edge "
          f"(hit {m.hit*100:.1f}%, CI_low {m.ci_low*100:.1f}%)")


def test_positive_control_injected_edge_detected():
    """On a strongly trending series, a momentum rule SHOULD beat a coin flip.

    This proves the harness can see a real edge when one exists — without it,
    a 'no edge' result would be meaningless.
    """
    series = [("MOM", _momentum_series())]

    def momentum_pred(asset, candles, train_end, test_end):
        preds = []
        for i in range(train_end + 5, test_end):
            # follow the drift: up if the last 3 candles rose
            up = candles[i][2] > candles[i - 3][2]
            preds.append(1 if up else 0)
        return preds

    r = purged_walk_forward(series, momentum_pred, payout=0.85, embargo=5)
    m = r.pooled
    assert m.n > 100, m.n
    assert m.hit > 0.55, (m.hit, m.n)      # real pattern -> clearly above 50%
    assert m.verdict == "above", (m.verdict, m.hit)
    print(f"PASS test_positive_control_injected_edge_detected "
          f"(hit {m.hit*100:.1f}%)")


def test_coin_flip_control_is_not_above_breakeven():
    """The built-in random predictor must not clear the breakeven."""
    import tempfile
    from candle_store import CandleStore
    with tempfile.TemporaryDirectory() as d:
        s = CandleStore(os.path.join(d, "c.db"))
        for a in ("A_otc", "B_otc", "C_otc"):
            s.save(a, 60, _random_walk_series(seed=hash(a) % 1000))
        r = coin_flip_control(s, 60, "otc", payout=0.85, embargo=5, seed=7)
        m = r.pooled
        assert m.n > 100, m.n
        assert m.verdict != "above", (m.verdict, m.hit)
        assert m.ci_low <= breakeven(0.85)
    print(f"PASS test_coin_flip_control_is_not_above_breakeven "
          f"(hit {m.hit*100:.1f}%)")


if __name__ == "__main__":
    test_wilson_known_values()
    test_p_value_known_values()
    test_max_drawdown()
    test_metrics_verdicts()
    test_make_folds_respects_embargo()
    test_negative_control_random_walk_no_edge()
    test_positive_control_injected_edge_detected()
    test_coin_flip_control_is_not_above_breakeven()
    print("\nall validation tests passed")
