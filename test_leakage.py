#!/usr/bin/env python3
"""Tests for the leakage audit: each check must pass on clean code and FAIL
on a deliberately leaky detector (so the audit itself is trustworthy)."""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_leakage import (future_permutation, fold_disjointness, label_overlap,
                           feature_causality, pooling_guard, run)
from candle_store import CandleStore


def _series(n=300, seed=0):
    rng = random.Random(seed)
    out = []
    price = 1.1
    for i in range(n):
        o = price
        price += rng.gauss(0, 0.0004)
        c = price
        out.append([i * 60, o, c, max(o, c) + 1e-4, min(o, c) - 1e-4])
    return out


def test_fold_disjointness_passes():
    r = fold_disjointness(1000, 5, 0.4, 5)
    assert r.passed, r.detail
    print("PASS test_fold_disjointness_passes")


def test_fold_disjointness_catches_zero_embargo():
    # with embargo=0 the check still holds (test_start == train_end is allowed)
    r = fold_disjointness(1000, 5, 0.4, 0)
    assert r.passed, r.detail
    print("PASS test_fold_disjointness_catches_zero_embargo")


def test_label_overlap():
    assert label_overlap(1, 5).passed
    assert not label_overlap(10, 5).passed   # embargo < horizon -> overlap
    print("PASS test_label_overlap")


def test_pooling_guard_passes():
    assert pooling_guard().passed
    print("PASS test_pooling_guard_passes")


def test_feature_causality_clean():
    cs = _series(200)

    def causal(candles, i):
        return candles[i][2] - candles[i - 1][2]

    r = feature_causality(cs, 100, causal)
    assert r.passed, r.detail
    print("PASS test_feature_causality_clean")


def test_feature_causality_catches_lookahead():
    cs = _series(200)

    def leaky(candles, i):
        # peeks one candle into the future
        return candles[i + 1][2] if i + 1 < len(candles) else None

    r = feature_causality(cs, 100, leaky)
    assert not r.passed, r.detail
    print("PASS test_feature_causality_catches_lookahead")


def test_future_permutation_clean_detector():
    cs = _series(300)

    def causal(candles, i):
        return round(candles[i][2], 6)

    r = future_permutation(cs, causal, 150)
    assert r.passed, r.detail
    print("PASS test_future_permutation_clean_detector")


def test_future_permutation_catches_leaky_detector():
    cs = _series(300)

    def leaky(candles, i):
        # peeks at the very next candle -> shuffling the future changes it
        return candles[i + 1][2] if i + 1 < len(candles) else None

    r = future_permutation(cs, leaky, 150)
    assert not r.passed, r.detail
    print("PASS test_future_permutation_catches_leaky_detector")


def test_full_audit_runs_clean_on_random_walk():
    with tempfile.TemporaryDirectory() as d:
        s = CandleStore(os.path.join(d, "c.db"))
        for a in ("A_otc", "B_otc"):
            s.save(a, 60, _series(300, seed=hash(a) % 999))
        results = run(s, 60, "otc", payout=0.85, embargo=5, limit=2)
        failed = [r for r in results if not r.passed]
        assert not failed, [f"{r.name}: {r.detail}" for r in failed]
        names = {r.name for r in results}
        assert "coin-flip-control" in names
        assert "future-permutation" in names
    print("PASS test_full_audit_runs_clean_on_random_walk")


if __name__ == "__main__":
    test_fold_disjointness_passes()
    test_fold_disjointness_catches_zero_embargo()
    test_label_overlap()
    test_pooling_guard_passes()
    test_feature_causality_clean()
    test_feature_causality_catches_lookahead()
    test_future_permutation_clean_detector()
    test_future_permutation_catches_leaky_detector()
    test_full_audit_runs_clean_on_random_walk()
    print("\nall leakage-audit tests passed")
