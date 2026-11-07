#!/usr/bin/env python3
"""Tests for features_v2.py: shape, causality, and the random-walk control."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features_v2
from ml_validate import run as ml_run


def _random_walk(n=900, seed=0, vol=0.0006, start=1.1):
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


def test_feature_names_unique_and_sized():
    assert len(features_v2.FEATURES_V2) == len(set(features_v2.FEATURES_V2))
    assert len(features_v2.FEATURES_V2) == 46, len(features_v2.FEATURES_V2)
    print("PASS test_feature_names_unique_and_sized")


def test_build_matrix_shape_and_labels():
    cs = _random_walk(600, seed=1)
    rows, labels, idx = features_v2.build_matrix(cs)
    assert rows and len(rows[0]) == len(features_v2.FEATURES_V2)
    assert len(rows) == len(labels) == len(idx)
    assert set(labels) <= {0, 1}
    # label must equal next-close direction
    for k in range(0, len(idx), 25):
        i = idx[k]
        assert labels[k] == (1 if cs[i + 1][2] > cs[i][2] else 0)
    print("PASS test_build_matrix_shape_and_labels")


def test_build_matrix_short_series():
    cs = _random_walk(30, seed=2)
    rows, labels, idx = features_v2.build_matrix(cs)
    assert rows == [] and labels == [] and idx == []
    print("PASS test_build_matrix_short_series")


def test_features_v2_causal():
    cs = _random_walk(500, seed=3)
    i_max = 400
    full_rows, _, full_idx = features_v2.build_matrix(cs)
    trunc_rows, _, trunc_idx = features_v2.build_matrix(cs[:i_max + 1])
    # rows computed on the truncated series must match the full series rows
    # for the same candle indices (no future data leaks into a feature)
    fmap = {idx: row for idx, row in zip(full_idx, full_rows)}
    for idx, row in zip(trunc_idx, trunc_rows):
        assert idx in fmap, idx
        assert row == fmap[idx], (idx, row, fmap[idx])
    print("PASS test_features_v2_causal")


def test_ml_random_walk_negative_control():
    """The v2 pipeline must not manufacture an edge on a random walk."""
    cs = _random_walk(2000, seed=7)
    r = ml_run(cs, n_splits=4, payout=0.85, label="rw")
    assert "error" not in r, r
    assert r["mean_edge"] < 0.05, r
    assert r["verdict"] in ("below", "indistinguishable"), r
    print(f"PASS test_ml_random_walk_negative_control "
          f"(acc {r['mean_acc']*100:.1f}% base {r['mean_baseline']*100:.1f}% "
          f"edge {r['mean_edge']:+.3f})")


if __name__ == "__main__":
    test_feature_names_unique_and_sized()
    test_build_matrix_shape_and_labels()
    test_build_matrix_short_series()
    test_features_v2_causal()
    test_ml_random_walk_negative_control()
    print("\nall features_v2 tests passed")
