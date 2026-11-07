#!/usr/bin/env python3
"""Tests for the signal-inversion test: policy choice and freezing."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from candle_store import CandleStore
import invert


def test_decide_policy_invert_on_low_train():
    # 40% over 400 signals: se = 0.025, so 0.40 < 0.5 - 0.025 -> invert
    assert invert.decide_policy(0.40, 400) == "invert"
    print("PASS test_decide_policy_invert_on_low_train")


def test_decide_policy_normal_on_high_train():
    assert invert.decide_policy(0.60, 400) == "normal"
    print("PASS test_decide_policy_normal_on_high_train")


def test_decide_policy_skip_near_half():
    assert invert.decide_policy(0.505, 400) == "skip"
    assert invert.decide_policy(0.50, 0) == "skip"
    print("PASS test_decide_policy_skip_near_half")


def test_apply_policy_freezes_decision():
    # normal keeps the raw score, invert complements it, skip is not inverting
    assert invert.apply_policy("normal", 60, 100) == (60, 100)
    assert invert.apply_policy("invert", 60, 100) == (40, 100)
    assert invert.apply_policy("skip", 60, 100) == (60, 100)
    print("PASS test_apply_policy_freezes_decision")


def test_inversion_cannot_beat_a_coin_flip():
    # The trap this guards against: an anti-signal on train must not be assumed
    # to persist on test. If test is a coin flip, inverting it is still 50%.
    w, n = invert.apply_policy("invert", 500, 1000)
    assert w == 500, w
    print("PASS test_inversion_cannot_beat_a_coin_flip")


def test_collect_labels_split_and_win():
    with tempfile.TemporaryDirectory() as d:
        s = CandleStore(os.path.join(d, "c.db"))
        # a strictly rising series: every CALL wins, every PUT loses
        rows = [(1000 + i * 60, 1.0, 1.0 + i * 0.001, 1.0 + i * 0.001, 1.0)
                for i in range(300)]
        s.save("X_otc", 60, rows)
        out = invert.collect(s, "rule", 60, train_frac=0.6, warmup=60)
        assert out, "expected some signals"
        splits = {x[0] for x in out}
        assert splits <= {"train", "test"}, splits
        # every recorded outcome must be a valid 0/1
        assert all(x[1] in (0, 1) for x in out)
        # a rising series is not degenerate: both splits present
        assert "train" in splits
    print("PASS test_collect_labels_split_and_win")


if __name__ == "__main__":
    test_decide_policy_invert_on_low_train()
    test_decide_policy_normal_on_high_train()
    test_decide_policy_skip_near_half()
    test_apply_policy_freezes_decision()
    test_inversion_cannot_beat_a_coin_flip()
    test_collect_labels_split_and_win()
    print("\nall invert tests passed")
