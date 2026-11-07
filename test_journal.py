#!/usr/bin/env python3
"""Tests for the live signal journal: logging, settlement, dedup, reporting."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from candle_store import CandleStore
from journal import Journal, settle_open, wilson


def _tmp(name):
    d = tempfile.mkdtemp()
    return os.path.join(d, name)


def test_settle_call_win():
    """CALL entered at T, expiry candle closes higher -> win."""
    store = CandleStore(_tmp("c.db"))
    # candle at t=100 closes 1.10 (entry), candle at t=105 closes 1.12 (expiry)
    store.save("X_otc", 5, [[100, 1.09, 1.10, 1.11, 1.08],
                            [105, 1.10, 1.12, 1.13, 1.09]])
    j = Journal(_tmp("j.db"))
    j.add({"asset": "X_otc", "period": 5, "action": "CALL",
           "entry_at": 100, "price": 1.10, "expiry_at": 105,
           "expiry_candles": 1, "score": 4}, "rule", now=101)
    assert settle_open(j, store, now=110) == 1
    row = j.rows("X_otc")[0]
    assert row[4] == "win", row
    print("PASS test_settle_call_win")


def test_settle_put_loss():
    """PUT but price rises -> loss."""
    store = CandleStore(_tmp("c.db"))
    store.save("Y_otc", 5, [[200, 1.10, 1.10, 1.11, 1.09],
                            [205, 1.10, 1.15, 1.16, 1.10]])
    j = Journal(_tmp("j.db"))
    j.add({"asset": "Y_otc", "period": 5, "action": "PUT",
           "entry_at": 200, "price": 1.10, "expiry_at": 205,
           "expiry_candles": 1, "score": 5}, "rule", now=201)
    settle_open(j, store, now=210)
    assert j.rows("Y_otc")[0][4] == "loss"
    print("PASS test_settle_put_loss")


def test_settle_flat_tie():
    """Unchanged price -> flat, not a loss."""
    store = CandleStore(_tmp("c.db"))
    store.save("Z_otc", 5, [[300, 1.10, 1.10, 1.11, 1.09],
                            [305, 1.10, 1.10, 1.11, 1.09]])
    j = Journal(_tmp("j.db"))
    j.add({"asset": "Z_otc", "period": 5, "action": "CALL",
           "entry_at": 300, "price": 1.10, "expiry_at": 305,
           "expiry_candles": 1, "score": 4}, "rule", now=301)
    settle_open(j, store, now=310)
    assert j.rows("Z_otc")[0][4] == "flat"
    print("PASS test_settle_flat_tie")


def test_not_settled_before_expiry():
    """A signal whose expiry is in the future stays open."""
    store = CandleStore(_tmp("c.db"))
    store.save("W_otc", 5, [[400, 1.10, 1.10, 1.11, 1.09]])
    j = Journal(_tmp("j.db"))
    j.add({"asset": "W_otc", "period": 5, "action": "CALL",
           "entry_at": 400, "price": 1.10, "expiry_at": 405,
           "expiry_candles": 1, "score": 4}, "rule", now=401)
    assert settle_open(j, store, now=402) == 0
    assert j.rows("W_otc")[0][4] == "open"
    print("PASS test_not_settled_before_expiry")


def test_dedup_same_signal():
    """The same (asset, period, entry_at, engine, action) is logged once."""
    j = Journal(_tmp("j.db"))
    sig = {"asset": "D_otc", "period": 5, "action": "CALL", "entry_at": 500,
           "price": 1.0, "expiry_at": 505, "expiry_candles": 1, "score": 4}
    assert j.add(sig, "rule", now=501) is True
    assert j.add(sig, "rule", now=501) is False
    assert len(j.rows()) == 1
    print("PASS test_dedup_same_signal")


def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61, (lo, hi)
    lo0, hi0 = wilson(0, 0)
    assert lo0 == 0.0 and hi0 == 0.0
    print("PASS test_wilson_bounds")


if __name__ == "__main__":
    test_settle_call_win()
    test_settle_put_loss()
    test_settle_flat_tie()
    test_not_settled_before_expiry()
    test_dedup_same_signal()
    test_wilson_bounds()
    print("\nall journal tests passed")
