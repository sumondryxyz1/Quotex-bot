#!/usr/bin/env python3
"""Tests for the go-live gate and paper account."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from journal import Journal
from golive import paper_curve, gate, _p_value


def _row(asset, period, action, result, entry_at, engine="rule"):
    # (asset, period, engine, action, result, entry_at, expiry_at,
    #  entry_price, settle_price, score, confidence)
    return (asset, period, engine, action, result, entry_at, entry_at + period,
            1.0, 1.0, 5.0, 0.6)


def test_paper_curve_win_loss():
    rows = [_row("A", 5, "CALL", "win", 1),
            _row("A", 5, "CALL", "loss", 2)]
    p = paper_curve(rows, stake_frac=0.1, payout=1.0, start=1000)
    # +10% then -10% of 1100 -> 1000 + 100 - 110 = 990
    assert abs(p["final"] - 990) < 0.01, p["final"]
    assert p["wins"] == 1 and p["losses"] == 1
    assert p["worst_losing_streak"] == 1
    print("PASS test_paper_curve_win_loss")


def test_paper_flat_returns_stake():
    rows = [_row("A", 5, "CALL", "flat", 1)]
    p = paper_curve(rows, stake_frac=0.5, payout=0.85, start=1000)
    assert abs(p["final"] - 1000) < 0.01, p["final"]
    assert p["flats"] == 1
    print("PASS test_paper_flat_returns_stake")


def test_drawdown_tracked():
    rows = [_row("A", 5, "CALL", "loss", i) for i in range(10)]
    p = paper_curve(rows, stake_frac=0.1, payout=0.85, start=1000)
    assert p["max_drawdown_pct"] > 50
    assert p["worst_losing_streak"] == 10
    print("PASS test_drawdown_tracked")


def test_pvalue():
    assert _p_value(50, 100) > 0.9
    assert _p_value(60, 100) < 0.1
    print("PASS test_pvalue")


def test_gate_demo_when_too_few():
    with tempfile.TemporaryDirectory() as d:
        j = Journal(os.path.join(d, "j.db"))
        for i in range(5):
            j._db.execute(
                "INSERT INTO signals(asset,period,engine,action,entry_at,"
                "expiry_at,result) VALUES('A',5,'rule','CALL',?,?,'win')",
                (i, i + 5))
        j._db.commit()
        res = gate(j, payout=0.85, min_n=300)
        assert res["stage"] == "DEMO", res["stage"]
    print("PASS test_gate_demo_when_too_few")


def test_gate_not_ready_when_losing():
    with tempfile.TemporaryDirectory() as d:
        j = Journal(os.path.join(d, "j.db"))
        for i in range(400):
            r = "win" if i % 3 == 0 else "loss"       # ~33% hit
            j._db.execute(
                "INSERT INTO signals(asset,period,engine,action,entry_at,"
                "expiry_at,result) VALUES('A',5,'rule','CALL',?,?,?)",
                (i, i + 5, r))
        j._db.commit()
        res = gate(j, payout=0.85, min_n=300)
        assert res["stage"] == "NOT READY", res["stage"]
        assert not res["paper"]["return_pct"] > 0
    print("PASS test_gate_not_ready_when_losing")


def test_gate_live_ready_when_strong():
    with tempfile.TemporaryDirectory() as d:
        j = Journal(os.path.join(d, "j.db"))
        # 62% hit, spread across many pairs, evenly
        pairs = [f"P{k}_otc" for k in range(10)]
        k = 0
        for i in range(600):
            asset = pairs[i % len(pairs)]
            r = "win" if (i * 7) % 100 < 62 else "loss"
            j._db.execute(
                "INSERT INTO signals(asset,period,engine,action,entry_at,"
                "expiry_at,result) VALUES(?,5,'rule','CALL',?,?,?)",
                (asset, i, i + 5, r))
        j._db.commit()
        res = gate(j, payout=0.85, min_n=300, max_drawdown=0.9)
        assert res["stage"] in ("LIVE-READY", "PAPER PASS"), res["stage"]
        assert res["ci_low"] > res["breakeven"]
    print("PASS test_gate_live_ready_when_strong")


if __name__ == "__main__":
    test_paper_curve_win_loss()
    test_paper_flat_returns_stake()
    test_drawdown_tracked()
    test_pvalue()
    test_gate_demo_when_too_few()
    test_gate_not_ready_when_losing()
    test_gate_live_ready_when_strong()
    print("\nall golive tests passed")
