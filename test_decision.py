#!/usr/bin/env python3
"""Tests for decision.py: weighted voting, EV gate, and NO TRADE conditions."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision import (EngineVote, decide, weighted_vote, reliability_weights,
                      CALL, PUT, ABSTAIN)
from payout import breakeven


def test_weighted_vote_agree_and_split():
    votes = [EngineVote("a", CALL, 1.0, 1.0), EngineVote("b", CALL, 1.0, 1.0)]
    net, total, side = weighted_vote(votes)
    assert net > 0 and side == 1, (net, side)
    votes2 = [EngineVote("a", CALL, 1.0, 1.0), EngineVote("b", PUT, 1.0, 1.0)]
    net2, _, side2 = weighted_vote(votes2)
    assert abs(net2) < 1e-9 and side2 == 0, (net2, side2)
    print("PASS test_weighted_vote_agree_and_split")


def test_decide_emits_call_when_confident_and_ev_positive():
    votes = [EngineVote("a", CALL, 1.0, 1.0), EngineVote("b", CALL, 1.0, 1.0),
             EngineVote("c", CALL, 0.8, 1.0)]
    d = decide(votes, p_up=0.60, payout=0.85)
    assert d.action == CALL, d.as_dict()
    assert d.ev > 0
    assert d.no_trade_reason is None
    print("PASS test_decide_emits_call_when_confident_and_ev_positive")


def test_decide_abstains_when_ev_negative():
    # strong agreement but the probability is below breakeven -> NO TRADE
    votes = [EngineVote("a", CALL, 1.0, 1.0), EngineVote("b", CALL, 1.0, 1.0)]
    d = decide(votes, p_up=0.50, payout=0.85)
    assert d.action == ABSTAIN, d.as_dict()
    assert "EV" in d.no_trade_reason
    print("PASS test_decide_abstains_when_ev_negative")


def test_decide_abstains_on_conflict():
    votes = [EngineVote("a", CALL, 1.0, 1.0), EngineVote("b", PUT, 1.0, 1.0)]
    d = decide(votes, p_up=0.9, payout=0.85)
    assert d.action == ABSTAIN, d.as_dict()
    assert "conflict" in d.no_trade_reason
    print("PASS test_decide_abstains_on_conflict")


def test_decide_abstains_when_regime_undefined():
    votes = [EngineVote("a", CALL, 1.0, 1.0)]
    d = decide(votes, p_up=0.9, regime_defined=False)
    assert d.action == ABSTAIN
    assert "regime" in d.no_trade_reason
    print("PASS test_decide_abstains_when_regime_undefined")


def test_decide_abstains_on_bad_data():
    votes = [EngineVote("a", CALL, 1.0, 1.0)]
    d = decide(votes, p_up=0.9, data_ok=False)
    assert d.action == ABSTAIN
    assert "data quality" in d.no_trade_reason
    print("PASS test_decide_abstains_on_bad_data")


def test_reliability_weights_zero_for_unproven():
    w = reliability_weights({"good": (600, 1000), "weak": (510, 1000),
                             "none": (0, 0)})
    assert w["good"] > 0
    assert w["weak"] == 0.0        # 51% is below the 54.05% breakeven
    assert w["none"] == 0.0
    print("PASS test_reliability_weights_zero_for_unproven")


if __name__ == "__main__":
    test_weighted_vote_agree_and_split()
    test_decide_emits_call_when_confident_and_ev_positive()
    test_decide_abstains_when_ev_negative()
    test_decide_abstains_on_conflict()
    test_decide_abstains_when_regime_undefined()
    test_decide_abstains_on_bad_data()
    test_reliability_weights_zero_for_unproven()
    print("\nall decision tests passed")
