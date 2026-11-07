#!/usr/bin/env python3
"""Decision layer: combine the engines, then decide whether to speak at all.

Two ideas the old ensemble did not have:

1. **Measured-reliability weighting.** Each engine is weighted by how it has
   actually performed (out-of-sample), not by a hand-set number. An engine with
   no measured record gets a low default weight.
2. **Expected-value NO TRADE gate.** A signal is only emitted when its
   payout-adjusted expected value clears a margin. Otherwise the answer is
   NO TRADE — the most valuable output when there is no edge.

Conflicting engines do not get averaged into a forced BUY/SELL; disagreement
above a threshold is itself a NO TRADE condition. Fewer, better signals.

Signals are alerts only — the bot never places a trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from payout import DEFAULT_PAYOUT, expected_value, breakeven as payout_breakeven

CALL, PUT, ABSTAIN = "CALL", "PUT", "ABSTAIN"


@dataclass
class EngineVote:
    name: str
    direction: Optional[str]     # CALL | PUT | None
    score: float = 1.0           # strength of this engine's read
    weight: float = 1.0          # measured reliability weight


@dataclass
class Decision:
    asset: str = ""
    period: int = 0
    source: str = ""
    time: int = 0
    action: str = ABSTAIN
    p_up: Optional[float] = None
    confidence: float = 0.0
    ev: float = 0.0
    payout: float = DEFAULT_PAYOUT
    net_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    no_trade_reason: Optional[str] = None
    context: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _sign(direction: Optional[str]) -> int:
    return 1 if direction == CALL else -1 if direction == PUT else 0


def weighted_vote(votes: list[EngineVote]) -> tuple[float, float, int]:
    """Return (net_score, total_weight, n_agree_side).

    net_score is in [-1, 1] after normalising by total weight.
    """
    total = sum(max(0.0, v.weight) for v in votes) or 1.0
    net = sum(_sign(v.direction) * max(0.0, v.weight) * max(0.0, v.score)
              for v in votes)
    net_norm = net / total
    return net_norm, total, (1 if net_norm > 0 else -1 if net_norm < 0 else 0)


def decide(votes: list[EngineVote], *, asset: str = "", period: int = 0,
           source: str = "", now: int = 0, payout: float = DEFAULT_PAYOUT,
           p_up: Optional[float] = None, regime_defined: bool = True,
           data_ok: bool = True, min_net: float = 0.25,
           max_conflict: float = 0.2, ev_margin: float = 0.0,
           context: Optional[dict] = None) -> Decision:
    """Combine engine votes and context into a CALL / PUT / ABSTAIN decision.

    NO TRADE (ABSTAIN) is returned when any of these holds:
      * the regime is undefined (transitional / ambiguous market),
      * the data quality is bad,
      * the engines conflict (net score below `min_net`, or balanced sides),
      * the payout-adjusted EV does not clear `ev_margin`.
    """
    d = Decision(asset=asset, period=period, source=source, time=now,
                 payout=payout, context=context or {})
    d.reasons = [f"{v.name}:{v.direction or '-'}(w{v.weight:.2f})" for v in votes]

    if not data_ok:
        d.no_trade_reason = "data quality below threshold"
        return d
    if not regime_defined:
        d.no_trade_reason = "market regime undefined (transitional)"
        return d
    if not votes:
        d.no_trade_reason = "no engine produced a read"
        return d

    net, total, side = weighted_vote(votes)
    d.net_score = round(net, 4)

    # conflict: engines disagree, or the net is too weak
    bull = sum(1 for v in votes if v.direction == CALL)
    bear = sum(1 for v in votes if v.direction == PUT)
    if bull and bear and abs(net) < max_conflict:
        d.no_trade_reason = f"engines conflict (bull {bull} vs bear {bear})"
        return d
    if abs(net) < min_net:
        d.no_trade_reason = f"net score {net:.2f} below {min_net}"
        return d

    action = CALL if net > 0 else PUT

    # probability + EV gate
    if p_up is not None:
        d.p_up = round(p_up, 4)
        prob = p_up if action == CALL else 1 - p_up
        d.confidence = round(prob, 4)
        d.ev = round(expected_value(prob, payout), 5)
        if d.ev <= ev_margin:
            d.no_trade_reason = (f"EV {d.ev:+.4f} <= margin {ev_margin:+.4f} "
                                 f"(breakeven {payout_breakeven(payout):.2%})")
            return d
    else:
        d.confidence = round(min(1.0, abs(net)), 4)

    d.action = action
    return d


def reliability_weights(hit_rates: dict[str, tuple[int, int]]) -> dict[str, float]:
    """Turn each engine's (wins, n) record into a weight.

    Weight = max(0, hit-rate − breakeven) × min(1, n/300). An engine that is at
    or below breakeven, or has too little data, gets ~0 weight — so an unproven
    engine cannot drive a signal.
    """
    out = {}
    for name, (wins, n) in hit_rates.items():
        if n <= 0:
            out[name] = 0.0
            continue
        hit = wins / n
        edge = max(0.0, hit - payout_breakeven(DEFAULT_PAYOUT))
        conf = min(1.0, n / 300.0)
        out[name] = round(edge * conf, 5)
    return out
