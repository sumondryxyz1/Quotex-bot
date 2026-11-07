#!/usr/bin/env python3
"""Intelligence engine: the market-structure / liquidity / SMC read at one candle.

This is the read-only analysis layer that sits on top of `structure.py`,
`liquidity.py`, `zones.py` and `regime.py`. It produces directional votes with
reasons (for the dashboard and the decision layer) — it does NOT place trades and
is not claimed to be profitable. Any edge it might have is a question for
`validation.py`, not for this file.

Every function is causal: index `i` uses only candles [0..i].

Signals are alerts only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import liquidity
import regime as regime_mod
import structure
import zones
from decision import EngineVote, CALL, PUT


@dataclass
class Intelligence:
    asset: str = ""
    period: int = 0
    index: int = 0
    structure_trend: str = "range"
    bos: Optional[str] = None
    choch: Optional[str] = None
    sweep: Optional[str] = None
    displacement: Optional[str] = None
    zone: Optional[str] = None
    regime: dict = field(default_factory=dict)
    votes: list[EngineVote] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"asset": self.asset, "period": self.period, "index": self.index,
                "structure": self.structure_trend, "bos": self.bos,
                "choch": self.choch, "sweep": self.sweep,
                "displacement": self.displacement, "zone": self.zone,
                "regime": self.regime,
                "votes": [f"{v.name}:{v.direction}({v.score})" for v in self.votes],
                "reasons": self.reasons}


def _arrays(candles: list[list]):
    return ([c[1] for c in candles], [c[2] for c in candles],
            [c[3] for c in candles], [c[4] for c in candles],
            [c[0] for c in candles])


def analyse(candles: list[list], i: Optional[int] = None,
            asset: str = "", period: int = 0) -> Optional[Intelligence]:
    """Full intelligence read at candle `i` (default: last). None if too short."""
    if i is None:
        i = len(candles) - 1
    if i < 60:
        return None
    o, c, h, l, t = _arrays(candles[:i + 1])
    intel = Intelligence(asset=asset, period=period, index=i)

    st = structure.structure_state(o, c, h, l, i)
    intel.structure_trend = st.trend
    if st.last_bos is not None:
        intel.bos = f"{st.last_bos.kind} {st.last_bos.direction} @ {st.last_bos.level:.6f}"
    if st.last_choch is not None:
        intel.choch = f"{st.last_choch.kind} {st.last_choch.direction} @ {st.last_choch.level:.6f}"

    ls = liquidity.liquidity_state(o, c, h, l, i)
    if ls.sweep is not None:
        intel.sweep = (f"{ls.sweep.direction} sweep @ {ls.sweep.level:.6f} "
                       f"(pen {ls.sweep.penetration:.2f}atr, "
                       f"disp={ls.sweep.displaced})")
    d = liquidity.displacement(o, c, h, l, i)
    if d["displaced"]:
        intel.displacement = f"{d['direction']} ({d['size_atr']:.2f}atr)"

    z = zones.active_zone_at(o, c, h, l, i)
    if z is not None:
        intel.zone = f"{z.kind} {z.direction} [{z.bottom:.6f},{z.top:.6f}]"

    reg = regime_mod.classify(o, c, h, l, t, i)
    intel.regime = reg.as_dict()

    # ---- votes (each engine contributes direction + strength) ----
    votes: list[EngineVote] = []
    # structure
    if st.trend == structure.UP:
        votes.append(EngineVote("structure", CALL, 0.6))
        intel.reasons.append("structure up")
    elif st.trend == structure.DOWN:
        votes.append(EngineVote("structure", PUT, 0.6))
        intel.reasons.append("structure down")
    if st.last_choch is not None:
        dirn = CALL if st.last_choch.direction == structure.UP else PUT
        votes.append(EngineVote("choch", dirn, 0.7))
        intel.reasons.append(f"CHoCH {st.last_choch.direction}")
    # liquidity sweep
    if ls.sweep is not None:
        dirn = CALL if ls.sweep.direction == liquidity.BULL else PUT
        strength = 0.8 if ls.sweep.displaced else 0.5
        votes.append(EngineVote("liquidity", dirn, strength))
        intel.reasons.append(f"{ls.sweep.direction} sweep"
                             + (" + displacement" if ls.sweep.displaced else ""))
    # zone
    if z is not None:
        dirn = CALL if z.direction == zones.BULL else PUT
        votes.append(EngineVote("zone", dirn, 0.5))
        intel.reasons.append(f"in {z.direction} {z.kind}")
    # regime (trend-following only when the regime supports it)
    if regime_mod.trend_following_ok(reg):
        dirn = CALL if reg.trend == regime_mod.STRONG_UP else PUT
        votes.append(EngineVote("regime", dirn, 0.4))
        intel.reasons.append(f"regime {reg.trend}")

    intel.votes = votes
    return intel
