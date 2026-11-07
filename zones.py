#!/usr/bin/env python3
"""Order blocks and fair-value gaps — and whether they actually work.

`smc.order_block()` and `smc.fair_value_gap()` only *find* zones. A zone that
price never reacts to is decoration. This module finds zones **and measures
their validity**: how often price respected the zone after it formed, how often
it was mitigated (traded back into), and how often it was violated (closed clean
through). The respect rate is a measured number, not an assumption.

Zone definitions (causal):

  order block (OB)   the last opposite candle before a displacement leg into the
                     zone. Bullish OB = last down candle before an up
                     displacement; bearish OB = last up candle before a down
                     displacement.
  fair value gap     a 3-candle imbalance: candle i's low above candle i-2's high
    (FVG)            (bullish) or candle i's high below candle i-2's low (bearish).

Zone lifecycle:

  fresh        not yet traded back into (untested)
  mitigated    price returned into the zone at least once
  respected    after mitigation, price reversed away from the zone (held)
  violated     price closed clean through the zone (broke)

Signals are alerts only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import indicators as ta
from liquidity import displacement

BULL, BEAR = "bull", "bear"


@dataclass
class Zone:
    kind: str            # "OB" | "FVG"
    direction: str       # "bull" | "bear"
    top: float
    bottom: float
    created_idx: int
    mitigated: bool = False
    respected: bool = False
    violated: bool = False
    mit_idx: Optional[int] = None

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def as_dict(self) -> dict:
        return {"kind": self.kind, "dir": self.direction,
                "range": (round(self.bottom, 6), round(self.top, 6)),
                "created": self.created_idx, "mitigated": self.mitigated,
                "respected": self.respected, "violated": self.violated}


@dataclass
class ZoneStats:
    n: int = 0
    mitigated: int = 0
    respected: int = 0
    violated: int = 0
    respect_rate: float = 0.0      # respected / mitigated
    violation_rate: float = 0.0    # violated / mitigated
    baseline: float = 0.5          # a coin-flip reference

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def find_order_block(opens: list[float], closes: list[float], highs: list[float],
                     lows: list[float], i: int, lookback: int = 20,
                     impulse_atr: float = 1.0) -> Optional[Zone]:
    """Order block created at/just before `i` by a displacement leg into `i`."""
    disp = displacement(opens, closes, highs, lows, i, atr_mult=impulse_atr,
                        body_frac=0.5)
    if not disp["displaced"]:
        return None
    lo = max(1, i - lookback)
    if disp["direction"] == BULL:
        for j in range(i - 1, lo - 1, -1):
            if closes[j] < opens[j]:
                return Zone("OB", BULL, highs[j], lows[j], j)
    else:
        for j in range(i - 1, lo - 1, -1):
            if closes[j] > opens[j]:
                return Zone("OB", BEAR, highs[j], lows[j], j)
    return None


def find_fvg(highs: list[float], lows: list[float], i: int,
             min_atr: float = 0.1) -> Optional[Zone]:
    """3-candle imbalance at `i` (candles i-2, i-1, i)."""
    if i < 3:
        return None
    atr = ta.atr(highs[:i + 1], lows[:i + 1],
                 [(highs[k] + lows[k]) / 2 for k in range(i + 1)], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if a <= 0:
        return None
    if lows[i] > highs[i - 2]:
        top, bottom = lows[i], highs[i - 2]
        if (top - bottom) >= min_atr * a:
            return Zone("FVG", BULL, top, bottom, i - 1)
    if highs[i] < lows[i - 2]:
        top, bottom = lows[i - 2], highs[i]
        if (top - bottom) >= min_atr * a:
            return Zone("FVG", BEAR, top, bottom, i - 1)
    return None


def advance_zone(zone: Zone, opens: list[float], closes: list[float],
                 highs: list[float], lows: list[float], j: int) -> Zone:
    """Advance a zone by exactly one candle `j` (incremental, O(1)).

    The incremental form keeps a live scan linear; it applies the same rules as
    `zone_state` but only for the newest candle, so it must be called in order.
    """
    if zone.violated or zone.respected:
        return zone
    if zone.direction == BULL:
        if closes[j] < zone.bottom:
            zone.violated = True
            return zone
        if lows[j] <= zone.top and highs[j] >= zone.bottom:
            if not zone.mitigated:
                zone.mitigated = True
                zone.mit_idx = j
        elif zone.mitigated and closes[j] > zone.top:
            zone.respected = True
    else:
        if closes[j] > zone.top:
            zone.violated = True
            return zone
        if lows[j] <= zone.top and highs[j] >= zone.bottom:
            if not zone.mitigated:
                zone.mitigated = True
                zone.mit_idx = j
        elif zone.mitigated and closes[j] < zone.bottom:
            zone.respected = True
    return zone


def zone_state(zone: Zone, opens: list[float], closes: list[float],
               highs: list[float], lows: list[float],
               up_to: Optional[int] = None) -> Zone:
    """Update a zone's lifecycle by scanning candles after it formed.

    Outcomes are exclusive: the scan stops at the first decisive event. A zone is
    either **violated** (price closed clean through the far side) or
    **respected** (price traded in, then closed back out the near side) — never
    both. `mitigated` records that price returned into the zone at all.
    """
    n = len(closes) if up_to is None else min(up_to + 1, len(closes))
    for j in range(zone.created_idx + 1, n):
        advance_zone(zone, opens, closes, highs, lows, j)
        if zone.violated or zone.respected:
            break
    return zone


def collect_zones(opens: list[float], closes: list[float], highs: list[float],
                  lows: list[float], i: int, lookback: int = 200,
                  kind: str = "OB") -> list[Zone]:
    """All OB (or FVG) zones created in [i-lookback, i]."""
    lo = max(4, i - lookback)
    out = []
    for j in range(lo, i + 1):
        z = (find_order_block(opens, closes, highs, lows, j) if kind == "OB"
             else find_fvg(highs, lows, j))
        if z is not None:
            out.append(z)
    return out


def zone_validity_stats(opens: list[float], closes: list[float],
                        highs: list[float], lows: list[float],
                        zones: list[Zone]) -> ZoneStats:
    """Measure the respect/violation rate of a set of zones on this series.

    The headline number is `respect_rate` = respected / mitigated. Compare it to
    the 0.5 baseline: if a zone type respects about as often as a coin flip, it
    carries no information.
    """
    st = ZoneStats()
    for z in zones:
        zone_state(z, opens, closes, highs, lows)
        st.n += 1
        if z.mitigated:
            st.mitigated += 1
            st.respected += int(z.respected)
            st.violated += int(z.violated)
    if st.mitigated:
        st.respect_rate = round(st.respected / st.mitigated, 4)
        st.violation_rate = round(st.violated / st.mitigated, 4)
    return st


def active_zone_at(opens: list[float], closes: list[float], highs: list[float],
                   lows: list[float], i: int, kind: str = "OB",
                   max_age: int = 60) -> Optional[Zone]:
    """Most recent un-violated zone containing price at `i` (a live setup)."""
    for z in reversed(collect_zones(opens, closes, highs, lows, i, kind=kind)):
        if i - z.created_idx > max_age:
            continue
        zone_state(z, opens, closes, highs, lows, up_to=i)
        if not z.violated and z.contains(closes[i]):
            return z
    return None
