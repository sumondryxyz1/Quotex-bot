#!/usr/bin/env python3
"""Market structure: swings, BOS, CHoCH, MSS — with explicit rules.

The old `smc.structure()` only returned up/down/range from the last two swings.
This module implements the concepts properly, and every label carries the rule
and the level that produced it, so a detection can be audited rather than
trusted.

Definitions used (all require a **close** beyond the level, never a wick alone):

  swing high/low   a fractal pivot: higher/lower than `left` bars before and
                   `right` bars after. Confirmed only once `right` bars exist,
                   so nothing at index `i` uses data after `i`.
  HH/HL/LH/LL      a swing vs the previous swing of the same kind.
  BOS              Break of Structure — a close beyond the most recent swing
                   *in the prevailing trend direction* (trend continuation).
  CHoCH            Change of Character — the first close beyond the most recent
                   *opposing* swing (a trend flip).
  MSS              Market Structure Shift — a CHoCH that is also accompanied by
                   displacement (a decisive, high-momentum leg).

Pure functions over OHLC lists. No look-ahead. Signals are alerts only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import indicators as ta

UP, DOWN, RANGE = "up", "down", "range"


@dataclass
class Swing:
    idx: int
    price: float
    kind: str            # "high" | "low"
    label: str = ""      # HH | HL | LH | LL


@dataclass
class StructureEvent:
    kind: str            # "BOS" | "CHoCH" | "MSS"
    direction: str       # "up" | "down"
    level: float
    idx: int
    detail: str = ""


@dataclass
class StructureState:
    trend: str = RANGE
    swings: list[Swing] = field(default_factory=list)
    events: list[StructureEvent] = field(default_factory=list)
    last_bos: Optional[StructureEvent] = None
    last_choch: Optional[StructureEvent] = None

    def as_dict(self) -> dict:
        return {
            "trend": self.trend,
            "last_bos": (f"{self.last_bos.direction} @ {self.last_bos.level}"
                         if self.last_bos else None),
            "last_choch": (f"{self.last_choch.direction} @ {self.last_choch.level}"
                           if self.last_choch else None),
            "swings": [(s.idx, s.kind, s.label) for s in self.swings[-6:]],
        }


def swing_series(highs: list[float], lows: list[float], i: int, left: int = 2,
                 right: int = 2, lookback: int = 120) -> list[Swing]:
    """Confirmed fractal pivots in [i-lookback, i-right], labelled HH/HL/LH/LL.

    Because a pivot needs `right` later bars, the newest pivot returned is at
    most `i - right` — this is what keeps the whole module causal.
    """
    lo = max(0, i - lookback)
    seg_h = highs[lo:i + 1]
    seg_l = lows[lo:i + 1]
    n = len(seg_h)
    out: list[Swing] = []
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None
    for j in range(left, n - right):
        if all(seg_h[j] > seg_h[j - k] for k in range(1, left + 1)) and \
           all(seg_h[j] > seg_h[j + k] for k in range(1, right + 1)):
            s = Swing(lo + j, seg_h[j], "high")
            s.label = _label_high(s, last_high)
            last_high = s
            out.append(s)
        if all(seg_l[j] < seg_l[j - k] for k in range(1, left + 1)) and \
           all(seg_l[j] < seg_l[j + k] for k in range(1, right + 1)):
            s = Swing(lo + j, seg_l[j], "low")
            s.label = _label_low(s, last_low)
            last_low = s
            out.append(s)
    out.sort(key=lambda s: s.idx)
    return out


def _label_high(s: Swing, prev: Optional[Swing]) -> str:
    if prev is None:
        return ""
    return "HH" if s.price > prev.price else "LH"


def _label_low(s: Swing, prev: Optional[Swing]) -> str:
    if prev is None:
        return ""
    return "HL" if s.price > prev.price else "LL"


def trend_from_swings(swings: list[Swing]) -> str:
    """HH+HL -> up, LH+LL -> down, else range. Uses the two latest of each kind."""
    highs = [s for s in swings if s.kind == "high"][-2:]
    lows = [s for s in swings if s.kind == "low"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return RANGE
    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price
    if hh and hl:
        return UP
    if lh and ll:
        return DOWN
    return RANGE


def _last_swing(swings: list[Swing], kind: str, before: int) -> Optional[Swing]:
    for s in reversed(swings):
        if s.kind == kind and s.idx < before:
            return s
    return None


def structure_state(opens: list[float], closes: list[float], highs: list[float],
                    lows: list[float], i: int, left: int = 2, right: int = 2,
                    lookback: int = 120, disp_atr: float = 1.2) -> StructureState:
    """Full structure read at `i`: trend, labelled swings, and the BOS/CHoCH events.

    Events are walked chronologically so the *first* opposing break is the CHoCH
    and same-direction breaks are BOS, matching how the concepts are taught.
    """
    swings = swing_series(highs, lows, i, left, right, lookback)
    st = StructureState(trend=trend_from_swings(swings), swings=swings)
    if len(swings) < 2:
        return st

    atr = ta.atr(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)

    # Walk candles in order; track the current trend and the last broken level.
    trend = RANGE
    swings_sorted = sorted(swings, key=lambda s: s.idx)
    si = 0
    pending_high: Optional[Swing] = None
    pending_low: Optional[Swing] = None
    start = max(left + 1, swings_sorted[0].idx if swings_sorted else 0)
    for j in range(start, i + 1):
        # advance swings that are already confirmed at j
        while si < len(swings_sorted) and swings_sorted[si].idx + right <= j:
            s = swings_sorted[si]
            if s.kind == "high":
                pending_high = s
            else:
                pending_low = s
            si += 1
        if pending_high is None or pending_low is None:
            continue
        c = closes[j]
        disp = (atr[j] is not None and atr[j] > 0
                and abs(closes[j] - opens[j]) >= disp_atr * atr[j])
        # break up: close above the last swing high
        if c > pending_high.price:
            if trend == DOWN:
                ev = StructureEvent("CHoCH", UP, pending_high.price, j,
                                    "displacement" if disp else "")
                if disp:
                    ev.kind = "MSS"
                st.last_choch = ev
                st.events.append(ev)
            else:
                # UP or the first break out of RANGE establishes/continues trend
                ev = StructureEvent("BOS", UP, pending_high.price, j)
                st.last_bos = ev
                st.events.append(ev)
            trend = UP
            pending_high = None
        # break down: close below the last swing low
        elif c < pending_low.price:
            if trend == UP:
                ev = StructureEvent("CHoCH", DOWN, pending_low.price, j,
                                    "displacement" if disp else "")
                if disp:
                    ev.kind = "MSS"
                st.last_choch = ev
                st.events.append(ev)
            else:
                ev = StructureEvent("BOS", DOWN, pending_low.price, j)
                st.last_bos = ev
                st.events.append(ev)
            trend = DOWN
            pending_low = None
    st.trend = trend if trend != RANGE else st.trend
    return st


def bos(closes: list[float], highs: list[float], lows: list[float], i: int,
        **kw) -> Optional[StructureEvent]:
    """Most recent Break of Structure at/just before `i`, or None."""
    st = structure_state(closes, closes, highs, lows, i, **kw)
    return st.last_bos


def choch(closes: list[float], highs: list[float], lows: list[float], i: int,
          **kw) -> Optional[StructureEvent]:
    """Most recent Change of Character at/just before `i`, or None."""
    st = structure_state(closes, closes, highs, lows, i, **kw)
    return st.last_choch


def internal_external(highs: list[float], lows: list[float], i: int) -> dict:
    """Structure at two degrees: minor (fast pivots) inside major (slow pivots)."""
    minor = trend_from_swings(swing_series(highs, lows, i, 1, 1, 60))
    major = trend_from_swings(swing_series(highs, lows, i, 3, 3, 200))
    return {"internal": minor, "external": major,
            "aligned": minor == major and major != RANGE}


def structure_series(opens: list[float], closes: list[float], highs: list[float],
                     lows: list[float], left: int = 2, right: int = 2,
                     lookback: int = 120, disp_atr: float = 1.2) -> dict:
    """Per-index structure in ONE forward pass (for feature building).

    Returns dict of arrays: trend[], bos_dir[] (1 up / -1 down / 0), and
    choch_dir[]. Equivalent to calling `structure_state` at each index but O(n),
    not O(n^2) — it never re-scans the past.
    """
    n = len(closes)
    trend_arr = [RANGE] * n
    bos_arr = [0] * n
    choch_arr = [0] * n
    # Use every confirmed pivot in the series. `swing_series` is causal by
    # construction (a pivot at j is only emitted once `right` bars follow it), so
    # a full-range scan stays look-ahead-free; a trailing window would not.
    swings_all = swing_series(highs, lows, n - 1, left, right, max(lookback, n))
    swings_all.sort(key=lambda s: s.idx)
    atr = ta.atr(highs, lows, closes, 14)
    si = 0
    pending_high = None
    pending_low = None
    trend = RANGE
    for j in range(n):
        while si < len(swings_all) and swings_all[si].idx + right <= j:
            s = swings_all[si]
            if s.kind == "high":
                pending_high = s
            else:
                pending_low = s
            si += 1
        if pending_high is not None and pending_low is not None:
            disp = (atr[j] is not None and atr[j] > 0
                    and abs(closes[j] - opens[j]) >= disp_atr * atr[j])
            if closes[j] > pending_high.price:
                if trend == DOWN:
                    choch_arr[j] = 1
                else:
                    bos_arr[j] = 1
                trend = UP
                pending_high = None
            elif closes[j] < pending_low.price:
                if trend == UP:
                    choch_arr[j] = -1
                else:
                    bos_arr[j] = -1
                trend = DOWN
                pending_low = None
        trend_arr[j] = trend
    return {"trend": trend_arr, "bos": bos_arr, "choch": choch_arr}
