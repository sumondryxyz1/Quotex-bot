#!/usr/bin/env python3
"""Classical chart-pattern detection — structural, not indicator-based.

Every pattern here is built from swing pivots (`smc.swings`) and closes only, and
every value at index `i` uses candles [0..i] — no look-ahead, so the same code is
safe in a backtest and live.

Patterns implemented
--------------------
  head & shoulders      three peaks, middle highest, neckline break (bearish)
  inverse H&S           three troughs, middle lowest (bullish)
  double top            two peaks at ~the same level (bearish)
  double bottom         two troughs at ~the same level (bullish)
  ascending triangle    flat top + rising lows -> break up (bullish)
  descending triangle   flat bottom + falling highs -> break down (bearish)
  support / resistance  clustered pivots; bounce (fade) or break (follow)
  flag / pennant        tight consolidation after an impulse, break continues

Each detector returns 'bull' / 'bear' / None. `pattern_vote` combines them into
one CALL/PUT/abstain vote with the reasons.

Signals are alerts only — never a trade instruction.
"""
from __future__ import annotations

from typing import Optional

import smc


def _tol(level: float, pct: float = 0.0015) -> float:
    return abs(level) * pct


def head_shoulders(highs: list[float], lows: list[float], i: int,
                   lookback: int = 90, tol: float = 0.002) -> Optional[str]:
    """H&S (bear) / inverse H&S (bull) confirmed by a neckline break at `i`."""
    lo = max(0, i - lookback)
    if i - lo < 20:
        return None
    sh, sl = smc.swings(highs[lo:i + 1], lows[lo:i + 1], 2, 2)
    sh = [x + lo for x in sh]
    sl = [x + lo for x in sl]

    # bearish: last three swing highs, middle clearly the highest
    if len(sh) >= 3:
        l, m, r = sh[-3], sh[-2], sh[-1]
        if (highs[m] > highs[l] and highs[m] > highs[r]
                and abs(highs[l] - highs[r]) <= _tol(highs[l], tol)
                and highs[m] - max(highs[l], highs[r]) > _tol(highs[m], tol)):
            between = [x for x in sl if l < x < r]
            if between:
                neck = min(lows[x] for x in between)
                if lows[i] < neck:            # neckline broken
                    return "bear"
    # bullish: last three swing lows, middle clearly the lowest
    if len(sl) >= 3:
        l, m, r = sl[-3], sl[-2], sl[-1]
        if (lows[m] < lows[l] and lows[m] < lows[r]
                and abs(lows[l] - lows[r]) <= _tol(lows[l], tol)
                and min(lows[l], lows[r]) - lows[m] > _tol(lows[m], tol)):
            between = [x for x in sh if l < x < r]
            if between:
                neck = max(highs[x] for x in between)
                if highs[i] > neck:
                    return "bull"
    return None


def double_top_bottom(highs: list[float], lows: list[float], i: int,
                      lookback: int = 70, tol: float = 0.0025) -> Optional[str]:
    """Two peaks/troughs at ~the same level, confirmed by the middle break."""
    lo = max(0, i - lookback)
    if i - lo < 15:
        return None
    sh, sl = smc.swings(highs[lo:i + 1], lows[lo:i + 1], 2, 2)
    sh = [x + lo for x in sh]
    sl = [x + lo for x in sl]

    if len(sh) >= 2:
        a, b = sh[-2], sh[-1]
        if abs(highs[a] - highs[b]) <= _tol(highs[a], tol):
            trough = [x for x in sl if a < x < b]
            if trough:
                neck = min(lows[x] for x in trough)
                if lows[i] < neck:
                    return "bear"
    if len(sl) >= 2:
        a, b = sl[-2], sl[-1]
        if abs(lows[a] - lows[b]) <= _tol(lows[a], tol):
            peak = [x for x in sh if a < x < b]
            if peak:
                neck = max(highs[x] for x in peak)
                if highs[i] > neck:
                    return "bull"
    return None


def triangle(highs: list[float], lows: list[float], i: int,
             lookback: int = 50) -> Optional[str]:
    """Ascending / descending triangle, broken at `i`."""
    lo = max(0, i - lookback)
    if i - lo < 18:
        return None
    sh, sl = smc.swings(highs[lo:i + 1], lows[lo:i + 1], 2, 2)
    sh = [x + lo for x in sh]
    sl = [x + lo for x in sl]
    if len(sh) < 2 or len(sl) < 2:
        return None
    flat_top = abs(highs[sh[-1]] - highs[sh[-2]]) <= _tol(highs[sh[-1]], 0.003)
    flat_bot = abs(lows[sl[-1]] - lows[sl[-2]]) <= _tol(lows[sl[-1]], 0.003)
    rising_lows = lows[sl[-1]] > lows[sl[-2]]
    falling_highs = highs[sh[-1]] < highs[sh[-2]]
    if flat_top and rising_lows and highs[i] > max(highs[x] for x in sh[-2:]):
        return "bull"
    if flat_bot and falling_highs and lows[i] < min(lows[x] for x in sl[-2:]):
        return "bear"
    return None


def sr_levels(highs: list[float], lows: list[float], i: int,
              lookback: int = 80, cluster: float = 0.002) -> list[float]:
    """Cluster recent swing pivots into horizontal support/resistance levels."""
    lo = max(0, i - lookback)
    if i - lo < 12:
        return []
    sh, sl = smc.swings(highs[lo:i + 1], lows[lo:i + 1], 2, 2)
    pivots = sorted([highs[x + lo] for x in sh] + [lows[x + lo] for x in sl])
    levels: list[float] = []
    for p in pivots:
        if levels and abs(p - levels[-1]) <= _tol(p, cluster):
            levels[-1] = (levels[-1] + p) / 2
        else:
            levels.append(p)
    return levels


def sr_price_action(highs: list[float], lows: list[float], closes: list[float],
                    opens: list[float], i: int, lookback: int = 80,
                    near: float = 0.0012) -> Optional[str]:
    """Price action at support/resistance.

    Rejection wick at a level -> fade (bounce). A decisive close through the
    level -> break (follow the break).
    """
    levels = sr_levels(highs, lows, i, lookback)
    if not levels:
        return None
    c, o, h, l = closes[i], opens[i], highs[i], lows[i]
    rng = h - l
    if rng <= 0:
        return None
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    for lv in levels:
        if abs(c - lv) <= _tol(lv, near):
            # rejection: long lower wick at the level -> bounce up
            if lower_wick > 0.5 * rng and c > o:
                return "bull"
            if upper_wick > 0.5 * rng and c < o:
                return "bear"
        if l < lv < c and c > lv + _tol(lv, near):   # broke up through it
            return "bull"
        if c < lv < h and c < lv - _tol(lv, near):   # broke down through it
            return "bear"
    return None


def flag(highs: list[float], lows: list[float], closes: list[float], i: int,
         impulse: int = 12, flag_len: int = 8) -> Optional[str]:
    """Impulse then tight consolidation; break continues the impulse."""
    if i < impulse + flag_len + 1:
        return None
    imp = closes[i - flag_len] - closes[i - flag_len - impulse]
    seg_hi = max(highs[i - flag_len:i])
    seg_lo = min(lows[i - flag_len:i])
    rng = closes[i - flag_len - impulse:i - flag_len]
    avg_rng = (max(rng) - min(rng)) or 1e-9
    tight = (seg_hi - seg_lo) < 0.6 * avg_rng
    if not tight:
        return None
    if imp > 0 and closes[i] > seg_hi:
        return "bull"
    if imp < 0 and closes[i] < seg_lo:
        return "bear"
    return None


def pattern_vote(opens: list[float], closes: list[float], highs: list[float],
                 lows: list[float], i: int) -> tuple[Optional[str], list[str]]:
    """Combine every detector into one CALL/PUT/abstain vote."""
    votes: list[tuple[str, str]] = []
    hs = head_shoulders(highs, lows, i)
    if hs:
        votes.append((hs, "head & shoulders"))
    db = double_top_bottom(highs, lows, i)
    if db:
        votes.append((db, "double top/bottom"))
    tr = triangle(highs, lows, i)
    if tr:
        votes.append((tr, "triangle break"))
    sr = sr_price_action(highs, lows, closes, opens, i)
    if sr:
        votes.append((sr, "S/R price action"))
    fl = flag(highs, lows, closes, i)
    if fl:
        votes.append((fl, "flag/pennant break"))

    bull = sum(1 for v, _ in votes if v == "bull")
    bear = sum(1 for v, _ in votes if v == "bear")
    reasons = [f"{name} ({'bull' if v == 'bull' else 'bear'})" for v, name in votes]
    if bull == bear:
        return None, reasons
    return ("CALL" if bull > bear else "PUT"), reasons
