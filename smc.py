#!/usr/bin/env python3
"""SMC (Smart Money Concepts) primitives — liquidity, price action, structure.

Pure functions over OHLC candles. Nothing here looks ahead: every value at index
`i` uses only candles [0..i]. That makes the same code safe in a backtest and
live.

Concepts implemented
--------------------
  swing highs / lows   fractal pivots (a bar higher/lower than its neighbours)
  market structure     HH+HL = uptrend, LH+LL = downtrend (BOS)
  liquidity sweep      price wicks through a prior swing then closes back inside
                       (a stop hunt / liquidity grab)
  order block          the last opposite candle before an impulsive move
  fair value gap       a 3-candle imbalance (a price void)
  premium/discount     where price sits inside the recent dealing range

Signals are alerts only — never a trade instruction.
"""
from __future__ import annotations

from typing import Optional

import indicators as ta


def swings(highs: list[float], lows: list[float], left: int = 2,
           right: int = 2) -> tuple[list[int], list[int]]:
    """Fractal swing points. A pivot needs `left`/`right` lower/higher bars.

    Returns (swing_high_indices, swing_low_indices), oldest -> newest.
    Only pivots confirmed by `right` bars are returned, so index <= len-1-right.
    """
    sh, sl = [], []
    n = len(highs)
    for i in range(left, n - right):
        if all(highs[i] > highs[i - j] for j in range(1, left + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, right + 1)):
            sh.append(i)
        if all(lows[i] < lows[i - j] for j in range(1, left + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, right + 1)):
            sl.append(i)
    return sh, sl


def structure(highs: list[float], lows: list[float], i: int,
              left: int = 2, right: int = 2, lookback: int = 60) -> str:
    """Market structure at `i`: 'up' (HH+HL), 'down' (LH+LL) or 'range'."""
    lo = max(0, i - lookback)
    sh, sl = swings(highs[lo:i + 1], lows[lo:i + 1], left, right)
    sh = [x + lo for x in sh]
    sl = [x + lo for x in sl]
    if len(sh) < 2 or len(sl) < 2:
        return "range"
    hh = highs[sh[-1]] > highs[sh[-2]]
    hl = lows[sl[-1]] > lows[sl[-2]]
    lh = highs[sh[-1]] < highs[sh[-2]]
    ll = lows[sl[-1]] < lows[sl[-2]]
    if hh and hl:
        return "up"
    if lh and ll:
        return "down"
    return "range"


def liquidity_sweep(highs: list[float], lows: list[float], closes: list[float],
                    i: int, lookback: int = 30) -> Optional[str]:
    """Did candle `i` sweep liquidity and reject?

    Bullish sweep: the low pierces a prior swing low but the close comes back
    above it (sellers trapped). Bearish: the high pierces a prior swing high but
    closes back below. Returns 'bull', 'bear' or None.
    """
    lo = max(0, i - lookback)
    if i - lo < 6:
        return None
    sh, sl = swings(highs[lo:i], lows[lo:i], 2, 2)
    sh = [x + lo for x in sh]
    sl = [x + lo for x in sl]
    if sl:
        prior_low = min(lows[x] for x in sl)
        if lows[i] < prior_low and closes[i] > prior_low:
            return "bull"
    if sh:
        prior_high = max(highs[x] for x in sh)
        if highs[i] > prior_high and closes[i] < prior_high:
            return "bear"
    return None


def order_block(opens: list[float], closes: list[float], highs: list[float],
                lows: list[float], i: int, lookback: int = 20,
                impulse_atr: float = 1.0) -> Optional[str]:
    """Last opposite candle before an impulsive move into `i`.

    Bullish: the most recent down candle immediately before a strong up leg.
    Bearish: the most recent up candle before a strong down leg.
    """
    lo = max(1, i - lookback)
    atr = ta.atr(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if a <= 0:
        return None
    leg = closes[i] - closes[i - 1]
    if leg > impulse_atr * a:              # strong up leg -> find the down candle
        for j in range(i - 1, lo - 1, -1):
            if closes[j] < opens[j]:
                return "bull"
    elif leg < -impulse_atr * a:           # strong down leg -> find the up candle
        for j in range(i - 1, lo - 1, -1):
            if closes[j] > opens[j]:
                return "bear"
    return None


def fair_value_gap(highs: list[float], lows: list[float], i: int) -> Optional[str]:
    """3-candle imbalance at `i` (candles i-2, i-1, i).

    Bullish FVG: candle i's low is above candle i-2's high (a gap up).
    Bearish FVG: candle i's high is below candle i-2's low.
    """
    if i < 2:
        return None
    if lows[i] > highs[i - 2]:
        return "bull"
    if highs[i] < lows[i - 2]:
        return "bear"
    return None


def dealing_range(highs: list[float], lows: list[float], i: int,
                  lookback: int = 40) -> tuple[float, float, float]:
    """Recent range (lo, hi, position 0..1) — premium above 0.5, discount below."""
    lo = max(0, i - lookback)
    hi_v = max(highs[lo:i + 1])
    lo_v = min(lows[lo:i + 1])
    if hi_v <= lo_v:
        return lo_v, hi_v, 0.5
    pos = (highs[i] - lo_v) / (hi_v - lo_v)
    return lo_v, hi_v, pos


def smc_vote(opens: list[float], closes: list[float], highs: list[float],
             lows: list[float], i: int) -> tuple[Optional[str], list[str]]:
    """One SMC read at `i`: 'CALL', 'PUT' or None, plus the reasons."""
    reasons: list[str] = []
    struct = structure(highs, lows, i)
    sweep = liquidity_sweep(highs, lows, closes, i)
    ob = order_block(opens, closes, highs, lows, i)
    fvg = fair_value_gap(highs, lows, i)
    _, _, pos = dealing_range(highs, lows, i)

    bull = bear = 0
    if sweep == "bull":
        bull += 2; reasons.append("bullish liquidity sweep")
    elif sweep == "bear":
        bear += 2; reasons.append("bearish liquidity sweep")
    if ob == "bull":
        bull += 1; reasons.append("bullish order block")
    elif ob == "bear":
        bear += 1; reasons.append("bearish order block")
    if fvg == "bull":
        bull += 1; reasons.append("bullish FVG")
    elif fvg == "bear":
        bear += 1; reasons.append("bearish FVG")
    if struct == "up":
        bull += 1; reasons.append("structure up")
    elif struct == "down":
        bear += 1; reasons.append("structure down")
    if pos < 0.35:
        bull += 1; reasons.append("discount zone")
    elif pos > 0.65:
        bear += 1; reasons.append("premium zone")

    if bull == bear:
        return None, reasons
    action = "CALL" if bull > bear else "PUT"
    return action, reasons
