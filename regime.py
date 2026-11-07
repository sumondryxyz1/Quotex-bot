#!/usr/bin/env python3
"""Market regime: what kind of market is this right now?

The smart engine only used ADX. Regime is multi-dimensional, and a strategy that
works in a trend is wrong in a range. This module classifies each candle's
context so a strategy can be *conditioned* on it (and switched off when the
regime does not suit it).

Dimensions
----------
  trend      strong-up / weak-up / sideways / weak-down / strong-down
             from ADX, EMA slope and Kaufman efficiency ratio.
  volatility high / normal / low from ATR percentile within the lookback.
  persistence  variance ratio (Lo-MacKay): >1 trending, <1 mean-reverting, ~1
               a random walk. A Hurst exponent (R/S) gives the same read.
  session      UTC session bucket, for real (non-OTC) pairs.

Pure and causal: every value at index `i` uses only candles [0..i].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import indicators as ta

STRONG_UP, WEAK_UP, SIDEWAYS, WEAK_DOWN, STRONG_DOWN = (
    "strong_up", "weak_up", "sideways", "weak_down", "strong_down")


@dataclass
class Regime:
    trend: str = SIDEWAYS
    adx: float = 0.0
    efficiency: float = 0.0
    volatility: str = "normal"
    atr_pct: float = 0.0
    variance_ratio: float = 1.0
    hurst: float = 0.5
    session: str = ""
    defined: bool = True     # False in transitional / ambiguous conditions

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def efficiency_ratio(closes: list[float], i: int, lookback: int = 20) -> float:
    """Kaufman efficiency: net move / sum of absolute moves. 1 = clean trend."""
    lo = max(0, i - lookback)
    if i - lo < 2:
        return 0.0
    net = abs(closes[i] - closes[lo])
    path = sum(abs(closes[k] - closes[k - 1]) for k in range(lo + 1, i + 1))
    return round(net / path, 4) if path else 0.0


def variance_ratio(closes: list[float], i: int, lookback: int = 60,
                    q: int = 5) -> float:
    """Lo-MacKay variance ratio: Var(q-step) / (q * Var(1-step)).

    > 1 trending (positive autocorrelation), < 1 mean-reverting, ~1 random walk.
    """
    lo = max(q + 1, i - lookback)
    if i - lo < q * 2:
        return 1.0
    r1 = [closes[k] - closes[k - 1] for k in range(lo + 1, i + 1)]
    rq = [closes[k] - closes[k - q] for k in range(lo + q, i + 1)]
    if len(r1) < 2 or len(rq) < 2:
        return 1.0
    m1 = sum(r1) / len(r1)
    v1 = sum((r - m1) ** 2 for r in r1) / (len(r1) - 1)
    mq = sum(rq) / len(rq)
    vq = sum((r - mq) ** 2 for r in rq) / (len(rq) - 1)
    if v1 <= 0:
        return 1.0
    return round(vq / (q * v1), 4)


def hurst(closes: list[float], i: int, lookback: int = 100) -> float:
    """Hurst exponent via rescaled range (R/S). >0.5 trending, <0.5 reverting."""
    lo = max(1, i - lookback)
    seg = closes[lo:i + 1]
    n = len(seg)
    if n < 20:
        return 0.5
    mean = sum(seg) / n
    dev = [s - mean for s in seg]
    cum = []
    acc = 0.0
    for d in dev:
        acc += d
        cum.append(acc)
    r = max(cum) - min(cum)
    s = math.sqrt(sum(d * d for d in dev) / n)
    if s <= 0 or r <= 0:
        return 0.5
    return round(math.log(r / s) / math.log(n), 4)


def volatility_bucket(highs: list[float], lows: list[float], closes: list[float],
                      i: int, lookback: int = 100) -> tuple[str, float]:
    """ATR percentile within the lookback -> high/normal/low, plus ATR% of price."""
    atr = ta.atr(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)
    if atr[i] is None or closes[i] == 0:
        return "normal", 0.0
    lo = max(14, i - lookback)
    vals = [atr[k] for k in range(lo, i + 1) if atr[k] is not None]
    if len(vals) < 10:
        return "normal", round(atr[i] / closes[i], 6)
    cur = atr[i]
    pct = sum(1 for v in vals if v <= cur) / len(vals)
    bucket = "high" if pct >= 0.75 else "low" if pct <= 0.25 else "normal"
    return bucket, round(cur / closes[i], 6)


def session_of(t: int) -> str:
    h = (t // 3600) % 24
    if 0 <= h < 8:
        return "asian"
    if 8 <= h < 13:
        return "london"
    if 13 <= h < 21:
        return "newyork"
    return "late"


def classify(opens: list[float], closes: list[float], highs: list[float],
             lows: list[float], times: list[int], i: int,
             lookback: int = 100) -> Regime:
    """Full regime read at `i`."""
    if i < 55:
        return Regime(defined=False)
    adx_line, _, _ = ta.adx(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)
    a = adx_line[i] if adx_line and adx_line[i] is not None else 0.0
    er = efficiency_ratio(closes, i, 20)
    ema_f = ta.ema(closes[:i + 1], 9)
    ema_s = ta.ema(closes[:i + 1], 21)
    slope = 0.0
    if ema_f[i] is not None and ema_s[i] is not None and closes[i]:
        slope = (ema_f[i] - ema_s[i]) / closes[i]
    vr = variance_ratio(closes, i, lookback)
    hu = hurst(closes, i, lookback)
    vol, atr_pct = volatility_bucket(highs, lows, closes, i, lookback)
    sess = session_of(times[i])

    trend = SIDEWAYS
    if a >= 25 and er >= 0.4 and abs(slope) > 0:
        trend = STRONG_UP if slope > 0 else STRONG_DOWN
    elif a >= 18 and er >= 0.25 and abs(slope) > 0:
        trend = WEAK_UP if slope > 0 else WEAK_DOWN

    # 'defined' = the regime is not transitional. A weak ADX with a middling
    # efficiency ratio is exactly the chop where signals should be suppressed.
    defined = not (14 < a < 22 and 0.2 < er < 0.35)
    return Regime(trend=trend, adx=round(a, 2), efficiency=er,
                  volatility=vol, atr_pct=atr_pct, variance_ratio=vr,
                  hurst=hu, session=sess, defined=defined)


def trend_following_ok(regime: Regime) -> bool:
    """A trend strategy is only appropriate in a real trend."""
    return regime.trend in (STRONG_UP, STRONG_DOWN) and regime.defined


def mean_reversion_ok(regime: Regime) -> bool:
    """Mean reversion suits a range or a mean-reverting (Hurst<0.5) market."""
    return (regime.trend == SIDEWAYS or regime.variance_ratio < 0.9
            or regime.hurst < 0.45)


def regime_series(opens: list[float], closes: list[float], highs: list[float],
                  lows: list[float], times: list[int], lookback: int = 100) -> dict:
    """Per-index regime in one pass (for feature building).

    Returns arrays: trend[], defined[], volatility[], adx[], efficiency[],
    variance_ratio[], hurst[]. Uses the same rules as `classify`; the caller can
    index any candle without recomputing the past.
    """
    n = len(closes)
    trend_arr = [SIDEWAYS] * n
    defined_arr = [False] * n
    vol_arr = ["normal"] * n
    adx_arr = [0.0] * n
    er_arr = [0.0] * n
    vr_arr = [1.0] * n
    hu_arr = [0.5] * n
    adx_line, _, _ = ta.adx(highs, lows, closes, 14)
    ema_f = ta.ema(closes, 9)
    ema_s = ta.ema(closes, 21)
    atr = ta.atr(highs, lows, closes, 14)
    for i in range(n):
        if i < 55:
            continue
        a = adx_line[i] if adx_line[i] is not None else 0.0
        er = efficiency_ratio(closes, i, 20)
        slope = 0.0
        if ema_f[i] is not None and ema_s[i] is not None and closes[i]:
            slope = (ema_f[i] - ema_s[i]) / closes[i]
        trend = SIDEWAYS
        if a >= 25 and er >= 0.4 and abs(slope) > 0:
            trend = STRONG_UP if slope > 0 else STRONG_DOWN
        elif a >= 18 and er >= 0.25 and abs(slope) > 0:
            trend = WEAK_UP if slope > 0 else WEAK_DOWN
        vol = "normal"
        if atr[i] is not None:
            lo = max(14, i - lookback)
            vals = [atr[k] for k in range(lo, i + 1) if atr[k] is not None]
            if len(vals) >= 10:
                pct = sum(1 for v in vals if v <= atr[i]) / len(vals)
                vol = "high" if pct >= 0.75 else "low" if pct <= 0.25 else "normal"
        trend_arr[i] = trend
        defined_arr[i] = not (14 < a < 22 and 0.2 < er < 0.35)
        vol_arr[i] = vol
        adx_arr[i] = round(a, 2)
        er_arr[i] = er
        vr_arr[i] = variance_ratio(closes, i, lookback)
        hu_arr[i] = hurst(closes, i, lookback)
    return {"trend": trend_arr, "defined": defined_arr, "volatility": vol_arr,
            "adx": adx_arr, "efficiency": er_arr,
            "variance_ratio": vr_arr, "hurst": hu_arr}
