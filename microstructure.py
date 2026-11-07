#!/usr/bin/env python3
"""Microstructure: short-horizon behaviour from ticks and candles.

Extends `tickflow.py` (which already extracted per-candle order-flow features)
with velocity, acceleration, bursts and a tick-imbalance proxy. Everything is a
*measured* property of the price/ticks we actually receive — we never present a
derived quantity as if it were real order-book depth, which this feed does not
provide.

Honest note: an earlier audit found the tick direction flag carries no
predictive information (≈49.7%, z≈−2) and tick-sign autocorrelation ≈ 0. This
module exists to *re-measure* those claims and to expose the extra dimensions
(bursts, velocity) that were never tested — not to assume they will work.

Pure functions; no look-ahead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import indicators as ta
from tickflow import candle_features


@dataclass
class MicroState:
    tick_count: int = 0
    up_ratio: float = 0.5
    dir_ratio: float = 0.5
    realised_vol: float = 0.0
    velocity: float = 0.0        # signed price change per tick, ATR-normalised
    acceleration: float = 0.0    # change in velocity vs the previous window
    burst: bool = False          # activity spike vs recent median
    imbalance: float = 0.0       # up-move share - down-move share, [-1, 1]

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def velocity(prices: list[float]) -> float:
    """Signed price change per tick, normalised by the tick range."""
    if len(prices) < 2:
        return 0.0
    rng = (max(prices) - min(prices)) or 1e-12
    return (prices[-1] - prices[0]) / rng


def tick_imbalance(prices: list[float]) -> float:
    """(up-move share − down-move share) in [-1, 1]. A one-sided pressure proxy."""
    if len(prices) < 2:
        return 0.0
    ups = downs = 0
    for k in range(1, len(prices)):
        d = prices[k] - prices[k - 1]
        if d > 0:
            ups += 1
        elif d < 0:
            downs += 1
    tot = ups + downs
    return round((ups - downs) / tot, 4) if tot else 0.0


def burst_detection(counts: list[int], window: int = 20,
                    mult: float = 3.0) -> bool:
    """Is the latest tick-count an activity burst vs the recent median?"""
    if len(counts) < window + 1:
        return False
    recent = sorted(counts[-window - 1:-1])
    med = recent[len(recent) // 2]
    return counts[-1] > max(med * mult, med + 3)


def analyse(ticks: list[tuple], opens: list[float], closes: list[float],
            highs: list[float], lows: list[float], t0: float, period: int,
            i: int, counts_history: Optional[list[int]] = None) -> MicroState:
    """Microstructure read for the candle starting at `t0` (candle index `i`)."""
    feat = candle_features(ticks, t0, period)
    st = MicroState()
    atr = ta.atr(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if feat is not None:
        win = [t for t in ticks if t0 <= t[0] < t0 + period]
        prices = [t[1] for t in win]
        st.tick_count = feat["n_ticks"]
        st.up_ratio = round(feat["up_ratio"], 4)
        st.dir_ratio = round(feat["dir_ratio"], 4)
        st.realised_vol = round(feat["volatility"], 8)
        st.velocity = round(velocity(prices), 4)
        st.imbalance = tick_imbalance(prices)
        if a > 0:
            st.acceleration = round((closes[i] - closes[i - 1]) / a
                                    - (closes[i - 1] - closes[i - 2]) / a, 4) \
                if i >= 2 else 0.0
    if counts_history is not None:
        st.burst = burst_detection(counts_history + [st.tick_count])
    return st


def tick_autocorr(prices: list[float], lag: int = 1) -> float:
    """Autocorrelation of tick returns at `lag` (momentum/reversal in the tape)."""
    if len(prices) < lag + 3:
        return 0.0
    r = [prices[k] - prices[k - 1] for k in range(1, len(prices))]
    n = len(r) - lag
    if n < 2:
        return 0.0
    m = sum(r) / len(r)
    num = sum((r[k] - m) * (r[k + lag] - m) for k in range(n))
    den = sum((x - m) ** 2 for x in r)
    return round(num / den, 4) if den else 0.0
