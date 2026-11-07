"""Feature engineering for the up/down prediction model.

Turns a list of OHLC candles into a feature vector. Everything is computed from
information available at the close of the last candle — no look-ahead — so the
same function is safe to call live and during training.
"""
from __future__ import annotations

import math
from typing import Optional

import indicators as ta

# Feature order is fixed; the trained model relies on it.
FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "body", "upper_wick", "lower_wick",
    "ema_fast_slow", "ema_slow_trend", "price_ema_fast", "price_ema_trend",
    "ema_slope_fast", "ema_slope_slow",
    "rsi", "rsi_slope",
    "macd_hist", "macd_hist_prev",
    "bb_pos", "bb_width",
    "atr_pct",
    "stoch_k", "stoch_kd",
    "consec_up", "consec_down",
    "hour_sin", "hour_cos",
]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_features(candles: list[list], i: Optional[int] = None) -> Optional[dict]:
    """Feature dict for candle index `i` (default: last). None if not enough data."""
    if i is None:
        i = len(candles) - 1
    if i < 50:  # EMA(50) is the longest lookback; everything else needs less
        return None

    window = candles[: i + 1]
    t = [c[0] for c in window]
    o = [c[1] for c in window]
    c = [c[2] for c in window]
    h = [c[3] for c in window]
    lo = [c[4] for c in window]

    ema_f = ta.ema(c, 9)
    ema_s = ta.ema(c, 21)
    ema_t = ta.ema(c, 50)
    rsi = ta.rsi(c, 14)
    macd_line, sig_line, hist = ta.macd(c, 12, 26, 9)
    bb_u, bb_m, bb_l = ta.bollinger(c, 20, 2.0)
    atr = ta.atr(h, lo, c, 14)
    st_k, st_d = ta.stochastic(h, lo, c, 14, 3, 3)

    if ema_f[i] is None or ema_s[i] is None or ema_t[i] is None or rsi[i] is None:
        return None

    price = c[i]
    rng = h[i] - lo[i]

    # consecutive same-direction candles
    cu = cd = 0
    for j in range(i, 0, -1):
        if c[j] > o[j]:
            if cd:
                break
            cu += 1
        elif c[j] < o[j]:
            if cu:
                break
            cd += 1
        else:
            break

    hour = (t[i] // 3600) % 24

    return {
        "ret_1": _safe_div(price - c[i - 1], c[i - 1]),
        "ret_3": _safe_div(price - c[i - 3], c[i - 3]),
        "ret_5": _safe_div(price - c[i - 5], c[i - 5]),
        "ret_10": _safe_div(price - c[i - 10], c[i - 10]),
        "body": _safe_div(c[i] - o[i], rng) if rng else 0.0,
        "upper_wick": _safe_div(h[i] - max(o[i], c[i]), rng) if rng else 0.0,
        "lower_wick": _safe_div(min(o[i], c[i]) - lo[i], rng) if rng else 0.0,
        "ema_fast_slow": _safe_div(ema_f[i] - ema_s[i], price),
        "ema_slow_trend": _safe_div(ema_s[i] - ema_t[i], price),
        "price_ema_fast": _safe_div(price - ema_f[i], price),
        "price_ema_trend": _safe_div(price - ema_t[i], price),
        "ema_slope_fast": _safe_div(ema_f[i] - ema_f[i - 1], price) if ema_f[i - 1] else 0.0,
        "ema_slope_slow": _safe_div(ema_s[i] - ema_s[i - 1], price) if ema_s[i - 1] else 0.0,
        "rsi": rsi[i] / 100.0,
        "rsi_slope": _safe_div((rsi[i] or 50) - (rsi[i - 1] or 50), 100.0),
        "macd_hist": _safe_div(hist[i] or 0.0, price),
        "macd_hist_prev": _safe_div(hist[i - 1] or 0.0, price),
        "bb_pos": _safe_div(price - bb_l[i], bb_u[i] - bb_l[i]) if bb_u[i] and bb_l[i] and bb_u[i] != bb_l[i] else 0.5,
        "bb_width": _safe_div(bb_u[i] - bb_l[i], price) if bb_u[i] and bb_l[i] else 0.0,
        "atr_pct": _safe_div(atr[i] or 0.0, price),
        "stoch_k": (st_k[i] or 50) / 100.0,
        "stoch_kd": _safe_div((st_k[i] or 50) - (st_d[i] or 50), 100.0),
        "consec_up": min(cu, 5),
        "consec_down": min(cd, 5),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }


def features_to_vector(feat: dict) -> list[float]:
    return [float(feat[k]) for k in FEATURES]
