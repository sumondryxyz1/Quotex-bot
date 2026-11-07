"""Technical indicators implemented in pure Python (no numpy/pandas needed).

Every function takes plain lists of floats (oldest -> newest) and returns a list
of the same length, left-padded with None until enough data is available.
"""
from __future__ import annotations

from typing import Optional


def sma(values: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    if period <= 0:
        return out
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: list[float], period: int) -> list[Optional[float]]:
    """Exponential moving average, seeded with the SMA of the first `period` values."""
    out: list[Optional[float]] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values: list[float], period: int = 14) -> list[Optional[float]]:
    """Wilder's RSI."""
    out: list[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: list[Optional[float]] = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    valid = [v for v in macd_line if v is not None]
    sig_valid = ema(valid, signal)
    signal_line: list[Optional[float]] = [None] * len(macd_line)
    offset = len(macd_line) - len(valid)
    for i, v in enumerate(sig_valid):
        signal_line[offset + i] = v
    hist: list[Optional[float]] = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


def bollinger(values: list[float], period: int = 20, mult: float = 2.0):
    """Returns (upper, middle, lower)."""
    mid = sma(values, period)
    upper: list[Optional[float]] = [None] * len(values)
    lower: list[Optional[float]] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        m = mid[i]
        if m is None:
            continue
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        upper[i] = m + mult * sd
        lower[i] = m - mult * sd
    return upper, mid, lower


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> list[Optional[float]]:
    """Average True Range (Wilder smoothing)."""
    n = len(closes)
    out: list[Optional[float]] = [None] * n
    if n <= period:
        return out
    trs = [0.0] * n
    for i in range(1, n):
        trs[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    prev = sum(trs[1:period + 1]) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


def stochastic(highs: list[float], lows: list[float], closes: list[float],
               period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """Returns (%K, %D)."""
    n = len(closes)
    raw_k: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        raw_k[i] = 50.0 if hh == ll else (closes[i] - ll) / (hh - ll) * 100
    k_valid = sma([v for v in raw_k if v is not None], smooth_k)
    k: list[Optional[float]] = [None] * n
    off = n - len(k_valid)
    for i, v in enumerate(k_valid):
        k[off + i] = v
    d_valid = sma([v for v in k if v is not None], smooth_d)
    d: list[Optional[float]] = [None] * n
    off = n - len(d_valid)
    for i, v in enumerate(d_valid):
        d[off + i] = v
    return k, d


def adx(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14):
    """Average Directional Index with +DI / -DI (Wilder smoothing).

    Returns (adx, plus_di, minus_di). ADX measures trend *strength* regardless of
    direction: <20 is a choppy/range regime (mean-reversion works), >25 is a
    trending regime (momentum/breakout works). This is the regime gate the smart
    engine uses to pick the right playbook.
    """
    n = len(closes)
    adx_out: list[Optional[float]] = [None] * n
    pdi_out: list[Optional[float]] = [None] * n
    mdi_out: list[Optional[float]] = [None] * n
    if n <= 2 * period:
        return adx_out, pdi_out, mdi_out

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))

    # Wilder smoothing seeded with the first `period` sums
    atr_s = sum(tr[1:period + 1])
    pdm_s = sum(plus_dm[1:period + 1])
    mdm_s = sum(minus_dm[1:period + 1])
    dx: list[Optional[float]] = [None] * n

    def di_pair(atr_v, pdm_v, mdm_v):
        if atr_v == 0:
            return 0.0, 0.0, 0.0
        pdi = 100 * pdm_v / atr_v
        mdi = 100 * mdm_v / atr_v
        denom = pdi + mdi
        return pdi, mdi, (100 * abs(pdi - mdi) / denom if denom else 0.0)

    pdi, mdi, d = di_pair(atr_s, pdm_s, mdm_s)
    pdi_out[period] = pdi
    mdi_out[period] = mdi
    dx[period] = d
    for i in range(period + 1, n):
        atr_s = atr_s - atr_s / period + tr[i]
        pdm_s = pdm_s - pdm_s / period + plus_dm[i]
        mdm_s = mdm_s - mdm_s / period + minus_dm[i]
        pdi, mdi, d = di_pair(atr_s, pdm_s, mdm_s)
        pdi_out[i] = pdi
        mdi_out[i] = mdi
        dx[i] = d

    # ADX = Wilder-smoothed DX
    first = 2 * period
    valid = [v for v in dx[period:first] if v is not None]
    if not valid:
        return adx_out, pdi_out, mdi_out
    prev = sum(valid) / len(valid)
    adx_out[first - 1] = prev
    for i in range(first, n):
        if dx[i] is None:
            continue
        prev = (prev * (period - 1) + dx[i]) / period
        adx_out[i] = prev
    return adx_out, pdi_out, mdi_out


def crossed_above(a: list[Optional[float]], b: list[Optional[float]], i: int) -> bool:
    """True if series a crossed above series b at index i."""
    if i < 1:
        return False
    a0, a1, b0, b1 = a[i - 1], a[i], b[i - 1], b[i]
    if None in (a0, a1, b0, b1):
        return False
    return a0 <= b0 and a1 > b1


def crossed_below(a: list[Optional[float]], b: list[Optional[float]], i: int) -> bool:
    if i < 1:
        return False
    a0, a1, b0, b1 = a[i - 1], a[i], b[i - 1], b[i]
    if None in (a0, a1, b0, b1):
        return False
    return a0 >= b0 and a1 < b1
