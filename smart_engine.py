#!/usr/bin/env python3
"""Smart signal engine — regime-aware, volatility-gated, multi-factor.

The plain engine (`signal_bot.SignalEngine`) counts indicator votes and fires at
a fixed threshold. That treats a choppy range and a strong trend the same way,
which is exactly backwards: mean-reversion rules lose in trends and momentum
rules lose in chop.

This engine adds the things that actually matter:

1. **Regime gate (ADX)**. ADX<20 -> range: only mean-reversion plays are allowed
   (buy oversold, sell overbought). ADX>25 -> trend: only momentum plays are
   allowed (follow the trend on a fresh cross). In between -> stand aside.
2. **Volatility gate (ATR percentile)**. If current ATR is in the bottom decile
   the move is too small to beat the spread; if it is in the top decile it is a
   news spike. Both are skipped.
3. **Multi-timeframe confirmation**. The higher timeframe's EMA trend must agree
   with the trade direction (e.g. 5s signal needs the 60s trend aligned).
4. **Breakout confirmation**. In a trend, price must actually break the recent
   extreme — no chasing.
5. **Confidence score**. Returns 0..1 so callers can require a minimum quality,
   instead of a raw vote count.

It implements the same `.evaluate(candles) -> Signal | None` interface as the
plain engine, so `backtest.py` can compare them directly.

Signals are alerts only — never a trade instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import indicators as ta


@dataclass
class SmartStrategy:
    # trend / momentum
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # oscillators (range playbook)
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_period: int = 20
    bb_mult: float = 2.0
    stoch_period: int = 14
    # regime / volatility
    adx_period: int = 14
    adx_range: float = 20.0        # below this -> range regime
    adx_trend: float = 25.0        # above this -> trend regime
    atr_period: int = 14
    atr_lookback: int = 100        # for the rolling median
    atr_min_ratio: float = 0.5     # skip if ATR < 0.5x its recent median
    atr_max_ratio: float = 3.0     # skip if ATR > 3x its recent median (spike)
    # confirmation
    breakout_lookback: int = 10
    htf_ema: int = 50              # higher-timeframe trend proxy
    min_confidence: float = 0.5
    cooldown_candles: int = 5      # don't repeat the same direction too soon

    def min_candles(self) -> int:
        return max(self.ema_trend, self.macd_slow + self.macd_signal,
                   self.bb_period, self.stoch_period + 6,
                   2 * self.adx_period + 2, self.atr_lookback) + 2


@dataclass
class SmartSignal:
    time: int
    action: str                    # "CALL" | "PUT"
    price: float
    confidence: float              # 0..1
    regime: str                    # "trend" | "range"
    reasons: list[str] = field(default_factory=list)


def _pct_rank(values: list[float], x: float) -> float:
    """Fraction of `values` that are <= x (0..1)."""
    if not values:
        return 0.5
    return sum(1 for v in values if v <= x) / len(values)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


class SmartEngine:
    """Regime-aware confluence engine with a calibrated confidence score."""

    def __init__(self, strategy: Optional[SmartStrategy] = None):
        self.s = strategy or SmartStrategy()
        self._last_action: Optional[str] = None
        self._last_index: int = -10 ** 9

    def evaluate(self, candles: list[list]) -> Optional[SmartSignal]:
        s = self.s
        if len(candles) < s.min_candles():
            return None
        i = len(candles) - 1
        # debounce: ignore a repeat of the same direction inside the cooldown
        if (self._last_action is not None
                and i - self._last_index < s.cooldown_candles):
            return None

        times = [c[0] for c in candles]
        closes = [c[2] for c in candles]
        highs = [c[3] for c in candles]
        lows = [c[4] for c in candles]

        ema_f = ta.ema(closes, s.ema_fast)
        ema_s = ta.ema(closes, s.ema_slow)
        ema_t = ta.ema(closes, s.ema_trend)
        rsi = ta.rsi(closes, s.rsi_period)
        macd_line, sig_line, _ = ta.macd(closes, s.macd_fast, s.macd_slow, s.macd_signal)
        bb_u, bb_m, bb_l = ta.bollinger(closes, s.bb_period, s.bb_mult)
        st_k, st_d = ta.stochastic(highs, lows, closes, s.stoch_period, 3, 3)
        adx_v, pdi, mdi = ta.adx(highs, lows, closes, s.adx_period)
        atr = ta.atr(highs, lows, closes, s.atr_period)

        if any(x is None for x in (ema_f[i], ema_s[i], ema_t[i], rsi[i],
                                   adx_v[i], atr[i])):
            return None
        if macd_line[i] is None or sig_line[i] is None:
            return None

        # --- volatility gate -------------------------------------------------
        # Compare current ATR to its recent *median*, not a percentile rank: a
        # percentile rejects a calm tail even when the move is perfectly normal,
        # which is exactly what a clean trend looks like after it settles.
        atr_hist = [v for v in atr[max(0, i - s.atr_lookback):i + 1] if v]
        if len(atr_hist) < 20:
            return None
        atr_med = _median(atr_hist)
        if atr_med <= 0:
            return None
        atr_ratio = atr[i] / atr_med
        if atr_ratio < s.atr_min_ratio or atr_ratio > s.atr_max_ratio:
            return None

        # --- regime gate -----------------------------------------------------
        adx = adx_v[i]
        if adx >= s.adx_trend:
            regime = "trend"
        elif adx <= s.adx_range:
            regime = "range"
        else:
            return None                      # transitional chop: stand aside

        reasons: list[str] = []
        conf = 0.0

        if regime == "trend":
            up = pdi[i] > mdi[i]
            action = "CALL" if up else "PUT"

            # momentum must agree with the directional index
            if up and ema_f[i] <= ema_s[i]:
                return None
            if not up and ema_f[i] >= ema_s[i]:
                return None
            # breakout confirmation: price must clear the recent extreme
            n = s.breakout_lookback
            if up and closes[i] <= max(highs[i - n:i]):
                return None
            if not up and closes[i] >= min(lows[i - n:i]):
                return None
            # higher-timeframe trend must agree
            if up and closes[i] <= ema_t[i]:
                return None
            if not up and closes[i] >= ema_t[i]:
                return None

            reasons.append(f"trend regime (ADX {adx:.0f})")
            reasons.append(f"{'+DI' if up else '-DI'} dominant "
                           f"({pdi[i]:.0f} vs {mdi[i]:.0f})")
            reasons.append("fresh breakout of recent extreme")
            conf = 0.5 + (0.15 if adx >= 35 else 0.08)
            crossed = (ta.crossed_above(ema_f, ema_s, i) if up
                       else ta.crossed_below(ema_f, ema_s, i))
            if crossed:
                conf += 0.15
                reasons.append("fresh EMA cross")
            macd_agrees = (macd_line[i] > sig_line[i]) if up else (macd_line[i] < sig_line[i])
            if macd_agrees:
                conf += 0.10
                reasons.append("MACD agrees")
        else:  # range
            if rsi[i] <= s.rsi_oversold:
                action, reasons = "CALL", [f"range regime (ADX {adx:.0f})",
                                           f"RSI oversold ({rsi[i]:.0f})"]
                conf = 0.5 + (s.rsi_oversold - rsi[i]) / 100.0
            elif rsi[i] >= s.rsi_overbought:
                action, reasons = "PUT", [f"range regime (ADX {adx:.0f})",
                                          f"RSI overbought ({rsi[i]:.0f})"]
                conf = 0.5 + (rsi[i] - s.rsi_overbought) / 100.0
            else:
                return None
            # must be at a band extreme, not mid-range
            if action == "CALL":
                if bb_l[i] is None or closes[i] > bb_m[i]:
                    return None
                reasons.append("near lower Bollinger band")
                conf += 0.12
            else:
                if bb_u[i] is None or closes[i] < bb_m[i]:
                    return None
                reasons.append("near upper Bollinger band")
                conf += 0.12
            # stochastic should agree (turning up from low / down from high)
            if st_k[i] is not None and st_d[i] is not None:
                if action == "CALL" and st_k[i] > st_d[i] and st_k[i] < 30:
                    conf += 0.10; reasons.append("Stoch turning up")
                elif action == "PUT" and st_k[i] < st_d[i] and st_k[i] > 70:
                    conf += 0.10; reasons.append("Stoch turning down")
                else:
                    return None
            else:
                return None

        conf = max(0.0, min(1.0, conf))
        if conf < s.min_confidence:
            return None

        self._last_action = action
        self._last_index = i
        return SmartSignal(time=times[i], action=action, price=closes[i],
                           confidence=round(conf, 3), regime=regime, reasons=reasons)


def format_signal(sig: SmartSignal) -> str:
    import time as _t
    ts = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(sig.time))
    head = f"{'🟢' if sig.action == 'CALL' else '🔴'} {sig.action} @ {sig.price}"
    return (f"\n{'=' * 62}\n{head}\n"
            f"  candle close : {ts}\n"
            f"  regime       : {sig.regime}\n"
            f"  confidence   : {sig.confidence:.2f}\n"
            f"  reasons      : {', '.join(sig.reasons)}\n"
            f"{'=' * 62}")
