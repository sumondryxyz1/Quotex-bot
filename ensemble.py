#!/usr/bin/env python3
"""Three-model confluence engine: SMC + trend + momentum.

Three independent "models" each vote CALL / PUT / abstain on the last closed
candle:

  trend     — EMA alignment (fast > slow > trend = up). This is also a HARD
              gate: the trade must point the same way as the trend. A counter-
              trend signal is dropped even if the other two agree.
  smc       — Smart Money Concepts: liquidity sweep, order block, fair value
              gap, market structure, premium/discount (see smc.py).
  momentum  — MACD histogram + RSI slope + stochastic turn.

A signal fires only when at least `min_agree` (default 2) of the three point the
same way AND that way matches the trend direction. That is the classic
confluence rule: one model can be wrong, two agreeing is the filter.

Implements the same `.evaluate(candles) -> Signal | None` interface as the other
engines, so backtest.py / edge.py / scan.py can compare it directly.

Signals are alerts only — never a trade instruction.

Usage
-----
    python ensemble.py EURUSD_otc 60            # live alerts
    python backtest.py --engine ensemble
    python scan.py 5 --engine ensemble
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import indicators as ta
import smc


@dataclass
class EnsembleStrategy:
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_period: int = 14
    min_agree: int = 2
    require_trend: bool = True     # hard trend-direction gate
    cooldown_candles: int = 5
    use_patterns: bool = False     # add chart-pattern vote as a 4th model

    def min_candles(self) -> int:
        return max(self.ema_trend, self.macd_slow + self.macd_signal,
                   self.stoch_period + 6, 70) + 2


@dataclass
class EnsembleSignal:
    time: int
    action: str                    # "CALL" | "PUT"
    price: float
    votes: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(1 for v in self.votes.values() if v == self.action)

    @property
    def confidence(self) -> float:
        return self.score / max(1, len(self.votes))


class EnsembleEngine:
    """Three voters, majority (>= min_agree) + trend-direction gate."""

    def __init__(self, strategy: Optional[EnsembleStrategy] = None):
        self.s = strategy or EnsembleStrategy()
        self._last_action: Optional[str] = None
        self._last_index: int = -10 ** 9

    def _trend_vote(self, closes, ema_f, ema_s, ema_t, i) -> tuple[Optional[str], str]:
        if ema_f[i] is None or ema_s[i] is None or ema_t[i] is None:
            return None, ""
        if ema_f[i] > ema_s[i] > ema_t[i]:
            return "CALL", "EMA stack up (fast>slow>trend)"
        if ema_f[i] < ema_s[i] < ema_t[i]:
            return "PUT", "EMA stack down (fast<slow<trend)"
        # partial alignment: still a lean, but weaker
        if ema_f[i] > ema_s[i] and closes[i] > ema_t[i]:
            return "CALL", "price>trend, fast>slow"
        if ema_f[i] < ema_s[i] and closes[i] < ema_t[i]:
            return "PUT", "price<trend, fast<slow"
        return None, "no clear trend"

    def _momentum_vote(self, closes, macd_line, sig_line, rsi, st_k, st_d, i):
        if macd_line[i] is None or sig_line[i] is None or rsi[i] is None:
            return None, ""
        hist = macd_line[i] - sig_line[i]
        reasons = []
        bull = bear = 0
        if hist > 0:
            bull += 1; reasons.append("MACD hist > 0")
        elif hist < 0:
            bear += 1; reasons.append("MACD hist < 0")
        if i >= 1 and rsi[i - 1] is not None:
            if rsi[i] > rsi[i - 1] and rsi[i] < 70:
                bull += 1; reasons.append("RSI rising")
            elif rsi[i] < rsi[i - 1] and rsi[i] > 30:
                bear += 1; reasons.append("RSI falling")
        if st_k[i] is not None and st_d[i] is not None:
            if ta.crossed_above(st_k, st_d, i) and st_k[i] < 60:
                bull += 1; reasons.append("Stoch cross up")
            elif ta.crossed_below(st_k, st_d, i) and st_k[i] > 40:
                bear += 1; reasons.append("Stoch cross down")
        if bull == bear:
            return None, ", ".join(reasons) or "momentum mixed"
        return ("CALL" if bull > bear else "PUT"), ", ".join(reasons)

    def evaluate(self, candles: list[list]) -> Optional[EnsembleSignal]:
        s = self.s
        if len(candles) < s.min_candles():
            return None
        i = len(candles) - 1
        if (self._last_action is not None
                and i - self._last_index < s.cooldown_candles):
            return None

        opens = [c[1] for c in candles]
        closes = [c[2] for c in candles]
        highs = [c[3] for c in candles]
        lows = [c[4] for c in candles]

        ema_f = ta.ema(closes, s.ema_fast)
        ema_s = ta.ema(closes, s.ema_slow)
        ema_t = ta.ema(closes, s.ema_trend)
        macd_line, sig_line, _ = ta.macd(closes, s.macd_fast, s.macd_slow, s.macd_signal)
        rsi = ta.rsi(closes, s.rsi_period)
        st_k, st_d = ta.stochastic(highs, lows, closes, s.stoch_period, 3, 3)

        t_vote, t_reason = self._trend_vote(closes, ema_f, ema_s, ema_t, i)
        m_vote, m_reason = self._momentum_vote(closes, macd_line, sig_line, rsi,
                                               st_k, st_d, i)
        s_vote, s_reasons = smc.smc_vote(opens, closes, highs, lows, i)

        votes = {"trend": t_vote, "smc": s_vote, "momentum": m_vote}
        reasons_by_model = {"trend": t_reason, "smc": ", ".join(s_reasons),
                            "momentum": m_reason}
        if s.use_patterns:
            import patterns as _p
            p_vote, p_reasons = _p.pattern_vote(opens, closes, highs, lows, i)
            votes["pattern"] = p_vote
            reasons_by_model["pattern"] = ", ".join(p_reasons)

        need = s.min_agree if not s.use_patterns else max(2, s.min_agree)
        bull = sum(1 for v in votes.values() if v == "CALL")
        bear = sum(1 for v in votes.values() if v == "PUT")
        if max(bull, bear) < need:
            return None
        if bull == bear:
            return None
        action = "CALL" if bull > bear else "PUT"

        # hard trend-direction gate: never trade against the trend
        if s.require_trend and t_vote != action:
            return None

        reasons = []
        for model, v in votes.items():
            if v == action:
                reasons.append(f"{model}: {reasons_by_model[model]}")

        self._last_action = action
        self._last_index = i
        return EnsembleSignal(time=candles[i][0], action=action, price=closes[i],
                              votes=votes, reasons=reasons)


def format_signal(sig: EnsembleSignal, asset: str = "") -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sig.time))
    head = f"{'🟢' if sig.action == 'CALL' else '🔴'} {sig.action} {asset} @ {sig.price}"
    vote_line = "  ".join(f"{k}={'C' if v == 'CALL' else 'P' if v == 'PUT' else '-'}"
                          for k, v in sig.votes.items())
    return (f"\n{'=' * 62}\n{head}\n"
            f"  candle close : {ts}\n"
            f"  votes        : {vote_line}  ({sig.score}/{len(sig.votes)} agree)\n"
            f"  reasons      : {' | '.join(sig.reasons)}\n"
            f"{'=' * 62}")


def main():
    ap = argparse.ArgumentParser(description="SMC + trend + momentum confluence bot")
    ap.add_argument("asset", nargs="?", default="EURUSD_otc")
    ap.add_argument("period", nargs="?", type=int, default=60)
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--no-trend-gate", action="store_true")
    ap.add_argument("--patterns", action="store_true",
                    help="add chart patterns (H&S, double top/bottom, S/R, flag) as a 4th vote")
    ap.add_argument("--ssid", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from market_qx_candles import CandleBuilder, MarketQX

    strat = EnsembleStrategy(min_agree=args.min_agree,
                             require_trend=not args.no_trend_gate,
                             use_patterns=args.patterns)
    engine = EnsembleEngine(strat)
    history: list[list] = []
    stats = {"candles": 0, "signals": 0}

    def on_closed(candle: list):
        history.append(candle)
        if len(history) > 1000:
            del history[:len(history) - 1000]
        stats["candles"] += 1
        sig = engine.evaluate(history)
        if sig:
            stats["signals"] += 1
            print(format_signal(sig, args.asset), flush=True)

    builder = CandleBuilder(args.period, on_closed=on_closed)
    qx = MarketQX(ssid=args.ssid, verbose=False).connect(wait_instruments=75)
    print(f"asset {args.asset} {args.period}s | 3 models (smc/trend/momentum), "
          f"need {strat.min_agree}/3 agree"
          f"{', trend gate ON' if strat.require_trend else ''}")

    def on_tick(asset, ts, price):
        if asset == args.asset:
            builder.add(ts, price)

    qx.on_tick(on_tick)
    qx.follow(args.asset, args.period)
    last = 0
    try:
        while True:
            time.sleep(2)
            if not args.quiet and time.time() - last >= 15:
                last = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] candles={stats['candles']} "
                      f"signals={stats['signals']}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        qx.close()


if __name__ == "__main__":
    main()
