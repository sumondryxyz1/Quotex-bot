#!/usr/bin/env python3
"""Signal bot for market-qx.info — EMA + indicator confluence.

Consumes the live candle stream from `market_qx_candles.py`, computes a set of
indicators on *closed* candles, and emits a CALL / PUT signal only when several
independent indicators agree.

Strategy (each rule votes, weighted):

  trend
    +2  EMA(fast) above EMA(slow)              -> bullish
    +1  price above EMA(trend)                 -> bullish
  momentum
    +2  EMA(fast) crossed EMA(slow)            -> fresh cross (strongest)
    +2  MACD line crossed signal line          -> momentum shift
  oscillator
    +1  RSI below oversold (mean-reversion up)
    +1  RSI above overbought (mean-reversion down)
    +1  Stochastic %K crossed %D
  volatility / mean-reversion
    +1  price closed below lower Bollinger band
    +1  price closed above upper Bollinger band

Bullish votes -> CALL, bearish votes -> PUT. A signal fires only when the net
score reaches `--threshold` (default 4) and the direction changed (debounced),
so the bot does not spam on every tick.

Signals are alerts only — the bot never places a trade.

Usage
-----
    python signal_bot.py                              # EURUSD_otc, M1, alerts only
    python signal_bot.py EURUSD_otc 60 --threshold 5
    python signal_bot.py EURUSD_otc 60 --quiet        # signals only, no heartbeat
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import indicators as ta
from market_qx_candles import CandleBuilder, MarketQX


@dataclass
class Strategy:
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_mult: float = 2.0
    stoch_period: int = 14
    stoch_k: int = 3
    stoch_d: int = 3

    def min_candles(self) -> int:
        return max(self.ema_trend, self.macd_slow + self.macd_signal,
                   self.bb_period, self.stoch_period + self.stoch_k + self.stoch_d) + 2


@dataclass
class Signal:
    time: int
    asset: str
    action: str            # "CALL" | "PUT"
    price: float
    score: int
    reasons: list[str] = field(default_factory=list)


class SignalEngine:
    """Turn a list of closed candles into a scored CALL/PUT decision."""

    def __init__(self, strategy: Strategy, threshold: int = 4):
        self.s = strategy
        self.threshold = threshold
        self._last_action: Optional[str] = None

    def evaluate(self, asset: str, candles: list[list]) -> Optional[Signal]:
        s = self.s
        if len(candles) < s.min_candles():
            return None

        times = [c[0] for c in candles]
        opens = [c[1] for c in candles]
        closes = [c[2] for c in candles]
        highs = [c[3] for c in candles]
        lows = [c[4] for c in candles]
        i = len(candles) - 1

        ema_f = ta.ema(closes, s.ema_fast)
        ema_s = ta.ema(closes, s.ema_slow)
        ema_t = ta.ema(closes, s.ema_trend)
        rsi = ta.rsi(closes, s.rsi_period)
        macd_line, sig_line, _ = ta.macd(closes, s.macd_fast, s.macd_slow, s.macd_signal)
        bb_u, bb_m, bb_l = ta.bollinger(closes, s.bb_period, s.bb_mult)
        st_k, st_d = ta.stochastic(highs, lows, closes, s.stoch_period, s.stoch_k, s.stoch_d)

        if ema_f[i] is None or ema_s[i] is None or ema_t[i] is None or rsi[i] is None:
            return None

        bull = bear = 0
        reasons: list[str] = []

        # --- trend ---
        if ema_f[i] > ema_s[i]:
            bull += 2; reasons.append("EMA fast > slow")
        else:
            bear += 2; reasons.append("EMA fast < slow")

        if closes[i] > ema_t[i]:
            bull += 1; reasons.append("price > EMA trend")
        else:
            bear += 1; reasons.append("price < EMA trend")

        # --- fresh EMA cross (strongest) ---
        if ta.crossed_above(ema_f, ema_s, i):
            bull += 2; reasons.append("EMA bullish cross")
        elif ta.crossed_below(ema_f, ema_s, i):
            bear += 2; reasons.append("EMA bearish cross")

        # --- MACD ---
        if macd_line[i] is not None and sig_line[i] is not None:
            if ta.crossed_above(macd_line, sig_line, i):
                bull += 2; reasons.append("MACD bullish cross")
            elif ta.crossed_below(macd_line, sig_line, i):
                bear += 2; reasons.append("MACD bearish cross")
            elif macd_line[i] > sig_line[i]:
                bull += 1; reasons.append("MACD > signal")
            else:
                bear += 1; reasons.append("MACD < signal")

        # --- RSI mean reversion ---
        if rsi[i] <= s.rsi_oversold:
            bull += 1; reasons.append(f"RSI oversold ({rsi[i]:.0f})")
        elif rsi[i] >= s.rsi_overbought:
            bear += 1; reasons.append(f"RSI overbought ({rsi[i]:.0f})")

        # --- Stochastic cross ---
        if st_k[i] is not None and st_d[i] is not None:
            if ta.crossed_above(st_k, st_d, i):
                bull += 1; reasons.append("Stoch bullish cross")
            elif ta.crossed_below(st_k, st_d, i):
                bear += 1; reasons.append("Stoch bearish cross")

        # --- Bollinger extremes ---
        if bb_l[i] is not None and closes[i] < bb_l[i]:
            bull += 1; reasons.append("close < lower BB")
        elif bb_u[i] is not None and closes[i] > bb_u[i]:
            bear += 1; reasons.append("close > upper BB")

        net = bull - bear
        if abs(net) < self.threshold:
            return None
        action = "CALL" if net > 0 else "PUT"

        # debounce: don't repeat the same direction back-to-back
        if action == self._last_action:
            return None
        self._last_action = action

        return Signal(time=times[i], asset=asset, action=action,
                      price=closes[i], score=abs(net), reasons=reasons)


def format_signal(sig: Signal) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sig.time))
    head = f"{'🟢' if sig.action == 'CALL' else '🔴'} {sig.action} {sig.asset} @ {sig.price}"
    return (f"\n{'=' * 62}\n{head}\n"
            f"  candle close : {ts}\n"
            f"  strength     : {sig.score}\n"
            f"  reasons      : {', '.join(sig.reasons)}\n"
            f"{'=' * 62}")


def main():
    ap = argparse.ArgumentParser(description="EMA + indicator confluence signal bot")
    ap.add_argument("asset", nargs="?", default="EURUSD_otc")
    ap.add_argument("period", nargs="?", type=int, default=60, help="candle seconds")
    ap.add_argument("--threshold", type=int, default=4, help="min net score to fire")
    ap.add_argument("--ssid", default="", help="SSID to enable server history / trading")
    ap.add_argument("--quiet", action="store_true", help="suppress heartbeat lines")
    args = ap.parse_args()

    strategy = Strategy()
    engine = SignalEngine(strategy, threshold=args.threshold)

    stats = {"candles": 0, "signals": 0, "calls": 0, "puts": 0}
    history: list[list] = []   # append-only closed candles, oldest -> newest
    max_history = 1000

    def on_closed(candle: list):
        history.append(candle)
        if len(history) > max_history:
            del history[:len(history) - max_history]
        stats["candles"] += 1
        sig = engine.evaluate(args.asset, history)
        if sig:
            stats["signals"] += 1
            stats["calls" if sig.action == "CALL" else "puts"] += 1
            print(format_signal(sig), flush=True)

    builder = CandleBuilder(args.period, on_closed=on_closed)

    qx = MarketQX(ssid=args.ssid, verbose=False).connect(wait_instruments=75)
    print(f"assets: {len(qx.instruments)}  |  strategy: EMA "
          f"{strategy.ema_fast}/{strategy.ema_slow}/{strategy.ema_trend} + RSI + MACD + BB + Stoch")
    print(f"asset: {args.asset}  period: {args.period}s  threshold: {args.threshold}")
    print(f"need {strategy.min_candles()} closed candles before the first signal\n")

    def on_tick(asset, ts, price):
        if asset == args.asset:
            builder.add(ts, price)

    qx.on_tick(on_tick)
    qx.follow(args.asset, args.period)

    last_beat = 0
    try:
        while True:
            time.sleep(2)
            now = time.time()
            if not args.quiet and now - last_beat >= 15:
                last_beat = now
                forming = builder.forming
                price = forming[2] if forming else "-"
                print(f"[{time.strftime('%H:%M:%S')}] candles={stats['candles']} "
                      f"signals={stats['signals']} (C{stats['calls']}/P{stats['puts']}) "
                      f"price={price}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nstopped. closed candles={stats['candles']} "
              f"signals={stats['signals']} (CALL={stats['calls']} PUT={stats['puts']})")
        qx.close()


if __name__ == "__main__":
    main()
