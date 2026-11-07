#!/usr/bin/env python3
"""Live AI prediction bot — predicts the next candle's direction.

Loads a model trained by `train_model.py`, consumes the live candle stream from
market-qx.info, and after every *closed* candle prints an up/down prediction with
a confidence level. It only speaks up when confidence is above a threshold.

Signals are alerts only — the bot never places a trade.

Usage
-----
    python ai_bot.py out/model.pkl EURUSD_otc 60
    python ai_bot.py out/model.pkl EURUSD_otc 60 --min-confidence 0.58
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from typing import Optional

from features import compute_features, features_to_vector
from market_qx_candles import CandleBuilder, MarketQX


class Predictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.features = bundle["features"]

    def predict(self, candles: list[list]) -> Optional[tuple[str, float]]:
        feat = compute_features(candles, len(candles) - 1)
        if feat is None:
            return None
        x = [features_to_vector(feat)]
        prob_up = float(self.model.predict_proba(x)[0][1])
        action = "UP" if prob_up >= 0.5 else "DOWN"
        conf = prob_up if action == "UP" else 1 - prob_up
        return action, conf


def main():
    ap = argparse.ArgumentParser(description="Live up/down prediction bot")
    ap.add_argument("model", help="path to model.pkl from train_model.py")
    ap.add_argument("asset", nargs="?", default="EURUSD_otc")
    ap.add_argument("period", nargs="?", type=int, default=60)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    predictor = Predictor(args.model)
    history: deque[list] = deque(maxlen=1000)
    stats = {"candles": 0, "signals": 0, "up": 0, "down": 0, "correct": 0, "graded": 0}
    pending: Optional[tuple[str, float, float]] = None  # action, conf, entry_price

    def on_closed(candle: list):
        nonlocal pending
        stats["candles"] += 1

        # grade the previous prediction against this candle's close
        if pending is not None:
            action, conf, entry = pending
            won = (action == "UP" and candle[2] > entry) or (action == "DOWN" and candle[2] < entry)
            stats["graded"] += 1
            stats["correct"] += int(won)
            if not args.quiet:
                print(f"    -> previous {action} @ {entry} "
                      f"{'WON ✅' if won else 'LOST ❌'} (running "
                      f"{stats['correct']}/{stats['graded']} = "
                      f"{stats['correct']/stats['graded']:.1%})", flush=True)
            pending = None

        history.append(candle)
        result = predictor.predict(list(history))
        if result is None:
            return
        action, conf = result
        if conf < args.min_confidence:
            return
        stats["signals"] += 1
        stats["up" if action == "UP" else "down"] += 1
        arrow = "🟢" if action == "UP" else "🔴"
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(candle[0]))
        print(f"\n{arrow} PREDICT {action}  {args.asset}  @ {candle[2]}")
        print(f"   candle close : {ts}")
        print(f"   confidence   : {conf:.1%}")
        pending = (action, conf, candle[2])

    builder = CandleBuilder(args.period, on_closed=on_closed)
    qx = MarketQX().connect(wait_instruments=75)
    qx.on_tick(lambda a, ts, p: builder.add(ts, p) if a == args.asset else None)
    qx.follow(args.asset, args.period)

    print(f"AI bot: {args.asset} {args.period}s  |  min-confidence {args.min_confidence:.0%}")
    print("predicting after every closed candle (Ctrl-C to stop)\n")

    last_beat = 0
    try:
        while True:
            time.sleep(2)
            now = time.time()
            if not args.quiet and now - last_beat >= 20:
                last_beat = now
                acc = (stats["correct"] / stats["graded"] * 100) if stats["graded"] else 0.0
                print(f"[{time.strftime('%H:%M:%S')}] candles={stats['candles']} "
                      f"signals={stats['signals']} (U{stats['up']}/D{stats['down']}) "
                      f"live-acc={acc:.1f}% ({stats['correct']}/{stats['graded']})", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        acc = (stats["correct"] / stats["graded"] * 100) if stats["graded"] else 0.0
        print(f"\nstopped. signals={stats['signals']} "
              f"live accuracy={acc:.1f}% ({stats['correct']}/{stats['graded']})")
        print("NOTE: a small live sample proves nothing — judge the model on the")
        print("      walk-forward numbers from train_model.py, not this run.")
        qx.close()


if __name__ == "__main__":
    main()
