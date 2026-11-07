#!/usr/bin/env python3
"""Collect live candles from market-qx.info into a CSV for model training.

The feed has no public history, so the only way to get a training set is to
record the live stream yourself. Run this in the background for a while
(hours -> days for a usable dataset), then feed the CSV to `train_model.py`.

CSV columns: time,open,close,high,low

Usage
-----
    python collect_data.py EURUSD_otc 60 --out data/EURUSD_otc_60.csv
    python collect_data.py EURUSD_otc 60 --hours 4        # auto-stop after 4h
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from market_qx_candles import CandleBuilder, MarketQX


def main():
    ap = argparse.ArgumentParser(description="Record live candles to CSV")
    ap.add_argument("asset", nargs="?", default="EURUSD_otc")
    ap.add_argument("period", nargs="?", type=int, default=60)
    ap.add_argument("--out", default="")
    ap.add_argument("--hours", type=float, default=0.0, help="0 = run forever")
    args = ap.parse_args()

    out = args.out or f"data/{args.asset}_{args.period}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    seen: set[int] = set()
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.reader(f):
                if row and row[0].isdigit():
                    seen.add(int(row[0]))
        print(f"resuming {out}: {len(seen)} candles already recorded")

    f = open(out, "a", newline="")
    writer = csv.writer(f)

    def on_closed(candle: list):
        t = int(candle[0])
        if t in seen:
            return
        seen.add(t)
        writer.writerow([t, candle[1], candle[2], candle[3], candle[4]])
        f.flush()
        if len(seen) % 10 == 0:
            print(f"  {len(seen)} candles  (last {time.strftime('%H:%M:%S', time.localtime(t))} "
                  f"o={candle[1]} c={candle[2]})", flush=True)

    builder = CandleBuilder(args.period, on_closed=on_closed)
    qx = MarketQX().connect(wait_instruments=75)
    qx.on_tick(lambda a, ts, p: builder.add(ts, p) if a == args.asset else None)
    qx.follow(args.asset, args.period)

    print(f"recording {args.asset} {args.period}s candles -> {out}")
    deadline = time.time() + args.hours * 3600 if args.hours else None
    try:
        while deadline is None or time.time() < deadline:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        qx.close()
        print(f"\nstopped. {len(seen)} candles in {out}")


if __name__ == "__main__":
    main()
