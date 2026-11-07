#!/usr/bin/env python3
"""Collect raw ticks from the Quotex feed — the order-flow data layer.

The candle feed only keeps OHLC, which throws away the two things that could
carry information: the broker's per-tick direction flag and the exact timing of
each price change. This collector subscribes to every pair and stores every tick
as (asset, ts, price, dir) in `data/candles.db` (table `ticks`), so the order
flow can be studied later.

It reconnects on drop and logs throughput, so it can run for days.

Usage
-----
    python collect_ticks.py                      # all pairs, forever
    python collect_ticks.py --pairs EURUSD_otc,BTCUSD_otc
    python collect_ticks.py --minutes 30         # collect for 30 min then stop
    python collect_ticks.py --stats              # show what is stored, exit
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from candle_store import CandleStore, MultiFeed


def show_stats(store: CandleStore):
    total = store.tick_count()
    assets = store.tick_assets()
    print(f"ticks stored: {total:,}  across {len(assets)} pairs")
    rows = store._db.execute(
        "SELECT asset, COUNT(*) n, MIN(ts), MAX(ts) FROM ticks "
        "GROUP BY asset ORDER BY n DESC LIMIT 10").fetchall()
    for asset, n, t0, t1 in rows:
        span = (t1 - t0) / 60.0 if t0 else 0
        print(f"  {asset:16} {n:>9,}  span {span:6.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Collect raw Quotex ticks")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--pairs", default="", help="comma list; default all pairs")
    ap.add_argument("--period", type=int, default=5, help="follow period (unused for ticks)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    store = CandleStore(args.db)
    if args.stats:
        show_stats(store)
        return

    assets = [a for a in args.pairs.split(",") if a] or []
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    deadline = time.time() + args.minutes * 60 if args.minutes else 0
    session = 0
    while not stop["flag"]:
        session += 1
        try:
            feed = MultiFeed(assets, [args.period], store)
        except Exception as e:
            print(f"connect failed: {e!r} — retrying in 10s", flush=True)
            time.sleep(10)
            continue
        print(f"[session {session}] collecting ticks for {len(feed.assets)} pairs "
              f"— db {args.db}", flush=True)
        last = time.time()
        try:
            while not stop["flag"]:
                time.sleep(10)
                if deadline and time.time() > deadline:
                    stop["flag"] = True
                    break
                if time.time() - last >= 60:
                    last = time.time()
                    print(f"  {time.strftime('%H:%M:%S')}  ticks saved "
                          f"{feed.ticks_saved:,}  (total {store.tick_count():,})  "
                          f"prices {len(feed.last_price)}", flush=True)
                if not feed.qx.connected:
                    raise ConnectionError("feed dropped")
        except Exception as e:
            print(f"feed error: {e!r} — reconnecting", flush=True)
        finally:
            try:
                feed.close()
            except Exception:
                pass
        if not stop["flag"]:
            time.sleep(5)

    print(f"stopped. ticks in db: {store.tick_count():,}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
