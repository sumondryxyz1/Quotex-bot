"""Evidence that the market-qx feed is genuinely live — and what it actually is.

Run this to verify, not to trust:

  1. liveness      : server timestamps track the local clock; prices change
  2. real vs OTC   : `_otc` symbols are the broker's own synthetic quotes;
                     non-`_otc` symbols mirror the real market (and freeze when
                     that market is closed, e.g. weekends).

Usage: python verify_feed.py
"""
import time

from market_qx_candles import MarketQX

ASSETS = ["EURUSD", "EURUSD_otc", "BTCUSD", "BTCUSD_otc", "XAUUSD", "XAUUSD_otc"]

qx = MarketQX(verbose=False).connect(wait_instruments=70)
latest = {a: (0, float("nan")) for a in ASSETS}
counts = {a: 0 for a in ASSETS}
seen_prices = {a: set() for a in ASSETS}


def on_tick(asset, ts, price):
    if asset in latest:
        latest[asset] = (ts, price)
        counts[asset] += 1
        seen_prices[asset].add(round(price, 5))


qx.on_tick(on_tick)
for a in ASSETS:
    qx.follow(a, 5)

print("capturing 30s of ticks across real + OTC symbols ...\n")
time.sleep(30)

now = time.time()
print(f"{'symbol':>12} | {'ticks':>5} | {'distinct':>8} | {'clock skew':>10} | price")
print("-" * 66)
for a in ASSETS:
    sts, price = latest.get(a, (0, float("nan")))
    skew = (sts - now) if sts else float("nan")
    print(f"{a:>12} | {counts[a]:>5} | {len(seen_prices[a]):>8} | {skew:>+9.2f}s | {price}")

print("\ninterpretation:")
print("  * clock skew ~ 0  -> frames are produced in real time (not replayed)")
print("  * distinct>1      -> the price actually moves")
print("  * non-OTC frozen  -> that real market is closed right now (weekend)")
print("  * OTC moving      -> broker-generated quotes, stream 24/7")

qx.close()
