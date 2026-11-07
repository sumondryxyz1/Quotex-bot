"""Offline test of SignalEngine with synthetic candle series.

Verifies the strategy actually fires on clear trends/reversals and that the
debounce stops repeat signals in the same direction.
"""
import math

from signal_bot import SignalEngine, Strategy


def make_candles(closes):
    """Build [time, open, close, high, low] rows from a close series."""
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        hi = max(o, c) * 1.0002
        lo = min(o, c) * 0.9998
        rows.append([1_700_000_000 + i * 60, o, c, hi, lo])
    return rows


def gen(n, start, step, noise=0.0):
    return [start + step * i + noise * math.sin(i / 3.0) for i in range(n)]


engine = SignalEngine(Strategy(), threshold=4)
fired = []


def replay(rows, label):
    print(f"\n--- {label} ---")
    # feed progressively longer prefixes, like the live stream does
    for end in range(60, len(rows) + 1):
        sig = engine.evaluate("TEST", rows[:end])
        if sig:
            fired.append((label, sig.action, sig.score))
            print(f"  signal @ candle {end}: {sig.action} score={sig.score} "
                  f"({', '.join(sig.reasons[:4])})")


replay(make_candles(gen(200, 1.10, 0.0002, 0.00005)), "uptrend")
replay(make_candles(gen(200, 1.30, -0.0002, 0.00005)), "downtrend")

# V-shape: down then up -> should flip direction
down = gen(120, 1.30, -0.0003)
up = gen(120, down[-1], 0.0003)
replay(make_candles(down + up), "V-reversal")

print("\n=== summary ===")
print("total signals:", len(fired))
print("CALL:", sum(1 for f in fired if f[1] == "CALL"))
print("PUT :", sum(1 for f in fired if f[1] == "PUT"))
# debounce check: consecutive signals must alternate direction
dirs = [f[1] for f in fired]
ok = all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1))
print("directions alternate (debounce works):", ok)
print("uptrend produced CALL:", any(f[0] == "uptrend" and f[1] == "CALL" for f in fired))
print("downtrend produced PUT:", any(f[0] == "downtrend" and f[1] == "PUT" for f in fired))
