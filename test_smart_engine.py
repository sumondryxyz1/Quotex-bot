"""Checks for the ADX indicator, the smart engine, and the backtest harness.

The engine is statistical, so these tests verify *mechanics* (gates fire, signals
are well-formed, the harness scores correctly), not that the engine is profitable.
A separate synthetic test proves the engine can detect a real, strong trend.
"""
import random

import indicators as ta
from backtest import Result, RandomEngine, backtest_pair
from smart_engine import SmartEngine, SmartStrategy

fails = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- ADX mechanics
# A clean, strong uptrend should push ADX up and +DI above -DI.
n = 120
highs, lows, closes = [], [], []
p = 100.0
for i in range(n):
    p += 0.5
    closes.append(p)
    highs.append(p + 0.3)
    lows.append(p - 0.3)
adx, pdi, mdi = ta.adx(highs, lows, closes, 14)
check("adx present in uptrend", adx[-1] is not None and adx[-1] > 25,
      f"adx={adx[-1]:.1f}" if adx[-1] is not None else "None")
check("+DI > -DI in uptrend", pdi[-1] > mdi[-1],
      f"{pdi[-1]:.0f} vs {mdi[-1]:.0f}")

# A flat series should keep ADX low (range regime).
highs = [100.0 + (0.2 if i % 2 else -0.2) for i in range(n)]
lows = [100.0 + (-0.2 if i % 2 else 0.2) for i in range(n)]
closes = [100.0] * n
adx2, _, _ = ta.adx(highs, lows, closes, 14)
check("adx low in range", adx2[-1] is not None and adx2[-1] < 25,
      f"adx={adx2[-1]:.1f}" if adx2[-1] is not None else "None")

# ------------------------------------------------------------------ smart gates
def synth_trend(n=400, step=0.6, noise=0.15, seed=1):
    rng = random.Random(seed)
    cs, h, l, p = [], [], [], 100.0
    for i in range(n):
        p += step + rng.uniform(-noise, noise)
        o = p - step
        cs.append([i * 5, o, p, max(o, p) + 0.1, min(o, p) - 0.1])
    return cs


def synth_chop(n=400, seed=2):
    rng = random.Random(seed)
    cs, p = [], 100.0
    for i in range(n):
        o = p
        p += rng.uniform(-0.3, 0.3)
        cs.append([i * 5, o, p, max(o, p) + 0.05, min(o, p) - 0.05])
    return cs


eng = SmartEngine(SmartStrategy())
sig = eng.evaluate(synth_trend())
check("smart fires on a strong trend", sig is not None and sig.action == "CALL",
      f"{sig.action} conf={sig.confidence}" if sig else "None")

# Too few candles -> no signal, no crash
check("smart needs warmup", SmartEngine().evaluate(synth_trend(30)) is None)

# ----------------------------------------------------------- backtest harness
# The harness must score a deliberately-correct engine at 100%.
class AlwaysUp:
    def evaluate(self, candles):
        if len(candles) < 60:
            return None

        class S:
            action = "CALL"

        return S()


cs = synth_trend()
r = backtest_pair(AlwaysUp(), cs, warmup=60, name="always-up")
check("harness scores a perfect engine at 100%", r.n > 0 and r.hit == 1.0,
      f"hit={r.hit:.2f} n={r.n}")

# The random control must land near 50% (the null). Wide tolerance for small n.
rc = backtest_pair(RandomEngine(rate=1.0, seed=7), cs, warmup=60, name="rand")
check("random control near 50%", 0.30 <= rc.hit <= 0.70, f"hit={rc.hit:.2f}")

# Wilson interval must contain the point estimate.
lo, hi = Result("x").ci()
r2 = backtest_pair(AlwaysUp(), cs, warmup=60, name="x")
lo, hi = r2.ci()
check("CI brackets the estimate", lo <= r2.hit <= hi, f"[{lo:.2f},{hi:.2f}] hit={r2.hit:.2f}")

print()
print("FAILURES:", fails if fails else "none")
