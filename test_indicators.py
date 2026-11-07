"""Sanity checks for indicators.py against hand-computable reference values."""
from indicators import ema, sma, rsi, macd, bollinger, atr, stochastic

fails = []


def check(name, got, want, tol=1e-6):
    ok = (got is not None) and abs(got - want) <= tol
    print(f"{'ok ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok:
        fails.append(name)


# SMA of 1..5 over period 3 -> 2, 3, 4
s = sma([1, 2, 3, 4, 5], 3)
check("sma[-1]", s[-1], 4.0)
check("sma[-2]", s[-2], 3.0)
check("sma[0] is None", 0.0 if s[0] is None else 1.0, 0.0)

# EMA: seed = SMA(3) of first 3 = 2; k=0.5
e = ema([1, 2, 3, 4, 5], 3)
check("ema[2] seed", e[2], 2.0)
check("ema[3]", e[3], 4 * 0.5 + 2 * 0.5)     # 3.0
check("ema[4]", e[4], 5 * 0.5 + 3.0 * 0.5)   # 4.0

# RSI: monotonically rising -> 100
r = rsi([float(i) for i in range(1, 30)], 14)
check("rsi all-up", r[-1], 100.0)

# RSI falling -> 0
r2 = rsi([float(i) for i in range(30, 1, -1)], 14)
check("rsi all-down", r2[-1], 0.0)

# Bollinger on constant series -> zero width
u, m, l = bollinger([5.0] * 25, 20, 2.0)
check("bb mid", m[-1], 5.0)
check("bb upper", u[-1], 5.0)
check("bb lower", l[-1], 5.0)

# ATR: constant range 2, close flat -> ATR 2
h = [10.0] * 20
lo = [8.0] * 20
c = [9.0] * 20
a = atr(h, lo, c, 14)
check("atr constant", a[-1], 2.0)

# Stochastic: close at top of range -> 100
h2 = [10.0] * 20
l2 = [0.0] * 20
c2 = [10.0] * 20
k, d = stochastic(h2, l2, c2, 14, 3, 3)
check("stoch k top", k[-1], 100.0)

# MACD on constant series -> 0
ml, sl, hist = macd([3.0] * 60, 12, 26, 9)
check("macd flat", ml[-1], 0.0)
check("macd signal flat", sl[-1], 0.0)

print()
print("FAILURES:", fails if fails else "none")
