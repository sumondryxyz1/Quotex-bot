#!/usr/bin/env python3
"""Statistical power: how much data before an edge can be believed?

The single most common way a backtest lies is by reporting a hit-rate from far
too few trades. This module answers "how many decisive signals do I need?" so
sample size is chosen up front, not discovered after a lucky run.

Two-sided test of a hit-rate p against the payout breakeven p0, at significance
`alpha` and power `1 - beta`, using the normal approximation:

    n ≈ ( z_{1-alpha/2} * sqrt(p0(1-p0)) + z_{1-beta} * sqrt(p(1-p)) )^2
        / (p - p0)^2

It also reports the minimum detectable effect: given the data actually available,
what is the smallest true edge that could have been detected?

Usage
-----
    python power.py --payout 0.85 --hit 0.56
    python power.py --payout 0.85 --table
    python power.py --payout 0.90 --n 500
"""
from __future__ import annotations

import argparse
import math

from payout import DEFAULT_PAYOUT, breakeven as payout_breakeven

# z-values for the standard confidence / power levels (two-sided).
_Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758, 0.001: 3.2905}
# one-sided z for power
_ZP = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def z_for(alpha: float) -> float:
    if alpha in _Z:
        return _Z[alpha]
    # inverse normal via Acklam-free simple rational approximation
    return _inv_norm(1 - alpha / 2)


def zp_for(power: float) -> float:
    if power in _ZP:
        return _ZP[power]
    return _inv_norm(power)


def _inv_norm(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm), for arbitrary p."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def required_n(hit: float, payout: float = DEFAULT_PAYOUT, alpha: float = 0.05,
               power: float = 0.80) -> int:
    """Decisive signals needed to detect a true hit-rate `hit` vs breakeven."""
    p0 = payout_breakeven(payout)
    p = hit
    if p <= p0:
        return -1  # no finite sample detects an edge that isn't there
    za = z_for(alpha)
    zb = zp_for(power)
    num = (za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p * (1 - p)))
    return math.ceil((num * num) / ((p - p0) ** 2))


def min_detectable_hit(n: int, payout: float = DEFAULT_PAYOUT,
                       alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest true hit-rate detectable with `n` signals (search upward)."""
    p0 = payout_breakeven(payout)
    lo, hi = p0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        need = required_n(mid, payout, alpha, power)
        if need < 0 or need > n:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ap = argparse.ArgumentParser(description="Sample-size / power calculator")
    ap.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    ap.add_argument("--hit", type=float, default=0.0,
                    help="true hit-rate to detect, e.g. 0.56")
    ap.add_argument("--n", type=int, default=0, help="available signals")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    be = payout_breakeven(args.payout)
    print(f"payout {args.payout:.0%} -> breakeven {be * 100:.2f}%  "
          f"(alpha {args.alpha}, power {args.power:.0%})")

    if args.table or (not args.hit and not args.n):
        print(f"\n  {'true hit':>9} {'edge':>7} {'required n':>12}")
        for hit in (0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62):
            n = required_n(hit, args.payout, args.alpha, args.power)
            tag = "not detectable" if n < 0 else f"{n:,}"
            print(f"  {hit * 100:>8.1f}% {(hit - be) * 100:>+6.2f}% {tag:>12}")

    if args.hit:
        n = required_n(args.hit, args.payout, args.alpha, args.power)
        print(f"\n  a true {args.hit * 100:.1f}% hit-rate needs "
              f"{'an impossible' if n < 0 else f'{n:,}'} decisive signals")

    if args.n:
        mdh = min_detectable_hit(args.n, args.payout, args.alpha, args.power)
        print(f"\n  with {args.n:,} signals you can detect a true hit-rate "
              f"down to {mdh * 100:.2f}% (edge {(mdh - be) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
