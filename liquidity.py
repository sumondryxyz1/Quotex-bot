#!/usr/bin/env python3
"""Liquidity intelligence: pools, sweeps, displacement, session levels.

The old `smc.liquidity_sweep()` only asked "did this wick pierce a prior swing
and close back?". This module adds the context that decides whether a sweep is
meaningful: the size and age of the pool it took, whether the reclaim was strong,
and whether a displacement leg followed.

Definitions (pure, causal — index `i` uses only candles [0..i]):

  equal highs/lows   two or more swing highs/lows within a tolerance -> a level
                     where resting orders (stops) cluster.
  liquidity pool     a clustered level plus how many times it was touched and how
                     old it is.
  liquidity sweep    price trades through a pool level then closes back inside,
                     trapping the orders there. Bullish sweep takes a low; bearish
                     sweep takes a high.
  displacement       a wide-range, high-body candle (an impulsive, one-sided move)
                     that leaves an imbalance — the footprint of size, not noise.

Signals are alerts only — never a trade instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import indicators as ta
from structure import swing_series

BULL, BEAR = "bull", "bear"


@dataclass
class Pool:
    level: float
    kind: str            # "high" | "low"
    touches: int
    first_idx: int
    last_idx: int
    age: int


@dataclass
class Sweep:
    direction: str       # "bull" (took a low) | "bear" (took a high)
    level: float
    idx: int
    penetration: float   # how far past the level, in ATR units
    reclaim: float       # 0..1, how far the close came back inside
    displaced: bool = False
    detail: str = ""


@dataclass
class LiquidityState:
    pools: list[Pool] = field(default_factory=list)
    sweep: Optional[Sweep] = None

    def as_dict(self) -> dict:
        return {
            "pools": [(round(p.level, 6), p.kind, p.touches, p.age)
                      for p in self.pools[-6:]],
            "sweep": (f"{self.sweep.direction} level {self.sweep.level} "
                      f"pen {self.sweep.penetration:.2f}atr "
                      f"reclaim {self.sweep.reclaim:.2f}"
                      if self.sweep else None),
        }


def equal_levels(highs: list[float], lows: list[float], i: int,
                 tol_atr: float = 0.25, lookback: int = 120,
                 left: int = 2, right: int = 2) -> tuple[list[float], list[float]]:
    """Equal-high and equal-low clusters in the lookback window.

    Swings of the same kind within `tol_atr` ATRs of each other form a cluster
    (two or more touches); the cluster mean is an equal level. Clustering — not
    adjacent-pair comparison — is what catches a double top with other swings
    between the two peaks. Returns (eq_high_levels, eq_low_levels).
    """
    swings = swing_series(highs, lows, i, left, right, lookback)
    atr = ta.atr(highs[:i + 1], lows[:i + 1],
                 [(highs[k] + lows[k]) / 2 for k in range(i + 1)], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if a <= 0:
        return [], []
    tol = tol_atr * a

    def _cluster(points: list[float]) -> list[float]:
        pts = sorted(points)
        out: list[float] = []
        group = [pts[0]] if pts else []
        for p in pts[1:]:
            if p - group[-1] <= tol:
                group.append(p)
            else:
                if len(group) >= 2:
                    out.append(sum(group) / len(group))
                group = [p]
        if len(group) >= 2:
            out.append(sum(group) / len(group))
        return out

    hs = [s.price for s in swings if s.kind == "high"]
    ls = [s.price for s in swings if s.kind == "low"]
    return _cluster(hs), _cluster(ls)


def liquidity_pools(highs: list[float], lows: list[float], i: int,
                    tol_atr: float = 0.25, lookback: int = 120) -> list[Pool]:
    """Clustered swing levels, ranked by touches (more touches = more stops)."""
    swings = swing_series(highs, lows, i, 2, 2, lookback)
    if not swings:
        return []
    atr = ta.atr(highs[:i + 1], lows[:i + 1],
                 [(highs[k] + lows[k]) / 2 for k in range(i + 1)], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if a <= 0:
        return []
    tol = tol_atr * a
    pools: list[Pool] = []
    for kind in ("high", "low"):
        pts = [s for s in swings if s.kind == kind]
        used = [False] * len(pts)
        for x in range(len(pts)):
            if used[x]:
                continue
            group = [pts[x]]
            used[x] = True
            for y in range(x + 1, len(pts)):
                if not used[y] and abs(pts[y].price - pts[x].price) <= tol:
                    group.append(pts[y])
                    used[y] = True
            if len(group) >= 2:
                level = sum(g.price for g in group) / len(group)
                pools.append(Pool(level, kind, len(group),
                                  group[0].idx, group[-1].idx, i - group[-1].idx))
    pools.sort(key=lambda p: (-p.touches, p.age))
    return pools


def displacement(opens: list[float], closes: list[float], highs: list[float],
                 lows: list[float], i: int, atr_mult: float = 1.5,
                 body_frac: float = 0.6) -> dict:
    """Is candle `i` a displacement (impulsive) candle?

    Needs a body at least `atr_mult` ATRs tall and at least `body_frac` of the
    candle's range. Returns {displaced, direction, size_atr, body_ratio}.
    """
    atr = ta.atr(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if a <= 0 or i < 1:
        return {"displaced": False, "direction": None, "size_atr": 0.0,
                "body_ratio": 0.0}
    rng = highs[i] - lows[i]
    body = abs(closes[i] - opens[i])
    size = body / a
    ratio = body / rng if rng else 0.0
    direction = BULL if closes[i] > opens[i] else BEAR
    return {"displaced": size >= atr_mult and ratio >= body_frac,
            "direction": direction, "size_atr": round(size, 3),
            "body_ratio": round(ratio, 3)}


def sweep_and_reclaim(opens: list[float], closes: list[float], highs: list[float],
                      lows: list[float], i: int, lookback: int = 40,
                      tol_atr: float = 0.25) -> Optional[Sweep]:
    """Did candle `i` sweep a pool and reclaim it?

    Bullish: the low pierces the lowest pool/swing low, the close comes back
    above it. Bearish: mirror on the high. `penetration` is in ATR units and
    `reclaim` is the fraction of the candle range that closed back inside.
    """
    lo = max(0, i - lookback)
    if i - lo < 6:
        return None
    atr = ta.atr(highs[:i + 1], lows[:i + 1], closes[:i + 1], 14)
    a = atr[i] if atr and atr[i] else 0.0
    if a <= 0:
        return None
    pools = liquidity_pools(highs, lows, i - 1, tol_atr, lookback)
    eq_h, eq_l = equal_levels(highs, lows, i - 1, tol_atr, lookback)
    levels_high = [p.level for p in pools if p.kind == "high"] + eq_h
    levels_low = [p.level for p in pools if p.kind == "low"] + eq_l
    # fall back to raw recent swings, per side, when a side has no clustered level
    sw = swing_series(highs, lows, i - 1, 2, 2, lookback)
    if not levels_high:
        levels_high = [s.price for s in sw if s.kind == "high"]
    if not levels_low:
        levels_low = [s.price for s in sw if s.kind == "low"]
    rng = highs[i] - lows[i] or a
    disp = displacement(opens, closes, highs, lows, i)
    # bullish sweep (took lows)
    if levels_low:
        lvl = min(levels_low)
        if lows[i] < lvl and closes[i] > lvl:
            pen = (lvl - lows[i]) / a
            reclaim = (closes[i] - lvl) / rng
            return Sweep(BULL, lvl, i, round(pen, 3), round(reclaim, 3),
                         disp["displaced"] and disp["direction"] == BULL,
                         "sweep+reclaim of low")
    if levels_high:
        lvl = max(levels_high)
        if highs[i] > lvl and closes[i] < lvl:
            pen = (highs[i] - lvl) / a
            reclaim = (lvl - closes[i]) / rng
            return Sweep(BEAR, lvl, i, round(pen, 3), round(reclaim, 3),
                         disp["displaced"] and disp["direction"] == BEAR,
                         "sweep+reclaim of high")
    return None


def session_levels(times: list[int], highs: list[float], lows: list[float],
                   i: int) -> dict:
    """Asian / London / New York high-low for the current UTC day.

    Meaningful for real (session-bound) pairs; on 24/7 OTC pairs all sessions
    fill in and the levels converge, which is itself a signal that the concept
    does not apply there.
    """
    day = times[i] // 86400
    out = {"asian": None, "london": None, "newyork": None}
    bounds = {"asian": (0, 8), "london": (8, 16), "newyork": (13, 21)}
    acc = {k: [float("inf"), float("-inf")] for k in out}
    for j in range(i, -1, -1):
        if times[j] // 86400 != day:
            break
        h = (times[j] // 3600) % 24
        for k, (a, b) in bounds.items():
            if a <= h < b:
                acc[k][0] = min(acc[k][0], lows[j])
                acc[k][1] = max(acc[k][1], highs[j])
    for k in out:
        lo_, hi_ = acc[k]
        if lo_ != float("inf"):
            out[k] = (lo_, hi_)
    return out


def liquidity_state(opens: list[float], closes: list[float], highs: list[float],
                    lows: list[float], i: int, lookback: int = 120) -> LiquidityState:
    st = LiquidityState(pools=liquidity_pools(highs, lows, i, lookback=lookback))
    st.sweep = sweep_and_reclaim(opens, closes, highs, lows, i)
    return st
