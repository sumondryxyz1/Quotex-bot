#!/usr/bin/env python3
"""Extended feature set: base + structure + liquidity + regime + zone features.

`features.py` covered price/indicator shape only. This module adds the market
intelligence dimensions (structure, liquidity, regime, zones) so a model can use
them — and, just as importantly, so their usefulness can be *measured*.

Everything is causal: features at index `i` use candles [0..i] only. The
per-index arrays come from one-pass series builders (`structure.structure_series`,
`regime.regime_series`) so this stays O(n), not O(n^2).

Label: 1 if candle i+1 closes higher than candle i, else 0 (same as everywhere).
"""
from __future__ import annotations

from typing import Optional

import features as base_features
import indicators as ta
import liquidity
import regime as regime_mod
import structure
import zones

# Extra feature names, in fixed order, appended after base_features.FEATURES.
EXTRA_FEATURES = [
    "st_trend",            # -1/0/1
    "st_bos", "st_choch",  # -1/0/1 direction of the last event
    "st_bos_age", "st_choch_age",   # candles since the event (capped)
    "reg_adx", "reg_eff", "reg_vr", "reg_hurst",
    "reg_defined", "reg_vol",       # vol: -1 low / 0 normal / 1 high
    "liq_sweep", "liq_pen", "liq_reclaim",
    "disp", "disp_size",
    "zone_in", "zone_dir", "zone_age",
    "atr_pct_rank",
]

FEATURES_V2 = base_features.FEATURES + EXTRA_FEATURES

_VOL_CODE = {"low": -1.0, "normal": 0.0, "high": 1.0}
_TREND_CODE = {structure.DOWN: -1.0, structure.RANGE: 0.0, structure.UP: 1.0}


def _last_event_age(arr: list[int], i: int, cap: int = 50) -> float:
    for j in range(i, max(-1, i - cap - 1), -1):
        if arr[j] != 0:
            return float(i - j)
    return float(cap)


def _zone_series(opens, closes, highs, lows, i_max,
                 max_age=60) -> tuple[list[float], list[float], list[float]]:
    """Forward pass tracking live OB/FVG zones: in-zone flag, direction, age.

    Starts at the first usable candle (not a trailing window) so the scan is
    causal and every index sees the same live zones it would in real time.
    """
    n = len(closes)
    in_zone = [0.0] * n
    zdir = [0.0] * n
    zage = [0.0] * n
    live: list[zones.Zone] = []
    for j in range(4, i_max + 1):
        z = zones.find_order_block(opens, closes, highs, lows, j)
        if z is None:
            z = zones.find_fvg(highs, lows, j)
        if z is not None:
            live.append(z)
        kept = []
        for zz in live:
            if zz.created_idx < j:
                zones.advance_zone(zz, opens, closes, highs, lows, j)
            if zz.violated or zz.respected or (j - zz.created_idx) > max_age:
                continue
            kept.append(zz)
        live = kept
        for zz in reversed(live):
            if zz.contains(closes[j]):
                in_zone[j] = 1.0
                zdir[j] = 1.0 if zz.direction == zones.BULL else -1.0
                zage[j] = float(j - zz.created_idx)
                break
    return in_zone, zdir, zage


def build_matrix(candles: list[list], min_index: int = 60) -> tuple[list[list[float]], list[int], list[int]]:
    """Return (rows, labels, indices) for a candle series.

    `rows[k]` is the feature vector for candle `indices[k]`; `labels[k]` is
    whether the next candle closed higher.
    """
    n = len(candles)
    if n < min_index + 2:
        return [], [], []
    o = [c[1] for c in candles]
    c = [c[2] for c in candles]
    h = [c[3] for c in candles]
    l = [c[4] for c in candles]
    t = [c[0] for c in candles]

    st = structure.structure_series(o, c, h, l)
    rg = regime_mod.regime_series(o, c, h, l, t)
    atr = ta.atr(h, l, c, 14)

    # liquidity sweep + displacement per index (bounded lookback)
    sweep_dir = [0.0] * n
    sweep_pen = [0.0] * n
    sweep_rec = [0.0] * n
    disp = [0.0] * n
    disp_size = [0.0] * n
    lb = 40
    for j in range(max(6, min_index), n):
        sw = liquidity.sweep_and_reclaim(o, c, h, l, j, lookback=lb)
        if sw is not None:
            sweep_dir[j] = 1.0 if sw.direction == liquidity.BULL else -1.0
            sweep_pen[j] = min(sw.penetration, 5.0)
            sweep_rec[j] = sw.reclaim
        d = liquidity.displacement(o, c, h, l, j)
        if d["displaced"]:
            disp[j] = 1.0 if d["direction"] == liquidity.BULL else -1.0
            disp_size[j] = min(d["size_atr"], 8.0)

    zin, zdir, zage = _zone_series(o, c, h, l, n - 1)

    # atr percentile rank (volatility context)
    atr_rank = [0.5] * n
    win = 100
    for j in range(min_index, n):
        lo = max(14, j - win)
        vals = [atr[k] for k in range(lo, j + 1) if atr[k] is not None]
        if len(vals) >= 10 and atr[j] is not None:
            atr_rank[j] = sum(1 for v in vals if v <= atr[j]) / len(vals)

    rows, labels, idx = [], [], []
    for i in range(min_index, n - 1):
        feat = base_features.compute_features(candles, i)
        if feat is None:
            continue
        row = base_features.features_to_vector(feat)
        row += [
            _TREND_CODE.get(st["trend"][i], 0.0),
            float(st["bos"][i]),
            float(st["choch"][i]),
            _last_event_age(st["bos"], i),
            _last_event_age(st["choch"], i),
            float(rg["adx"][i]) / 100.0,
            float(rg["efficiency"][i]),
            min(float(rg["variance_ratio"][i]), 4.0),
            float(rg["hurst"][i]),
            1.0 if rg["defined"][i] else 0.0,
            _VOL_CODE.get(rg["volatility"][i], 0.0),
            sweep_dir[i],
            sweep_pen[i],
            sweep_rec[i],
            disp[i],
            disp_size[i],
            zin[i],
            zdir[i],
            min(zage[i], 60.0) / 60.0,
            atr_rank[i],
        ]
        rows.append(row)
        labels.append(1 if c[i + 1] > c[i] else 0)
        idx.append(i)
    return rows, labels, idx
