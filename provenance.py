#!/usr/bin/env python3
"""Data provenance: which market each candle/row came from, and how clean it is.

Three datasets must never be mixed. Pooling them (or pooling raw returns across
pairs with different price scales) manufactures significance that is not real —
this was already observed with the candle data.

    otc        broker-synthetic quotes (`*_otc`): run 24/7, deliberately
               unpredictable
    real       the real underlying market (non-`_otc`): only moves in its
               session; frozen otherwise
    synthetic  locally generated test data (`make_synthetic.py`): never real

Every analysis in the validation harness filters by source, so a result always
states which universe it came from.

Also provides a lightweight data-quality read (gaps, spikes, flat rate) so a
series can be rejected before it pollutes a statistic.

Usage
-----
    python provenance.py --db data/candles.db
    python provenance.py --db data/candles.db --period 60
"""
from __future__ import annotations

import argparse
import math
from typing import Iterable, Optional

SOURCE_OTC = "otc"
SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"
SOURCES = (SOURCE_OTC, SOURCE_REAL, SOURCE_SYNTHETIC)

# Feed identity, so a row can be traced back to the collector that wrote it.
FEED_ID = "market-qx.info"


def classify_source(symbol: str) -> str:
    """Map an asset symbol to its dataset.

    `synthetic` wins first (locally generated files carry these words), then the
    `_otc` suffix, else the real market.
    """
    s = (symbol or "").lower()
    if "synthetic" in s or s.startswith("random") or s.startswith("momentum"):
        return SOURCE_SYNTHETIC
    if s.endswith("_otc"):
        return SOURCE_OTC
    return SOURCE_REAL


def group_by_source(symbols: Iterable[str]) -> dict[str, list[str]]:
    """Partition symbols into {source: [symbols]}. Order within a group is kept."""
    out: dict[str, list[str]] = {s: [] for s in SOURCES}
    for sym in symbols:
        out[classify_source(sym)].append(sym)
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #

def gap_stats(candles: list[list], period: int) -> dict:
    """Count candle boundaries that are missing, given the expected `period`."""
    if len(candles) < 2:
        return {"n": len(candles), "gaps": 0, "gap_ratio": 0.0, "max_gap_s": 0}
    gaps = 0
    max_gap = 0
    for a, b in zip(candles, candles[1:]):
        dt = int(b[0]) - int(a[0])
        if dt <= 0:
            continue
        missing = dt // period - 1
        if missing > 0:
            gaps += missing
            max_gap = max(max_gap, dt)
    span = len(candles) + gaps
    return {"n": len(candles), "gaps": gaps,
            "gap_ratio": round(gaps / span, 6) if span else 0.0,
            "max_gap_s": max_gap}


def spike_stats(candles: list[list], k: float = 10.0) -> dict:
    """Count candles whose body is an outlier, using a robust (median/MAD) scale.

    A plain mean/std z-score hides a lone spike: the spike inflates the std that
    would flag it. Median + MAD is not fooled that way. When the bodies have no
    spread at all (a frozen market), any non-zero body counts as a spike.
    """
    bodies = [abs(c[2] - c[1]) for c in candles]
    if len(bodies) < 30:
        return {"spikes": 0, "spike_ratio": 0.0, "k": k}
    med = _median(bodies)
    mad = _median([abs(b - med) for b in bodies])
    scale = 1.4826 * mad
    if scale <= 0:
        spikes = sum(1 for b in bodies if b > med)
        return {"spikes": spikes, "spike_ratio": round(spikes / len(bodies), 6),
                "k": k}
    spikes = sum(1 for b in bodies if (b - med) / scale > k)
    return {"spikes": spikes, "spike_ratio": round(spikes / len(bodies), 6), "k": k}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def flat_rate(candles: list[list]) -> float:
    """Fraction of candles with open == close (a frozen / illiquid market)."""
    if not candles:
        return 0.0
    flat = sum(1 for c in candles if c[1] == c[2])
    return round(flat / len(candles), 6)


def quality(candles: list[list], period: int, *, max_gap_ratio: float = 0.05,
            max_flat_ratio: float = 0.5, spike_k: float = 10.0) -> dict:
    """Summarise a series' cleanliness and give it a 0..1 score.

    The score is deliberately simple and conservative: it starts at 1 and loses
    points for missing candles, frozen stretches, and price spikes. `tradable`
    is False when any hard threshold is breached, so the caller can drop the
    series instead of measuring noise on it.
    """
    g = gap_stats(candles, period)
    sp = spike_stats(candles, spike_k)
    fr = flat_rate(candles)
    score = 1.0
    score -= min(1.0, g["gap_ratio"] / max_gap_ratio) * 0.4
    score -= min(1.0, fr / max_flat_ratio) * 0.4
    score -= min(1.0, sp["spike_ratio"] / 0.01) * 0.2
    score = max(0.0, round(score, 4))
    return {
        "n": len(candles),
        "period": period,
        "gaps": g["gaps"],
        "gap_ratio": g["gap_ratio"],
        "flat_ratio": fr,
        "spikes": sp["spikes"],
        "spike_ratio": sp["spike_ratio"],
        "score": score,
        "tradable": (g["gap_ratio"] <= max_gap_ratio
                     and fr <= max_flat_ratio
                     and sp["spike_ratio"] <= 0.01),
    }


def series_report(store, source: Optional[str] = None,
                  period: Optional[int] = None,
                  min_candles: int = 80) -> list[dict]:
    """Quality summary for every (asset, period) in the store, filtered by source."""
    out = []
    for asset, per in store.series(min_candles=min_candles):
        if source and classify_source(asset) != source:
            continue
        if period is not None and per != period:
            continue
        cs = store.load(asset, per, limit=20000)
        q = quality(cs, per)
        q["asset"] = asset
        q["source"] = classify_source(asset)
        out.append(q)
    return out


def main():
    from candle_store import CandleStore
    ap = argparse.ArgumentParser(description="Data provenance & quality report")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--source", choices=list(SOURCES), default=None)
    ap.add_argument("--period", type=int, default=None)
    ap.add_argument("--min-candles", type=int, default=80)
    args = ap.parse_args()

    store = CandleStore(args.db)
    rows = series_report(store, args.source, args.period, args.min_candles)
    if not rows:
        print("no series match the filter")
        return
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    for src in SOURCES:
        grp = by_src.get(src)
        if not grp:
            continue
        good = sum(1 for r in grp if r["tradable"])
        print(f"\n=== source: {src}  ({len(grp)} series, {good} tradable) ===")
        grp.sort(key=lambda r: r["score"])
        for r in grp[:15]:
            flag = "ok " if r["tradable"] else "BAD"
            print(f"  [{flag}] {r['asset']:16} {r['period']:>4}s  n={r['n']:>6}  "
                  f"score={r['score']:.2f}  gap={r['gap_ratio'] * 100:4.1f}%  "
                  f"flat={r['flat_ratio'] * 100:4.1f}%  spikes={r['spikes']}")


if __name__ == "__main__":
    main()
