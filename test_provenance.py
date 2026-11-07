#!/usr/bin/env python3
"""Tests for P0 provenance: source classification, quality, store migration."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from provenance import (classify_source, group_by_source, gap_stats, spike_stats,
                        flat_rate, quality, series_report, SOURCE_OTC,
                        SOURCE_REAL, SOURCE_SYNTHETIC)
from candle_store import CandleStore


def test_classify_source():
    assert classify_source("EURUSD_otc") == SOURCE_OTC
    assert classify_source("BTCUSD_otc") == SOURCE_OTC
    assert classify_source("EURUSD") == SOURCE_REAL
    assert classify_source("GBPUSD") == SOURCE_REAL
    assert classify_source("synthetic_random") == SOURCE_SYNTHETIC
    assert classify_source("synthetic_momentum") == SOURCE_SYNTHETIC
    print("PASS test_classify_source")


def test_group_by_source_keeps_universes_apart():
    g = group_by_source(["EURUSD_otc", "EURUSD", "synthetic_random", "BTCUSD_otc"])
    assert g[SOURCE_OTC] == ["EURUSD_otc", "BTCUSD_otc"], g
    assert g[SOURCE_REAL] == ["EURUSD"], g
    assert g[SOURCE_SYNTHETIC] == ["synthetic_random"], g
    print("PASS test_group_by_source_keeps_universes_apart")


def _candles(times, closes):
    # build [t, o, c, h, l] with o = previous close
    out = []
    prev = closes[0]
    for t, c in zip(times, closes):
        out.append([t, prev, c, max(prev, c), min(prev, c)])
        prev = c
    return out


def test_gap_stats_detects_missing():
    cs = _candles([0, 60, 120, 300, 360], [1, 2, 3, 4, 5])  # 180s gap
    g = gap_stats(cs, 60)
    assert g["gaps"] == 2, g   # 300-120 = 180 -> 2 missing
    print("PASS test_gap_stats_detects_missing")


def test_flat_rate_and_quality():
    cs = [[t, 1.0, 1.0, 1.0, 1.0] for t in range(0, 600, 60)]  # all flat
    assert flat_rate(cs) == 1.0
    q = quality(cs, 60)
    assert q["tradable"] is False, q   # frozen series is not tradable
    print("PASS test_flat_rate_and_quality")


def _random_walk_candles(n=200, seed=0, start=1.1, vol=0.0004):
    import random
    rng = random.Random(seed)
    out = []
    price = start
    for i in range(n):
        o = price
        price += rng.gauss(0, vol)
        c = price
        out.append([i * 60, o, c, max(o, c) + abs(rng.gauss(0, vol / 2)),
                    min(o, c) - abs(rng.gauss(0, vol / 2))])
    return out


def test_quality_clean_series_is_tradable():
    cs = _random_walk_candles(200)
    q = quality(cs, 60)
    assert q["tradable"] is True, q
    assert q["gap_ratio"] == 0.0
    print("PASS test_quality_clean_series_is_tradable")


def test_spike_stats_finds_outlier():
    cs = _random_walk_candles(200)
    cs[50][2] = cs[50][1] + 0.5   # huge body once
    s = spike_stats(cs, k=10)
    assert s["spikes"] >= 1, s
    print("PASS test_spike_stats_finds_outlier")


def test_store_migration_backfills_source():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.db")
        s = CandleStore(path)
        s.save("EURUSD_otc", 60, [[0, 1, 2, 2, 1]])
        s.save("EURUSD", 60, [[0, 1, 2, 2, 1]])
        rows = dict(s._db.execute(
            "SELECT asset, source FROM candles").fetchall())
        assert rows["EURUSD_otc"] == "otc", rows
        assert rows["EURUSD"] == "real", rows
        # reopening the store must not break (columns already exist)
        s2 = CandleStore(path)
        assert s2.count("EURUSD_otc", 60) == 1
    print("PASS test_store_migration_backfills_source")


def test_store_series_report_by_source():
    with tempfile.TemporaryDirectory() as d:
        s = CandleStore(os.path.join(d, "c.db"))
        cs = _candles([i * 60 for i in range(100)],
                      [1.0 + 0.0001 * (i % 5) for i in range(100)])
        s.save("EURUSD_otc", 60, cs)
        s.save("EURUSD", 60, cs)
        otc = series_report(s, source="otc", period=60, min_candles=80)
        real = series_report(s, source="real", period=60, min_candles=80)
        assert len(otc) == 1 and otc[0]["asset"] == "EURUSD_otc", otc
        assert len(real) == 1 and real[0]["asset"] == "EURUSD", real
    print("PASS test_store_series_report_by_source")


if __name__ == "__main__":
    test_classify_source()
    test_group_by_source_keeps_universes_apart()
    test_gap_stats_detects_missing()
    test_flat_rate_and_quality()
    test_quality_clean_series_is_tradable()
    test_spike_stats_finds_outlier()
    test_store_migration_backfills_source()
    test_store_series_report_by_source()
    print("\nall provenance tests passed")
