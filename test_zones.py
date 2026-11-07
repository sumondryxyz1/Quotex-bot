#!/usr/bin/env python3
"""Tests for zones.py: OB/FVG detection, lifecycle, and measured validity."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zones
from zones import (find_order_block, find_fvg, zone_state, collect_zones,
                   zone_validity_stats, active_zone_at, Zone, BULL, BEAR)


def _rw(n=300, seed=0, vol=0.0006, start=1.1):
    rng = random.Random(seed)
    cs = []
    price = start
    for i in range(n):
        o = price
        price += rng.gauss(0, vol)
        c = price
        cs.append([i * 60, o, c, max(o, c) + abs(rng.gauss(0, vol / 2)),
                   min(o, c) - abs(rng.gauss(0, vol / 2))])
    return cs


def _arr(cs):
    return ([c[1] for c in cs], [c[2] for c in cs],
            [c[3] for c in cs], [c[4] for c in cs], [c[0] for c in cs])


def test_find_order_block_bullish():
    cs = _rw(60, seed=1)
    o, c, h, l, t = _arr(cs)
    # down candle then a big up displacement
    o.append(1.10); c.append(1.0995); h.append(1.1001); l.append(1.0990)  # down candle
    o.append(1.0995); c.append(1.1060); h.append(1.1061); l.append(1.0994)  # up displacement
    z = find_order_block(o, c, h, l, len(c) - 1, impulse_atr=0.5)
    assert z is not None and z.direction == BULL, z
    assert z.bottom <= z.top
    print("PASS test_find_order_block_bullish")


def test_find_fvg_bullish():
    cs = _rw(20, seed=2)
    o, c, h, l, t = _arr(cs)
    # craft a bullish gap: candle i low above candle i-2 high
    o.extend([1.0, 1.01, 1.05]); c.extend([1.005, 1.04, 1.06])
    h.extend([1.006, 1.041, 1.061]); l.extend([0.999, 1.005, 1.045])
    z = find_fvg(h, l, len(c) - 1, min_atr=0.0)
    assert z is not None and z.direction == BULL, z
    assert z.bottom < z.top
    print("PASS test_find_fvg_bullish")


def test_zone_lifecycle_exclusive_outcomes():
    # a bull zone that is touched then violated
    z = Zone("OB", BULL, top=1.10, bottom=1.09, created_idx=0)
    o = [1.0, 1.095, 1.08]; c = [1.095, 1.10, 1.08]; h = [1.096, 1.101, 1.085]
    l = [0.999, 1.093, 1.079]
    zone_state(z, o, c, h, l)
    assert z.mitigated and z.violated and not z.respected, z.as_dict()
    # a bull zone touched then respected (close back above top)
    z2 = Zone("OB", BULL, top=1.10, bottom=1.09, created_idx=0)
    o2 = [1.0, 1.095, 1.105]; c2 = [1.095, 1.095, 1.11]; h2 = [1.096, 1.101, 1.111]
    l2 = [0.999, 1.093, 1.104]
    zone_state(z2, o2, c2, h2, l2)
    assert z2.respected and not z2.violated, z2.as_dict()
    print("PASS test_zone_lifecycle_exclusive_outcomes")


def test_validity_stats_shapes():
    cs = _rw(300, seed=4)
    o, c, h, l, t = _arr(cs)
    zs = collect_zones(o, c, h, l, len(cs) - 1, kind="OB")
    st = zone_validity_stats(o, c, h, l, zs)
    assert st.n == len(zs)
    assert 0.0 <= st.respect_rate <= 1.0
    assert 0.0 <= st.violation_rate <= 1.0
    # respected and violated are disjoint per zone
    for z in zs:
        assert not (z.respected and z.violated), z.as_dict()
    print("PASS test_validity_stats_shapes")


def test_zone_state_is_causal():
    cs = _rw(200, seed=7)
    o, c, h, l, t = _arr(cs)
    i = len(cs) - 40
    z = collect_zones(o, c, h, l, i, kind="OB")[-1]
    zone_state(z, o[:i + 1], c[:i + 1], h[:i + 1], l[:i + 1], up_to=i)
    before = (z.mitigated, z.respected, z.violated)
    # extending the data must not change a past snapshot (recompute fresh zone)
    z2 = collect_zones(o, c, h, l, i, kind="OB")[-1]
    zone_state(z2, o, c, h, l, up_to=i)
    after = (z2.mitigated, z2.respected, z2.violated)
    assert before == after, (before, after)
    print("PASS test_zone_state_is_causal")


if __name__ == "__main__":
    test_find_order_block_bullish()
    test_find_fvg_bullish()
    test_zone_lifecycle_exclusive_outcomes()
    test_validity_stats_shapes()
    test_zone_state_is_causal()
    print("\nall zones tests passed")
