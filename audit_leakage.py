#!/usr/bin/env python3
"""Look-ahead bias and data-leakage audit — run this before trusting anything.

A backtest with leakage looks brilliant and is worthless. This module tries to
*prove* leakage rather than assume its absence. It runs a set of independent
checks and fails loudly if any trip.

Checks
------
1. **Future-permutation test.** Shuffle the *future* of a series and re-run a
   detector. If its present-time output changes, it read the future.
2. **Time-disjointness.** Every walk-forward fold must satisfy
   `train_end <= test_start` (with the embargo), asserted for the harness.
3. **Label-overlap test.** With `horizon=1` the label at `i` uses `close[i+1]`;
   the embargo must be >= horizon or the first test labels overlap train.
4. **Feature causality.** Every feature at index `i` must be identical whether it
   is computed on `candles[:i+1]` or on `candles[:i+2]`.
5. **Pooling guard.** The harness must refuse to pool raw returns across pairs;
   this checks `classify_source` keeps the universes apart.
6. **Coin-flip control.** A random predictor must not beat the breakeven — if it
   does, the harness leaks somewhere.

Usage
-----
    python audit_leakage.py --db data/candles.db --period 60
    python audit_leakage.py --db data/candles.db --period 60 --json
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Callable

from candle_store import CandleStore
from provenance import SOURCE_OTC, classify_source
from validation import make_folds, purged_walk_forward, coin_flip_control, _load_series


class AuditResult:
    def __init__(self, name: str, passed: bool, detail: str):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name:28} {self.detail}"


# --------------------------------------------------------------------------- #
# 1. Future-permutation
# --------------------------------------------------------------------------- #

def future_permutation(series: list[list], detector: Callable[[list[list], int], object],
                       i: int, seed: int = 0) -> AuditResult:
    """A detector's value at `i` must not depend on candles after `i`.

    The detector is handed the *whole* series (so it is free to peek) and asked
    for its value at `i`. We then shuffle everything after `i` and ask again. A
    causal detector returns the same value; a leaky one changes.
    """
    if i < 5 or i + 5 >= len(series):
        return AuditResult("future-permutation", True, "series too short (skipped)")
    base = detector(series, i)
    rng = random.Random(seed)
    changed = False
    for _ in range(5):
        shuffled = list(series)
        tail = shuffled[i + 1:]
        rng.shuffle(tail)
        shuffled[i + 1:] = tail
        if detector(shuffled, i) != base:
            changed = True
            break
    ok = not changed
    return AuditResult("future-permutation", ok,
                       "output independent of future" if ok
                       else "OUTPUT CHANGED when future was shuffled — look-ahead!")


# --------------------------------------------------------------------------- #
# 2 & 3. Fold disjointness + embargo
# --------------------------------------------------------------------------- #

def fold_disjointness(n: int, n_splits: int = 5, min_train: float = 0.4,
                      embargo: int = 5) -> AuditResult:
    folds = make_folds(n, n_splits, min_train, embargo)
    if not folds:
        return AuditResult("fold-disjointness", True, "no folds (short series, skipped)")
    for k, (tr0, tr_end, te_start, te_end) in enumerate(folds):
        if te_start < tr_end:
            return AuditResult("fold-disjointness", False,
                               f"fold {k}: test starts before train ends")
        if te_start - tr_end < embargo:
            return AuditResult("fold-disjointness", False,
                               f"fold {k}: embargo {te_start - tr_end} < {embargo}")
        if te_end <= te_start:
            return AuditResult("fold-disjointness", False, f"fold {k}: empty test")
    return AuditResult("fold-disjointness", True,
                       f"{len(folds)} folds, train_end+embargo<=test_start")


def label_overlap(horizon: int = 1, embargo: int = 5) -> AuditResult:
    ok = embargo >= horizon
    return AuditResult("label-overlap", ok,
                       f"embargo {embargo} >= horizon {horizon}" if ok
                       else f"embargo {embargo} < horizon {horizon}: labels overlap!")


# --------------------------------------------------------------------------- #
# 4. Feature causality
# --------------------------------------------------------------------------- #

def feature_causality(candles: list[list], i: int,
                      feature_fn: Callable[[list[list], int], object]) -> AuditResult:
    """Feature at `i` from candles[:i+1] must equal the same from candles[:i+2]."""
    if i < 55 or i + 1 >= len(candles):
        return AuditResult("feature-causality", True, "series too short (skipped)")
    a = feature_fn(candles[:i + 1], i)
    b = feature_fn(candles[:i + 2], i)
    ok = a == b
    return AuditResult("feature-causality", ok,
                       "features causal" if ok
                       else "feature at i changed when a later candle was added — look-ahead!")


# --------------------------------------------------------------------------- #
# 5. Pooling guard
# --------------------------------------------------------------------------- #

def pooling_guard() -> AuditResult:
    samples = ["EURUSD_otc", "BTCUSD_otc", "EURUSD", "GBPUSD",
               "synthetic_random", "synthetic_momentum"]
    got = [classify_source(s) for s in samples]
    want = ["otc", "otc", "real", "real", "synthetic", "synthetic"]
    ok = got == want
    return AuditResult("pooling-guard", ok,
                       f"classification correct {got}" if ok
                       else f"classification wrong: {got}")


# --------------------------------------------------------------------------- #
# 6. Coin-flip control
# --------------------------------------------------------------------------- #

def coin_flip_check(store: CandleStore, period: int, source: str,
                    payout: float, n_splits: int = 5, embargo: int = 5,
                    limit: int = 0, seed: int = 0) -> AuditResult:
    r = coin_flip_control(store, period, source, payout, n_splits, embargo,
                          seed=seed, limit=limit)
    m = r.pooled
    if not m.n:
        return AuditResult("coin-flip-control", True, "no data (skipped)")
    # the control must NOT clear the breakeven; its CI should straddle it
    ok = m.ci_low <= m.breakeven
    return AuditResult("coin-flip-control", ok,
                       f"hit {m.hit * 100:.1f}% CI [{m.ci_low * 100:.1f},"
                       f"{m.ci_high * 100:.1f}] vs breakeven {m.breakeven * 100:.1f}%"
                       + ("" if ok else "  — CONTROL BEAT BREAKEVEN: harness leaks!"))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run(store: CandleStore, period: int, source: str = SOURCE_OTC,
        payout: float = 0.85, embargo: int = 5, limit: int = 3,
        json_out: bool = False) -> list[AuditResult]:
    import features as feat
    import smc

    series = _load_series(store, period, source, limit)
    results: list[AuditResult] = []

    # fold disjointness on a representative length
    n_rep = max((len(cs) for _, cs in series), default=0)
    results.append(fold_disjointness(n_rep, 5, 0.4, embargo))
    results.append(label_overlap(1, embargo))
    results.append(pooling_guard())

    # future-permutation + feature causality on the first series
    if series:
        asset, cs = series[0]
        if len(cs) > 80:
            i = len(cs) // 2

            def struct_detector(candles, idx):
                highs = [c[3] for c in candles]
                lows = [c[4] for c in candles]
                return smc.structure(highs, lows, idx)

            def feature_detector(candles, idx):
                return feat.compute_features(candles, idx)

            results.append(future_permutation(cs, struct_detector, i))
            results.append(future_permutation(cs, feature_detector, i))
            results.append(feature_causality(cs, i, feature_detector))

    results.append(coin_flip_check(store, period, source, payout, 5, embargo,
                                   limit=limit))

    if json_out:
        print(json.dumps([{"name": r.name, "passed": r.passed, "detail": r.detail}
                          for r in results], indent=2))
    else:
        print(f"\n=== leakage audit | {period}s | source={source} | "
              f"payout {payout:.0%} ===")
        for r in results:
            print(r)
        bad = [r for r in results if not r.passed]
        print(f"\n  {len(results) - len(bad)}/{len(results)} checks passed"
              + ("" if not bad else f"  — {len(bad)} FAILED"))
    return results


def main():
    ap = argparse.ArgumentParser(description="Look-ahead / leakage audit")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--source", default=SOURCE_OTC)
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    run(CandleStore(args.db), args.period, args.source, args.payout,
        args.embargo, args.limit, args.json)


if __name__ == "__main__":
    main()
