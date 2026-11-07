#!/usr/bin/env python3
"""Honest validation of the v2 feature set — does the extra context predict?

Trains a model on `features_v2.FEATURES_V2` and judges it the only way that
matters: purged walk-forward accuracy vs the majority-class baseline, the
payout-adjusted hit-rate, and permutation importance on held-out folds (to see
which features, if any, carry signal rather than noise).

Then it runs the **random-walk negative control**: the same pipeline on
synthetic random-walk candles. If the v2 features look good on a random walk,
the pipeline is leaking, not learning. A pipeline that reports ~baseline on a
random walk and ~baseline on real data is working correctly and honestly saying
"no edge found".

Usage
-----
    python ml_validate.py --period 60 --source otc
    python ml_validate.py --period 60 --source otc --random-walk
"""
from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import features_v2
from candle_store import CandleStore
from validation import make_folds, metrics_from_outcomes
from payout import DEFAULT_PAYOUT


def _dataset_from_candles(candles: list[list]):
    rows, labels, idx = features_v2.build_matrix(candles)
    if not rows:
        return None
    return np.array(rows, dtype=float), np.array(labels, dtype=int), np.array(idx)


def _random_walk_candles(n: int, seed: int = 0, vol: float = 0.0006,
                         start: float = 1.1) -> list[list]:
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


def run(candles: list[list], n_splits: int = 5, min_train: float = 0.4,
        embargo: int = 1, payout: float = DEFAULT_PAYOUT, label: str = "") -> dict:
    """Purged walk-forward on the v2 features. Returns a result dict."""
    ds = _dataset_from_candles(candles)
    if ds is None:
        return {"label": label, "error": "not enough data"}
    X, y, idx = ds
    folds = make_folds(len(X), n_splits=n_splits, min_train=min_train,
                       embargo=embargo)
    if not folds:
        return {"label": label, "error": f"only {len(X)} samples"}

    fold_acc, fold_base, fold_hit, fold_n = [], [], [], []
    outcomes: list[bool] = []
    imp_sum = np.zeros(X.shape[1])
    imp_cnt = 0
    for tr_s, tr_e, te_s, te_e in folds:
        Xtr, ytr = X[tr_s:tr_e], y[tr_s:tr_e]
        Xte, yte = X[te_s:te_e], y[te_s:te_e]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        model = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                           learning_rate=0.05, subsample=0.9,
                                           random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        fold_acc.append(accuracy_score(yte, pred))
        fold_base.append(max(float(np.mean(yte)), 1 - float(np.mean(yte))))
        hits = int(np.sum(pred == yte))
        fold_hit.append(hits / len(yte))
        fold_n.append(len(yte))
        outcomes.extend([bool(p == t) for p, t in zip(pred, yte)])
        # permutation importance on this held-out fold
        base_acc = accuracy_score(yte, model.predict(Xte))
        for k in range(X.shape[1]):
            Xp = Xte.copy()
            rng = np.random.default_rng(1000 + k)
            rng.shuffle(Xp[:, k])
            imp_sum[k] += base_acc - accuracy_score(yte, model.predict(Xp))
        imp_cnt += 1

    m = metrics_from_outcomes(outcomes, payout=payout, name=label)
    imp = (imp_sum / imp_cnt) if imp_cnt else np.zeros(X.shape[1])
    order = np.argsort(imp)[::-1][:10]
    top = [(features_v2.FEATURES_V2[k], round(float(imp[k]), 5)) for k in order]
    return {
        "label": label, "n_samples": len(X), "n_folds": len(fold_acc),
        "mean_acc": float(np.mean(fold_acc)) if fold_acc else 0.0,
        "mean_baseline": float(np.mean(fold_base)) if fold_base else 0.0,
        "mean_edge": (float(np.mean(fold_acc)) - float(np.mean(fold_base))
                      if fold_acc else 0.0),
        "hit": m.hit, "ci": (m.ci_low, m.ci_high), "verdict": m.verdict,
        "n_trades": m.n, "ev": m.ev, "breakeven": m.breakeven,
        "top_features": top,
    }


def format_result(r: dict) -> str:
    if "error" in r:
        return f"{r['label']}: {r['error']}"
    return (f"{r['label']}\n"
            f"  samples   : {r['n_samples']}   folds: {r['n_folds']}\n"
            f"  mean acc  : {r['mean_acc']:.4f}   baseline: {r['mean_baseline']:.4f}"
            f"   edge: {r['mean_edge']:+.4f}\n"
            f"  hit-rate  : {r['hit']*100:.2f}%  CI[{r['ci'][0]*100:.2f},"
            f"{r['ci'][1]*100:.2f}]  breakeven {r['breakeven']*100:.2f}%\n"
            f"  verdict   : {r['verdict'].upper()}   (n={r['n_trades']})\n"
            f"  top feats : " + ", ".join(f"{k}:{v:+.4f}" for k, v in r["top_features"]))


def main():
    ap = argparse.ArgumentParser(description="Validate the v2 feature set honestly")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--source", default="otc", choices=["otc", "real", "synthetic"])
    ap.add_argument("--asset", default=None)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    ap.add_argument("--random-walk", action="store_true",
                    help="run the negative control on synthetic random-walk data")
    ap.add_argument("--rw-seeds", type=int, default=3)
    args = ap.parse_args()

    if args.random_walk:
        print("=== random-walk negative control (v2 features) ===")
        for s in range(args.rw_seeds):
            cs = _random_walk_candles(3000, seed=s)
            r = run(cs, n_splits=args.splits, payout=args.payout,
                    label=f"random-walk seed={s}")
            print(format_result(r))
            print()
        print("A pipeline that shows a real edge here is broken, not brilliant.")
        return

    store = CandleStore()
    from validation import _load_series
    series = _load_series(store, args.period, args.source, args.asset)
    if not series:
        print(f"no {args.source} data for period {args.period}")
        return
    for asset, candles in series:
        r = run(candles, n_splits=args.splits, payout=args.payout,
                label=f"{asset} {args.period}s [{args.source}]")
        print(format_result(r))
        print()
    print("NOTE: 'edge' near 0 and an INDISTINGUISHABLE verdict mean the extra")
    print("      features did not beat the baseline on this data. That is the")
    print("      honest result, not a failure to hide.")


if __name__ == "__main__":
    main()
