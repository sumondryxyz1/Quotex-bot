#!/usr/bin/env python3
"""Train an up/down classifier on recorded candles.

Honest evaluation is the whole point here:

  * Time-series split (never shuffle) — training on future data is cheating.
  * Walk-forward validation — train on a growing window, test on the next block,
    repeated across the series.
  * Reports accuracy AND a baseline (always predicting the majority class), plus
    the AUC, so a "model" that just predicts the common direction is exposed.

Label: 1 if the *next* candle closes higher than this one, else 0.

Usage
-----
    python train_model.py data/EURUSD_otc_60.csv
    python train_model.py data/EURUSD_otc_60.csv --model out/model.pkl
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from features import FEATURES, compute_features, features_to_vector


def load_candles(path: str) -> list[list]:
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip().lstrip("-").isdigit():
                continue
            try:
                rows.append([int(float(row[0])), float(row[1]), float(row[2]),
                             float(row[3]), float(row[4])])
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def build_dataset(candles: list[list]):
    X, y, idx = [], [], []
    for i in range(len(candles) - 1):
        feat = compute_features(candles, i)
        if feat is None:
            continue
        X.append(features_to_vector(feat))
        y.append(1 if candles[i + 1][2] > candles[i][2] else 0)
        idx.append(i)
    return np.array(X), np.array(y), idx


def walk_forward(X, y, n_splits=5, min_train=0.4):
    """Expanding-window walk-forward. Returns list of (accuracy, auc, baseline)."""
    n = len(X)
    start = int(n * min_train)
    if start < 50 or n - start < n_splits * 5:
        return []
    block = (n - start) // n_splits
    out = []
    for k in range(n_splits):
        tr_end = start + k * block
        te_end = tr_end + block
        if te_end > n or tr_end < 50:
            break
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xte, yte = X[tr_end:te_end], y[tr_end:te_end]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        model = GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                           learning_rate=0.05, subsample=0.9,
                                           random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, pred)
        try:
            auc = roc_auc_score(yte, prob)
        except ValueError:
            auc = float("nan")
        baseline = max(np.mean(yte), 1 - np.mean(yte))
        out.append((acc, auc, baseline))
    return out


def main():
    ap = argparse.ArgumentParser(description="Train up/down model on candle CSV")
    ap.add_argument("csv_path")
    ap.add_argument("--model", default="out/model.pkl")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    candles = load_candles(args.csv_path)
    print(f"loaded {len(candles)} candles from {args.csv_path}")
    if len(candles) < 300:
        print("WARNING: <300 candles — results will be unreliable. "
              "Collect more data with collect_data.py")

    X, y, idx = build_dataset(candles)
    print(f"dataset: {len(X)} samples, {X.shape[1]} features, "
          f"up-rate={y.mean():.3f}")

    results = walk_forward(X, y, n_splits=args.splits)
    if results:
        accs = [r[0] for r in results]
        aucs = [r[1] for r in results if not np.isnan(r[1])]
        bases = [r[2] for r in results]
        print("\n=== walk-forward validation ===")
        for k, (acc, auc, base) in enumerate(results):
            print(f"  fold {k+1}: acc={acc:.4f}  auc={auc:.4f}  baseline={base:.4f}  "
                  f"edge={acc - base:+.4f}")
        print(f"\n  mean acc      : {np.mean(accs):.4f}")
        print(f"  mean auc      : {np.mean(aucs):.4f}" if aucs else "  auc: n/a")
        print(f"  mean baseline : {np.mean(bases):.4f}")
        print(f"  mean edge     : {np.mean(accs) - np.mean(bases):+.4f}")
        print("\n  NOTE: an edge near 0 means the model is no better than")
        print("        always guessing the majority direction. That is the")
        print("        usual, honest result for this kind of data.")
    else:
        print("\nnot enough data for walk-forward validation")

    # final model trained on everything, for live use
    final = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.9,
                                       random_state=42)
    final.fit(X, y)

    # feature importances
    imp = sorted(zip(FEATURES, final.feature_importances_),
                 key=lambda kv: kv[1], reverse=True)
    print("\ntop features:")
    for name, val in imp[:8]:
        print(f"  {name:16} {val:.4f}")

    import os
    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    with open(args.model, "wb") as f:
        pickle.dump({"model": final, "features": FEATURES,
                     "trained_on": len(X)}, f)
    print(f"\nsaved model -> {args.model}")


if __name__ == "__main__":
    main()
