#!/usr/bin/env python3
"""Three trained AI models, majority vote + trend gate — measured honestly.

Three genuinely different model families are trained on the same 26 features
(see features.py) to predict whether the *next* candle closes higher:

  gbm  — GradientBoostingClassifier
  rf   — RandomForestClassifier
  logit — LogisticRegression (scaled)

Evaluation is a strict time split: train on the first 60% of each pair's history,
test on the last 40% (never shuffled). For every test candle we collect each
model's up/down call, then check:

  * each model alone
  * >=2 of 3 agreeing
  * >=2 of 3 agreeing AND agreeing with the EMA trend direction

and report hit-rate with a Wilson CI and a binomial p-value against the payout
breakeven. This is the "train three AI models, act when two agree" idea — tested,
not asserted.

Signals are alerts only — never a trade instruction.

Usage
-----
    python ai_ensemble.py --period 5
    python ai_ensemble.py --period 15 --payout 0.85 --json
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import indicators as ta
from candle_store import CandleStore
from edge import p_value
from features import compute_features, features_to_vector


def wilson(w: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def build_dataset(panels: list[tuple[str, list[list]]], cut_frac: float = 0.6):
    """Feature matrix + labels + trend vote + time-split index, pooled per pair.

    Returns (Xtr, ytr, Xte, yte, trend_te). The split is per pair (time split),
    so the test set is strictly after the training set for every pair.
    """
    Xtr, ytr, Xte, yte, trend_te = [], [], [], [], []
    for asset, cs in panels:
        n = len(cs)
        cut = int(n * cut_frac)
        if cut < 60 or n - cut < 30:
            continue
        closes = [c[2] for c in cs]
        ema_f = ta.ema(closes, 9)
        ema_s = ta.ema(closes, 21)
        for i in range(50, n - 1):
            feat = compute_features(cs, i)
            if feat is None:
                continue
            nxt = cs[i + 1][2] - cs[i + 1][1]
            if nxt == 0:
                continue
            x = features_to_vector(feat)
            y = 1 if nxt > 0 else 0
            if ema_f[i] is None or ema_s[i] is None:
                tv = 0
            else:
                tv = 1 if ema_f[i] > ema_s[i] else -1
            if i < cut:
                Xtr.append(x); ytr.append(y)
            else:
                Xte.append(x); yte.append(y); trend_te.append(tv)
    return (np.array(Xtr), np.array(ytr), np.array(Xte), np.array(yte),
            np.array(trend_te))


def run(store: CandleStore, period: int, payout: float, kind: str = "otc",
        limit: int = 0, json_out: bool = False) -> dict:
    breakeven = 1.0 / (1.0 + payout)
    assets = [r[0] for r in store._db.execute(
        "SELECT DISTINCT asset FROM candles WHERE period=?", (period,)).fetchall()]
    if kind == "otc":
        assets = [a for a in assets if a.endswith("_otc")]
    elif kind == "real":
        assets = [a for a in assets if not a.endswith("_otc")]
    if limit:
        assets = assets[:limit]
    panels = [(a, store.load(a, period, limit=20000)) for a in assets]
    panels = [(a, cs) for a, cs in panels if len(cs) >= 150]

    Xtr, ytr, Xte, yte, trend = build_dataset(panels)
    if len(Xte) == 0:
        return {"error": "no test rows"}

    models = {
        "gbm": GradientBoostingClassifier(random_state=0),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=6,
                                     min_samples_leaf=50, random_state=0, n_jobs=-1),
        "logit": None,   # scaled, built below
    }
    scaler = StandardScaler().fit(Xtr)
    models["logit"] = LogisticRegression(max_iter=2000, C=0.5)

    preds: dict[str, np.ndarray] = {}
    for name, m in models.items():
        Xin, Xout = (Xtr, Xte) if name != "logit" else (scaler.transform(Xtr), scaler.transform(Xte))
        m.fit(Xin, ytr)
        p = m.predict_proba(Xout)[:, 1]
        preds[name] = (p >= 0.5).astype(int)   # 1 = up (CALL), 0 = down (PUT)

    up = sum(preds[k] for k in preds)          # votes for CALL (0..3)
    down = 3 - up
    action = np.where(up > down, 1, np.where(down > up, 0, -1))  # -1 = abstain
    agree2 = (up >= 2) | (down >= 2)
    trend_up = (trend == 1)
    trend_down = (trend == -1)
    gate2 = agree2 & (((action == 1) & trend_up) | ((action == 0) & trend_down))
    agree3 = (up == 3) | (down == 3)

    def score(mask, pred_up):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return dict(n=0, hit=0.0, ci=[0, 0], p=1.0)
        w = int(np.sum(yte[idx] == pred_up[idx]))
        lo, hi = wilson(w, len(idx))
        return dict(n=int(len(idx)), hit=w / len(idx),
                    ci=[round(lo, 4), round(hi, 4)],
                    p=round(p_value(w, len(idx)), 4))

    out = {"period": period, "payout": payout, "breakeven": round(breakeven, 4),
           "test_rows": int(len(yte)), "pairs": len(panels), "models": {}, "rules": {}}
    for name in preds:
        out["models"][name] = score(np.ones(len(yte), bool), preds[name])
    out["rules"]["any_single_gbm"] = out["models"]["gbm"]
    out["rules"]["agree_2of3"] = score(agree2, (action == 1).astype(int))
    out["rules"]["agree_2of3_trendgate"] = score(gate2, (action == 1).astype(int))
    out["rules"]["agree_3of3"] = score(agree3, (action == 1).astype(int))

    if json_out:
        print(json.dumps(out, indent=2))
        return out

    print(f"\n=== 3 AI models, {period}s, {kind}, {len(panels)} pairs, "
          f"payout {payout:.0%} (breakeven {breakeven:.1%}) ===")
    print(f"  test rows: {len(yte):,}  (train {len(ytr):,}, strict time split)\n")
    print("  single models:")
    for name, r in out["models"].items():
        print(f"    {name:6} n={r['n']:>6}  hit={r['hit']*100:5.1f}%  "
              f"CI[{r['ci'][0]*100:4.1f},{r['ci'][1]*100:4.1f}]  p={r['p']:.3f}")
    print("  combined rules:")
    for name, r in out["rules"].items():
        if name == "any_single_gbm":
            continue
        print(f"    {name:20} n={r['n']:>6}  hit={r['hit']*100:5.1f}%  "
              f"CI[{r['ci'][0]*100:4.1f},{r['ci'][1]*100:4.1f}]  p={r['p']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Three trained AI models, majority vote")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--kind", choices=["otc", "real", "all"], default="otc")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    run(CandleStore(args.db), args.period, args.payout, args.kind,
        args.limit, args.json)


if __name__ == "__main__":
    main()
