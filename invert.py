#!/usr/bin/env python3
"""Is the signal *anti*-predictive? Test inversion the honest way.

The live journal settled at 44.2% (p=0.03) — below a coin flip. If that is real,
flipping every signal to the opposite side would win ~55.8%. But a low hit-rate
found by looking at the data is exactly how false discoveries are born, so this
tool never inverts on the same data it measures.

Protocol (no look-ahead, no data snooping):
  1. For each (engine, period), measure the hit-rate on the FIRST 60% of history.
  2. Decide a policy from train only: normal / invert / skip
     - invert only if train hit is below 50% by more than one standard error
     - skip if train has too few signals
  3. Apply that frozen policy to the LAST 40% and report the result with a
     Wilson CI and a binomial p-value against the payout breakeven.

If inversion is a real effect it survives step 3. If it was noise it collapses.

Usage
-----
    python invert.py                       # all engines x key periods
    python invert.py --period 60
    python invert.py --engine ensemble --period 15
"""
from __future__ import annotations

import argparse
import math

from candle_store import CandleStore
from edge import make_engine, p_value


def wilson(w: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z = 1.96
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def collect(store: CandleStore, engine_name: str, period: int,
            train_frac: float = 0.6, warmup: int = 60, horizon: int = 1,
            kind: str = "otc"):
    """Return per-signal (split, up_move) for one engine/period.

    For every historical signal we record which split it falls in (train/test)
    and whether the next candle actually moved up. Whether the engine said CALL
    or PUT is kept so we can score both normal and inverted.
    """
    assets = [r[0] for r in store._db.execute(
        "SELECT DISTINCT asset FROM candles WHERE period=?", (period,)).fetchall()]
    if kind == "otc":
        assets = [a for a in assets if a.endswith("_otc")]
    elif kind == "real":
        assets = [a for a in assets if not a.endswith("_otc")]

    out = []
    for a in assets:
        cs = store.load(a, period, limit=20000)
        if len(cs) < warmup + 40:
            continue
        cut = int(len(cs) * train_frac)
        ev = make_engine(engine_name, a)
        for i in range(warmup, len(cs) - horizon):
            sig = ev(cs[:i + 1])
            if sig is None:
                continue
            move = cs[i + horizon][2] - cs[i][2]
            if move == 0:
                continue
            split = "train" if i < cut else "test"
            # did the engine's own call win?
            win = (move > 0) if sig.action == "CALL" else (move < 0)
            out.append((split, 1 if win else 0))
    return out


def decide_policy(hit_train: float, n_train: int, se: float = 1.0) -> str:
    """Choose normal / invert / skip from TRAIN data only.

    Invert only when the train hit-rate is below 50% by more than `se` standard
    errors; normal only when it is above by the same margin; otherwise skip.
    """
    if not n_train:
        return "skip"
    se_train = math.sqrt(0.25 / n_train)
    if hit_train < 0.5 - se * se_train:
        return "invert"
    if hit_train > 0.5 + se * se_train:
        return "normal"
    return "skip"


def apply_policy(policy: str, wins: int, n: int) -> tuple[int, int]:
    """Return (wins, n) after applying the frozen policy to test data."""
    if policy == "invert":
        return n - wins, n
    return wins, n


def hit(rows, split):
    r = [x for x in rows if x[0] == split]
    n = len(r)
    w = sum(x[1] for x in r)
    return w, n, (w / n if n else 0.0)


def run_pair(store: CandleStore, engine_name: str, period: int, payout: float,
             train_frac: float = 0.6, min_train: int = 100, se: float = 1.0):
    rows = collect(store, engine_name, period, train_frac)
    if not rows:
        return None
    w_tr, n_tr, h_tr = hit(rows, "train")
    w_te, n_te, h_te = hit(rows, "test")
    if n_tr < min_train or n_te < 30:
        return None

    # decide the policy using TRAIN ONLY
    policy = decide_policy(h_tr, n_tr, se)
    w_te_inv, n_te_inv = apply_policy(policy, w_te, n_te)

    lo, hi = wilson(w_te_inv, n_te_inv)
    return {
        "engine": engine_name, "period": period,
        "train_n": n_tr, "train_hit": h_tr,
        "test_n": n_te, "test_hit_raw": h_te,
        "policy": policy, "test_hit": w_te_inv / n_te_inv,
        "lo": lo, "hi": hi, "p": p_value(w_te_inv, n_te_inv),
        "breakeven": 1.0 / (1.0 + payout),
    }


def main():
    ap = argparse.ArgumentParser(description="Test whether signals are anti-predictive")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--engine", default="all",
                    choices=["all", "rule", "smart", "ensemble"])
    ap.add_argument("--period", type=int, default=0)
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--kind", choices=["otc", "real", "all"], default="otc")
    ap.add_argument("--min-train", type=int, default=100)
    args = ap.parse_args()

    store = CandleStore(args.db)
    engines = ["rule", "smart", "ensemble"] if args.engine == "all" else [args.engine]
    periods = [args.period] if args.period else [5, 15, 30, 60, 120]

    print(f"\ninversion test | payout {args.payout:.0%} | breakeven "
          f"{1/(1+args.payout)*100:.1f}% | train 60% / test 40%")
    print("policy is chosen on train only, then frozen and applied to test\n")
    print(f"  {'engine':9} {'tf':>4} {'train n':>8} {'train hit':>10} "
          f"{'policy':>7} {'test n':>7} {'test hit':>9} {'95% CI':>16} {'p':>7}")

    results = []
    for eng in engines:
        for p in periods:
            r = run_pair(store, eng, p, args.payout, min_train=args.min_train)
            if r is None:
                continue
            results.append(r)
            print(f"  {r['engine']:9} {r['period']:>4} {r['train_n']:>8} "
                  f"{r['train_hit']*100:>9.1f}% {r['policy']:>7} {r['test_n']:>7} "
                  f"{r['test_hit']*100:>8.1f}% "
                  f"[{r['lo']*100:>5.1f},{r['hi']*100:>5.1f}] {r['p']:>7.3f}")

    usable = [r for r in results if r["policy"] == "invert" and r["lo"] > r["breakeven"]]
    print(f"\n  {len(usable)} of {len(results)} (engine,tf) combos: "
          f"inversion survived out-of-sample AND beats breakeven.")
    if not usable:
        print("  Inversion does NOT hold out-of-sample. The 44% live hit-rate was")
        print("  not a stable anti-signal — flipping the signals would not help.\n")
    else:
        for r in usable:
            print(f"    {r['engine']} {r['period']}s -> invert, test hit "
                  f"{r['test_hit']*100:.1f}% (CI low {r['lo']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
