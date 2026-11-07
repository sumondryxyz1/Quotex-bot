#!/usr/bin/env python3
"""Validation harness: purged walk-forward, embargo, and honest metrics.

This is the gate every strategy must pass before it is called anything. It
answers one question only: *does this strategy beat the payout breakeven on data
it has never seen, and is that result statistically real?*

Discipline (why each piece exists)
----------------------------------
* **Purged walk-forward.** Time is cut into sequential folds. A fold trains on
  everything before it and tests on the block after it. Never shuffled — a
  shuffled split on time series is leakage.
* **Embargo.** A gap is left between the train window and the test window. Without
  it, a label at the end of train (`close[i+1]` vs `close[i]`) overlaps the first
  test candles, so the model has effectively seen the future.
* **One settlement definition.** A prediction on candle `i` is scored against
  `close[i+1] - close[i]` — the close-to-close rule the journal already uses.
  (`backtest.py` historically used open-to-close; the two must not disagree.)
* **Per-source, never pooled across universes.** OTC, real and synthetic are
  measured separately. Raw returns are never pooled across pairs.
* **Payout is a parameter.** Every expectancy figure states the payout it used.
* **No edge claim without evidence.** A result is only "positive" when the
  Wilson CI's lower bound clears the breakeven; otherwise it is reported as
  indistinguishable from a coin flip.

Usage
-----
    python validation.py --db data/candles.db --engine rule --period 60
    python validation.py --db data/candles.db --engine all --period 15 --source otc
    python validation.py --control                 # random-walk / coin-flip sanity
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from candle_store import CandleStore
from provenance import SOURCE_OTC, SOURCE_REAL, SOURCE_SYNTHETIC, classify_source
from payout import DEFAULT_PAYOUT, breakeven as payout_breakeven

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


# --------------------------------------------------------------------------- #
# Statistics (pure, unit-testable against hand-computed answers)
# --------------------------------------------------------------------------- #

def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if not n:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def p_value(wins: int, n: int) -> float:
    """Two-sided binomial p-value vs 50% (normal approx for n>=30, exact below)."""
    if not n:
        return 1.0
    if n >= 30:
        z = (wins - n / 2) / math.sqrt(n / 4)
        return math.erfc(abs(z) / math.sqrt(2))
    k = min(wins, n - wins)
    total = sum(math.comb(n, j) for j in range(k + 1))
    return min(1.0, 2 * total / (2 ** n))


def max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough drop of an equity curve, as a fraction of the peak."""
    peak = equity[0] if equity else 0.0
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class Metrics:
    name: str = ""
    source: str = ""
    payout: float = DEFAULT_PAYOUT
    n: int = 0
    wins: int = 0
    hit: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    breakeven: float = 0.0
    p_value: float = 1.0
    ev: float = 0.0                 # payout-adjusted EV per unit staked
    expectancy_atr: float = 0.0     # avg signed move in ATR units (cost-adjusted)
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    verdict: str = "insufficient"   # above | below | indistinguishable | insufficient

    def as_dict(self) -> dict:
        return asdict(self)


def metrics_from_outcomes(outcomes: list[bool], payout: float = DEFAULT_PAYOUT,
                          moves_atr: Optional[list[float]] = None,
                          name: str = "", source: str = "") -> Metrics:
    """Build a Metrics record from per-trade outcomes (True = win).

    `moves_atr` (optional) is the cost-adjusted signed move per trade in ATR
    units, used for the expectancy figure. Equity for drawdown assumes a flat
    unit stake, so it is comparable across strategies.
    """
    n = len(outcomes)
    wins = sum(1 for o in outcomes if o)
    be = payout_breakeven(payout)
    m = Metrics(name=name, source=source, payout=payout, n=n, wins=wins,
                breakeven=round(be, 4))
    if n:
        m.hit = wins / n
        lo, hi = wilson(wins, n)
        m.ci_low, m.ci_high = lo, hi
        m.p_value = p_value(wins, n)
        m.ev = m.hit * (1 + payout) - 1.0
        losses = n - wins
        gross_win = wins * payout
        m.profit_factor = (gross_win / losses) if losses else (
            float("inf") if gross_win > 0 else 0.0)
        eq = [0.0]
        for o in outcomes:
            eq.append(eq[-1] + (payout if o else -1.0))
        m.max_drawdown = max_drawdown(eq)
        if m.ci_low > be:
            m.verdict = "above"
        elif m.ci_high < be:
            m.verdict = "below"
        else:
            m.verdict = "indistinguishable"
    if moves_atr:
        m.expectancy_atr = sum(moves_atr) / len(moves_atr)
    return m


# --------------------------------------------------------------------------- #
# Purged walk-forward
# --------------------------------------------------------------------------- #

def make_folds(n: int, n_splits: int = 5, min_train: float = 0.4,
               embargo: int = 0) -> list[tuple[int, int, int, int]]:
    """Sequential expanding-window folds with an embargo gap.

    Returns [(train_start, train_end, test_start, test_end), ...] where
    `test_start = train_end + embargo`. Train is always [0, train_end) so the
    window expands; the test block is the next `block` candles.

    The embargo is reserved inside the tail so every fold's test block fits:
    with `n=1000, min_train=0.4, embargo=5, n_splits=5` the first test starts at
    405 and the last ends at 1000.
    """
    if n <= 0:
        return []
    start = int(n * min_train)
    if start < 1:
        return []
    tail = n - start - embargo          # usable candles after train + embargo
    if tail < n_splits:
        return []
    block = tail // n_splits
    if block <= 0:
        return []
    folds = []
    for k in range(n_splits):
        tr_end = start + k * block
        te_start = tr_end + embargo
        te_end = te_start + block
        if te_end > n:
            break
        folds.append((0, tr_end, te_start, te_end))
    return folds


@dataclass
class FoldResult:
    fold: int
    train_n: int
    test_n: int
    metrics: Metrics


@dataclass
class WalkForwardResult:
    name: str
    source: str
    payout: float
    folds: list[FoldResult] = field(default_factory=list)
    pooled: Metrics = field(default_factory=Metrics)

    @property
    def oos_metrics(self) -> Metrics:
        """Out-of-sample metrics pooled across folds (the number that matters)."""
        return self.pooled


def _settle(prev_close: float, next_close: float, pred: int) -> Optional[bool]:
    """Score a prediction with the close-to-close rule. None for a flat candle."""
    if next_close == prev_close:
        return None
    up = next_close > prev_close
    return (pred == 1) == up


def purged_walk_forward(series: list[tuple[str, list[list]]],
                        fit_predict: Callable[[str, list[list], int, int], list[int]],
                        *, payout: float = DEFAULT_PAYOUT, n_splits: int = 5,
                        min_train: float = 0.4, embargo: int = 0,
                        horizon: int = 1, name: str = "",
                        source: str = "") -> WalkForwardResult:
    """Run purged walk-forward over one universe (all series share a `source`).

    `fit_predict(asset, candles, train_end, test_end)` must fit on candles
    `[0, train_end)` and return one prediction per index in `[test_start,
    test_end)`, where `test_start = train_end + embargo`. It must NOT read beyond
    `test_end`, and must not use candles at/after the index it predicts.
    A prediction of 1 = CALL, 0 = PUT.
    """
    res = WalkForwardResult(name=name, source=source, payout=payout)
    pooled_outcomes: list[bool] = []
    for asset, candles in series:
        n = len(candles)
        folds = make_folds(n, n_splits, min_train, embargo)
        for k, (tr0, tr_end, te_start, te_end) in enumerate(folds):
            preds = fit_predict(asset, candles, tr_end, te_end)
            # preds align with [te_start, te_end); score close-to-close
            outcomes: list[bool] = []
            for off, pred in enumerate(preds):
                i = te_start + off
                if pred is None or i < 1 or i + horizon >= n:
                    continue
                o = _settle(candles[i][2], candles[i + horizon][2], pred)
                if o is not None:
                    outcomes.append(o)
            if not outcomes:
                continue
            fm = metrics_from_outcomes(outcomes, payout, name=f"{asset}",
                                       source=source)
            res.folds.append(FoldResult(k, tr_end - tr0, len(outcomes), fm))
            pooled_outcomes.extend(outcomes)
    res.pooled = metrics_from_outcomes(pooled_outcomes, payout, name=name,
                                       source=source)
    return res


# --------------------------------------------------------------------------- #
# Engine adapters (read-only wrappers around the EXISTING engines)
# --------------------------------------------------------------------------- #

def _existing_engine_predictor(engine_name: str):
    """Return a per-candle predictor backed by an existing engine.

    A fresh engine is built per (asset, fold) so debounce/cooldown state does not
    leak across folds. The engines are used exactly as they are — not modified.

    The full causal history `candles[:i+1]` is passed to `evaluate` on every step
    so each engine's own indexing (smart/ensemble cooldown uses the absolute
    index) stays correct. The rule engine takes `(asset, candles)`; smart and
    ensemble take `(candles)`.
    """
    def fit_predict(asset, candles, train_end, test_end, embargo=0):
        if engine_name == "rule":
            from signal_bot import SignalEngine, Strategy
            eng = SignalEngine(Strategy(), threshold=4)

            def call(i):
                return eng.evaluate(asset, candles[:i + 1])
        elif engine_name == "smart":
            from smart_engine import SmartEngine, SmartStrategy
            eng = SmartEngine(SmartStrategy())

            def call(i):
                return eng.evaluate(candles[:i + 1])
        elif engine_name == "ensemble":
            from ensemble import EnsembleEngine, EnsembleStrategy
            eng = EnsembleEngine(EnsembleStrategy())

            def call(i):
                return eng.evaluate(candles[:i + 1])
        else:
            raise ValueError(f"unknown engine {engine_name!r}")

        # Replay the train window so any online state reflects train only.
        for i in range(60, train_end):
            call(i)
        preds = []
        for i in range(train_end + embargo, test_end):
            sig = call(i)
            preds.append(None if sig is None
                         else (1 if sig.action == "CALL" else 0))
        return preds

    return fit_predict


def run_engine_report(store: CandleStore, engine_name: str, period: int,
                      source: str, payout: float = DEFAULT_PAYOUT,
                      n_splits: int = 5, min_train: float = 0.4,
                      embargo: int = 0, limit: int = 0) -> WalkForwardResult:
    """Purged walk-forward for one existing engine on one source."""
    series = _load_series(store, period, source, limit)
    fp = _existing_engine_predictor(engine_name)
    # bind embargo into the predictor
    def bound(asset, candles, train_end, test_end):
        return fp(asset, candles, train_end, test_end, embargo)
    return purged_walk_forward(series, bound, payout=payout, n_splits=n_splits,
                               min_train=min_train, embargo=embargo,
                               name=engine_name, source=source)


def _load_series(store: CandleStore, period: int, source: str,
                 limit: int = 0, min_candles: int = 120) -> list[tuple[str, list[list]]]:
    out = []
    for asset, per in store.series(min_candles=min_candles, periods=[period]):
        if classify_source(asset) != source:
            continue
        out.append((asset, store.load(asset, per, limit=20000)))
    out.sort(key=lambda t: t[0])
    if limit:
        out = out[:limit]
    return out


# --------------------------------------------------------------------------- #
# Negative control: a coin flip must NOT beat the payout
# --------------------------------------------------------------------------- #

def coin_flip_control(store: CandleStore, period: int, source: str,
                      payout: float = DEFAULT_PAYOUT, n_splits: int = 5,
                      embargo: int = 0, seed: int = 0, limit: int = 0,
                      rate: float = 1.0) -> WalkForwardResult:
    """A random predictor — the null every strategy must beat.

    With `rate=1.0` it predicts every candle with a fair coin, so its hit-rate
    must land near 50% and its CI must straddle the breakeven. If this ever
    reports an edge, the harness itself is leaking.
    """
    import random
    rng = random.Random(seed)
    series = _load_series(store, period, source, limit)

    def fit_predict(asset, candles, train_end, test_end):
        preds = []
        for i in range(train_end + embargo, test_end):
            preds.append(None if rng.random() > rate
                         else (1 if rng.random() < 0.5 else 0))
        return preds

    return purged_walk_forward(series, fit_predict, payout=payout,
                               n_splits=n_splits, embargo=embargo,
                               name="coin-flip", source=source)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def format_result(r: WalkForwardResult) -> str:
    m = r.pooled
    label = PERIOD_LABEL.get(0, "")
    lines = [f"\n=== {r.name} | source={r.source} | payout {r.payout:.0%} "
             f"(breakeven {m.breakeven * 100:.2f}%) ==="]
    if not m.n:
        lines.append("  no out-of-sample predictions (not enough data)")
        return "\n".join(lines)
    lines.append(f"  folds            : {len(r.folds)}")
    lines.append(f"  OOS trades       : {m.n}  (wins {m.wins})")
    lines.append(f"  hit-rate         : {m.hit * 100:.2f}%  "
                 f"95% CI [{m.ci_low * 100:.2f}, {m.ci_high * 100:.2f}]")
    lines.append(f"  payout-adj EV    : {m.ev:+.4f} per unit staked")
    lines.append(f"  profit factor    : {m.profit_factor:.3f}")
    lines.append(f"  max drawdown     : {m.max_drawdown * 100:.2f}%")
    lines.append(f"  p-value vs 50%   : {m.p_value:.4f}")
    lines.append(f"  verdict          : {m.verdict.upper()}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Purged walk-forward validation harness")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--engine", choices=["rule", "smart", "ensemble", "all"], default="rule")
    ap.add_argument("--source", choices=[SOURCE_OTC, SOURCE_REAL, SOURCE_SYNTHETIC],
                    default=SOURCE_OTC)
    ap.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=5,
                    help="candles between train and test (default 5)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="also run the coin-flip negative control")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    store = CandleStore(args.db)
    engines = (["rule", "smart", "ensemble"] if args.engine == "all"
               else [args.engine])
    import json as _json
    out = {}
    for eng in engines:
        r = run_engine_report(store, eng, args.period, args.source,
                              args.payout, args.splits, embargo=args.embargo,
                              limit=args.limit)
        if args.json:
            out[eng] = {"pooled": r.pooled.as_dict(),
                        "folds": [f.metrics.as_dict() for f in r.folds]}
        else:
            print(format_result(r))
    if args.control:
        c = coin_flip_control(store, args.period, args.source, args.payout,
                              args.splits, embargo=args.embargo, limit=args.limit)
        if args.json:
            out["coin-flip"] = {"pooled": c.pooled.as_dict()}
        else:
            print(format_result(c))
    if args.json:
        print(_json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
