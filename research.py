#!/usr/bin/env python3
"""Fast, honest research harness for finding a real edge.

The old `backtest.py` recomputes every indicator at every candle, so testing
more than a couple of ideas is hopeless. This harness precomputes each pair's
feature arrays once, then evaluates hundreds of hypotheses as boolean masks.

Discipline (this is the whole point):

  * **Time split.** Each pair's candles are cut into train / validation / test
    by time. Hypotheses are *chosen* on validation and *confirmed* on test.
  * **No peeking.** A signal at candle i is only ever scored against candle i+1.
  * **Multiple testing.** Searching 100 rules and reporting the best one is how
    people fool themselves. We count how many were tried and apply a
    Bonferroni-style bar, and we require the *test* set to clear the payout
    breakeven, not just validation.
  * **Pooled stats.** Per-pair sample sizes are tiny, so results are pooled
    across pairs, with a Wilson interval and a binomial p-value.

A binary option breaks even at 1/(1+payout): ~54.1% at 85% payout.

Usage
-----
    python research.py --period 5
    python research.py --period 60 --min-val 30
    python research.py --period 5 --list        # show the hypotheses
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional

import indicators as ta
from candle_store import CandleStore

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


# --------------------------------------------------------------------------- #
# panel: one (asset, period) with every feature precomputed once
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    asset: str
    period: int
    t: list[int]
    o: list[float]
    c: list[float]
    h: list[float]
    l: list[float]
    f: dict[str, list]        # feature name -> per-index value (None = warmup)

    def __len__(self):
        return len(self.c)


def _pct_rank_series(values: list[Optional[float]], lookback: int) -> list[Optional[float]]:
    """Fraction of the last `lookback` values that are <= the current one."""
    out: list[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if values[i] is None:
            continue
        lo = max(0, i - lookback + 1)
        window = [v for v in values[lo:i + 1] if v is not None]
        if len(window) < 5:
            continue
        out[i] = sum(1 for v in window if v <= values[i]) / len(window)
    return out


def build_panel(asset: str, period: int, candles: list[list]) -> Optional[Panel]:
    n = len(candles)
    if n < 80:
        return None
    t = [int(r[0]) for r in candles]
    o = [float(r[1]) for r in candles]
    c = [float(r[2]) for r in candles]
    h = [float(r[3]) for r in candles]
    l = [float(r[4]) for r in candles]

    ema9 = ta.ema(c, 9)
    ema21 = ta.ema(c, 21)
    ema50 = ta.ema(c, 50)
    rsi = ta.rsi(c, 14)
    _, _, hist = ta.macd(c, 12, 26, 9)
    bb_u, bb_m, bb_l = ta.bollinger(c, 20, 2.0)
    atr = ta.atr(h, l, c, 14)
    adx = ta.adx(h, l, c, 14)
    st_k, st_d = ta.stochastic(h, l, c, 14, 3, 3)

    def ret(k: int) -> list[Optional[float]]:
        return [None if i < k else (c[i] - c[i - k]) / c[i - k] for i in range(n)]

    # consecutive up/down closes
    consec_up = [0] * n
    consec_down = [0] * n
    for i in range(1, n):
        if c[i] > c[i - 1]:
            consec_up[i] = consec_up[i - 1] + 1
        elif c[i] < c[i - 1]:
            consec_down[i] = consec_down[i - 1] + 1

    # position inside the recent high-low range, 0 = at low, 1 = at high
    range_pos: list[Optional[float]] = [None] * n
    for i in range(20, n):
        hh = max(h[i - 19:i + 1])
        ll = min(l[i - 19:i + 1])
        range_pos[i] = (c[i] - ll) / (hh - ll) if hh > ll else 0.5

    # candle body / wick shape
    body: list[Optional[float]] = [None] * n
    for i in range(n):
        rng = h[i] - l[i]
        body[i] = (c[i] - o[i]) / rng if rng else 0.0

    # hour of day (UTC) as sin/cos
    hour = [(t[i] // 3600) % 24 for i in range(n)]
    hour_sin = [math.sin(2 * math.pi * x / 24) for x in hour]
    hour_cos = [math.cos(2 * math.pi * x / 24) for x in hour]

    atr_pct = _pct_rank_series(atr, 100)

    f = {
        "close": c, "open": o, "high": h, "low": l,
        "ret1": ret(1), "ret3": ret(3), "ret5": ret(5),
        "ema9": ema9, "ema21": ema21, "ema50": ema50,
        "rsi": rsi, "macd_hist": hist,
        "bb_u": bb_u, "bb_m": bb_m, "bb_l": bb_l,
        "atr": atr, "atr_pct": atr_pct, "adx": adx,
        "stoch_k": st_k, "stoch_d": st_d,
        "consec_up": consec_up, "consec_down": consec_down,
        "range_pos": range_pos, "body": body,
        "hour": hour, "hour_sin": hour_sin, "hour_cos": hour_cos,
    }
    return Panel(asset, period, t, o, c, h, l, f)


# --------------------------------------------------------------------------- #
# hypotheses: each returns (mask, direction) arrays over the panel
# direction: +1 = CALL, -1 = PUT
# --------------------------------------------------------------------------- #
def _zeros(p: Panel) -> list[int]:
    return [0] * len(p)


def _ok(p: Panel, i: int, *names: str) -> bool:
    return all(p.f[n][i] is not None for n in names)


def _generic(p: Panel, cond) -> tuple[list[bool], list[int]]:
    mask = [False] * len(p)
    direction = _zeros(p)
    for i in range(len(p)):
        r = cond(p, i)
        if r:
            mask[i] = True
            direction[i] = r
    return mask, direction


def h_streak_reversal(k: int):
    """N up closes -> PUT, N down closes -> CALL (fade the streak)."""
    def rule(p, i):
        if p.f["consec_up"][i] >= k:
            return -1
        if p.f["consec_down"][i] >= k:
            return 1
        return 0
    return f"streak_reversal_{k}", rule


def h_streak_continuation(k: int):
    """N up closes -> CALL, N down -> PUT (ride the streak)."""
    def rule(p, i):
        if p.f["consec_up"][i] >= k:
            return 1
        if p.f["consec_down"][i] >= k:
            return -1
        return 0
    return f"streak_cont_{k}", rule


def h_rsi_extreme(lo: float, hi: float):
    def rule(p, i):
        r = p.f["rsi"][i]
        if r is None:
            return 0
        if r <= lo:
            return 1
        if r >= hi:
            return -1
        return 0
    return f"rsi_extreme_{lo:.0f}_{hi:.0f}", rule


def h_range_fade(edge: float):
    """Close in the outer 10% of its recent range -> fade it."""
    def rule(p, i):
        rp = p.f["range_pos"][i]
        if rp is None:
            return 0
        if rp <= edge:
            return 1
        if rp >= 1 - edge:
            return -1
        return 0
    return f"range_fade_{edge:.2f}", rule


def h_trend_pullback():
    """EMA50 up and price dips to EMA21 -> CALL; mirror for down."""
    def rule(p, i):
        if i < 1:
            return 0
        e21, e50, c = p.f["ema21"][i], p.f["ema50"][i], p.f["close"][i]
        e50p = p.f["ema50"][i - 1]
        if e21 is None or e50 is None or e50p is None:
            return 0
        if e50 > e50p and c <= e21:
            return 1
        if e50 < e50p and c >= e21:
            return -1
        return 0
    return "trend_pullback", rule


def h_vol_regime_breakout():
    """High ATR percentile + close above recent high -> CALL (breakout)."""
    def rule(p, i):
        ap = p.f["atr_pct"][i]
        if ap is None or i < 20:
            return 0
        if ap < 0.7:
            return 0
        hh = max(p.f["high"][i - 19:i])
        ll = min(p.f["low"][i - 19:i])
        if p.f["close"][i] > hh:
            return 1
        if p.f["close"][i] < ll:
            return -1
        return 0
    return "vol_breakout", rule


def h_low_vol_fade():
    """Low ATR percentile + close at range edge -> fade (range play)."""
    def rule(p, i):
        ap = p.f["atr_pct"][i]
        rp = p.f["range_pos"][i]
        if ap is None or rp is None:
            return 0
        if ap > 0.35:
            return 0
        if rp <= 0.1:
            return 1
        if rp >= 0.9:
            return -1
        return 0
    return "low_vol_fade", rule


def h_bb_reversion():
    def rule(p, i):
        c = p.f["close"][i]
        u, m, lo = p.f["bb_u"][i], p.f["bb_m"][i], p.f["bb_l"][i]
        if u is None or lo is None:
            return 0
        if c <= lo:
            return 1
        if c >= u:
            return -1
        return 0
    return "bb_reversion", rule


def h_bb_breakout():
    def rule(p, i):
        c = p.f["close"][i]
        u, lo = p.f["bb_u"][i], p.f["bb_l"][i]
        if u is None or lo is None:
            return 0
        if c > u:
            return 1
        if c < lo:
            return -1
        return 0
    return "bb_breakout", rule


def h_hour(hour: int, direction: int):
    def rule(p, i):
        return direction if p.f["hour"][i] == hour else 0
    return f"hour_{hour}_{'C' if direction > 0 else 'P'}", rule


def h_stoch_cross():
    def rule(p, i):
        k, d = p.f["stoch_k"][i], p.f["stoch_d"][i]
        if k is None or d is None:
            return 0
        pk, pd = p.f["stoch_k"][i - 1], p.f["stoch_d"][i - 1]
        if pk is None or pd is None:
            return 0
        if pk <= pd and k > d and k < 30:
            return 1
        if pk >= pd and k < d and k > 70:
            return -1
        return 0
    return "stoch_cross", rule


def h_macd_cross():
    def rule(p, i):
        h = p.f["macd_hist"][i]
        ph = p.f["macd_hist"][i - 1] if i else None
        if h is None or ph is None:
            return 0
        if ph <= 0 and h > 0:
            return 1
        if ph >= 0 and h < 0:
            return -1
        return 0
    return "macd_cross", rule


def h_body_reversal():
    """Strong body candle -> fade the next one (overreaction)."""
    def rule(p, i):
        b = p.f["body"][i]
        if b is None:
            return 0
        if b >= 0.7:
            return -1
        if b <= -0.7:
            return 1
        return 0
    return "body_reversal", rule


def h_always(direction: int):
    def rule(p, i):
        return direction
    return f"always_{'C' if direction > 0 else 'P'}", rule


def all_rules() -> list[tuple[str, Callable]]:
    rules: list[tuple[str, Callable]] = []
    for k in (2, 3, 4, 5):
        rules.append(h_streak_reversal(k))
        rules.append(h_streak_continuation(k))
    for lo, hi in ((20, 80), (25, 75), (30, 70), (35, 65)):
        rules.append(h_rsi_extreme(lo, hi))
    for e in (0.05, 0.1, 0.2):
        rules.append(h_range_fade(e))
    rules += [h_trend_pullback(), h_vol_regime_breakout(), h_low_vol_fade(),
              h_bb_reversion(), h_bb_breakout(), h_stoch_cross(), h_macd_cross(),
              h_body_reversal(), h_always(1), h_always(-1)]
    for hour in range(24):
        rules.append(h_hour(hour, 1))
    return rules


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def wilson(wins: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z, p = 1.96, wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def p_value(wins: int, n: int) -> float:
    if not n:
        return 1.0
    if n >= 30:
        z = (wins - n / 2) / math.sqrt(n / 4)
        return math.erfc(abs(z) / math.sqrt(2))
    k = min(wins, n - wins)
    total = sum(math.comb(n, j) for j in range(k + 1))
    return min(1.0, 2 * total / (2 ** n))


@dataclass
class Score:
    n: int = 0
    wins: int = 0

    def hit(self) -> float:
        return self.wins / self.n if self.n else 0.0


def score_rule(rule, panels: list[Panel], lo_i: int, hi_i: int,
               horizon: int = 1) -> Score:
    """Score a rule over the index window [lo_i, hi_i) of every panel.

    `horizon` is the expiry in candles: the trade enters at the close of the
    signal candle and settles at the close `horizon` candles later — the same
    close-to-close basis the live journal uses.
    """
    s = Score()
    for p in panels:
        mask, direction = _generic(p, rule)
        n = len(p)
        hi = min(hi_i, n - horizon)     # need i+horizon to exist
        for i in range(max(lo_i, 1), hi):
            if not mask[i]:
                continue
            move = p.c[i + horizon] - p.c[i]
            if move == 0:
                continue
            s.n += 1
            won = (move > 0) if direction[i] > 0 else (move < 0)
            s.wins += 1 if won else 0
    return s


def is_otc(asset: str) -> bool:
    return asset.endswith("_otc")


def flat_rate(panel: Panel, horizon: int = 1) -> float:
    """Fraction of candles with no net move — a dead/closed-market pair."""
    n = flat = 0
    for i in range(len(panel) - horizon):
        n += 1
        if panel.c[i + horizon] == panel.c[i]:
            flat += 1
    return flat / n if n else 1.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Fast honest edge search")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--engine", default="", help="(unused; harness is rule-based)")
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--min-val", type=int, default=40,
                    help="min signals on validation to consider a rule")
    ap.add_argument("--min-test", type=int, default=30,
                    help="min signals on test to accept a rule")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--horizon", type=int, default=1,
                    help="expiry in candles (1 = next candle)")
    ap.add_argument("--kind", choices=["all", "otc", "real"], default="all",
                    help="restrict to OTC (synthetic) or real pairs")
    ap.add_argument("--max-flat", type=float, default=0.5,
                    help="drop pairs with more than this flat rate (dead/closed)")
    ap.add_argument("--list", action="store_true", help="list hypotheses and exit")
    ap.add_argument("--db", default="data/candles.db")
    args = ap.parse_args()

    rules = all_rules()
    if args.list:
        print(f"{len(rules)} hypotheses:")
        for name, _ in rules:
            print(" ", name)
        return

    store = CandleStore(args.db)
    assets = [r[0] for r in store._db.execute(
        "SELECT DISTINCT asset FROM candles WHERE period=?", (args.period,)).fetchall()]
    if args.kind == "otc":
        assets = [a for a in assets if is_otc(a)]
    elif args.kind == "real":
        assets = [a for a in assets if not is_otc(a)]

    panels, dropped_flat = [], 0
    for a in assets:
        cs = store.load(a, args.period, limit=20000)
        p = build_panel(a, args.period, cs)
        if not p:
            continue
        if flat_rate(p, args.horizon) > args.max_flat:
            dropped_flat += 1
            continue
        panels.append(p)

    if not panels:
        print(f"no usable data for {PERIOD_LABEL.get(args.period, args.period)} "
              f"(kind={args.kind})")
        return

    n = min(len(p) for p in panels)
    cut1 = int(n * args.train_frac)
    cut2 = int(n * (args.train_frac + args.val_frac))
    label = PERIOD_LABEL.get(args.period, f"{args.period}s")
    breakeven = 1.0 / (1.0 + args.payout)

    print(f"\nresearch | {label} | {len(panels)} pairs (kind={args.kind}) | ~{n} candles/pair")
    if dropped_flat:
        print(f"dropped {dropped_flat} dead pairs (flat rate > {args.max_flat:.0%})")
    print(f"split: train [0,{cut1}) val [{cut1},{cut2}) test [{cut2},{n})")
    print(f"horizon {args.horizon} candle(s) | payout {args.payout * 100:.0f}% "
          f"-> breakeven {breakeven * 100:.1f}%")
    print(f"{len(rules)} hypotheses tried (multiple-testing aware)\n")

    # selection on validation
    candidates = []
    all_val = []
    for name, rule in rules:
        s = score_rule(rule, panels, cut1, cut2, args.horizon)
        if s.n >= args.min_val:
            all_val.append((name, rule, s))
            if s.hit() > breakeven:
                candidates.append((name, rule, s))
    candidates.sort(key=lambda r: -r[2].hit())
    all_val.sort(key=lambda r: -r[2].hit())

    print(f"top validation hit-rates (breakeven {breakeven * 100:.1f}%):")
    for name, _, s in all_val[:12]:
        flag = " clears" if s.hit() > breakeven else ""
        print(f"    {name:<22} n={s.n:>6}  {s.hit() * 100:>5.1f}%{flag}")
    print()

    print(f"rules clearing breakeven on validation: {len(candidates)}")
    if not candidates:
        print("  none. Nothing to confirm out-of-sample.\n")
        return

    # Bonferroni bar on the test set: with m candidates, require p < 0.05/m
    m = len(candidates)
    bar = 0.05 / m
    print(f"Bonferroni bar for confirmation: p < {bar:.4f} (m={m})\n")
    print(f"  {'rule':<22} {'val n':>6} {'val%':>6} {'test n':>7} {'test%':>7} "
          f"{'95% CI':>16} {'p':>8}  verdict")

    survived = 0
    for name, rule, sval in candidates:
        st = score_rule(rule, panels, cut2, n, args.horizon)
        lo, hi = wilson(st.wins, st.n)
        pv = p_value(st.wins, st.n)
        ok = (st.n >= args.min_test and lo > breakeven and pv < bar)
        survived += ok
        print(f"  {name:<22} {sval.n:>6} {sval.hit() * 100:>5.1f}% "
              f"{st.n:>7} {st.hit() * 100:>6.1f}% [{lo * 100:>5.1f},{hi * 100:>5.1f}] "
              f"{pv:>8.4f}  {'EDGE' if ok else '-'}")

    print(f"\n  {survived} of {m} validation survivors confirmed on held-out test.")
    if not survived:
        print("  No real edge found. These were validation noise.\n")
    else:
        print("  Confirmed edge(s) — re-verify with more data before trusting.\n")


if __name__ == "__main__":
    main()
