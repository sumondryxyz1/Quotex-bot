#!/usr/bin/env python3
"""Go-live gate + paper-trading account.

The point of this file is to answer one question honestly: *is it safe to move
from demo to real money yet?* It runs a simulated (paper) account through every
signal the journal has actually settled, then checks a set of hard, statistical
criteria. Real money is only "unlocked" when those criteria pass.

This never connects to the broker and never places a trade. It is a paper
account — no money is at risk. It exists so you do not go live on a hunch.

Stages
------
  DEMO        — watching. Not enough settled signals to judge.
  NOT READY   — enough data, but no measured edge (or drawdown too deep).
  PAPER PASS  — edge holds on the settled sample; keep it on demo longer.
  LIVE-READY  — every gate passed with a safety margin; a *small* real stake
                is defensible. Scale up only if it keeps holding.

Usage
-----
    python golive.py --report              # the gate verdict + paper equity
    python golive.py --simulate            # just the paper-account curve
    python golive.py --stake 0.01 --payout 0.85 --min-n 300
    python golive.py --report --json       # machine-readable
"""
from __future__ import annotations

import argparse
import json
import math

from journal import Journal, wilson

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


def paper_curve(rows: list[tuple], stake_frac: float = 0.01,
                payout: float = 0.85, start: float = 1000.0) -> dict:
    """Run a paper account through the settled signals, oldest first.

    Each trade stakes `stake_frac` of the current balance. A win returns
    `stake * payout`, a loss costs `stake`, a flat (tie) returns the stake.
    """
    settled = [r for r in rows if r[4] in ("win", "loss", "flat")]
    settled.sort(key=lambda r: r[5] or 0)          # by entry time
    bal = peak = start
    max_dd = 0.0
    streak = worst_streak = 0
    wins = losses = flats = 0
    curve = [start]
    for r in settled:
        result = r[4]
        stake = bal * stake_frac
        if result == "win":
            bal += stake * payout
            wins += 1
            streak = 0
        elif result == "loss":
            bal -= stake
            losses += 1
            streak += 1
            worst_streak = max(worst_streak, streak)
        else:
            flats += 1
            streak = 0
        peak = max(peak, bal)
        if peak > 0:
            max_dd = max(max_dd, (peak - bal) / peak)
        curve.append(bal)
    return {
        "start": start,
        "final": round(bal, 2),
        "return_pct": round((bal / start - 1) * 100, 2) if start else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "worst_losing_streak": worst_streak,
        "stake_frac": stake_frac,
        "trades": wins + losses + flats,
        "wins": wins, "losses": losses, "flats": flats,
        "curve": [round(x, 2) for x in curve],
    }


def _p_value(wins: int, n: int) -> float:
    """Two-sided binomial p-value for the hit-rate vs 50%."""
    if not n:
        return 1.0
    if n >= 30:
        z = (wins - n / 2) / math.sqrt(n / 4)
        return math.erfc(abs(z) / math.sqrt(2))
    total = 0.0
    for i in range(n + 1):
        pr = math.comb(n, i) * 0.5 ** n
        if abs(i - n / 2) >= abs(wins - n / 2) - 1e-9:
            total += pr
    return min(1.0, total)


def gate(j: Journal, payout: float = 0.85, min_n: int = 300,
         margin: float = 0.01, min_pairs: int = 5,
         max_drawdown: float = 0.25, stake_frac: float = 0.01) -> dict:
    """Evaluate the go-live criteria. Returns a dict with a verdict + reasons."""
    breakeven = 1.0 / (1.0 + payout)
    rows = j.rows()
    settled = [r for r in rows if r[4] in ("win", "loss", "flat")]
    decisive = [r for r in settled if r[4] in ("win", "loss")]
    wins = sum(1 for r in decisive if r[4] == "win")
    n = len(decisive)

    # per-pair consistency
    by_pair: dict[tuple, list[int]] = {}
    for r in decisive:
        k = (r[0], r[1])
        by_pair.setdefault(k, [0, 0])
        by_pair[k][0] += 1
        by_pair[k][1] += 1 if r[4] == "win" else 0
    pairs_with_data = {k: v for k, v in by_pair.items() if v[0] >= 20}
    pair_hits = sorted(v[1] / v[0] for v in pairs_with_data.values())
    median_pair_hit = pair_hits[len(pair_hits) // 2] if pair_hits else 0.0
    top_share = (max(v[0] for v in by_pair.values()) / n) if n and by_pair else 0.0

    lo, hi = wilson(wins, n) if n else (0.0, 0.0)
    pv = _p_value(wins, n)
    paper = paper_curve(rows, stake_frac, payout)

    checks = [
        ("enough settled signals", n >= min_n,
         f"{n} decisive (need {min_n})"),
        ("hit-rate above breakeven", lo > breakeven + margin,
         f"CI low {lo * 100:.1f}% vs breakeven {breakeven * 100:.1f}% (+{margin * 100:.0f}%)"),
        ("statistically significant", pv < 0.01,
         f"p={pv:.4f} (need < 0.01)"),
        ("positive paper return", paper["return_pct"] > 0,
         f"{paper['return_pct']:+.2f}% over {paper['trades']} trades"),
        ("drawdown within limits", paper["max_drawdown_pct"] <= max_drawdown * 100,
         f"max DD {paper['max_drawdown_pct']:.1f}% (limit {max_drawdown * 100:.0f}%)"),
        ("consistent across pairs", len(pairs_with_data) >= min_pairs
         and median_pair_hit > breakeven,
         f"{len(pairs_with_data)} pairs with 20+ trades, median hit "
         f"{median_pair_hit * 100:.1f}%"),
        ("not one pair carrying it", top_share <= 0.5,
         f"busiest pair = {top_share * 100:.0f}% of trades (limit 50%)"),
    ]

    passed = all(ok for _, ok, _ in checks)
    if n < min_n:
        stage = "DEMO"
    elif passed:
        stage = "LIVE-READY"
    elif paper["return_pct"] > 0 and lo > breakeven:
        stage = "PAPER PASS"
    else:
        stage = "NOT READY"

    return {
        "stage": stage,
        "payout": payout,
        "breakeven": round(breakeven, 4),
        "settled": len(settled),
        "decisive": n,
        "wins": wins,
        "hit_rate": round(wins / n, 4) if n else 0.0,
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(pv, 4),
        "median_pair_hit": round(median_pair_hit, 4),
        "paper": paper,
        "checks": [{"name": nm, "pass": ok, "detail": d} for nm, ok, d in checks],
    }


def report(res: dict):
    stage_icon = {"DEMO": "⏳", "NOT READY": "🔴", "PAPER PASS": "🟡",
                  "LIVE-READY": "🟢"}[res["stage"]]
    print(f"\n=== go-live gate | {stage_icon} {res['stage']} ===")
    print(f"  decisive signals {res['decisive']}  |  hit {res['hit_rate'] * 100:.1f}%  "
          f"95% CI [{res['ci_low'] * 100:.1f}, {res['ci_high'] * 100:.1f}]  "
          f"p={res['p_value']:.4f}")
    print(f"  breakeven {res['breakeven'] * 100:.1f}%  |  "
          f"median pair hit {res['median_pair_hit'] * 100:.1f}%")
    print("\n  criteria:")
    for c in res["checks"]:
        print(f"    [{'x' if c['pass'] else ' '}] {c['name']:<28} {c['detail']}")
    p = res["paper"]
    print(f"\n  paper account (stake {p['stake_frac'] * 100:.1f}% per trade, "
          f"start {p['start']:.0f}):")
    print(f"    final {p['final']:.2f}  ({p['return_pct']:+.2f}%)  "
          f"max DD {p['max_drawdown_pct']:.1f}%  "
          f"worst losing streak {p['worst_losing_streak']}")
    print(f"    trades {p['trades']}  (W {p['wins']} / L {p['losses']} / flat {p['flats']})")
    if res["stage"] == "DEMO":
        print("\n  -> Still on demo: not enough settled signals to judge yet.")
    elif res["stage"] == "NOT READY":
        print("\n  -> Stay on demo. The measured edge is not there. Do not use real money.")
    elif res["stage"] == "PAPER PASS":
        print("\n  -> Looking promising, but keep it on demo longer before risking money.")
    else:
        print("\n  -> Gate passed. A SMALL real stake is defensible. Scale up slowly,")
        print("     and re-run this gate often — the edge can vanish.")
    print()


def main():
    ap = argparse.ArgumentParser(description="Demo -> real money readiness gate")
    ap.add_argument("--db", default="data/journal.db")
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--min-n", type=int, default=300,
                    help="minimum settled decisive signals to judge")
    ap.add_argument("--stake", type=float, default=0.01, help="paper stake fraction")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--max-drawdown", type=float, default=0.25)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report", action="store_true", help="the gate verdict (default)")
    ap.add_argument("--simulate", action="store_true", help="paper curve only")
    args = ap.parse_args()

    j = Journal(args.db)
    if args.simulate:
        p = paper_curve(j.rows(), args.stake, args.payout)
        print(json.dumps(p, indent=2) if args.json else
              f"paper: {p['start']:.0f} -> {p['final']:.2f} "
              f"({p['return_pct']:+.2f}%), max DD {p['max_drawdown_pct']:.1f}%, "
              f"{p['trades']} trades")
        return
    res = gate(j, args.payout, args.min_n, stake_frac=args.stake,
               min_pairs=args.min_pairs, max_drawdown=args.max_drawdown)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        report(res)


if __name__ == "__main__":
    main()
