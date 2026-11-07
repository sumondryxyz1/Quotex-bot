#!/usr/bin/env python3
"""Live signal journal — record every signal and score it honestly.

The dashboard shows signals. This records them, then waits for the expiry candle
to close and settles each one as win/loss. Over a few days you get your own
measured track record, not a backtest and not a hunch.

How settlement works
--------------------
A signal on (asset, period) at candle T enters at the next boundary and expires
`expiry_candles` later. We settle it against the close of the expiry candle
versus the entry price. That is the same thing a binary option pays on, so the
journal's win-rate is directly comparable to the payout breakeven.

Because the feed keeps feeding candles into the same store, settlement needs no
extra data: we just look up the candle that closed at the expiry time.

Usage
-----
    python journal.py --once                 # record current signals, settle done ones
    python journal.py --watch                # loop forever (every 30s)
    python journal.py --report               # show the track record so far
    python journal.py --report --asset BTCUSD_otc
"""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import time
from collections import defaultdict

from candle_store import CandleStore, PERIODS
from scan import scan_all

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}


class Journal:
    """SQLite log of signals and their settled outcomes."""

    def __init__(self, path: str = "data/journal.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = __import__("threading").Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS signals(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   logged_at INTEGER,
                   asset TEXT, period INTEGER,
                   engine TEXT,
                   action TEXT,
                   entry_at INTEGER, entry_price REAL,
                   expiry_at INTEGER, expiry_candles INTEGER,
                   score REAL, confidence REAL, regime TEXT,
                   result TEXT DEFAULT 'open',      -- open|win|loss|flat
                   settle_price REAL, settled_at INTEGER,
                   UNIQUE(asset, period, entry_at, engine, action))""")
        self._db.commit()

    def add(self, row: dict, engine: str, now: int) -> bool:
        """Insert a signal; returns False if it was already logged."""
        with self._lock:
            cur = self._db.execute(
                """INSERT OR IGNORE INTO signals(
                       logged_at, asset, period, engine, action,
                       entry_at, entry_price, expiry_at, expiry_candles,
                       score, confidence, regime)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now, row["asset"], row["period"], engine, row["action"],
                 row["entry_at"], row.get("price", 0.0), row["expiry_at"],
                 row["expiry_candles"], row.get("score", 0.0),
                 row.get("confidence", 0.0), row.get("regime", "")))
            self._db.commit()
            return cur.rowcount > 0

    def open_signals(self) -> list[tuple]:
        with self._lock:
            return self._db.execute(
                "SELECT id,asset,period,action,entry_at,entry_price,expiry_at "
                "FROM signals WHERE result='open'").fetchall()

    def settle(self, sid: int, result: str, price: float, now: int):
        with self._lock:
            self._db.execute(
                "UPDATE signals SET result=?, settle_price=?, settled_at=? WHERE id=?",
                (result, price, now, sid))
            self._db.commit()

    def rows(self, asset: str = "", engine: str = "") -> list[tuple]:
        sql = ("SELECT asset,period,engine,action,result,entry_at,expiry_at,"
               "entry_price,settle_price,score,confidence FROM signals")
        where, params = [], []
        if asset:
            where.append("asset=?"); params.append(asset)
        if engine:
            where.append("engine=?"); params.append(engine)
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            return self._db.execute(sql, params).fetchall()


def settle_open(j: Journal, store: CandleStore, now: int) -> int:
    """Close every signal whose expiry candle has finished. Returns count settled."""
    settled = 0
    for sid, asset, period, action, entry_at, entry_price, expiry_at in j.open_signals():
        if expiry_at > now:
            continue
        cs = store.load(asset, period, limit=5000)
        # the candle that closed at expiry_at is the settlement candle
        settle_candle = next((c for c in cs if c[0] == expiry_at), None)
        if settle_candle is None:
            continue                      # feed has not delivered it yet
        # entry price: the candle that closed at entry_at (its close is the entry)
        entry_candle = next((c for c in cs if c[0] == entry_at), None)
        ep = entry_candle[2] if entry_candle else entry_price
        sp = settle_candle[2]
        if sp == ep:
            result = "flat"
        elif (sp > ep) if action == "CALL" else (sp < ep):
            result = "win"
        else:
            result = "loss"
        j.settle(sid, result, sp, now)
        settled += 1
    return settled


def wilson(wins: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z, p = 1.96, wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def summary(j: "Journal", asset: str = "", engine: str = "",
            payout: float = 0.85) -> dict:
    """Machine-readable track record, shared by the CLI report and the API."""
    rows = j.rows(asset, engine)
    breakeven = 1.0 / (1.0 + payout)
    settled = [r for r in rows if r[4] in ("win", "loss", "flat")]
    wins = sum(1 for r in settled if r[4] == "win")
    losses = sum(1 for r in settled if r[4] == "loss")
    flats = sum(1 for r in settled if r[4] == "flat")
    decisive = wins + losses

    by_asset = defaultdict(lambda: [0, 0])
    for r in settled:
        by_asset[f"{r[0]}|{r[1]}"][0] += 1
        by_asset[f"{r[0]}|{r[1]}"][1] += 1 if r[4] == "win" else 0

    def _hit(w, n):
        if not n:
            return {"n": 0, "hit": 0.0, "lo": 0.0, "hi": 0.0}
        lo, hi = wilson(w, n)
        return {"n": n, "hit": round(w / n, 4),
                "lo": round(lo, 4), "hi": round(hi, 4)}

    verdict = "insufficient"
    if decisive:
        dlo, dhi = wilson(wins, decisive)
        verdict = ("above" if dlo > breakeven else
                   "below" if dhi < breakeven else "indistinguishable")

    pairs = []
    for key, (cnt, w) in by_asset.items():
        a, p = key.rsplit("|", 1)
        pairs.append({"asset": a, "period": int(p), "n": cnt,
                      "wins": w, "hit": round(w / cnt, 4) if cnt else 0.0})
    pairs.sort(key=lambda r: -r["hit"])

    return {
        "logged": len(rows),
        "settled": len(settled),
        "open": sum(1 for r in rows if r[4] == "open"),
        "wins": wins, "losses": losses, "flats": flats,
        "overall": _hit(wins, len(settled)),
        "decisive": _hit(wins, decisive),
        "payout": payout,
        "breakeven": round(breakeven, 4),
        "verdict": verdict,
        "pairs": pairs,
    }


def report(j: Journal, asset: str = "", engine: str = "", payout: float = 0.85):
    s = summary(j, asset, engine, payout)
    breakeven = s["breakeven"]
    print(f"\n=== signal journal{' — ' + asset if asset else ''}"
          f"{' — ' + engine if engine else ''} ===")
    print(f"  logged {s['logged']}  |  settled {s['settled']}  |  open {s['open']}")
    if not s["settled"]:
        print("  nothing settled yet — signals need their expiry candle to close.\n")
        return
    o = s["overall"]
    d = s["decisive"]
    print(f"  win {s['wins']}  loss {s['losses']}  flat {s['flats']}")
    print(f"  hit-rate {o['hit'] * 100:.1f}%  95% CI [{o['lo'] * 100:.1f}, {o['hi'] * 100:.1f}]")
    if d["n"]:
        print(f"  decisive (excl. flat ties): {d['hit'] * 100:.1f}%  "
              f"95% CI [{d['lo'] * 100:.1f}, {d['hi'] * 100:.1f}]  n={d['n']}")
    print(f"  breakeven at {s['payout'] * 100:.0f}% payout = {breakeven * 100:.1f}%")
    verdict = {"above": "ABOVE breakeven", "below": "BELOW breakeven",
               "indistinguishable": "not distinguishable from breakeven"}[s["verdict"]]
    print(f"  -> {verdict}")
    if s["flats"] > s["settled"] * 0.25:
        print(f"  note: {s['flats'] / s['settled'] * 100:.0f}% flat — those are pairs whose price did")
        print(f"  not move (mostly non-OTC pairs while the real market is closed).")

    shown = [p for p in s["pairs"] if p["n"] >= 10]
    if shown:
        print("\n  pairs (n>=10):")
        for p in shown[:5]:
            print(f"    {p['asset']:<14} {PERIOD_LABEL.get(p['period'], p['period']):>4}"
                  f"  {p['wins']}/{p['n']}  {p['hit'] * 100:.0f}%")
    print()


def main():
    ap = argparse.ArgumentParser(description="Live signal journal")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--journal", default="data/journal.db")
    ap.add_argument("--periods", default="5")
    ap.add_argument("--engine", choices=["rule", "smart"], default="rule")
    ap.add_argument("--expiry", type=int, default=1)
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--min-score", type=int, default=0,
                    help="only journal rule signals with score >= this")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--asset", default="")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    if args.report:
        report(Journal(args.journal), args.asset, args.engine, args.payout)
        return

    if not os.path.exists(args.db):
        raise SystemExit(f"no candle store at {args.db} — start the dashboard first")

    store = CandleStore(args.db)
    j = Journal(args.journal)

    from scan import parse_periods
    periods = parse_periods(args.periods)

    def cycle():
        now = int(time.time())
        settled = settle_open(j, store, now)
        rep = scan_all(store, periods, args.engine, top=0,
                       expiry_candles=args.expiry)
        added = 0
        for row in rep["signals"]:
            if args.engine == "rule" and row["score"] < args.min_score:
                continue
            # entry price is the close of the candle that produced the signal
            cs = store.load(row["asset"], row["period"], limit=2)
            row["price"] = cs[-1][2] if cs else 0.0
            if j.add(row, args.engine, now):
                added += 1
        print(f"  {time.strftime('%H:%M:%S')}  settled {settled}  "
              f"logged {added} new  ({rep['signal_count']} fired)", flush=True)

    if args.once:
        cycle()
        report(j, args.asset, args.engine, args.payout)
        return

    if not args.watch:
        args.watch = True
    print(f"journal watching {periods} (engine {args.engine}) — Ctrl-C to stop")
    try:
        while True:
            cycle()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
        report(j, args.asset, args.engine, args.payout)


if __name__ == "__main__":
    main()
