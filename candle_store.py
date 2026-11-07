#!/usr/bin/env python3
"""Multi-asset, multi-timeframe candle store.

One `MarketQX` connection feeds candle builders for every (asset, period) pair
we care about, and closed candles are persisted to a single SQLite database so
history survives restarts (the feed itself exposes no public history).

Usage
-----
    from candle_store import CandleStore, MultiFeed
    store = CandleStore("data/candles.db")
    feed = MultiFeed(["EURUSD_otc", "BTCUSD_otc"], [60, 300], store)
    feed.candles("EURUSD_otc", 60)     # list of closed candles
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Callable, Iterable, Optional

from market_qx_candles import CandleBuilder, MarketQX

# timeframes offered in the UI (seconds -> label)
TIMEFRAMES = [
    (5, "5s"), (15, "15s"), (30, "30s"),
    (60, "1m"), (120, "2m"), (300, "5m"),
    (600, "10m"), (900, "15m"),
]
PERIODS = [p for p, _ in TIMEFRAMES]


class CandleStore:
    """SQLite-backed candle history, keyed by (asset, period, time)."""

    def __init__(self, path: str = "data/candles.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS candles(
                   asset TEXT, period INTEGER, time INTEGER,
                   open REAL, close REAL, high REAL, low REAL,
                   PRIMARY KEY(asset, period, time))""")
        # Raw ticks, not aggregated. This is the only data with a chance of
        # order-flow structure: the feed sends one row per price change with the
        # broker's tick direction, which the candle table throws away.
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS ticks(
                   asset TEXT, ts REAL, price REAL, dir INTEGER)""")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ticks_asset_ts ON ticks(asset, ts)")
        self._db.commit()
        self._total = self._db.execute("SELECT COUNT(*) FROM candles").fetchone()[0]

    def save(self, asset: str, period: int, candles: Iterable[list]):
        rows = [(asset, period, c[0], c[1], c[2], c[3], c[4]) for c in candles]
        if not rows:
            return
        with self._lock:
            cur = self._db.executemany(
                "INSERT OR IGNORE INTO candles VALUES(?,?,?,?,?,?,?)", rows)
            self._db.commit()
            self._total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    @property
    def total(self) -> int:
        return self._total

    def save_ticks(self, rows: Iterable[tuple]):
        """Persist raw (asset, ts, price, dir) ticks. Batched by the caller."""
        rows = list(rows)
        if not rows:
            return
        with self._lock:
            self._db.executemany("INSERT INTO ticks VALUES(?,?,?,?)", rows)
            self._db.commit()

    def load_ticks(self, asset: str, limit: int = 100000) -> list[tuple]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts,price,dir FROM ticks WHERE asset=? "
                "ORDER BY ts DESC LIMIT ?", (asset, limit)).fetchall()
        return [tuple(r) for r in reversed(rows)]

    def tick_count(self, asset: Optional[str] = None) -> int:
        with self._lock:
            if asset:
                return self._db.execute(
                    "SELECT COUNT(*) FROM ticks WHERE asset=?", (asset,)).fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]

    def tick_assets(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute(
                "SELECT DISTINCT asset FROM ticks ORDER BY asset").fetchall()]

    def load(self, asset: str, period: int, limit: int = 5000) -> list[list]:
        with self._lock:
            rows = self._db.execute(
                "SELECT time,open,close,high,low FROM candles "
                "WHERE asset=? AND period=? ORDER BY time DESC LIMIT ?",
                (asset, period, limit)).fetchall()
        return [list(r) for r in reversed(rows)]

    def count(self, asset: str, period: int) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM candles WHERE asset=? AND period=?",
                (asset, period)).fetchone()[0]

    def assets(self) -> list[str]:
        """Every asset we already have candles for (used when discovery fails)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT asset FROM candles ORDER BY asset").fetchall()
        return [r[0] for r in rows]

    def series(self, min_candles: int = 1,
               periods: Optional[Iterable[int]] = None) -> list[tuple[str, int]]:
        """Every (asset, period) that has at least `min_candles` stored candles."""
        sql = ("SELECT asset, period, COUNT(*) c FROM candles "
               "GROUP BY asset, period HAVING c >= ?")
        params: list = [min_candles]
        if periods:
            periods = list(periods)
            sql += f" AND period IN ({','.join('?' * len(periods))})"
            params += periods
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY asset, period", params).fetchall()
        return [(r[0], r[1]) for r in rows]


class MultiFeed:
    """Feed one connection into many candle builders; persist every closed candle."""

    def __init__(self, assets: list[str], periods: list[int], store: CandleStore,
                 on_closed: Optional[Callable[[str, int, list], None]] = None,
                 on_tick: Optional[Callable[[str, float, float], None]] = None,
                 wait_instruments: float = 75.0, retries: int = 3):
        self.store = store
        self.periods = list(periods)
        self.on_closed = on_closed
        self.on_tick = on_tick
        self.lock = threading.Lock()
        self.instruments: dict[str, dict] = {}
        self.last_price: dict[str, float] = {}
        self.connected = False
        self.started = time.time()

        # Instrument discovery is occasionally empty on a fresh connection even
        # though streaming works. Retry a few times, then fall back to whatever
        # we already have candles for so the dashboard never comes up empty.
        self.qx = None
        for attempt in range(retries):
            qx = MarketQX().connect(wait_instruments=wait_instruments)
            if qx.instruments:
                self.qx = qx
                break
            qx.close()
            if attempt < retries - 1:
                time.sleep(3)
        if self.qx is None:
            self.qx = MarketQX().connect(wait_instruments=wait_instruments)

        self.instruments = dict(self.qx.instruments)
        # one connection serves discovery and streaming
        self.assets = list(assets) or sorted(self.instruments.keys())
        if not self.assets:
            self.assets = store.assets()
            print(f"discovery empty — falling back to {len(self.assets)} stored pairs",
                  flush=True)
        self._asset_set = set(self.assets)

        self.history: dict[tuple, list] = defaultdict(list)
        self._builders: dict[tuple, CandleBuilder] = {}
        for a in self.assets:
            for p in self.periods:
                self.history[(a, p)] = store.load(a, p, limit=5000)
                self._builders[(a, p)] = CandleBuilder(p, on_closed=self._cb(a, p))

        # Raw ticks are buffered and flushed in batches: one INSERT per tick
        # would serialize the whole feed on SQLite.
        self._tick_buf: list[tuple] = []
        self._tick_lock = threading.Lock()
        self.ticks_saved = 0
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

        self.qx.on_tick(self._on_tick)
        for a in self.assets:
            self.qx.follow(a, self.periods[0])
        self.connected = True

    def _cb(self, asset: str, period: int):
        def handler(candle: list):
            with self.lock:
                self.history[(asset, period)].append(candle)
                if len(self.history[(asset, period)]) > 20000:
                    del self.history[(asset, period)][:5000]
            self.store.save(asset, period, [candle])
            if self.on_closed:
                self.on_closed(asset, period, candle)
        return handler

    def _flush_loop(self):
        while True:
            time.sleep(5)
            with self._tick_lock:
                if not self._tick_buf:
                    continue
                batch, self._tick_buf = self._tick_buf, []
            self.store.save_ticks(batch)
            self.ticks_saved += len(batch)

    def _on_tick(self, asset: str, ts: float, price: float, direction=None):
        if asset not in self._asset_set:
            return
        self.last_price[asset] = price
        d = 0
        try:
            d = int(direction)
        except (TypeError, ValueError):
            d = 0
        with self._tick_lock:
            self._tick_buf.append((asset, float(ts), float(price), d))
        for p in self.periods:
            self._builders[(asset, p)].add(ts, price)
        if self.on_tick:
            self.on_tick(asset, ts, price)

    def candles(self, asset: str, period: int) -> list[list]:
        with self.lock:
            return list(self.history.get((asset, period), []))

    def forming(self, asset: str, period: int) -> Optional[list]:
        b = self._builders.get((asset, period))
        return b.forming if b else None

    def close(self):
        self.qx.close()


_CRYPTO = ("BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "SOL", "DOGE", "DOT",
           "TON", "TRX", "BNB", "XLM", "AVAX", "LINK", "MATIC", "AXS", "TRU")
_COMMOD = ("XAU", "XAG", "XPT", "OIL", "BRENT", "WTI", "NGAS", "GAS")
_FIAT = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "CNH",
         "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY", "PLN",
         "BRL", "INR", "DZD", "PKR", "ARS", "COP", "EGP", "IDR", "PHP",
         "RON", "THB", "VND", "NGN", "KES", "BDT", "LKR", "TND", "MAD")


def _guess_type(symbol: str) -> str:
    s = symbol.upper()
    if any(c in s for c in _CRYPTO):
        return "cryptocurrency"
    if any(c in s for c in _COMMOD):
        return "commodity"
    if sum(1 for c in _FIAT if c in s) >= 1 and len(s) >= 6:
        return "currency"
    return "stock"


def asset_catalog(instruments: dict[str, dict],
                  assets: Optional[list[str]] = None) -> list[dict]:
    """Group instruments by type for the UI dropdown.

    Falls back to a name-based guess when instrument metadata is unavailable
    (discovery sometimes comes back empty), so the dropdown is never blank.
    """
    groups: dict[str, list] = defaultdict(list)
    if instruments:
        for sym, info in instruments.items():
            groups[info.get("type", "other")].append(sym)
    else:
        for sym in assets or []:
            groups[_guess_type(sym)].append(sym)
    order = ["currency", "cryptocurrency", "commodity", "stock", "other"]
    out = []
    for t in order + [k for k in groups if k not in order]:
        if groups.get(t):
            out.append({"type": t, "symbols": sorted(groups[t])})
    return out
