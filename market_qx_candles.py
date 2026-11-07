#!/usr/bin/env python3
"""Live candle data from market-qx.info (Quotex / QxBroker white-label).

How it works
------------
market-qx.info itself is a Cloudflare + geo-gated website, but its trading
terminal talks to a plain Socket.IO (Engine.IO v3) feed:

    wss://ws2.market-qx.info/socket.io/?EIO=3&transport=websocket

That feed serves public market data without a login:

  * 42["instruments/follow","<ASSET>"]   -> live price ticks
        binary frame:  [[ "<ASSET>", <epoch_seconds>, <price>, <dir> ], ...]
  * 42["instruments/update",{...}]        -> instrument metadata / period
  * 42["tick"]                            -> keep-alive / pull
  * 451-["instruments/list", ...]         -> list of all tradable assets

Authenticated accounts additionally get server-side OHLC history
(451-["history/list/v2", ...] with {"asset","period","history","candles"}).
Pass an SSID to `MarketQX(ssid=...)` to enable that path.

Candle rows are [time, open, close, high, low].

Usage
-----
    python market_qx_candles.py                 # live 1-minute candles (EURUSD_otc)
    python market_qx_candles.py EURUSD_otc 60   # asset, period seconds
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import threading
import time
import traceback
from collections import OrderedDict
from typing import Callable, Optional

import websocket


class CandleBuilder:
    """Aggregate a live price stream into OHLC candles.

    Candle row layout: [time, open, close, high, low].
    `on_closed` fires once per completed candle (when a tick opens the next one).
    """

    def __init__(self, period: int, on_closed: Optional[Callable[[list], None]] = None,
                 maxlen: int = 500):
        self.period = period
        self.candles: "OrderedDict[int, list]" = OrderedDict()
        self.maxlen = maxlen
        self.on_closed = on_closed

    def add(self, ts: float, price: float) -> list:
        bucket = int(ts // self.period) * self.period
        c = self.candles.get(bucket)
        if c is None:
            # any older buckets are now final
            if self.candles and self.on_closed:
                for closed in self._drain_before(bucket):
                    try:
                        self.on_closed(closed)
                    except Exception:
                        # a user callback error must be visible, not swallowed
                        traceback.print_exc()
            c = [bucket, price, price, price, price]
            self.candles[bucket] = c
            while len(self.candles) > self.maxlen:
                self.candles.popitem(last=False)
        else:
            c[2] = price
            c[3] = max(c[3], price)
            c[4] = min(c[4], price)
        return c

    def _drain_before(self, bucket: int) -> list[list]:
        """Pop and return every stored candle whose time is < `bucket`."""
        closed = []
        for key in list(self.candles):
            if key < bucket:
                closed.append(self.candles.pop(key))
            else:
                break
        return closed

    @property
    def closed(self) -> list[list]:
        """All candles except the still-forming last one."""
        rows = list(self.candles.values())
        return rows[:-1] if rows else []

    @property
    def forming(self) -> Optional[list]:
        return list(self.candles.values())[-1] if self.candles else None


class MarketQX:
    HOST = "ws2.market-qx.info"
    ORIGIN = "https://market-qx.info"

    def __init__(self, ssid: str = "", host: str = HOST, origin: str = ORIGIN,
                 verbose: bool = False):
        self.ssid = ssid
        self.host = host
        self.origin = origin
        self.verbose = verbose
        self.url = f"wss://{host}/socket.io/?EIO=3&transport=websocket"

        self.instruments: dict[str, dict] = {}
        self.history: dict[str, list] = {}
        self.last_price: dict[str, float] = {}
        self.instruments_ready = threading.Event()

        self._ws: Optional[websocket.WebSocket] = None
        self._stop = threading.Event()
        self._pending: Optional[str] = None
        self._tick_cb: Optional[Callable[[str, float, float], None]] = None

    # ---- connection ----
    def _headers(self) -> dict:
        return {
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "Upgrade",
            "Host": self.host,
            "Origin": self.origin,
            "Pragma": "no-cache",
            "Sec-WebSocket-Version": "13",
            "Upgrade": "websocket",
            "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode(),
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"),
        }

    def connect(self, wait_instruments: float = 75.0) -> "MarketQX":
        self._ws = websocket.create_connection(
            self.url, header=self._headers(), timeout=20,
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )
        self._ws.settimeout(3)
        self._ws.recv()  # Engine.IO open: 0{"sid":...}
        self._ws.recv()  # Socket.IO connect ack: 40
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()
        if self.ssid:
            self.send(f'42["authorization",{json.dumps({"session": self.ssid, "isDemo": 1, "tournamentId": 0, "isFastHistory": True})}]')
        if wait_instruments:
            self.instruments_ready.wait(wait_instruments)
        return self

    def _keepalive(self):
        while not self._stop.is_set():
            try:
                self.send('42["tick"]')
            except Exception:
                pass
            self._stop.wait(5)

    def send(self, msg: str):
        if self._ws is not None:
            self._ws.send(msg)

    # ---- receive / decode ----
    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                self._dispatch(self._ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                time.sleep(0.3)

    def _dispatch(self, msg):
        if self.verbose:
            print("RAW:", str(msg)[:200], flush=True)
        if isinstance(msg, bytes):
            body = msg[1:] if msg[:1] == b"\x04" else msg
            try:
                data = json.loads(body.decode("utf-8", "ignore"))
            except Exception:
                return
            event, self._pending = self._pending, None
            self._store(event, data)
            return
        text = msg.strip()
        if not (text.startswith("42") or text.startswith("45")):
            return
        try:
            arr = json.loads(text[text.find("["):])
        except Exception:
            return
        if not isinstance(arr, list) or not arr:
            return
        self._pending = arr[0]
        if len(arr) >= 2 and not (isinstance(arr[1], dict) and arr[1].get("_placeholder")):
            self._store(arr[0], arr[1])
            self._pending = None

    def _store(self, event, data):
        if event == "instruments/list" and isinstance(data, list):
            for row in data:
                if isinstance(row, list) and len(row) > 3:
                    self.instruments[row[1]] = {"id": row[0], "symbol": row[1],
                                                "name": row[2], "type": row[3]}
            if self.instruments:
                self.instruments_ready.set()
        elif event in ("history/list/v2", "chart_notification/get") and isinstance(data, dict):
            asset = data.get("asset")
            candles = data.get("candles") or data.get("history")
            if asset and isinstance(candles, list):
                self.history[asset] = candles
        # live ticks arrive as headerless list payloads
        if isinstance(data, list) and data and isinstance(data[0], list):
            for row in data:
                if len(row) == 4 and isinstance(row[0], str):
                    asset, ts, price, _dir = row
                    self.last_price[asset] = price
                    if self._tick_cb:
                        self._tick_cb(asset, float(ts), float(price))

    # ---- requests ----
    def follow(self, asset: str, period: int = 60):
        """Subscribe to live ticks for an asset (public, no auth)."""
        self.send(f'42["instruments/follow","{asset}"]')
        self.send(f'42["instruments/update",{json.dumps({"asset": asset, "period": period})}]')
        self.send('42["tick"]')

    def load_history(self, asset: str, period: int = 60, offset: int = 0):
        """Server-side OHLC history (requires an authenticated SSID)."""
        self.send(f'42["history/load",{json.dumps({"asset": asset, "index": 0, "offset": offset, "period": period, "time": int(time.time())})}]')
        self.send(f'42["chart_notification/get",{json.dumps({"asset": asset, "version": "1.0.0"})}]')

    def on_tick(self, cb: Callable[[str, float, float], None]):
        self._tick_cb = cb

    def close(self):
        self._stop.set()
        try:
            self._ws.close()
        except Exception:
            pass


def main():
    asset = sys.argv[1] if len(sys.argv) > 1 else "EURUSD_otc"
    period = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    qx = MarketQX(verbose=False).connect(wait_instruments=75)
    print(f"assets available: {len(qx.instruments)}")
    if qx.instruments and asset not in qx.instruments:
        print(f"'{asset}' not found; using '{next(iter(qx.instruments))}'")
        asset = next(iter(qx.instruments))

    builder = CandleBuilder(period)
    printed = set()

    def on_tick(a, ts, price):
        if a != asset:
            return
        candle = builder.add(ts, price)
        if candle[0] not in printed:
            printed.add(candle[0])
            print(f"  new candle  t={candle[0]} o={candle[1]} c={candle[2]} h={candle[3]} l={candle[4]}")

    qx.on_tick(on_tick)
    qx.follow(asset, period)
    print(f"streaming live {period}s candles for {asset} (Ctrl-C to stop)\n")

    last = 0
    try:
        while True:
            time.sleep(5)
            candles = list(builder.candles.values())
            if candles:
                c = candles[-1]
                print(f"[{time.strftime('%H:%M:%S')}] {asset} {period}s candle  "
                      f"o={c[1]} h={c[3]} l={c[4]} c={c[2]}  (ticks={len(candles)})")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for ticks...")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nrecent candles:")
        for c in list(builder.candles.values())[-10:]:
            print(" ", c)
        qx.close()


if __name__ == "__main__":
    main()
