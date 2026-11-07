#!/usr/bin/env python3
"""Telegram signal alerts.

Watches the stored candles, runs the same rule engine as the dashboard's
"Analyze" button, and pushes a Telegram message whenever a *new* CALL/PUT
signal appears. Nothing is ever traded — these are alerts only.

Setup (one time)
----------------
1. In Telegram, talk to @BotFather -> /newbot -> copy the bot token.
2. Send any message to your new bot (so it can reply to you).
3. Find your chat id:

       python alerts.py --whoami --token <BOT_TOKEN>

4. Export the credentials (or put them in telegram.json):

       export TELEGRAM_BOT_TOKEN=123456:ABC...
       export TELEGRAM_CHAT_ID=987654321

Usage
-----
    python alerts.py --test                 # send a test message, then exit
    python alerts.py                        # watch 5s+15s, alert on new signals
    python alerts.py --periods 5,15,60
    python alerts.py --min-score 6          # only strong setups
    python alerts.py --once                 # one scan pass (for cron)

Config file `telegram.json` (optional, git-ignored):
    {"bot_token": "...", "chat_id": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from candle_store import CandleStore, PERIODS
from scan import parse_periods, scan_all

PERIOD_LABEL = {5: "5s", 15: "15s", 30: "30s", 60: "1m",
                120: "2m", 300: "5m", 600: "10m", 900: "15m"}
STATE_PATH = "data/alert_state.json"
CONFIG_PATH = "telegram.json"


# ---------------------------------------------------------------- credentials
def load_config(args) -> tuple[str, str]:
    token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = args.chat or os.environ.get("TELEGRAM_CHAT_ID", "")
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            token = token or cfg.get("bot_token", "")
            chat = chat or str(cfg.get("chat_id", ""))
        except (OSError, ValueError) as exc:
            print(f"warning: could not read {CONFIG_PATH}: {exc}", file=sys.stderr)
    return token.strip(), chat.strip()


def api(token: str, method: str, params: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params or {}).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"telegram {method} failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"telegram {method} unreachable: {exc.reason}") from exc


def send(token: str, chat: str, text: str) -> None:
    res = api(token, "sendMessage",
              {"chat_id": chat, "text": text, "disable_web_page_preview": "true"})
    if not res.get("ok"):
        raise RuntimeError(f"telegram rejected the message: {res}")


def whoami(token: str) -> None:
    """Print chat ids that have recently messaged the bot."""
    res = api(token, "getUpdates")
    seen = {}
    for upd in res.get("result", []):
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat = msg.get("chat") or {}
        if chat.get("id"):
            name = chat.get("title") or " ".join(
                p for p in (chat.get("first_name"), chat.get("last_name")) if p)
            seen[chat["id"]] = f"{name or '?'} ({chat.get('type')})"
    if not seen:
        print("no chats found. Send a message to your bot in Telegram, then re-run.")
        return
    print("chat ids found — put the one you want in TELEGRAM_CHAT_ID:")
    for cid, desc in seen.items():
        print(f"  {cid}  {desc}")


# -------------------------------------------------------------------- message
def _clock(ts) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _countdown(secs: int) -> str:
    if secs <= 0:
        return "now"
    if secs < 60:
        return f"in {secs}s"
    return f"in {secs // 60}m{secs % 60:02d}s"


def format_signal(res) -> str:
    """Format an analysis result (Analysis object or scan_all row dict)."""
    get = res.get if isinstance(res, dict) else (lambda k, d=None: getattr(res, k, d))
    action = get("action")
    icon = {"CALL": "\U0001F7E2 CALL", "PUT": "\U0001F534 PUT"}.get(action, action)
    period = get("period")
    label = PERIOD_LABEL.get(period, f"{period}s")
    lines = [f"{icon}  {get('asset')}  {label}"]
    if get("engine") == "smart" or get("regime"):
        conf = get("confidence") or 0.0
        lines.append(f"confidence {conf:.2f}  ({get('regime') or 'range'})")
    else:
        lines.append(f"score {get('score')}  (bull {get('bull')} / bear {get('bear')})")
    entry_at, expiry_at = get("entry_at"), get("expiry_at")
    if entry_at and expiry_at:
        n = get("expiry_candles") or 1
        lines.append(f"entry {_clock(entry_at)} ({_countdown(get('entry_in') or 0)})"
                     f"  →  expiry {_clock(expiry_at)}"
                     f" ({_countdown(get('expires_in') or 0)}, {n} candle"
                     f"{'s' if n != 1 else ''})")
    price = (get("values") or {}).get("price") if not isinstance(res, dict) else None
    if price is not None:
        lines.append(f"price {price}")
    reasons = get("reasons") or []
    if reasons:
        lines.append("")
        lines += [f"- {r}" for r in reasons]
    lines += ["", "alert only — not a trade instruction."]
    return "\n".join(lines)


# ----------------------------------------------------------------------- state
def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def should_alert(prev: dict, res, cooldown: float, now: float) -> bool:
    """Alert on a new/fresh signal, not on every candle of the same run."""
    action = res["action"] if isinstance(res, dict) else res.action
    if action == "NEUTRAL":
        return False
    if prev.get("action") != action:
        return True                                  # new signal, or a flip
    return (now - prev.get("t", 0)) >= cooldown       # same direction, cooldown up


# ------------------------------------------------------------------------ scan
def scan_once(store: CandleStore, periods: list[int], min_score: int,
              token: str, chat: str, state: dict, cooldown: float,
              dry: bool, engine: str = "rule", min_conf: float = 0.5,
              expiry: int = 1) -> int:
    """One pass over every (pair, timeframe); alert on new signals only."""
    now = time.time()
    sent = 0
    rep = scan_all(store, periods, engine, include_neutral=True,
                   expiry_candles=expiry)
    for row in rep["signals"]:
        if row["action"] == "NEUTRAL":
            continue
        if engine == "smart":
            if row["confidence"] < min_conf:
                continue
        elif row["score"] < min_score:
            continue
        key = f"{row['asset']}|{row['period']}"
        prev = state.get(key, {})
        if should_alert(prev, row, cooldown, now):
            msg = format_signal({**row, "engine": engine})
            if dry:
                print(f"[dry-run]\n{msg}\n")
            else:
                send(token, chat, msg)
                strength = (f"conf {row['confidence']:.2f}" if engine == "smart"
                            else f"score {row['score']}")
                print(f"sent: {row['action']} {row['asset']} "
                      f"{row['period']}s {strength}")
            sent += 1
        state[key] = {"action": row["action"], "score": row["score"],
                      "confidence": row["confidence"], "t": now}
    return sent


def main():
    ap = argparse.ArgumentParser(description="Telegram signal alerts")
    ap.add_argument("--token", help="bot token (else $TELEGRAM_BOT_TOKEN)")
    ap.add_argument("--chat", help="chat id (else $TELEGRAM_CHAT_ID)")
    ap.add_argument("--periods", default="5,15",
                    help="timeframes: 5,15,60 or 'all' (every timeframe)")
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--min-conf", type=float, default=0.6,
                    help="smart engine: only alert at/above this confidence")
    ap.add_argument("--expiry", type=int, default=1, metavar="N",
                    help="expiry in candles after entry, shown in the alert")
    ap.add_argument("--engine", choices=["rule", "smart"], default="rule",
                    help="rule = indicator votes; smart = regime + volatility")
    ap.add_argument("--cooldown", type=float, default=300,
                    help="seconds before repeating the same signal")
    ap.add_argument("--interval", type=float, default=20, help="scan interval (s)")
    ap.add_argument("--db", default="data/candles.db")
    ap.add_argument("--whoami", action="store_true", help="list chat ids, then exit")
    ap.add_argument("--test", action="store_true", help="send a test message, then exit")
    ap.add_argument("--once", action="store_true", help="one scan pass, then exit")
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending")
    args = ap.parse_args()

    token, chat = load_config(args)

    if args.whoami:
        if not token:
            raise SystemExit("need a bot token: --token or $TELEGRAM_BOT_TOKEN")
        whoami(token)
        return

    if not args.dry_run:
        if not token:
            raise SystemExit("no bot token. Set TELEGRAM_BOT_TOKEN (or pass --token), "
                             "or create telegram.json")
        if not chat:
            raise SystemExit("no chat id. Set TELEGRAM_CHAT_ID (or pass --chat). "
                             "Find it with: python alerts.py --whoami")

    if args.test and not args.dry_run:
        send(token, chat, "\U0001F514 Market-QX signal alerts connected. "
                          "You will get a message when a new signal fires.")
        print("test message sent — check Telegram")
        return

    periods = parse_periods(args.periods)
    store = CandleStore(args.db)
    state = load_state()
    label = "all" if len(periods) == len(PERIODS) else \
        ",".join(PERIOD_LABEL.get(p, str(p)) for p in periods)
    strength = (f"min conf {args.min_conf}" if args.engine == "smart"
                else f"min score {args.min_score}")
    print(f"watching {label} ({len(periods)} timeframes) | engine {args.engine} | "
          f"{strength} | cooldown {int(args.cooldown)}s | every {int(args.interval)}s")

    if args.once:
        n = scan_once(store, periods, args.min_score, token, chat, state,
                      args.cooldown, args.dry_run, args.engine, args.min_conf,
                      args.expiry)
        save_state(state)
        print(f"done — {n} alert(s)")
        return

    try:
        while True:
            scan_once(store, periods, args.min_score, token, chat, state,
                      args.cooldown, args.dry_run, args.engine, args.min_conf,
                      args.expiry)
            save_state(state)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        save_state(state)
        print("\nstopped")


if __name__ == "__main__":
    main()
