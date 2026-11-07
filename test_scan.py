"""Checks for the all-pairs scanner (scan.py) and its alert integration."""
import os
import tempfile

from alerts import format_signal, scan_once, should_alert
from candle_store import CandleStore
from scan import entry_plan, parse_periods, scan_all

fails = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


# ------------------------------------------------------------ parse_periods
check("parse '5'", parse_periods("5") == [5])
check("parse '5,15,60'", parse_periods("5,15,60") == [5, 15, 60])
check("parse 'all' is every timeframe", len(parse_periods("all")) == 8)
check("parse '' -> all", parse_periods("") == parse_periods("all"))


# ----------------------------------------------------- scan_all over a store
def synth(n=200, step=0.5, seed=1):
    import random
    rng = random.Random(seed)
    cs, p = [], 100.0
    for i in range(n):
        o = p
        p += step + rng.uniform(-0.1, 0.1)
        cs.append([i * 5, o, p, max(o, p) + 0.05, min(o, p) - 0.05])
    return cs


tmp = tempfile.mkdtemp()
store = CandleStore(os.path.join(tmp, "t.db"))
for asset in ("AAA_otc", "BBB_otc"):
    store.save(asset, 5, synth(seed=len(asset)))
    store.save(asset, 15, synth(n=60, seed=len(asset)))
store.save("TINY_otc", 5, synth(n=10))          # too few -> should be "waiting"

# series() only returns what has data, and honours min_candles
check("series finds every stored (asset, tf)",
      len(store.series(periods=[5, 15])) == 5,
      str(sorted(store.series(periods=[5, 15]))))
check("min_candles filters short series",
      ("TINY_otc", 5) not in store.series(min_candles=52, periods=[5]))

rep = scan_all(store, [5, 15], "rule")
check("scanned counts all stored series", rep["scanned"] == 5, str(rep["scanned"]))
check("waiting counts short series", rep["waiting"] == 1, str(rep["waiting"]))
check("signals is a list", isinstance(rep["signals"], list))
check("report carries the engine", rep["engine"] == "rule")

# single-timeframe scan must not include other timeframes
rep5 = scan_all(store, [5], "rule")
check("single tf only scans that tf",
      all(s["period"] == 5 for s in rep5["signals"]) and rep5["scanned"] == 3,
      str(rep5["scanned"]))

# smart engine path runs and yields well-formed rows
reps = scan_all(store, [5, 15], "smart")
ok_rows = all(set(("asset", "period", "action", "confidence", "reasons")) <= set(r)
              for r in reps["signals"])
check("smart rows are well-formed", ok_rows)

# `top` truncates the returned list, not the count
rept = scan_all(store, [5, 15], "rule", top=1)
check("top truncates rows", len(rept["signals"]) <= 1)
check("top keeps signal_count", rept["signal_count"] >= len(rept["signals"]))


# -------------------------------------------------------------- entry / expiry
# A signal read off closed candles can only be taken from the next boundary on.
p = entry_plan(60, last_time=1000, now=1000.0)     # exactly on the boundary
check("entry at boundary when now == last+period", p["entry_at"] == 1060, str(p))
check("expiry one candle later", p["expiry_at"] == 1120, str(p))
check("expiry_candles=1 by default", p["expiry_candles"] == 1)

p = entry_plan(60, last_time=1000, now=1010.0)     # mid-candle, after boundary
check("entry is never in the past", p["entry_at"] >= 1010, str(p))
check("entry waits for the last candle's boundary", p["entry_at"] == 1060, str(p))
check("entry_in is a real countdown", p["entry_in"] == 50, str(p))

# if the scan lags a whole candle behind, entry snaps forward to a live boundary
p = entry_plan(60, last_time=1000, now=1150.0)
check("stale signal snaps to a future boundary", p["entry_at"] == 1200, str(p))

p = entry_plan(60, last_time=1000, now=999.0)      # before last close (race)
check("entry waits for the signal to close", p["entry_at"] == 1060, str(p))

p = entry_plan(5, last_time=1000, now=1003.0, expiry_candles=3)
check("expiry spans N candles", p["expiry_at"] == p["entry_at"] + 15, str(p))
check("expiry_candles reported", p["expiry_candles"] == 3)
check("expires_in = entry_in + N*period",
      p["expires_in"] == p["entry_in"] + 15, str(p))

p = entry_plan(60, last_time=1000, now=1000.0, expiry_candles=0)
check("expiry_candles floors at 1", p["expiry_candles"] == 1)

# every row the scanner emits must carry a usable, future-facing plan
rep = scan_all(store, [5, 15], "rule")
ok_plan = all(r["entry_at"] >= rep["now"] and r["expiry_at"] > r["entry_at"]
              and r["entry_in"] >= 0 and r["expires_in"] > r["entry_in"]
              for r in rep["signals"])
check("every scan row has a future entry/expiry", ok_plan)
check("report exposes 'now'", isinstance(rep["now"], int))

rep3 = scan_all(store, [5], "rule", expiry_candles=3)
ok_span = all(r["expiry_at"] - r["entry_at"] == 15 for r in rep3["signals"])
check("expiry_candles threads through scan_all", ok_span)


# ------------------------------------------------------------- alert plumbing
r = {"action": "CALL", "asset": "X", "period": 5, "score": 6,
     "bull": 6, "bear": 0, "regime": "trend", "confidence": 0.75,
     "reasons": ["uptrend"], "engine": "smart"}
check("should_alert on a new action", should_alert({}, r, 300, 1000))
check("no repeat inside cooldown",
      not should_alert({"action": "CALL", "t": 900}, r, 300, 1000))
check("repeat after cooldown",
      should_alert({"action": "CALL", "t": 600}, r, 300, 1000))
check("flip alerts immediately",
      should_alert({"action": "PUT", "t": 999}, r, 300, 1000))
check("NEUTRAL never alerts",
      not should_alert({}, {"action": "NEUTRAL"}, 300, 1000))

msg = format_signal(r)
check("smart message shows confidence", "confidence 0.75" in msg, msg.splitlines()[1])
check("message warns it is not a trade", "not a trade" in msg)

r2 = {"action": "CALL", "asset": "X", "period": 60, "score": 6,
      "bull": 6, "bear": 0, "reasons": ["uptrend"], "engine": "rule",
      "entry_at": 1000, "expiry_at": 1060, "entry_in": 10,
      "expires_in": 70, "expiry_candles": 1}
msg2 = format_signal(r2)
check("rule message shows the entry clock", "entry" in msg2 and "expiry" in msg2)
check("rule message shows the countdown", "in 10s" in msg2, msg2)
check("rule message names the expiry length", "1 candle" in msg2, msg2)

# a dry-run pass over the synthetic store must not raise and must respect min_conf
n = scan_once(store, [5, 15], 4, "", "", {}, 300, dry=True, engine="rule")
check("scan_once dry-run returns a count", isinstance(n, int), str(n))

print()
print("FAILURES:", fails if fails else "none")
