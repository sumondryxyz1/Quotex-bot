#!/usr/bin/env python3
"""Payout: one configurable place instead of a hardcoded 0.85 everywhere.

A binary option breaks even when the hit-rate clears `1 / (1 + payout)`:

    70% payout -> 58.8%    85% -> 54.1%    95% -> 51.3%

The Quotex feed does not transmit payout, so it was hardcoded to 0.85 in several
modules. That is a hidden assumption: every expectancy figure silently depended
on it. This module makes it explicit and configurable.

The payout is **user-supplied** — never scraped from an authenticated session or
any private endpoint. Set it to whatever your account actually offers per pair.

Precedence when resolving a pair's payout:
    1. an explicit per-pair value in the config file
    2. the config file's `default`
    3. `DEFAULT_PAYOUT` (conservative 0.85)

Config file `payout.json` (optional, git-ignored):
    {"default": 0.85, "pairs": {"EURUSD_otc": 0.92, "BTCUSD_otc": 0.80}}

Usage
-----
    python payout.py --show
    python payout.py --set 0.90
    python payout.py --set-pair EURUSD_otc 0.92
    python payout.py --breakeven 0.92
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

# Conservative default: the lowest plausible payout, so no result is overstated.
DEFAULT_PAYOUT = 0.85
CONFIG_PATH = "payout.json"


def breakeven(payout: float) -> float:
    """Hit-rate a strategy must beat to break even at this payout."""
    if payout <= 0:
        return 1.0
    return 1.0 / (1.0 + payout)


def expected_value(p_up: float, payout: float) -> float:
    """EV per unit staked for a CALL at probability `p_up` and this payout.

    EV = p_up * payout - (1 - p_up) = p_up * (1 + payout) - 1.
    Zero exactly at the breakeven hit-rate; positive only above it.
    """
    return p_up * (1.0 + payout) - 1.0


def profit_factor(wins: int, losses: int, payout: float) -> float:
    """Gross profit / gross loss. inf when there are no losses."""
    gross_win = wins * payout
    gross_loss = float(losses)
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


class PayoutConfig:
    """Per-pair payout with a global default, loaded from an optional JSON file."""

    def __init__(self, default: float = DEFAULT_PAYOUT,
                 pairs: Optional[dict[str, float]] = None):
        self.default = float(default)
        self.pairs = {k: float(v) for k, v in (pairs or {}).items()}

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "PayoutConfig":
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return cls(data.get("default", DEFAULT_PAYOUT),
                           data.get("pairs", {}))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        return cls()

    def save(self, path: str = CONFIG_PATH):
        with open(path, "w") as f:
            json.dump({"default": self.default, "pairs": self.pairs}, f, indent=2)

    def get(self, asset: str = "") -> float:
        """Resolve the payout for `asset` (falls back to default)."""
        return self.pairs.get(asset, self.default)

    def breakeven(self, asset: str = "") -> float:
        return breakeven(self.get(asset))


def main():
    ap = argparse.ArgumentParser(description="Payout configuration (user-supplied)")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--set", type=float, metavar="P", help="set the default payout")
    ap.add_argument("--set-pair", nargs=2, metavar=("ASSET", "P"),
                    help="set a per-pair payout")
    ap.add_argument("--breakeven", type=float, metavar="P",
                    help="print the breakeven hit-rate for payout P")
    args = ap.parse_args()

    if args.breakeven is not None:
        print(f"payout {args.breakeven:.0%} -> breakeven hit-rate "
              f"{breakeven(args.breakeven) * 100:.2f}%")
        return

    cfg = PayoutConfig.load(args.config)
    changed = False
    if args.set is not None:
        cfg.default = float(args.set)
        changed = True
    if args.set_pair:
        asset, p = args.set_pair
        cfg.pairs[asset] = float(p)
        changed = True
    if changed:
        cfg.save(args.config)
        print(f"wrote {args.config}")

    if args.show or not changed:
        print(f"config: {args.config} (exists={os.path.exists(args.config)})")
        print(f"  default payout : {cfg.default:.0%}  "
              f"(breakeven {breakeven(cfg.default) * 100:.2f}%)")
        if cfg.pairs:
            print("  per-pair overrides:")
            for a in sorted(cfg.pairs):
                print(f"    {a:16} {cfg.pairs[a]:.0%}  "
                      f"(breakeven {breakeven(cfg.pairs[a]) * 100:.2f}%)")
        else:
            print("  per-pair overrides: none")


if __name__ == "__main__":
    main()
