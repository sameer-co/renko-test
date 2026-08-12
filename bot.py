"""
╔══════════════════════════════════════════════════════════════════╗
║      Renko ATR Strategy — 1-Year Backtester (Binance Public API) ║
║  Same signal logic as the live forward tester, run over history  ║
╚══════════════════════════════════════════════════════════════════╝

CHANGELOG — fixes applied vs the original version
---------------------------------------------------


9. [LOW] Historical fetch: added a guard against a stalled cursor
   (Binance returning a page that doesn't advance startTime), which
   could previously spin in an infinite loop.
"""

import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
#  CONFIG — everything you'd want to change lives here
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "symbol"          : "SOLUSDT",
    "timeframe"       : "30m",     # any Binance interval: 1m,3m,5m,15m,30m,1h,4h,1d ...
    "lookback_days"   : 1095,      # how far back to backtest

    "atr_period"      : 14,
    "renko_mult"      : 1.0,       # brick size = renko_mult x ATR (re-anchored per new brick)
    "sl_mult"         : 1.5,       # SL distance = sl_mult x ATR
    "risk_reward_ratio": 3.5,      # TP distance = risk_reward_ratio x SL distance (renamed from tp_mult)
    "min_sell_bricks" : 2,         # min consecutive bearish bricks before a LONG entry
    "atr_gap_mult"    : 1.0,       # duplicate-entry guard, same as live bot

    "allow_shorts"    : False,     # if True, also take SHORT trades on bearish reversals
    "min_buy_bricks"  : 2,         # min consecutive bullish bricks before a SHORT entry

    # execution assumptions
    "fee_pct"         : 0.02,      # taker fee, % of notional PER SIDE (0.04 = 0.04%, i.e. 4bps)
    "slippage_pct"    : 0.02,      # slippage, % applied to the actual fill price PER SIDE
    "exit_priority"   : "heuristic",  # "heuristic" (infer from candle open/close) or force "SL"/"TP"
                                       # for any single candle that touches both TP and SL

    "initial_capital" : 1000.0,       # USD, starting equity for the curve
    "position_sizing_mode": "risk_based",  # "risk_based": size the position so the SL distance
                                            #    equals risk_pct_per_trade of capital.
                                            # "fixed_fraction": always allocate risk_pct_per_trade
                                            #    of capital regardless of SL distance (legacy behaviour).
    "risk_pct_per_trade": 2.0,        # meaning depends on position_sizing_mode above
    "max_leverage"    : 3.0,          # cap on position size as a multiple of capital
}

BINANCE_URL = "https://api.binance.com/api/v3/klines"

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000,
}


# ─────────────────────────────────────────────────────────────
#  HISTORICAL DATA — paginated fetch of `lookback_days`
# ─────────────────────────────────────────────────────────────
def fetch_historical_klines(symbol: str, interval: str, lookback_days: int):
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval '{interval}'. "
                          f"Choose from: {list(INTERVAL_MS)}")

    step_ms = INTERVAL_MS[interval]
    end_ms  = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * 86_400_000

    all_rows = []
    cursor = start_ms
    print(f"[FETCH] {symbol} {interval} — downloading {lookback_days} days...")

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": 1000,
        }
        r = requests.get(BINANCE_URL, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        new_cursor = last_open_time + step_ms

        # FIX (low sev.): guard against a stalled cursor -> infinite loop
        if new_cursor <= cursor:
            break
        cursor = new_cursor

        # be polite to the API
        time.sleep(0.25)

        if len(rows) < 1000:
            break

    if not all_rows:
        raise RuntimeError("No candles returned — check symbol/interval.")

    # drop the last candle if it's still open (in-progress)
    if all_rows[-1][6] > end_ms:   # closeTime > now => not closed yet
        all_rows = all_rows[:-1]

    # de-dupe by open time, keep order
    seen = set()
    dedup = []
    for row in all_rows:
        if row[0] not in seen:
            seen.add(row[0])
            dedup.append(row)

    opens  = np.array([float(c[1]) for c in dedup])
    highs  = np.array([float(c[2]) for c in dedup])
    lows   = np.array([float(c[3]) for c in dedup])
    closes = np.array([float(c[4]) for c in dedup])
    times  = np.array([int(c[0])   for c in dedup])

    print(f"[FETCH] Got {len(dedup)} candles "
          f"({datetime.fromtimestamp(times[0]/1000, tz=timezone.utc):%Y-%m-%d} "
          f"→ {datetime.fromtimestamp(times[-1]/1000, tz=timezone.utc):%Y-%m-%d})")
    return opens, highs, lows, closes, times


# ─────────────────────────────────────────────────────────────
#  ATR (Wilder smoothing) — identical to the live bot
# ─────────────────────────────────────────────────────────────
def calc_atr(highs, lows, closes, period: int) -> np.ndarray:
    n   = len(closes)
    tr  = np.zeros(n)
    atr = np.zeros(n)
    s   = 0.0
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
        if i < period:
            s += tr[i]
        elif i == period:
            s += tr[i]
            atr[i] = s / period
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ─────────────────────────────────────────────────────────────
#  RENKO BUILDER — identical to the live bot, keeps candle idx
#  NOTE: brick size re-anchors to the latest ATR each time a new
#  brick forms (adaptive sizing). This is a deliberate strategy
#  choice inherited from the live bot, not a bug — flagged here
#  because it materially affects brick counts vs a fixed-ATR scheme.
# ─────────────────────────────────────────────────────────────
def build_renko(closes, atr_arr, mult: float):
    bricks  = []
    ref     = None
    ref_atr = None

    for i in range(len(closes)):
        a = atr_arr[i]
        if a == 0:
            continue
        if ref is None:
            ref     = closes[i]
            ref_atr = a
            continue

        price    = closes[i]
        brick_sz = ref_atr * mult

        while price >= ref + brick_sz:
            bricks.append({"dir": 1, "open": ref, "close": ref + brick_sz,
                           "idx": i, "atr": ref_atr})
            ref     += brick_sz
            ref_atr  = a
            brick_sz = ref_atr * mult

        while price <= ref - brick_sz:
            bricks.append({"dir": -1, "open": ref, "close": ref - brick_sz,
                           "idx": i, "atr": ref_atr})
            ref     -= brick_sz
            ref_atr  = a
            brick_sz = ref_atr * mult

    return bricks


# ─────────────────────────────────────────────────────────────
#  EXIT-AMBIGUITY RESOLUTION
#  When a single candle's range touches BOTH TP and SL, we don't
#  know which was hit first. "heuristic" infers a likely intrabar
#  path from the candle's open/close (bullish => low visited before
#  high; bearish => high before low). "SL"/"TP" force the old
#  static behaviour if you want the conservative / optimistic case.
# ─────────────────────────────────────────────────────────────
def resolve_ambiguous_exit(cfg, open_price, close_price, side):
    mode = cfg.get("exit_priority", "heuristic")
    if mode in ("SL", "TP"):
        return mode

    bullish = close_price >= open_price
    if side == "LONG":
        # bullish candle: low before high -> SL (below) hit first
        # bearish candle: high before low -> TP (above) hit first
        return "SL" if bullish else "TP"
    else:  # SHORT
        # bullish candle: low before high -> TP (below, for a short) hit first
        # bearish candle: high before low -> SL (above, for a short) hit first
        return "TP" if bullish else "SL"


def apply_fill_prices(side, entry, exit_price, slip_frac):
    """Slippage moves each fill against you, applied directly to price."""
    if side == "LONG":
        entry_fill = entry * (1 + slip_frac)        # buying costs slightly more
        exit_fill  = exit_price * (1 - slip_frac)    # selling nets slightly less
    else:  # SHORT
        entry_fill = entry * (1 - slip_frac)         # selling short nets slightly less
        exit_fill  = exit_price * (1 + slip_frac)     # buying back costs slightly more
    return entry_fill, exit_fill


# ─────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
#  Walks bricks in time order. Same signal condition + duplicate
#  guard as the live bot. Once a trade is open, no new signals are
#  considered until it exits (matches the live bot's behaviour of
#  only tracking one open trade at a time). Exit is a plain fixed
#  TP or SL — no trailing-stop logic.
# ─────────────────────────────────────────────────────────────
def run_backtest(opens, highs, lows, closes, times, bricks, cfg: dict):
    trades = []
    sell_run = 0   # consecutive bearish bricks -> feeds LONG entries
    buy_run  = 0   # consecutive bullish bricks -> feeds SHORT entries
    open_trade = None
    last_entry_price = 0.0

    fee_frac  = cfg["fee_pct"] / 100.0
    slip_frac = cfg["slippage_pct"] / 100.0
    allow_shorts = cfg.get("allow_shorts", False)

    i = 0
    n_bricks = len(bricks)
    while i < n_bricks:
        b = bricks[i]

        if open_trade is None:
            if b["dir"] == -1:
                # ---- possible SHORT entry: bearish brick after a bullish run ----
                if allow_shorts and buy_run >= cfg["min_buy_bricks"]:
                    entry = b["close"]
                    atr = b["atr"]
                    entry_idx = b["idx"]
                    dup = (last_entry_price > 0 and
                           abs(entry - last_entry_price) < cfg["atr_gap_mult"] * atr)
                    if not dup:
                        sl_dist = cfg["sl_mult"] * atr
                        sl = entry + sl_dist
                        tp = entry - cfg["risk_reward_ratio"] * sl_dist
                        open_trade = {
                            "side": "SHORT", "entry": entry, "sl": sl, "tp": tp, "atr": atr,
                            "entry_idx": entry_idx, "entry_time": times[entry_idx],
                        }
                buy_run = 0
                sell_run += 1
            else:
                # ---- possible LONG entry: bullish brick after a bearish run ----
                if sell_run >= cfg["min_sell_bricks"]:
                    entry = b["close"]
                    atr = b["atr"]
                    entry_idx = b["idx"]
                    dup = (last_entry_price > 0 and
                           abs(entry - last_entry_price) < cfg["atr_gap_mult"] * atr)
                    if not dup:
                        sl_dist = cfg["sl_mult"] * atr
                        sl = entry - sl_dist
                        tp = entry + cfg["risk_reward_ratio"] * sl_dist
                        open_trade = {
                            "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "atr": atr,
                            "entry_idx": entry_idx, "entry_time": times[entry_idx],
                        }
                sell_run = 0
                buy_run += 1
            i += 1
            continue

        # ── we're in a trade: scan forward candle-by-candle for exit ──
        t = open_trade
        exit_found = False
        exit_j = None

        for j in range(t["entry_idx"] + 1, len(closes)):
            if t["side"] == "LONG":
                hit_tp = highs[j] >= t["tp"]
                hit_sl = lows[j]  <= t["sl"]
            else:
                hit_tp = lows[j]  <= t["tp"]
                hit_sl = highs[j] >= t["sl"]

            if not (hit_tp or hit_sl):
                continue

            if hit_tp and hit_sl:
                outcome = resolve_ambiguous_exit(cfg, opens[j], closes[j], t["side"])
            elif hit_tp:
                outcome = "TP"
            else:
                outcome = "SL"

            exit_price = t["tp"] if outcome == "TP" else t["sl"]
            entry_fill, exit_fill = apply_fill_prices(t["side"], t["entry"], exit_price, slip_frac)

            if t["side"] == "LONG":
                gross_pct = (exit_fill - entry_fill) / entry_fill
            else:
                gross_pct = (entry_fill - exit_fill) / entry_fill

            net_pct = gross_pct - 2 * fee_frac   # fee charged on entry AND exit

            trades.append({
                "side"      : t["side"],
                "entry_time": datetime.fromtimestamp(t["entry_time"]/1000, tz=timezone.utc),
                "exit_time" : datetime.fromtimestamp(times[j]/1000, tz=timezone.utc),
                "entry"     : t["entry"],
                "sl"        : t["sl"],
                "tp"        : t["tp"],
                "atr"       : t["atr"],
                "outcome"   : outcome,
                "gross_pct" : gross_pct * 100,
                "net_pct"   : net_pct * 100,
                "bars_held" : j - t["entry_idx"],
            })

            last_entry_price = t["entry"]
            open_trade = None
            exit_found = True
            exit_j = j
            break

        if exit_found:
            # FIX (CRITICAL): skip every brick that formed WHILE this trade
            # was open. Those bricks occurred concurrently with a position
            # the live bot would never have acted on (one trade at a time),
            # so they must not be replayed as signals once the trade closes.
            while i < n_bricks and bricks[i]["idx"] <= exit_j:
                i += 1
            sell_run = 0
            buy_run = 0
            continue
        else:
            # trade never closed within available data (still "open" at end)
            side = t["side"]
            entry_fill, _ = apply_fill_prices(side, t["entry"], t["entry"], slip_frac)
            if side == "LONG":
                gross_pct = (closes[-1] - entry_fill) / entry_fill
            else:
                gross_pct = (entry_fill - closes[-1]) / entry_fill
            net_pct = gross_pct - fee_frac  # FIX (medium): only entry-side fee has actually been paid

            trades.append({
                "side"      : side,
                "entry_time": datetime.fromtimestamp(t["entry_time"]/1000, tz=timezone.utc),
                "exit_time" : None,
                "entry"     : t["entry"],
                "sl"        : t["sl"],
                "tp"        : t["tp"],
                "atr"       : t["atr"],
                "outcome"   : "OPEN_AT_END",
                "gross_pct" : gross_pct * 100,
                "net_pct"   : net_pct * 100,
                "bars_held" : len(closes) - 1 - t["entry_idx"],
            })
            open_trade = None
            break  # no more candle data left to resolve anything further

    return trades


# ─────────────────────────────────────────────────────────────
#  PERFORMANCE SUMMARY
# ─────────────────────────────────────────────────────────────
def summarize(trades: list, cfg: dict):
    if not trades:
        print("\nNo trades were generated over this period — "
              "try loosening min_sell_bricks or the lookback window.")
        return None

    df = pd.DataFrame(trades)
    closed = df[df["outcome"] != "OPEN_AT_END"]

    n_total  = len(df)
    n_closed = len(closed)
    n_tp     = (closed["outcome"] == "TP").sum()
    n_sl     = (closed["outcome"] == "SL").sum()

    win_rate  = n_tp / n_closed * 100 if n_closed else 0.0

    gains  = closed.loc[closed["net_pct"] > 0, "net_pct"].sum()
    losses = -closed.loc[closed["net_pct"] < 0, "net_pct"].sum()
    profit_factor = gains / losses if losses > 0 else float("inf")

    avg_win   = closed.loc[closed["outcome"] == "TP", "net_pct"].mean() if n_tp else 0
    avg_loss  = closed.loc[closed["outcome"] == "SL", "net_pct"].mean() if n_sl else 0
    expectancy = closed["net_pct"].mean() if n_closed else 0

    # ---- equity curve ----
    # FIX (HIGH): position size now actually derives from the SL distance
    # in "risk_based" mode, so risk_pct_per_trade truly reflects capital
    # at risk if the stop is hit. "fixed_fraction" keeps the old behaviour
    # (allocate a flat % of capital to every trade's raw return).
    mode = cfg.get("position_sizing_mode", "risk_based")
    risk_frac_cfg = cfg["risk_pct_per_trade"] / 100.0
    max_lev = cfg.get("max_leverage", 1.0)

    equity = cfg["initial_capital"]
    curve = [equity]
    for _, row in closed.iterrows():
        sl_dist_pct = abs(row["entry"] - row["sl"]) / row["entry"]
        if mode == "risk_based":
            position_fraction = 0.0 if sl_dist_pct <= 0 else min(risk_frac_cfg / sl_dist_pct, max_lev)
        else:  # fixed_fraction (legacy behaviour)
            position_fraction = min(risk_frac_cfg, max_lev)

        equity *= (1 + (row["net_pct"] / 100.0) * position_fraction)
        curve.append(equity)
    curve = np.array(curve)

    running_max = np.maximum.accumulate(curve)
    drawdowns = (curve - running_max) / running_max * 100
    max_dd = drawdowns.min()

    total_return_pct = (curve[-1] / curve[0] - 1) * 100

    print("\n" + "═" * 60)
    print(f"  BACKTEST SUMMARY — {cfg['symbol']} {cfg['timeframe']} "
          f"({cfg['lookback_days']}d)")
    print("═" * 60)
    print(f"  Params        : ATR({cfg['atr_period']}) | "
          f"brick={cfg['renko_mult']}xATR | SL={cfg['sl_mult']}xATR | "
          f"R:R={cfg['risk_reward_ratio']} | min_sell={cfg['min_sell_bricks']} | "
          f"shorts={'on' if cfg.get('allow_shorts') else 'off'}")
    print(f"  Sizing        : mode={mode} | risk_pct_per_trade={cfg['risk_pct_per_trade']}% "
          f"| max_leverage={max_lev}x")
    print(f"  Total signals : {n_total}  ({n_closed} closed, "
          f"{n_total - n_closed} still open at data end)")
    print(f"  Wins / Losses : {n_tp} / {n_sl}")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Avg win       : {avg_win:+.2f}%")
    print(f"  Avg loss      : {avg_loss:+.2f}%")
    print(f"  Expectancy/tr : {expectancy:+.3f}%")
    print(f"  Profit factor : {profit_factor:.2f}")
    print(f"  Total return  : {total_return_pct:+.1f}%  "
          f"(${cfg['initial_capital']:.0f} → ${curve[-1]:.0f})")
    print(f"  Max drawdown  : {max_dd:.1f}%")
    print(f"  Avg bars held : {closed['bars_held'].mean():.1f}")
    print("═" * 60)

    return {"df": df, "curve": curve, "max_dd": max_dd,
            "total_return_pct": total_return_pct}


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main(cfg=None):
    cfg = cfg or CONFIG

    opens, highs, lows, closes, times = fetch_historical_klines(
        cfg["symbol"], cfg["timeframe"], cfg["lookback_days"])

    atr_arr = calc_atr(highs, lows, closes, cfg["atr_period"])
    bricks  = build_renko(closes, atr_arr, cfg["renko_mult"])
    print(f"[RENKO] Built {len(bricks)} bricks from {len(closes)} candles")

    trades = run_backtest(opens, highs, lows, closes, times, bricks, cfg)
    result = summarize(trades, cfg)

    if result is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result["curve"], color="#2563eb", linewidth=1.5)
            ax.set_title(f"{cfg['symbol']} {cfg['timeframe']} Renko-ATR — "
                         f"Equity Curve ({cfg['lookback_days']}d)")
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Equity ($)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            out_png = "/mnt/user-data/outputs/renko_backtest_equity_curve.png"
            fig.savefig(out_png, dpi=150)
            print(f"Saved equity curve → {out_png}")
        except Exception as e:
            print(f"[WARN] Could not save equity curve chart: {e}")

    return trades, result


if __name__ == "__main__":
    main()
