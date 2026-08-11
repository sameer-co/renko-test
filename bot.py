"""
╔══════════════════════════════════════════════════════════════════╗
║      Renko ATR Strategy — 1-Year Backtester (Binance Public API) ║
║  Same signal logic as the live forward tester, run over history  ║
╚══════════════════════════════════════════════════════════════════╝

What it does:
  1. Downloads N days of historical candles from Binance (paginated).
  2. Builds ATR + ATR-based Renko bricks (identical logic to your bot).
  3. Walks the bricks, opens a trade on the same BUY signal condition
     (bullish brick after >= min_sell_bricks bearish ones, with the
     same duplicate-entry ATR-gap guard).
  4. Determines the exit (TP or SL only — simple fixed exits, no
     trailing stop) using intrabar HIGH/LOW of the candles that
     follow the entry — not just candle closes — so the backtest
     reflects what would really have happened.
  5. Prints a performance summary and saves an equity curve chart.

Every strategy parameter lives in CONFIG below — nothing else needs
to be touched to test a different symbol, timeframe, ATR period,
brick size, SL/TP multiples, or lookback window.
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
    "symbol"          : "BTCUSDT",
    "timeframe"       : "30m",     # any Binance interval: 1m,3m,5m,15m,30m,1h,4h,1d ...
    "lookback_days"   : 365,      # how far back to backtest

    "atr_period"      : 14,
    "renko_mult"      : 1.0,      # brick size = renko_mult x ATR
    "sl_mult"         : 1.5,      # SL = sl_mult x ATR below entry
    "tp_mult"         : 3.5,        # TP = tp_mult x SL above entry
    "min_sell_bricks" : 2,        # min consecutive bearish bricks before entry
    "atr_gap_mult"    : 1.0,      # duplicate-entry guard, same as live bot

    # execution assumptions (set to 0 to test the "perfect fill" case)
    "fee_pct"         : 0.04,     # taker fee per side, % (Binance spot default ~0.1%, many use 0.04% w/ BNB)
    "slippage_pct"    : 0.02,     # extra slippage per side, %
    "exit_priority"   : "SL",     # if a single candle's range touches BOTH tp & sl: "SL" (conservative) or "TP"

    "initial_capital" : 1000.0,   # USD, for equity curve / % return only (no real position sizing)
    "risk_pct_per_trade": 100.0,  # % of capital "risked" per trade for equity curve (100 = full compounding)
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
        cursor = last_open_time + step_ms

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
#  BACKTEST ENGINE
#  Walks bricks in time order. Same signal condition + duplicate
#  guard as the live bot. Once a trade is open, no new signals are
#  considered until it exits (matches the live bot's behaviour of
#  only tracking one open trade at a time). Exit is a plain fixed
#  TP or SL — no trailing-stop logic.
# ─────────────────────────────────────────────────────────────
def run_backtest(highs, lows, closes, times, bricks, cfg: dict):
    trades = []
    sell_run = 0
    open_trade = None
    last_entry_price = 0.0
    fee_slip = (cfg["fee_pct"] + cfg["slippage_pct"]) / 100.0  # per side

    i = 0
    while i < len(bricks):
        b = bricks[i]

        if open_trade is None:
            if b["dir"] == -1:
                sell_run += 1
            else:
                if sell_run >= cfg["min_sell_bricks"]:
                    entry     = b["close"]
                    atr       = b["atr"]
                    entry_idx = b["idx"]

                    dup = (last_entry_price > 0 and
                           abs(entry - last_entry_price) < cfg["atr_gap_mult"] * atr)

                    if not dup:
                        sl = entry - cfg["sl_mult"] * atr
                        tp = entry + cfg["tp_mult"] * cfg["sl_mult"] * atr
                        open_trade = {
                            "entry": entry, "sl": sl, "tp": tp, "atr": atr,
                            "entry_idx": entry_idx,
                            "entry_time": times[entry_idx],
                            "sell_run": sell_run,
                        }
                sell_run = 0
            i += 1
            continue

        # ── we're in a trade: scan forward candle-by-candle for exit ──
        t = open_trade
        exit_found = False

        for j in range(t["entry_idx"] + 1, len(closes)):
            hit_tp = highs[j] >= t["tp"]
            hit_sl = lows[j]  <= t["sl"]

            if hit_tp and hit_sl:
                # same candle touches both — ambiguous, use configured priority
                outcome = cfg["exit_priority"]
                exit_price = t["tp"] if outcome == "TP" else t["sl"]
            elif hit_tp:
                outcome, exit_price = "TP", t["tp"]
            elif hit_sl:
                outcome, exit_price = "SL", t["sl"]
            else:
                continue

            gross_pct = (exit_price - t["entry"]) / t["entry"]
            net_pct   = gross_pct - 2 * fee_slip   # entry + exit friction

            trades.append({
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
            break

        if not exit_found:
            # trade never closed within available data (still "open" at end)
            trades.append({
                "entry_time": datetime.fromtimestamp(t["entry_time"]/1000, tz=timezone.utc),
                "exit_time" : None,
                "entry"     : t["entry"],
                "sl"        : t["sl"],
                "tp"        : t["tp"],
                "atr"       : t["atr"],
                "outcome"   : "OPEN_AT_END",
                "gross_pct" : (closes[-1] - t["entry"]) / t["entry"] * 100,
                "net_pct"   : ((closes[-1] - t["entry"]) / t["entry"] - fee_slip) * 100,
                "bars_held" : len(closes) - 1 - t["entry_idx"],
            })
            open_trade = None

        i += 1

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
    loss_rate = n_sl / n_closed * 100 if n_closed else 0.0

    gains  = closed.loc[closed["net_pct"] > 0, "net_pct"].sum()
    losses = -closed.loc[closed["net_pct"] < 0, "net_pct"].sum()
    profit_factor = gains / losses if losses > 0 else float("inf")

    avg_win   = closed.loc[closed["outcome"] == "TP", "net_pct"].mean() if n_tp else 0
    avg_loss  = closed.loc[closed["outcome"] == "SL", "net_pct"].mean() if n_sl else 0
    expectancy = closed["net_pct"].mean() if n_closed else 0

    # equity curve (compounding, risk_pct_per_trade of capital per trade)
    equity = cfg["initial_capital"]
    curve = [equity]
    risk_frac = cfg["risk_pct_per_trade"] / 100.0
    for pct in closed["net_pct"]:
        equity *= (1 + (pct / 100.0) * risk_frac)
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
          f"TP={cfg['tp_mult']}xSL | min_sell={cfg['min_sell_bricks']}")
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

    trades = run_backtest(highs, lows, closes, times, bricks, cfg)
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
