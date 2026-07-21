"""
╔══════════════════════════════════════════════════════════════╗
║   SOL/USDT  |  9 EMA × 9 SMA(EMA)  |  BACKTEST             ║
║   EXACT same logic as the live tracker — replayed on history ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
  pip install requests pandas

EDIT THE CONFIG BLOCK BELOW, THEN RUN:
  python sol_backtest.py

OUTPUT:
  - Printed summary in terminal
  - trades_backtest.csv  (every closed trade)
  - equity_curve.csv     (capital after every closed trade)
"""

# ─────────────────────── CONFIG ───────────────────────────── #

SYMBOL         = "SOLUSDT"          # Binance pair
INTERVAL       = "1m"               # Candle interval

# ── Date range ───────────────────────────────────────────── #
# Format: "YYYY-MM-DD"  (UTC midnight used automatically)
START_DATE     = "2023-01-01"
END_DATE       = "2025-03-31"

EMA_PERIOD     = 9                  # Base EMA period
SMA_PERIOD     = 9                  # SMA applied ON TOP of the EMA

SL_BUFFER_PCT  = 0.20               # Buffer % added to raw SL distance
RISK_REWARD    = 2.0                # TP = entry + SL_dist × RISK_REWARD

# ── Position Sizing ──────────────────────────────────────── #
RISK_MODE      = "percent"            # "fixed" → flat USD | "percent" → % of capital
RISK_FIXED_USD = 100                # USD risked per trade (if RISK_MODE = "fixed")
RISK_PCT       = 1.0                # % of capital risked  (if RISK_MODE = "percent")
CAPITAL        = 10_000             # Starting capital

# ── Output ───────────────────────────────────────────────── #
SAVE_TRADES_CSV = True              # Save every closed trade
SAVE_EQUITY_CSV = True              # Save equity curve
TRADES_CSV_PATH = "trades_backtest.csv"
EQUITY_CSV_PATH = "equity_curve.csv"

# ──────────────────────────────────────────────────────────── #

import requests
import time
import csv
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ═══════════════════════ DATA FETCH ════════════════════════════

BINANCE_BASE = "https://api.binance.com"

def date_to_ms(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to millisecond timestamp (UTC midnight)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_candles_range(symbol: str, interval: str,
                        start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch ALL 1m candles between start_ms and end_ms.
    Binance returns max 1000 candles per call — we paginate automatically.
    """
    all_rows = []
    current_start = start_ms

    print(f"  Fetching candles from Binance (this may take a while for long ranges)...")
    batch = 0

    while current_start < end_ms:
        batch += 1
        resp = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={
                "symbol"    : symbol,
                "interval"  : interval,
                "startTime" : current_start,
                "endTime"   : end_ms,
                "limit"     : 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()

        if not raw:
            break

        all_rows.extend(raw)
        last_open_time = raw[-1][0]

        # Advance past the last candle fetched
        # Each 1m candle is 60 000 ms
        current_start = last_open_time + 60_000

        if batch % 10 == 0:
            pct = (current_start - start_ms) / (end_ms - start_ms) * 100
            print(f"    ...{pct:.0f}% downloaded ({len(all_rows):,} candles so far)")

        # Respect Binance rate-limit (1200 weight/min; klines = 1 weight)
        time.sleep(0.15)

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)

    # Drop the last (possibly still-forming) candle — same as live script
    df = df.iloc[:-1].reset_index(drop=True)

    print(f"  ✓ Total candles loaded: {len(df):,}\n")
    return df


# ═══════════════════════ INDICATORS ════════════════════════════
# ── IDENTICAL to live script ────────────────────────────────── #

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema"]            = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["sma_ema"]        = df["ema"].rolling(SMA_PERIOD).mean()
    df["ema_above"]      = df["ema"] > df["sma_ema"]
    df["ema_above_prev"] = df["ema_above"].shift(1).fillna(False).astype(bool)
    df["signal_buy"]     = (~df["ema_above_prev"]) & df["ema_above"]
    return df


# ═══════════════════════ POSITION SIZING ═══════════════════════
# ── IDENTICAL to live script ────────────────────────────────── #

def compute_risk(capital: float) -> float:
    if RISK_MODE == "fixed":
        return RISK_FIXED_USD
    return capital * (RISK_PCT / 100)


# ═══════════════════════ TRADE DATACLASS ═══════════════════════

@dataclass
class Trade:
    entry_time  : str
    entry       : float
    sl          : float
    tp          : float
    sl_dist     : float
    risk_usd    : float
    qty         : float
    trigger_low : float


# ═══════════════════════ CSV OUTPUT ════════════════════════════

TRADE_HEADERS = [
    "trade_num","entry_time","exit_time","entry","sl","tp","exit_price",
    "sl_dist","qty","risk_usd","pnl_usd","pnl_r","result","capital"
]

EQUITY_HEADERS = ["trade_num","exit_time","capital","net_pnl","net_r"]


def save_trades(records: list):
    if not SAVE_TRADES_CSV:
        return
    with open(TRADES_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_HEADERS)
        w.writeheader()
        w.writerows(records)
    print(f"  Trades saved → {TRADES_CSV_PATH}")


def save_equity(equity: list):
    if not SAVE_EQUITY_CSV:
        return
    with open(EQUITY_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EQUITY_HEADERS)
        w.writeheader()
        w.writerows(equity)
    print(f"  Equity curve saved → {EQUITY_CSV_PATH}")


# ═══════════════════════ BACKTEST ENGINE ═══════════════════════
# ── Logic mirrors the live main() loop exactly ──────────────── #

def run_backtest(df: pd.DataFrame):
    capital   = float(CAPITAL)
    net_pnl   = 0.0
    total = wins = losses = 0

    open_trade : Optional[Trade] = None
    trade_records = []
    equity_curve  = []

    warmup = EMA_PERIOD + SMA_PERIOD  # skip rows before indicators are valid

    print("  Running backtest...\n")

    for i in range(warmup, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]      # already processed in prior iteration

        # ── Step 1: Check open trade for SL / TP ────────────
        # (Live script checks this BEFORE looking for new signals)
        if open_trade is not None:
            low  = row["low"]
            high = row["high"]

            exit_price = exit_reason = None

            # SL checked first — same priority as live script
            if low <= open_trade.sl:
                exit_price  = open_trade.sl
                exit_reason = "SL"
            elif high >= open_trade.tp:
                exit_price  = open_trade.tp
                exit_reason = "TP"

            if exit_reason:
                pnl_usd  = (exit_price - open_trade.entry) * open_trade.qty
                pnl_r    = pnl_usd / open_trade.risk_usd
                net_pnl  += pnl_usd
                capital  += pnl_usd
                total    += 1
                if exit_reason == "TP":
                    wins += 1
                else:
                    losses += 1

                trade_records.append({
                    "trade_num" : total,
                    "entry_time": open_trade.entry_time,
                    "exit_time" : row["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
                    "entry"     : round(open_trade.entry, 4),
                    "sl"        : round(open_trade.sl, 4),
                    "tp"        : round(open_trade.tp, 4),
                    "exit_price": round(exit_price, 4),
                    "sl_dist"   : round(open_trade.sl_dist, 4),
                    "qty"       : round(open_trade.qty, 4),
                    "risk_usd"  : round(open_trade.risk_usd, 2),
                    "pnl_usd"   : round(pnl_usd, 2),
                    "pnl_r"     : round(pnl_r, 3),
                    "result"    : exit_reason,
                    "capital"   : round(capital, 2),
                })
                equity_curve.append({
                    "trade_num" : total,
                    "exit_time" : row["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
                    "capital"   : round(capital, 2),
                    "net_pnl"   : round(net_pnl, 2),
                    "net_r"     : round(wins * RISK_REWARD - losses * 1.0, 2),
                })

                open_trade = None

        # ── Step 2: Check for new BUY signal on the PREVIOUS candle ──
        # In live script: signal is on 'latest' (last closed candle),
        # entry is on df.iloc[-1]["open"] (next/current forming candle open).
        # In backtest: signal candle = row i-1, entry candle = row i.
        # We use prev for signal detection and row["open"] for entry.

        if open_trade is None and prev["signal_buy"]:
            next_open = row["open"]        # entry price = current candle open
            trig_low  = prev["low"]        # SL anchor = signal candle low

            raw_sl_dist = next_open - trig_low
            if raw_sl_dist > 0:
                sl_dist  = raw_sl_dist * (1 + SL_BUFFER_PCT / 100)
                sl       = next_open - sl_dist
                tp       = next_open + sl_dist * RISK_REWARD
                risk_usd = compute_risk(capital)
                qty      = risk_usd / sl_dist

                open_trade = Trade(
                    entry_time  = row["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
                    entry       = next_open,
                    sl          = sl,
                    tp          = tp,
                    sl_dist     = sl_dist,
                    risk_usd    = risk_usd,
                    qty         = qty,
                    trigger_low = trig_low,
                )

    # ── Handle trade still open at end of data ───────────────
    if open_trade is not None:
        last_close = df.iloc[-1]["close"]
        pnl_usd   = (last_close - open_trade.entry) * open_trade.qty
        pnl_r     = pnl_usd / open_trade.risk_usd
        net_pnl  += pnl_usd
        capital  += pnl_usd
        total    += 1
        print(f"  ⚠  Trade still open at end of data — closed at last price ${last_close:.4f}")
        trade_records.append({
            "trade_num" : total,
            "entry_time": open_trade.entry_time,
            "exit_time" : df.iloc[-1]["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
            "entry"     : round(open_trade.entry, 4),
            "sl"        : round(open_trade.sl, 4),
            "tp"        : round(open_trade.tp, 4),
            "exit_price": round(last_close, 4),
            "sl_dist"   : round(open_trade.sl_dist, 4),
            "qty"       : round(open_trade.qty, 4),
            "risk_usd"  : round(open_trade.risk_usd, 2),
            "pnl_usd"   : round(pnl_usd, 2),
            "pnl_r"     : round(pnl_r, 3),
            "result"    : "OPEN_AT_END",
            "capital"   : round(capital, 2),
        })

    return total, wins, losses, net_pnl, capital, trade_records, equity_curve


# ═══════════════════════ SUMMARY PRINT ═════════════════════════

def print_summary(total, wins, losses, net_pnl, start_cap, end_cap,
                  trade_records):
    win_rate   = wins / total * 100 if total else 0
    net_r      = wins * RISK_REWARD - losses * 1.0
    return_pct = (end_cap - start_cap) / start_cap * 100

    # Max drawdown from equity curve
    capitals = [start_cap] + [r["capital"] for r in trade_records]
    peak = capitals[0]
    max_dd = 0.0
    for c in capitals:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Avg win / avg loss
    pnls = [r["pnl_usd"] for r in trade_records if r["result"] != "OPEN_AT_END"]
    win_pnls  = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    avg_win   = sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0
    avg_loss  = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║           BACKTEST RESULTS SUMMARY           ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Symbol      : {SYMBOL}  {INTERVAL}")
    print(f"  Period      : {START_DATE}  →  {END_DATE}")
    print(f"  Strategy    : {EMA_PERIOD} EMA × {SMA_PERIOD} SMA(EMA)")
    print(f"  R:R         : 1 : {RISK_REWARD}   |  SL Buffer: {SL_BUFFER_PCT}%")
    print(f"  Risk/Trade  : "
          + (f"${RISK_FIXED_USD} fixed" if RISK_MODE == "fixed"
             else f"{RISK_PCT}% of capital"))
    print(f"  Start Cap   : ${start_cap:,.2f}")
    print("─" * 48)
    print(f"  Total Trades: {total}")
    print(f"  Wins        : {wins}   ({win_rate:.1f}%)")
    print(f"  Losses      : {losses}")
    print(f"  Net P&L     : {'+'if net_pnl>=0 else ''}${net_pnl:,.2f}")
    print(f"  Net R       : {'+'if net_r>=0 else ''}{net_r:.1f} R")
    print(f"  Return      : {'+'if return_pct>=0 else ''}{return_pct:.2f}%")
    print(f"  End Capital : ${end_cap:,.2f}")
    print("─" * 48)
    print(f"  Avg Win     : ${avg_win:,.2f}")
    print(f"  Avg Loss    : ${avg_loss:,.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print("─" * 48)


# ═══════════════════════ MAIN ══════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  SOL/USDT  9EMA × 9SMA(EMA)  Backtest       ║")
    print("╚══════════════════════════════════════════════╝\n")
    print(f"  Date range : {START_DATE}  →  {END_DATE}")
    print(f"  Pair       : {SYMBOL}  |  Interval: {INTERVAL}\n")

    start_ms = date_to_ms(START_DATE)
    # end_ms = end of the END_DATE day
    end_ms   = date_to_ms(END_DATE) + 86_400_000 - 1

    # Fetch all candles
    df = fetch_candles_range(SYMBOL, INTERVAL, start_ms, end_ms)

    # Add indicators (same function as live script)
    df = add_indicators(df)

    # Run backtest
    total, wins, losses, net_pnl, end_cap, trade_records, equity_curve = run_backtest(df)

    # Print summary
    print_summary(total, wins, losses, net_pnl, CAPITAL, end_cap, trade_records)

    # Save CSVs
    save_trades(trade_records)
    save_equity(equity_curve)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
