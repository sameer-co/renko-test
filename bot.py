"""
=============================================================
  NIFTY 50 — Renko Backtest (1H timeframe)
=============================================================
Strategy:
  - Build ATR-14 based Renko chart from 1H OHLC data
  - Entry: After a sell-side move (one or more red bricks),
           wait for the FIRST completed GREEN brick → BUY
  - SL  : Entry price - (ATR * 1.5)
  - TP  : Entry price + (SL_distance * 3)   → RR = 1:3

Requirements:
  pip install yfinance pandas numpy stocktrends tabulate openpyxl
=============================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from stocktrends import Renko
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ATR_PERIOD        = 14
ATR_MULTIPLIER_SL = 1.5
RR_RATIO          = 3.0          # TP = SL_distance * RR_RATIO
PERIOD            = "3650d"       # ~2 years (max for 1H on yfinance)
INTERVAL          = "1d"
CAPITAL           = 100_000      # per-trade capital in INR
DELAY_BETWEEN     = 1.5          # seconds between downloads (rate-limit safe)

# ─────────────────────────────────────────────
# NIFTY 50 SYMBOLS  (NSE suffix)
# ─────────────────────────────────────────────
NIFTY50 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS",
    "BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS",
    "DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS",
    "HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","INDUSINDBK.NS","INFY.NS","ITC.NS","JSWSTEEL.NS",
    "KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS",
    "ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS",
    "SUNPHARMA.NS","TATACONSUM.NS","TATASTEEL.NS",
    "TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS","UPL.NS",
    "WIPRO.NS"
]


# ─────────────────────────────────────────────
# HELPER: Compute ATR (Wilder / RMA)
# ─────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


# ─────────────────────────────────────────────
# HELPER: Build ATR-Renko from OHLC
# ─────────────────────────────────────────────
def build_renko(df: pd.DataFrame, atr_val: float) -> pd.DataFrame | None:
    """
    Feed OHLC to stocktrends Renko.
    Returns renko DataFrame with columns: date, open, high, low, close, uptrend
    uptrend=True  → green brick
    uptrend=False → red brick
    """
    renko_input = df[["Open", "High", "Low", "Close"]].copy().reset_index()
    renko_input.columns = ["date", "open", "high", "low", "close"]

    # stocktrends also needs a volume column
    renko_input["volume"] = 0

    try:
        r = Renko(renko_input)
        r.brick_size = round(float(atr_val), 4)
        renko_df = r.get_ohlc_data()
        if renko_df is None or len(renko_df) < 3:
            return None
        return renko_df
    except Exception:
        return None


# ─────────────────────────────────────────────
# CORE: Backtest a single symbol
# ─────────────────────────────────────────────
def backtest_symbol(symbol: str) -> dict | None:
    # 1. Download 1H data
    raw = yf.download(symbol, period=PERIOD, interval=INTERVAL,
                      progress=False, auto_adjust=True)

    if raw is None or len(raw) < 50:
        return None

    # Flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.dropna(inplace=True)
    if len(raw) < 50:
        return None

    # 2. Compute ATR on raw OHLC
    atr_series = compute_atr(raw, ATR_PERIOD)
    atr_val    = atr_series.dropna().median()   # use median for stable brick size
    if np.isnan(atr_val) or atr_val <= 0:
        return None

    # 3. Build Renko
    renko = build_renko(raw, atr_val)
    if renko is None or len(renko) < 5:
        return None

    # 4. Simulate trades
    trades       = []
    in_trade     = False
    sell_streak  = 0   # count consecutive red bricks

    for i in range(1, len(renko)):
        brick      = renko.iloc[i]
        prev_brick = renko.iloc[i - 1]
        is_green   = bool(brick["uptrend"])
        is_red     = not is_green
        was_red    = not bool(prev_brick["uptrend"])

        if not in_trade:
            # Track consecutive red bricks
            if is_red:
                sell_streak += 1
            elif is_green and sell_streak >= 1:
                # ─── ENTRY SIGNAL ───────────────────────────────────────
                # After at least 1 red brick, first completed green brick
                entry_price  = float(brick["close"])
                sl_distance  = atr_val * ATR_MULTIPLIER_SL
                sl           = entry_price - sl_distance
                tp           = entry_price + sl_distance * RR_RATIO
                entry_date   = brick["date"]

                qty          = max(1, int(CAPITAL / entry_price))

                in_trade    = True
                sell_streak = 0

                trade_info = {
                    "entry_date"  : entry_date,
                    "entry_price" : entry_price,
                    "sl"          : sl,
                    "tp"          : tp,
                    "qty"         : qty,
                    "exit_date"   : None,
                    "exit_price"  : None,
                    "result"      : None,
                    "pnl"         : None,
                }
            else:
                sell_streak = 0   # green appeared without prior red — reset

        else:
            # ─── TRADE MANAGEMENT ───────────────────────────────────────
            # Check against remaining Renko bricks
            cur_close = float(brick["close"])
            cur_low   = float(brick["low"])
            cur_high  = float(brick["high"])

            if cur_low <= trade_info["sl"]:
                # Stop loss hit
                trade_info["exit_date"]  = brick["date"]
                trade_info["exit_price"] = trade_info["sl"]
                trade_info["result"]     = "SL"
                trade_info["pnl"]        = (trade_info["sl"] - trade_info["entry_price"]) * trade_info["qty"]
                trades.append(trade_info)
                in_trade = False

            elif cur_high >= trade_info["tp"]:
                # Take profit hit
                trade_info["exit_date"]  = brick["date"]
                trade_info["exit_price"] = trade_info["tp"]
                trade_info["result"]     = "TP"
                trade_info["pnl"]        = (trade_info["tp"] - trade_info["entry_price"]) * trade_info["qty"]
                trades.append(trade_info)
                in_trade = False

    # Close any open trade at last price
    if in_trade:
        last = renko.iloc[-1]
        trade_info["exit_date"]  = last["date"]
        trade_info["exit_price"] = float(last["close"])
        trade_info["result"]     = "OPEN"
        trade_info["pnl"]        = (float(last["close"]) - trade_info["entry_price"]) * trade_info["qty"]
        trades.append(trade_info)
        in_trade = False

    if not trades:
        return None

    df_trades = pd.DataFrame(trades)
    total     = len(df_trades)
    wins      = (df_trades["result"] == "TP").sum()
    losses    = (df_trades["result"] == "SL").sum()
    open_t    = (df_trades["result"] == "OPEN").sum()
    win_rate  = wins / total * 100 if total else 0
    total_pnl = df_trades["pnl"].sum()
    avg_win   = df_trades.loc[df_trades["result"] == "TP",  "pnl"].mean() if wins   else 0
    avg_loss  = df_trades.loc[df_trades["result"] == "SL",  "pnl"].mean() if losses else 0
    exp_factor= (avg_win * (win_rate/100) + avg_loss * (1 - win_rate/100)) if total else 0

    return {
        "Symbol"      : symbol.replace(".NS", ""),
        "Trades"      : total,
        "Wins"        : int(wins),
        "Losses"      : int(losses),
        "Open"        : int(open_t),
        "Win%"        : round(win_rate, 1),
        "Total PnL"   : round(total_pnl, 2),
        "Avg Win"     : round(avg_win,   2),
        "Avg Loss"    : round(avg_loss,  2),
        "Expectancy"  : round(exp_factor, 2),
        "Brick Size"  : round(atr_val, 2),
        "_trades"     : df_trades,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  NIFTY 50  |  Renko Backtest  |  1H  |  ATR-14  |  RR 1:3")
    print("=" * 65)
    print(f"  Universe  : {len(NIFTY50)} stocks")
    print(f"  Period    : Last 2 years (730d)")
    print(f"  Entry     : First green Renko brick after sell-side move")
    print(f"  SL        : Entry - ATR × {ATR_MULTIPLIER_SL}")
    print(f"  TP        : Entry + SL_dist × {RR_RATIO}  (1:{int(RR_RATIO)} RR)")
    print("=" * 65, "\n")

    summary     = []
    all_trades  = []
    failed      = []

    for idx, sym in enumerate(NIFTY50, 1):
        print(f"[{idx:2d}/{len(NIFTY50)}] Fetching {sym:<20}", end=" ", flush=True)
        try:
            result = backtest_symbol(sym)
            if result:
                trades_df          = result.pop("_trades")
                trades_df["Symbol"] = result["Symbol"]
                all_trades.append(trades_df)
                summary.append(result)
                print(f"✓  trades={result['Trades']}  Win%={result['Win%']}%  PnL=₹{result['Total PnL']:,.0f}")
            else:
                failed.append(sym)
                print("✗  insufficient data")
        except Exception as e:
            failed.append(sym)
            print(f"✗  error: {e}")

        time.sleep(DELAY_BETWEEN)

    if not summary:
        print("\n❌  No results — check your internet connection.")
        return

    # ─── Build summary DF ───────────────────────────────────────
    df_summary = pd.DataFrame(summary).sort_values("Total PnL", ascending=False).reset_index(drop=True)
    df_all_trades = pd.concat(all_trades, ignore_index=True)

    # ─── Portfolio aggregate ─────────────────────────────────────
    total_trades = df_summary["Trades"].sum()
    total_wins   = df_summary["Wins"].sum()
    total_losses = df_summary["Losses"].sum()
    overall_wr   = total_wins / total_trades * 100 if total_trades else 0
    overall_pnl  = df_summary["Total PnL"].sum()

    print("\n" + "=" * 65)
    print("  PORTFOLIO SUMMARY")
    print("=" * 65)
    print(f"  Stocks backtested : {len(summary)}")
    print(f"  Stocks failed     : {len(failed)}")
    print(f"  Total Trades      : {total_trades}")
    print(f"  Total Wins (TP)   : {total_wins}")
    print(f"  Total Losses (SL) : {total_losses}")
    print(f"  Overall Win Rate  : {overall_wr:.1f}%")
    print(f"  Total Net PnL     : ₹{overall_pnl:,.2f}")
    print("=" * 65)

    # ─── Top / Bottom performers ─────────────────────────────────
    print("\n📈  TOP 10 PERFORMERS (by Total PnL):")
    top10 = df_summary.head(10)[["Symbol","Trades","Win%","Total PnL","Expectancy","Brick Size"]]
    print(tabulate(top10, headers="keys", tablefmt="rounded_outline", showindex=False,
                   floatfmt=("", "", ".1f", ",.2f", ".2f", ".2f")))

    print("\n📉  BOTTOM 10 PERFORMERS:")
    bot10 = df_summary.tail(10)[["Symbol","Trades","Win%","Total PnL","Expectancy","Brick Size"]]
    print(tabulate(bot10, headers="keys", tablefmt="rounded_outline", showindex=False,
                   floatfmt=("", "", ".1f", ",.2f", ".2f", ".2f")))

    # ─── Save to Excel ───────────────────────────────────────────
    out_file = "nifty50_renko_results.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_all_trades.to_excel(writer, sheet_name="All Trades", index=False)

        # Per-symbol sheets
        for sym_row in summary:
            sym_name = sym_row["Symbol"]
            sym_df   = df_all_trades[df_all_trades["Symbol"] == sym_name].copy()
            sheet    = sym_name[:31]   # Excel sheet name limit
            sym_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\n✅  Results saved to: {out_file}")
    if failed:
        print(f"⚠️   Skipped ({len(failed)} stocks): {', '.join(s.replace('.NS','') for s in failed)}")


if __name__ == "__main__":
    main()
