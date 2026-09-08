"""
SOL/USDT (Binance Spot) - Price x HMA(50) Crossover Strategy Backtest
=======================================================================
Python re-implementation of the Pine Script v6 strategy:

    - Long entry: close crosses above HMA(50), filled at the OPEN of the
      following candle (matches Pine's default calc_on_every_tick=false).
    - Stop-loss: the LOW of the crossover candle.
    - Take-profit: entry_price + tpMult * (entry_price - stop_loss)
      tpMult = 12 (i.e. TP = 12x the SL distance).
    - Position sizing: 100% of equity per trade (default_qty_type =
      strategy.percent_of_equity, default_qty_value = 100), no pyramiding.
    - Commission: 0.02% per fill (entry AND exit), 0 slippage.

Usage:
    python sol_hma_backtest.py

Requires: pandas, numpy, matplotlib, requests
    pip install pandas numpy matplotlib requests
"""

import os
import time
import datetime as dt
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================== CONFIG ================================
SYMBOL          = "SOLUSDT"
INTERVAL        = "5m"
YEARS_BACK      = 5
HMA_LEN         = 50
TP_MULT         = 12.0          # Take Profit = SL distance x 12
COMMISSION_RATE = 0.0002        # 0.02% per fill
INITIAL_CAPITAL = 10_000.0
MIN_TICK        = 0.001         # safety-net risk distance if SL == entry (SOLUSDT tick ~ 0.01, kept conservative)
CACHE_DIR       = "data_cache"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Assume worst case (stop-loss fills first) when both SL and TP are touched
# within the same candle. Set to False to assume TP fills first instead.
CONSERVATIVE_SAME_BAR_FILL = True


# =========================== DATA FETCHING =============================
def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated fetch of historical klines from Binance public REST API.
    Caches the result to CSV so repeat runs don't re-download 5 years of data."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f"{symbol}_{interval}_{start_ms}_{end_ms}.csv"
    )
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["open_time", "close_time"])
        return df

    print(f"Downloading {symbol} {interval} klines from Binance "
          f"({dt.datetime.utcfromtimestamp(start_ms/1000)} -> "
          f"{dt.datetime.utcfromtimestamp(end_ms/1000)}) ...")

    limit = 1000
    all_rows = []
    cur = start_ms
    session = requests.Session()

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        for attempt in range(5):
            try:
                resp = session.get(BINANCE_KLINES_URL, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    print(f"Rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                rows = resp.json()
                break
            except requests.RequestException as e:
                print(f"Request failed ({e}), retry {attempt+1}/5...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("Failed to fetch klines after 5 retries")

        if not rows:
            break

        all_rows.extend(rows)
        cur = rows[-1][6] + 1  # last candle's close_time + 1ms
        print(f"  fetched {len(all_rows)} candles, up to "
              f"{dt.datetime.utcfromtimestamp(cur/1000)}", end="\r")
        time.sleep(0.15)  # be polite to the rate limit

    print()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    df.to_csv(cache_path, index=False)
    return df


# ============================ INDICATORS ================================
def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average, matches Pine's ta.hma()."""
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(round(np.sqrt(length))), 1)
    wma_half = wma(series, half_len)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_len)


# ============================= BACKTEST ==================================
def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.reset_index(drop=True).copy()
    df["hma"] = hma(df["close"], HMA_LEN)
    df["cross_up"] = (df["close"] > df["hma"]) & (df["close"].shift(1) <= df["hma"].shift(1))

    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = np.full(len(df), np.nan)

    state = "FLAT"           # FLAT -> PENDING_ENTRY -> IN_POSITION
    sl_price = None
    entry_price = None
    tp_price = None
    entry_idx = None

    n = len(df)
    for i in range(n):
        row = df.iloc[i]

        if state == "FLAT":
            equity_curve[i] = equity
            if row["cross_up"] and not np.isnan(row["hma"]):
                sl_price = row["low"]
                state = "PENDING_ENTRY"
            continue

        if state == "PENDING_ENTRY":
            # Fill at the open of this (the next) candle
            entry_price = row["open"]
            risk_dist = entry_price - sl_price
            if risk_dist <= 0:
                risk_dist = MIN_TICK * 10
            tp_price = entry_price + TP_MULT * risk_dist
            entry_idx = i
            equity_at_entry = equity  # equity right before this trade opened
            state = "IN_POSITION"
            equity_curve[i] = equity
            continue

        if state == "IN_POSITION":
            hit_tp = row["high"] >= tp_price
            hit_sl = row["low"] <= sl_price
            exit_price = None
            exit_reason = None

            if hit_tp and hit_sl:
                if CONSERVATIVE_SAME_BAR_FILL:
                    exit_price, exit_reason = sl_price, "SL"
                else:
                    exit_price, exit_reason = tp_price, "TP"
            elif hit_tp:
                exit_price, exit_reason = tp_price, "TP"
            elif hit_sl:
                exit_price, exit_reason = sl_price, "SL"

            if exit_price is not None:
                gross_mult = exit_price / entry_price
                equity_before = equity
                equity = equity * gross_mult * (1 - COMMISSION_RATE) ** 2
                trades.append({
                    "entry_time": df.iloc[entry_idx]["open_time"],
                    "exit_time": row["open_time"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "reason": exit_reason,
                    "bars_held": i - entry_idx,
                    "return_pct": (equity / equity_before - 1) * 100,
                    "equity_after": equity,
                })
                state = "FLAT"
                sl_price = entry_price = tp_price = entry_idx = None
                equity_curve[i] = equity
            else:
                # Mark-to-market the open position so the equity curve/drawdown
                # reflect intra-trade price action, not just realized P&L.
                equity_curve[i] = equity_at_entry * (row["close"] / entry_price) * (1 - COMMISSION_RATE)

    df["equity"] = equity_curve
    df["equity"] = df["equity"].ffill().fillna(INITIAL_CAPITAL)
    trades_df = pd.DataFrame(trades)
    return trades_df, df


# ============================== METRICS ==================================
def summarize(trades: pd.DataFrame, df: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}

    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]
    total_return_pct = (df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100

    years = (df["open_time"].iloc[-1] - df["open_time"].iloc[0]).days / 365.25
    cagr = ((df["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    running_max = df["equity"].cummax()
    drawdown = (df["equity"] - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    gross_profit = wins["equity_after"].sub(
        wins["equity_after"] / (1 + wins["return_pct"] / 100)
    ).sum() if not wins.empty else 0
    gross_loss = -losses["equity_after"].sub(
        losses["equity_after"] / (1 + losses["return_pct"] / 100)
    ).sum() if not losses.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100,
        "avg_win_pct": wins["return_pct"].mean() if not wins.empty else 0,
        "avg_loss_pct": losses["return_pct"].mean() if not losses.empty else 0,
        "profit_factor": profit_factor,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd_pct,
        "final_equity": df["equity"].iloc[-1],
        "years": years,
    }


def print_summary(stats: dict):
    if stats.get("trades", 0) == 0:
        print("No trades were generated.")
        return
    print("\n===================== BACKTEST SUMMARY =====================")
    print(f"Period covered        : {stats['years']:.2f} years")
    print(f"Total trades          : {stats['trades']}")
    print(f"Win rate              : {stats['win_rate_pct']:.2f}%")
    print(f"Avg win               : {stats['avg_win_pct']:.2f}%")
    print(f"Avg loss              : {stats['avg_loss_pct']:.2f}%")
    print(f"Profit factor         : {stats['profit_factor']:.2f}")
    print(f"Total return          : {stats['total_return_pct']:.2f}%")
    print(f"CAGR                  : {stats['cagr_pct']:.2f}%")
    print(f"Max drawdown          : {stats['max_drawdown_pct']:.2f}%")
    print(f"Final equity          : ${stats['final_equity']:,.2f}")
    print("==============================================================\n")


def plot_equity_curve(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["open_time"], df["equity"], color="#1f9d55", linewidth=1.2)
    ax.set_title(f"{SYMBOL} {INTERVAL} - HMA({HMA_LEN}) Crossover Strategy (TP={TP_MULT}x SL)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================ MAIN ====================================
def main():
    end_dt = dt.datetime.utcnow()
    start_dt = end_dt - dt.timedelta(days=365 * YEARS_BACK)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    df = fetch_binance_klines(SYMBOL, INTERVAL, start_ms, end_ms)
    print(f"Loaded {len(df)} candles from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")

    trades, df = run_backtest(df)
    stats = summarize(trades, df)
    print_summary(stats)

    trades_path = "trades.csv"
    equity_path = "equity_curve.png"
    trades.to_csv(trades_path, index=False)
    plot_equity_curve(df, equity_path)
    print(f"Saved trade log to {trades_path}")
    print(f"Saved equity curve chart to {equity_path}")


if __name__ == "__main__":
    main()
