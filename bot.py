"""
╔══════════════════════════════════════════════════════════════╗
║        SOL Renko ATR Backtester — Binance Public API        ║
║  Strategy : Buy first bullish Renko after sell-side move    ║
║  SL       : 1.5x ATR below entry                           ║
║  TP       : 3x SL (4.5x ATR) above entry                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import requests
import time
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
#  SETTINGS  ← edit everything here
# ─────────────────────────────────────────────────────────────
SETTINGS = {
    "symbol"          : "SOLUSDT",
    "timeframe"       : "15m",          # 1m 3m 5m 15m 30m 1h 4h 1d
    "years"           : 15,            # how many years of data
    "atr_period"      : 14,
    "renko_mult"      : 1.0,           # brick size = renko_mult × ATR
    "sl_mult"         : 1.5,           # SL = sl_mult × ATR below entry
    "tp_mult"         : 3.0,           # TP = tp_mult × SL above entry
    "capital"         : 1000,          # starting capital in USD
    "fee_pct"         : 0.06,          # round-trip fee %
    "min_sell_bricks" : 2,             # min consecutive bearish bricks before entry
}

BINANCE_URL = "https://api.binance.com/api/v3/klines"
LIMIT       = 1000   # max candles per request


# ─────────────────────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────────────────────
def fetch_all_klines(symbol, interval, years):
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365.25 * 24 * 3600 * 1000)
    all_data = []
    cur      = start_ms
    print(f"\n📡  Fetching {years}y of {symbol} {interval} from Binance…")

    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cur, "endTime": end_ms, "limit": LIMIT}
        r = requests.get(BINANCE_URL, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_data.extend(batch)
        cur = batch[-1][0] + 1
        pct = min(100, int((cur - start_ms) / (end_ms - start_ms) * 100))
        print(f"    {pct:3d}%  ({len(all_data):,} candles)", end="\r")
        if len(batch) < LIMIT:
            break
        time.sleep(0.3)

    print(f"\n✅  {len(all_data):,} candles loaded")
    df = pd.DataFrame(all_data, columns=[
        "t","open","high","low","close","vol",
        "ct","qvol","nt","tbvol","tqvol","_"])
    df = df[["t","open","high","low","close"]].astype(
        {"t": int, "open": float, "high": float, "low": float, "close": float})
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  ATR  (Wilder smoothing)
# ─────────────────────────────────────────────────────────────
def calc_atr(df, period):
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    tr  = np.zeros(len(df))
    atr = np.zeros(len(df))
    s   = 0.0
    for i in range(1, len(df)):
        tr[i] = max(hi[i] - lo[i],
                    abs(hi[i] - cl[i-1]),
                    abs(lo[i] - cl[i-1]))
        if i < period:
            s += tr[i]
        elif i == period:
            s += tr[i]
            atr[i] = s / period
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ─────────────────────────────────────────────────────────────
#  RENKO BUILDER
#  Brick size is locked to the ATR when the brick OPENS so the
#  size cannot shift mid-sequence on a single candle.
# ─────────────────────────────────────────────────────────────
def build_renko(df, atr_arr, mult):
    bricks  = []
    ref     = None
    ref_atr = None

    for i in range(len(df)):
        a = atr_arr[i]
        if a == 0:
            continue
        if ref is None:
            ref     = df["close"].iat[i]
            ref_atr = a
            continue

        price    = df["close"].iat[i]
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
#  STRATEGY
#
#  KEY DESIGN: the exit scan is bounded to candles between the
#  entry candle (ci_entry + 1) and the NEXT brick's candle
#  index (b_next["idx"]).  This prevents looking into the
#  future beyond what a real-time system would have seen.
#
#  Trade lifecycle:
#   OPEN  : when a bullish brick closes after ≥ min_sell_bricks
#            consecutive bearish bricks.  Entry = brick close.
#   CHECK : on every subsequent brick, scan candles from the
#            last-checked candle up to this brick's candle to
#            see if SL or TP was hit.
#   CLOSE : first candle where lo ≤ SL  → exit at SL
#           first candle where hi ≥ TP  → exit at TP
#           end of dataset               → exit at last close
# ─────────────────────────────────────────────────────────────
def run_strategy(bricks, df, s):
    sl_m   = s["sl_mult"]
    tp_m   = s["tp_mult"]
    min_sb = s["min_sell_bricks"]
    fee_rt = s["fee_pct"] / 100   # round-trip fee as a fraction

    trades     = []
    in_trade   = None
    sell_count = 0
    scan_from  = 0   # next candle index to scan for SL/TP

    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    dt = df["date"].values
    n  = len(df)

    for i in range(len(bricks)):
        b = bricks[i]

        # ── not in a trade ──────────────────────────────────
        if in_trade is None:
            if b["dir"] == -1:
                sell_count += 1
            else:
                # bullish brick — check entry condition
                if sell_count >= min_sb:
                    entry     = b["close"]
                    sl        = entry - sl_m * b["atr"]
                    tp        = entry + tp_m * sl_m * b["atr"]
                    in_trade  = {
                        "entry" : entry,
                        "sl"    : sl,
                        "tp"    : tp,
                        "atr"   : b["atr"],
                        "date"  : dt[b["idx"]],
                    }
                    # start scanning from the candle AFTER entry
                    scan_from = b["idx"] + 1
                sell_count = 0   # bullish brick always resets run

        # ── in a trade: check candles up to THIS brick ──────
        else:
            entry = in_trade["entry"]
            sl    = in_trade["sl"]
            tp    = in_trade["tp"]

            # scan only candles we haven't checked yet,
            # up to (and including) the current brick's candle
            scan_to   = min(b["idx"] + 1, n)   # exclusive upper bound
            exit_p, reason = None, None

            for ci in range(scan_from, scan_to):
                if lo[ci] <= sl:
                    exit_p, reason = sl, "SL"
                    break
                if hi[ci] >= tp:
                    exit_p, reason = tp, "TP"
                    break

            # advance scan cursor regardless of exit
            scan_from = scan_to

            if exit_p is None:
                # no exit yet — trade still open, wait for next brick
                continue

            # ── record closed trade ─────────────────────────
            gross_pct = (exit_p - entry) / entry * 100
            # fee_rt is already a fraction (e.g. 0.0006)
            # multiply by 100 to express in %, subtract from gross %
            net_pct   = gross_pct - fee_rt * 100

            trades.append({
                "date"      : pd.Timestamp(in_trade["date"]).strftime("%Y-%m-%d %H:%M"),
                "entry"     : entry,
                "sl"        : sl,
                "tp"        : tp,
                "exit"      : exit_p,
                "reason"    : reason,
                "atr"       : in_trade["atr"],
                "gross_pct" : gross_pct,
                "net_pct"   : net_pct,
                "win"       : net_pct > 0,
            })
            in_trade   = None
            sell_count = 0

    # ── close any still-open trade at last price ────────────
    if in_trade is not None:
        entry     = in_trade["entry"]
        exit_p    = cl[-1]
        gross_pct = (exit_p - entry) / entry * 100
        net_pct   = gross_pct - fee_rt * 100
        trades.append({
            "date"      : pd.Timestamp(in_trade["date"]).strftime("%Y-%m-%d %H:%M"),
            "entry"     : entry,
            "sl"        : in_trade["sl"],
            "tp"        : in_trade["tp"],
            "exit"      : exit_p,
            "reason"    : "EOD",
            "atr"       : in_trade["atr"],
            "gross_pct" : gross_pct,
            "net_pct"   : net_pct,
            "win"       : net_pct > 0,
        })

    return trades


# ─────────────────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────────────────
def calc_stats(trades, capital, fee_pct):
    if not trades:
        return None

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    win_rate = len(wins) / len(trades) * 100
    avg_win  = np.mean([t["net_pct"] for t in wins])   if wins   else 0.0
    avg_loss = np.mean([t["net_pct"] for t in losses]) if losses else 0.0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # ── fixed sizing (always bet on base capital) ─────────────
    eq_f       = capital
    peak_f     = capital
    max_dd_f   = 0.0
    gross_f    = 0.0
    net_f      = 0.0
    eq_f_curve = [capital]

    for t in trades:
        g = capital * t["gross_pct"] / 100
        n = capital * t["net_pct"]   / 100
        gross_f += g
        net_f   += n
        eq_f    += n
        peak_f   = max(peak_f, eq_f)
        dd       = (peak_f - eq_f) / peak_f * 100 if peak_f > 0 else 0.0
        max_dd_f = max(max_dd_f, dd)
        eq_f_curve.append(eq_f)

    # ── compounded sizing (bet on current equity) ─────────────
    eq_c       = capital
    peak_c     = capital
    max_dd_c   = 0.0
    gross_c    = 0.0
    net_c      = 0.0
    eq_c_curve = [capital]

    for t in trades:
        g = eq_c * t["gross_pct"] / 100
        n = eq_c * t["net_pct"]   / 100
        gross_c += g
        net_c   += n
        eq_c    += n
        peak_c   = max(peak_c, eq_c)
        dd       = (peak_c - eq_c) / peak_c * 100 if peak_c > 0 else 0.0
        max_dd_c = max(max_dd_c, dd)
        eq_c_curve.append(eq_c)

    return {
        "total"    : len(trades),
        "wins"     : len(wins),
        "losses"   : len(losses),
        "win_rate" : win_rate,
        "avg_win"  : avg_win,
        "avg_loss" : avg_loss,
        "rr"       : rr,
        "fixed"    : {
            "gross_pnl" : gross_f,
            "net_pnl"   : net_f,
            "final_eq"  : eq_f,
            "max_dd"    : max_dd_f,
            "ret_pct"   : (eq_f - capital) / capital * 100,
            "fee_drag"  : gross_f - net_f,
        },
        "comp"     : {
            "gross_pnl" : gross_c,
            "net_pnl"   : net_c,
            "final_eq"  : eq_c,
            "max_dd"    : max_dd_c,
            "ret_pct"   : (eq_c - capital) / capital * 100,
            "fee_drag"  : gross_c - net_c,
        },
        "eq_fixed_curve" : eq_f_curve,
        "eq_comp_curve"  : eq_c_curve,
    }


# ─────────────────────────────────────────────────────────────
#  PRINT REPORT
# ─────────────────────────────────────────────────────────────
def print_report(stats, settings, trades):
    c   = settings["capital"]
    fee = settings["fee_pct"]
    W    = "\033[0m"
    G    = "\033[92m"
    R    = "\033[91m"
    Y    = "\033[93m"
    B    = "\033[94m"
    BOLD = "\033[1m"

    def usd(v): return f"${v:>12,.2f}"
    def pct(v): return f"{v:>+8.2f}%"
    def sep():  print(f"{Y}{'─'*60}{W}")

    print(f"\n{BOLD}{B}{'═'*60}")
    print(f"  SOL RENKO ATR BACKTEST RESULTS")
    print(f"  {settings['symbol']} · {settings['timeframe']} · {settings['years']}y")
    print(f"{'═'*60}{W}")

    sep()
    print(f"  TRADE SUMMARY")
    sep()
    print(f"  Total Trades        : {BOLD}{stats['total']}{W}")
    print(f"  Wins                : {G}{stats['wins']}{W}")
    print(f"  Losses              : {R}{stats['losses']}{W}")
    print(f"  Win Rate            : {BOLD}{stats['win_rate']:.2f}%{W}")
    print(f"  Avg Win  (net)      : {G}{stats['avg_win']:+.2f}%{W}")
    print(f"  Avg Loss (net)      : {R}{stats['avg_loss']:+.2f}%{W}")
    print(f"  Risk : Reward       : 1 : {stats['rr']:.2f}")

    for label, key in [("FIXED SIZING", "fixed"), ("COMPOUNDED SIZING", "comp")]:
        st = stats[key]
        sep()
        print(f"  {label}  (capital: ${c:,.0f})")
        sep()
        col = G if st["gross_pnl"] >= 0 else R
        print(f"  Gross P&L           : {col}{usd(st['gross_pnl'])}{W}   (before fees)")
        col = G if st["net_pnl"] >= 0 else R
        print(f"  Net   P&L           : {col}{usd(st['net_pnl'])}{W}   (after {fee}% fee)")
        col = G if st["final_eq"] >= c else R
        print(f"  Final Equity        : {col}{usd(st['final_eq'])}{W}")
        col = G if st["ret_pct"] >= 0 else R
        print(f"  Total Return        : {col}{pct(st['ret_pct'])}{W}")
        print(f"  Max Drawdown        : {R}{st['max_dd']:>8.2f}%{W}")
        print(f"  Fee Drag            : {Y}{usd(st['fee_drag'])}{W}")

    sep()
    print(f"  FEE IMPACT SUMMARY")
    sep()
    print(f"  Fee per round trip  : {fee}%")
    print(f"  Total fees (fixed)  : {Y}{usd(stats['fixed']['fee_drag'])}{W}")
    print(f"  Total fees (comp)   : {Y}{usd(stats['comp']['fee_drag'])}{W}")
    print(f"  Trades × fee        : {stats['total']} × {fee}% = {stats['total']*fee:.2f}%")
    sep()

    # ── trade log (last 20) ──────────────────────────────────
    print(f"\n  Last 20 Trades:")
    hdr = (f"  {'#':>4}  {'Date':<17}  {'Entry':>8}  {'SL':>8}  "
           f"{'TP':>8}  {'Exit':>8}  {'Why':<4}  {'Gross':>7}  {'Net':>7}  Result")
    print(hdr)
    print(f"  {'─'*len(hdr)}")
    for i, t in enumerate(trades[-20:], start=max(1, len(trades) - 19)):
        res = f"{G}WIN {W}" if t["win"] else f"{R}LOSS{W}"
        gc  = G if t["gross_pct"] >= 0 else R
        nc  = G if t["net_pct"]   >= 0 else R
        print(f"  {i:>4}  {t['date']:<17}  {t['entry']:>8.3f}  {t['sl']:>8.3f}  "
              f"{t['tp']:>8.3f}  {t['exit']:>8.3f}  {t['reason']:<4}  "
              f"{gc}{t['gross_pct']:>+6.2f}%{W}  {nc}{t['net_pct']:>+6.2f}%{W}  {res}")


# ─────────────────────────────────────────────────────────────
#  SAVE CSV
# ─────────────────────────────────────────────────────────────
def save_csv(trades, settings):
    df    = pd.DataFrame(trades)
    fname = f"backtest_{settings['symbol']}_{settings['timeframe']}.csv"
    df.to_csv(fname, index=False)
    print(f"\n💾  Trade log saved → {fname}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    s = SETTINGS
    print(__doc__)
    print(f"  Symbol     : {s['symbol']}")
    print(f"  Timeframe  : {s['timeframe']}")
    print(f"  Years      : {s['years']}")
    print(f"  ATR period : {s['atr_period']}")
    print(f"  Brick size : {s['renko_mult']}× ATR")
    print(f"  SL         : {s['sl_mult']}× ATR")
    print(f"  TP         : {s['tp_mult']}× SL  ({s['tp_mult']*s['sl_mult']}× ATR)")
    print(f"  Capital    : ${s['capital']:,}")
    print(f"  Fee        : {s['fee_pct']}% round trip")
    print(f"  Min sell   : {s['min_sell_bricks']} consecutive bearish bricks")

    df      = fetch_all_klines(s["symbol"], s["timeframe"], s["years"])
    atr_arr = calc_atr(df, s["atr_period"])

    print(f"\n🧱  Building Renko bricks (mult={s['renko_mult']}×ATR)…")
    bricks  = build_renko(df, atr_arr, s["renko_mult"])
    print(f"    {len(bricks):,} bricks built")

    print(f"⚡  Running strategy…")
    trades  = run_strategy(bricks, df, s)
    print(f"    {len(trades):,} trades found")

    stats = calc_stats(trades, s["capital"], s["fee_pct"])
    if stats is None:
        print("❌  No trades generated. Try adjusting settings.")
        return

    print_report(stats, s, trades)
    save_csv(trades, s)


if __name__ == "__main__":
    main()
