"""
SOL/USDT Renko Back-Testing Script
====================================
Exchange   : Binance public REST API (no API key needed)
Data       : 3-minute candles, last 365 days (~175,200 bars)
Box Size   : ATR-14 (adaptive — recalculated every closed candle, same as live bot)
Buy Signal : First GREEN brick after a trend reversal (red → green)
Stop-Loss  : 1.5 × ATR below entry
Take-Profit: 3 × SL distance above entry  (i.e. 4.5 × ATR)

Output     : Console summary + backtest_results.csv + equity_curve.png
"""

import asyncio
import csv
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (keep in sync with live bot)
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL          = "SOLUSDT"
ATR_PERIOD      = 14
ATR_MULTIPLIER  = 1.0          # box_size = ATR × multiplier
SL_ATR_MULT     = 1.5          # stop-loss  = entry − 1.5 × ATR
TP_SL_MULT      = 3.0          # take-profit distance = 3 × SL distance

# Backtest window
LOOKBACK_DAYS   = 365          # how many days of history to test
SEED_CANDLES    = 200          # warm-up bars (excluded from trading)

CANDLE_INTERVAL  = "3m"            # Binance interval string
BARS_PER_DAY     = 480             # 1440 min/day ÷ 3 min = 480 bars/day

BINANCE_REST_URL = "https://api.binance.com"
MAX_CANDLES_PER_REQUEST = 1000  # Binance hard limit per call

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("renko_backtest_3m")


# ══════════════════════════════════════════════════════════════════════════════
# ATR CALCULATOR  (identical to live bot — Wilder / RMA smoothing)
# ══════════════════════════════════════════════════════════════════════════════
class ATR:
    def __init__(self, period: int = 14):
        self.period       = period
        self._prev_close: Optional[float] = None
        self._rma:        Optional[float] = None
        self._count       = 0
        self._warm        = False
        self._sum_tr      = 0.0

    @property
    def value(self) -> Optional[float]:
        return self._rma if self._warm else None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low  - self._prev_close))

        self._prev_close = close
        self._count += 1

        if not self._warm:
            self._sum_tr += tr
            if self._count >= self.period:
                self._rma  = self._sum_tr / self.period
                self._warm = True
        else:
            alpha     = 1.0 / self.period
            self._rma = self._rma * (1 - alpha) + tr * alpha

        return self._rma if self._warm else None


# ══════════════════════════════════════════════════════════════════════════════
# RENKO ENGINE  (identical to live bot)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RenkoBrick:
    direction:   int
    open_price:  float
    close_price: float
    formed_at:   datetime


@dataclass
class RenkoState:
    bricks:       list  = field(default_factory=list)
    last_close:   Optional[float] = None
    current_dir:  Optional[int]   = None
    box_size:     Optional[float] = None
    pending_open: Optional[float] = None

    def set_box(self, box: float) -> None:
        self.box_size = round(box, 4)

    def _snap(self, price: float, box: float) -> float:
        return math.floor(price / box) * box

    def seed_price(self, price: float) -> None:
        if self.last_close is None and self.box_size:
            self.last_close   = self._snap(price, self.box_size)
            self.pending_open = self.last_close

    def feed(self, price: float, ts: datetime) -> list:
        if self.box_size is None or self.last_close is None:
            return []

        box    = self.box_size
        new_bx = []

        while True:
            up_target   = self.last_close + box
            down_target = self.last_close - box

            if price >= up_target:
                open_p = self.last_close
                brick  = RenkoBrick(+1, open_p, open_p + box, ts)
                new_bx.append(brick)
                self.bricks.append(brick)
                self.last_close  = open_p + box
                self.current_dir = +1

            elif price <= down_target:
                open_p = self.last_close
                brick  = RenkoBrick(-1, open_p, open_p - box, ts)
                new_bx.append(brick)
                self.bricks.append(brick)
                self.last_close  = open_p - box
                self.current_dir = -1
            else:
                break

        return new_bx


# ══════════════════════════════════════════════════════════════════════════════
# TRADE  (mirrors live bot)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_price:   float
    sl:            float
    tp:            float
    atr_at_entry:  float
    box_at_entry:  float
    entered_at:    datetime
    status:        str             = "OPEN"
    exit_price:    Optional[float] = None
    exited_at:     Optional[datetime] = None

    @property
    def pnl_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.exited_at is None:
            return None
        return (self.exited_at - self.entered_at).total_seconds() / 60


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHER  (paginated — handles 365 days of 3-min bars)
# ══════════════════════════════════════════════════════════════════════════════
async def fetch_all_klines(session: aiohttp.ClientSession,
                           days: int = LOOKBACK_DAYS) -> list[dict]:
    """
    Fetch 3-min candles for the past `days` days from Binance.
    Binance caps at 1 000 candles per request, so we paginate.
    Total bars ≈ days × 480.
    """
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000

    url      = f"{BINANCE_REST_URL}/api/v3/klines"
    candles  = []
    cursor   = start_ms
    total_expected = days * BARS_PER_DAY

    log.info("Downloading %d days of 3-min data (~%d bars) …", days, total_expected)

    while cursor < end_ms:
        params = {
            "symbol":    SYMBOL,
            "interval":  CANDLE_INTERVAL,
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     MAX_CANDLES_PER_REQUEST,
        }
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            raw = await r.json()

        if not raw:
            break

        for k in raw:
            candles.append({
                "open":  float(k[1]),
                "high":  float(k[2]),
                "low":   float(k[3]),
                "close": float(k[4]),
                "ts":    datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            })

        # Next page starts 1 ms after last candle open time
        cursor = int(raw[-1][0]) + 1

        log.info("  Downloaded %d / ~%d candles (3m) …", len(candles), total_expected)

        # Polite rate-limit pause (Binance allows 1200 weight/min; each kline = 1)
        await asyncio.sleep(0.25)

    # Drop the last (potentially unclosed) candle
    if candles:
        candles = candles[:-1]

    log.info("Total candles fetched (excl. last open): %d", len(candles))
    return candles


# ══════════════════════════════════════════════════════════════════════════════
# BACK-TEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(candles: list[dict]) -> list[Trade]:
    """
    Replay candles with the same ATR / Renko / trade logic as the live bot.
    Returns list of completed trades.
    """
    atr      = ATR(ATR_PERIOD)
    renko    = RenkoState()
    trades:  list[Trade] = []
    open_trade: Optional[Trade] = None

    dir_before: Optional[int] = None

    log.info("Running backtest on %d candles …", len(candles))

    for i, c in enumerate(candles):
        price = c["close"]
        high  = c["high"]
        low   = c["low"]
        ts    = c["ts"]

        # ── Check open trade against this candle's High/Low ───────────────
        # Use candle body to simulate intra-candle SL/TP touches.
        # Convention: check SL first (conservative / realistic).
        if open_trade is not None:
            # Did price touch SL?
            if low <= open_trade.sl:
                open_trade.status     = "HIT_SL"
                open_trade.exit_price = open_trade.sl
                open_trade.exited_at  = ts
                trades.append(open_trade)
                open_trade = None
            # Did price touch TP? (only if SL not already hit)
            elif high >= open_trade.tp:
                open_trade.status     = "HIT_TP"
                open_trade.exit_price = open_trade.tp
                open_trade.exited_at  = ts
                trades.append(open_trade)
                open_trade = None

        # ── Update ATR with closed candle ──────────────────────────────────
        atr_val = atr.update(high, low, price)
        if atr_val is None:
            continue   # still warming up

        # ── Adaptive box size (same 5% change threshold as live bot) ──────
        new_box = round(atr_val * ATR_MULTIPLIER, 4)
        if renko.box_size is None:
            renko.set_box(new_box)
            renko.seed_price(price)
        else:
            if abs(new_box - renko.box_size) / renko.box_size > 0.05:
                renko.set_box(new_box)

        # ── Feed close to Renko ────────────────────────────────────────────
        dir_before_tick = renko.current_dir
        new_bricks = renko.feed(price, ts)

        for brick in new_bricks:
            # Signal: first green brick after red→green reversal
            if (
                brick.direction == +1
                and dir_before_tick == -1
                and open_trade is None        # no concurrent trade
                and i >= SEED_CANDLES         # past warm-up period
            ):
                entry   = brick.close_price
                sl      = round(entry - SL_ATR_MULT * atr_val, 4)
                sl_dist = entry - sl
                tp      = round(entry + TP_SL_MULT * sl_dist, 4)
                open_trade = Trade(
                    entry_price  = entry,
                    sl           = sl,
                    tp           = tp,
                    atr_at_entry = atr_val,
                    box_at_entry = renko.box_size,
                    entered_at   = ts,
                )

            # Keep dir_before updated within same batch of bricks
            dir_before_tick = brick.direction

    # Mark any still-open trade as cancelled at last candle close
    if open_trade is not None:
        last_price = candles[-1]["close"]
        open_trade.status     = "CANCELLED"
        open_trade.exit_price = last_price
        open_trade.exited_at  = candles[-1]["ts"]
        trades.append(open_trade)

    return trades


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(trades: list[Trade]) -> None:
    closed = [t for t in trades if t.status != "CANCELLED"]
    wins   = [t for t in closed if t.status == "HIT_TP"]
    losses = [t for t in closed if t.status == "HIT_SL"]
    cancelled = [t for t in trades if t.status == "CANCELLED"]

    total  = len(closed)
    if total == 0:
        log.warning("No completed trades found.")
        return

    pnls      = [t.pnl_pct for t in closed]
    cum_pnl   = sum(pnls)
    avg_pnl   = cum_pnl / total
    win_rate  = len(wins) / total * 100

    avg_win   = sum(t.pnl_pct for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0

    # Max drawdown on cumulative PnL curve
    equity    = []
    running   = 0.0
    peak      = 0.0
    max_dd    = 0.0
    for p in pnls:
        running += p
        equity.append(running)
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    avg_dur = sum(t.duration_minutes for t in closed if t.duration_minutes) / total

    # Profit factor
    gross_win  = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses)) or 1e-9
    pf = gross_win / gross_loss

    bar = "═" * 50
    print(f"\n{bar}")
    print(f"  RENKO BACKTEST RESULTS — {SYMBOL}  ({LOOKBACK_DAYS}d of 3-min data)")
    print(bar)
    print(f"  Total trades      : {total}  (+{len(cancelled)} cancelled/open)")
    print(f"  Wins (TP hit)     : {len(wins)}")
    print(f"  Losses (SL hit)   : {len(losses)}")
    print(f"  Win Rate          : {win_rate:.1f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Avg PnL / trade   : {avg_pnl:+.2f}%")
    print(f"  Avg Win           : {avg_win:+.2f}%")
    print(f"  Avg Loss          : {avg_loss:+.2f}%")
    print(f"  Profit Factor     : {pf:.2f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Cumulative PnL    : {cum_pnl:+.2f}%  (equal-weight, no compounding)")
    print(f"  Max Drawdown      : -{max_dd:.2f}%")
    print(f"  Avg Trade Duration: {avg_dur:.0f} min")
    print(bar)

    if trades:
        first = trades[0].entered_at.strftime("%Y-%m-%d")
        last  = trades[-1].entered_at.strftime("%Y-%m-%d")
        print(f"  Period            : {first} → {last}")
        print(bar)


def save_csv(trades: list[Trade], path: str = "backtest_results.csv") -> None:
    if not trades:
        return
    fields = [
        "trade_no", "status", "entry_price", "sl", "tp",
        "exit_price", "pnl_pct", "atr_at_entry", "box_at_entry",
        "entered_at", "exited_at", "duration_min",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(trades, 1):
            w.writerow({
                "trade_no":      i,
                "status":        t.status,
                "entry_price":   t.entry_price,
                "sl":            t.sl,
                "tp":            t.tp,
                "exit_price":    t.exit_price,
                "pnl_pct":       f"{t.pnl_pct:+.4f}" if t.pnl_pct else "",
                "atr_at_entry":  f"{t.atr_at_entry:.4f}",
                "box_at_entry":  f"{t.box_at_entry:.4f}",
                "entered_at":    t.entered_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "exited_at":     t.exited_at.strftime("%Y-%m-%d %H:%M:%S UTC") if t.exited_at else "",
                "duration_min":  f"{t.duration_minutes:.0f}" if t.duration_minutes else "",
            })
    log.info("Trade log saved → %s", path)


def save_equity_chart(trades: list[Trade], path: str = "equity_curve.png") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("matplotlib not installed — skipping chart. Run: pip install matplotlib")
        return

    closed = [t for t in trades if t.status != "CANCELLED"]
    if not closed:
        return

    dates  = [t.exited_at for t in closed]
    equity = []
    cum    = 0.0
    for t in closed:
        cum += t.pnl_pct
        equity.append(cum)

    # Build drawdown series
    peak   = float("-inf")
    dd     = []
    for e in equity:
        if e > peak:
            peak = e
        dd.append(peak - e)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{SYMBOL} Renko Backtest — {LOOKBACK_DAYS}d of 3-min data\n"
                 f"ATR-14 adaptive box | SL={SL_ATR_MULT}×ATR | TP={TP_SL_MULT}×SL",
                 fontsize=13)

    # Equity curve
    ax1.plot(dates, equity, color="#00c853", linewidth=1.5, label="Cumulative PnL %")
    ax1.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.fill_between(dates, 0, equity,
                     where=[e >= 0 for e in equity], alpha=0.15, color="#00c853")
    ax1.fill_between(dates, 0, equity,
                     where=[e < 0 for e in equity],  alpha=0.15, color="#ff1744")
    ax1.set_ylabel("Cumulative PnL %", color="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#1a1a2e")
    ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3, color="#555")

    # Drawdown
    ax2.fill_between(dates, 0, [-d for d in dd], color="#ff1744", alpha=0.6)
    ax2.set_ylabel("Drawdown %", color="white")
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#1a1a2e")
    ax2.spines[:].set_color("#444")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="white")

    fig.patch.set_facecolor("#0d0d1a")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Equity curve saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
async def main() -> None:
    connector = aiohttp.TCPConnector(limit=5, ssl=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        candles = await fetch_all_klines(session, days=LOOKBACK_DAYS)

    if len(candles) < SEED_CANDLES + ATR_PERIOD:
        log.error("Not enough candles to run backtest.")
        return

    trades = run_backtest(candles)

    print_summary(trades)
    save_csv(trades, "backtest_results.csv")
    save_equity_chart(trades, "equity_curve.png")


if __name__ == "__main__":
    asyncio.run(main())
