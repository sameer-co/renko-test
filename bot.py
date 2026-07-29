"""
╔══════════════════════════════════════════════════════════════════╗
║      SOL Renko ATR Forward Tester — Binance Public API          ║
║  Runs two loops in parallel: 3m and 5m timeframes               ║
║  Sends Telegram alerts on ENTRY, TP, SL, and heartbeat         ║
╚══════════════════════════════════════════════════════════════════╝

Strategy (same as backtest):
  • Build ATR-based Renko bricks from live candles
  • BUY when a bullish brick forms after ≥ min_sell_bricks bearish ones
  • SL  = entry − sl_mult × ATR
  • TP  = entry + tp_mult × sl_mult × ATR
  • Alerts sent via Telegram for every key event
"""

import requests
import time
import threading
import numpy as np
from datetime import datetime, timezone
from collections import deque

# ─────────────────────────────────────────────────────────────
#  TELEGRAM CONFIG
# ─────────────────────────────────────────────────────────────
TG_TOKEN   = "8661081060:AAGtNViZMS6FSl_7vQeMz1TcCnzrFddu7z4"
TG_CHAT_ID = "1950462171"
TG_URL     = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

# ─────────────────────────────────────────────────────────────
#  STRATEGY SETTINGS
# ─────────────────────────────────────────────────────────────
SETTINGS = {
    "symbol"          : "SOLUSDT",
    "timeframes"      : ["3m", "5m"],   # both run in parallel
    "atr_period"      : 14,
    "renko_mult"      : 1.0,            # brick size = renko_mult × ATR
    "sl_mult"         : 1.5,            # SL = sl_mult × ATR below entry
    "tp_mult"         : 3.0,            # TP = tp_mult × SL above entry
    "min_sell_bricks" : 2,              # min consecutive bearish bricks before entry
    "lookback_candles": 200,            # candles to fetch for ATR + Renko context
    "heartbeat_mins"  : 60,             # send "still alive" ping every N minutes
}

BINANCE_URL = "https://api.binance.com/api/v3/klines"
PRICE_URL   = "https://api.binance.com/api/v3/ticker/price"

# Interval in seconds between each poll per timeframe
POLL_SECONDS = {
    "3m": 3 * 60,   # poll every 3 minutes
    "5m": 5 * 60,   # poll every 5 minutes
}


# ─────────────────────────────────────────────────────────────
#  TELEGRAM HELPERS
# ─────────────────────────────────────────────────────────────
def send_tg(message: str, retries: int = 3):
    """Send a Telegram message with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.post(TG_URL, json={
                "chat_id"   : TG_CHAT_ID,
                "text"      : message,
                "parse_mode": "HTML",
            }, timeout=10)
            if r.status_code == 200:
                return True
            print(f"[TG] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[TG] Error (attempt {attempt+1}): {e}")
        time.sleep(2)
    return False


def fmt_price(p: float) -> str:
    return f"{p:.4f}"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ─────────────────────────────────────────────────────────────
#  DATA FETCH — last N closed candles
# ─────────────────────────────────────────────────────────────
def fetch_candles(symbol: str, interval: str, limit: int = 200):
    """
    Fetch the last `limit` CLOSED candles.
    Binance returns the in-progress candle last, so we drop it.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit + 1}
    r = requests.get(BINANCE_URL, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()[:-1]   # drop the open/live candle

    opens  = np.array([float(c[1]) for c in raw])
    highs  = np.array([float(c[2]) for c in raw])
    lows   = np.array([float(c[3]) for c in raw])
    closes = np.array([float(c[4]) for c in raw])
    times  = np.array([int(c[0])   for c in raw])
    return opens, highs, lows, closes, times


def fetch_current_price(symbol: str) -> float:
    r = requests.get(PRICE_URL, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


# ─────────────────────────────────────────────────────────────
#  ATR (Wilder smoothing) — same as backtest
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
#  RENKO BUILDER — same logic as backtest
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
#  SIGNAL DETECTION
#  Returns the signal dict if a new trade signal is found,
#  otherwise None.
#  State carries: last known brick list (as snapshot) so we
#  can detect when new bricks appear.
# ─────────────────────────────────────────────────────────────
def detect_signal(bricks, min_sell_bricks: int, sl_mult: float,
                  tp_mult: float, last_brick_count: int,
                  last_entry_price: float = 0.0,
                  atr_gap_mult: float = 1.0):
    """
    Walk from where we last left off.
    Return (signal_dict | None, new_last_brick_count).

    FIX - duplicate entry guard:
    Bricks are rebuilt fresh each poll, so the same historical
    bullish brick can re-trigger after a SL/TP reset if last_n
    shifts slightly. We block any new signal whose entry is within
    (atr_gap_mult x ATR) of the previous entry price. A genuine
    new signal will be at least 1 ATR away from the last entry.
    """
    n = len(bricks)
    if n == last_brick_count or n < 3:
        return None, n

    sell_run = 0
    signal   = None

    for i in range(n):
        b = bricks[i]
        if b["dir"] == -1:
            sell_run += 1
        else:
            # bullish brick - only consider NEW bricks (beyond last cursor)
            if sell_run >= min_sell_bricks and i >= last_brick_count:
                entry = b["close"]
                atr   = b["atr"]
                # duplicate guard: skip if entry is within atr_gap_mult x ATR
                # of the last entry - it is the same signal replaying
                if last_entry_price > 0:
                    gap = abs(entry - last_entry_price)
                    if gap < atr_gap_mult * atr:
                        sell_run = 0
                        continue
                sl    = entry - sl_mult * atr
                tp    = entry + tp_mult * sl_mult * atr
                signal = {
                    "entry"    : entry,
                    "sl"       : sl,
                    "tp"       : tp,
                    "atr"      : atr,
                    "sell_run" : sell_run,
                }
            sell_run = 0

    return signal, n


# ─────────────────────────────────────────────────────────────
#  TRADE MONITOR — checks open trade against live price
# ─────────────────────────────────────────────────────────────
def check_trade(trade: dict, symbol: str) -> tuple:
    """
    Returns ("TP" | "SL" | "OPEN", current_price)
    """
    price = fetch_current_price(symbol)
    if price >= trade["tp"]:
        return "TP", price
    if price <= trade["sl"]:
        return "SL", price
    return "OPEN", price


# ─────────────────────────────────────────────────────────────
#  PER-TIMEFRAME WORKER
# ─────────────────────────────────────────────────────────────
class RenkoWorker:
    def __init__(self, symbol: str, timeframe: str, settings: dict):
        self.symbol    = symbol
        self.tf        = timeframe
        self.s         = settings
        self.trade            = None   # current open trade or None
        self.last_n           = 0      # last known brick count
        self.last_entry_price = 0.0    # price of last entry (duplicate guard)
        self.last_beat        = time.time()
        self.total_trades     = 0
        self.wins             = 0

        tag = f"[{symbol} {timeframe}]"
        send_tg(
            f"🚀 <b>Forward Tester Started</b>\n"
            f"Pair: <code>{symbol}</code> | TF: <code>{timeframe}</code>\n"
            f"ATR period: {settings['atr_period']} | "
            f"Brick: {settings['renko_mult']}×ATR\n"
            f"SL: {settings['sl_mult']}×ATR | "
            f"TP: {settings['tp_mult']}×SL\n"
            f"Min sell bricks: {settings['min_sell_bricks']}\n"
            f"Time: {now_utc()}"
        )
        print(f"{tag} Worker initialised")

    # ── main loop ───────────────────────────────────────────
    def run(self):
        poll = POLL_SECONDS.get(self.tf, 180)
        tag  = f"[{self.symbol} {self.tf}]"

        while True:
            try:
                self._tick()
            except Exception as e:
                msg = f"⚠️ {tag} Error: {e}"
                print(msg)
                send_tg(msg)

            # heartbeat
            if time.time() - self.last_beat >= self.s["heartbeat_mins"] * 60:
                self._heartbeat()
                self.last_beat = time.time()

            time.sleep(poll)

    # ── one poll cycle ───────────────────────────────────────
    def _tick(self):
        tag  = f"[{self.symbol} {self.tf}]"
        _, highs, lows, closes, _ = fetch_candles(
            self.symbol, self.tf, self.s["lookback_candles"])
        atr_arr = calc_atr(highs, lows, closes, self.s["atr_period"])
        bricks  = build_renko(closes, atr_arr, self.s["renko_mult"])

        # ── if in a trade, check for exit ───────────────────
        if self.trade is not None:
            status, price = check_trade(self.trade, self.symbol)
            t = self.trade

            if status == "TP":
                pct = (t["tp"] - t["entry"]) / t["entry"] * 100
                self.wins += 1
                self.total_trades += 1
                send_tg(
                    f"✅ <b>TAKE PROFIT HIT</b> — {self.symbol} {self.tf}\n"
                    f"Entry : <code>{fmt_price(t['entry'])}</code>\n"
                    f"TP    : <code>{fmt_price(t['tp'])}</code>\n"
                    f"Price : <code>{fmt_price(price)}</code>\n"
                    f"P&L   : <b>+{pct:.2f}%</b>\n"
                    f"Win rate: {self.wins}/{self.total_trades} "
                    f"({100*self.wins/self.total_trades:.1f}%)\n"
                    f"⏰ {now_utc()}"
                )
                print(f"{tag} ✅ TP hit at {price:.4f}  (+{pct:.2f}%)")
                self.trade = None
                self.last_n = len(bricks)   # reset brick cursor

            elif status == "SL":
                pct = (t["sl"] - t["entry"]) / t["entry"] * 100
                self.total_trades += 1
                send_tg(
                    f"❌ <b>STOP LOSS HIT</b> — {self.symbol} {self.tf}\n"
                    f"Entry : <code>{fmt_price(t['entry'])}</code>\n"
                    f"SL    : <code>{fmt_price(t['sl'])}</code>\n"
                    f"Price : <code>{fmt_price(price)}</code>\n"
                    f"P&L   : <b>{pct:.2f}%</b>\n"
                    f"Win rate: {self.wins}/{self.total_trades} "
                    f"({100*self.wins/self.total_trades:.1f}%)\n"
                    f"⏰ {now_utc()}"
                )
                print(f"{tag} ❌ SL hit at {price:.4f}  ({pct:.2f}%)")
                self.trade = None
                self.last_n = len(bricks)

            else:
                # trade still open — log to console only
                pct = (price - t["entry"]) / t["entry"] * 100
                print(f"{tag} 📊 Open trade | Price: {price:.4f} | "
                      f"Entry: {t['entry']:.4f} | P&L: {pct:+.2f}%")
            return

        # ── not in a trade — scan for new signal ────────────
        signal, new_n = detect_signal(
            bricks,
            self.s["min_sell_bricks"],
            self.s["sl_mult"],
            self.s["tp_mult"],
            self.last_n,
            last_entry_price = self.last_entry_price,
            atr_gap_mult     = 1.0,
        )
        self.last_n = new_n

        if signal:
            self.trade            = signal
            self.last_entry_price = signal["entry"]
            rr = self.s["tp_mult"] * self.s["sl_mult"]
            send_tg(
                f"🟢 <b>BUY SIGNAL</b> — {self.symbol} {self.tf}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Entry  : <code>{fmt_price(signal['entry'])}</code>\n"
                f"SL     : <code>{fmt_price(signal['sl'])}</code>  "
                f"(−{self.s['sl_mult']}×ATR)\n"
                f"TP     : <code>{fmt_price(signal['tp'])}</code>  "
                f"(+{rr}×ATR)\n"
                f"ATR    : <code>{fmt_price(signal['atr'])}</code>\n"
                f"SL dist: <code>{fmt_price(signal['entry']-signal['sl'])}</code>  "
                f"({(signal['entry']-signal['sl'])/signal['entry']*100:.2f}%)\n"
                f"TP dist: <code>{fmt_price(signal['tp']-signal['entry'])}</code>  "
                f"({(signal['tp']-signal['entry'])/signal['entry']*100:.2f}%)\n"
                f"Sell bricks before: {signal['sell_run']}\n"
                f"⏰ {now_utc()}"
            )
            print(f"{tag} 🟢 BUY signal | Entry: {signal['entry']:.4f} | "
                  f"SL: {signal['sl']:.4f} | TP: {signal['tp']:.4f}")
        else:
            print(f"{tag} 👀 No signal | Bricks: {new_n} | "
                  f"Last close: {closes[-1]:.4f}")

    # ── heartbeat ────────────────────────────────────────────
    def _heartbeat(self):
        tag   = f"[{self.symbol} {self.tf}]"
        price = fetch_current_price(self.symbol)
        in_t  = "YES" if self.trade else "NO"
        wr    = (f"{self.wins}/{self.total_trades} "
                 f"({100*self.wins/self.total_trades:.1f}%)"
                 if self.total_trades else "No trades yet")

        msg = (
            f"💓 <b>Heartbeat</b> — {self.symbol} {self.tf}\n"
            f"Price    : <code>{fmt_price(price)}</code>\n"
            f"In trade : {in_t}\n"
            f"Win rate : {wr}\n"
            f"⏰ {now_utc()}"
        )
        if self.trade:
            t   = self.trade
            pct = (price - t["entry"]) / t["entry"] * 100
            msg += (
                f"\n\n📌 Open Trade\n"
                f"Entry: <code>{fmt_price(t['entry'])}</code>\n"
                f"SL   : <code>{fmt_price(t['sl'])}</code>\n"
                f"TP   : <code>{fmt_price(t['tp'])}</code>\n"
                f"P&L  : <b>{pct:+.2f}%</b>"
            )

        send_tg(msg)
        print(f"{tag} 💓 Heartbeat | Price: {price:.4f} | "
              f"Trade: {in_t} | {wr}")


# ─────────────────────────────────────────────────────────────
#  MAIN — spin up one thread per timeframe
# ─────────────────────────────────────────────────────────────
def main():
    print(__doc__)
    s = SETTINGS

    # Send a single startup message listing both timeframes
    send_tg(
        f"🤖 <b>SOL Renko ATR Forward Tester</b>\n"
        f"Launching workers for: "
        f"{', '.join(s['timeframes'])}\n"
        f"Symbol: <code>{s['symbol']}</code>\n"
        f"⏰ {now_utc()}"
    )

    threads = []
    for tf in s["timeframes"]:
        worker = RenkoWorker(s["symbol"], tf, s)
        t = threading.Thread(
            target=worker.run,
            name=f"Worker-{tf}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        print(f"[MAIN] Started thread for {tf}")
        time.sleep(2)   # stagger starts slightly

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
            # Print a console status line every minute
            alive = [t.name for t in threads if t.is_alive()]
            print(f"[MAIN] {now_utc()} | Alive threads: {alive}")
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down…")
        send_tg(
            f"🛑 <b>Forward Tester Stopped</b>\n"
            f"Symbol: {s['symbol']}\n"
            f"⏰ {now_utc()}"
        )


if __name__ == "__main__":
    main()
