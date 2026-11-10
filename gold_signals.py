#!/usr/bin/env python3
"""
Gold (XAU/USD) Signal System v2  —  BUY / SELL / WAIT
Timeframes: 5m, 15m, 1h   (1m removed)

Changes in v2:
  * 1m removed; 5m/15m use 1h as the higher-timeframe trend filter
  * RSI filter: no BUY when RSI > 70, no SELL when RSI < 30
  * Only CLOSED candles are used (no repainting)
  * Paper-trading tracker: signals are logged to trades.csv, closed when
    SL or TP is hit, and profit/loss statistics are printed

Run:
    pip install pandas numpy yfinance
    python gold_signals.py                  # live data (yfinance GC=F) + paper tracking
    python gold_signals.py --loop 60        # refresh every 60 s (screen is cleared)
    python gold_signals.py --stats          # only show paper-trading results
    python gold_signals.py --backtest       # quick historical backtest
    python gold_signals.py --demo           # synthetic data (offline test, no tracking)
    python gold_signals.py --csv-dir data   # data/5m.csv, 15m.csv, 1h.csv

CSV columns: time,open,high,low,close,volume

NOTE: Educational tool. Not financial advice. No signal is guaranteed.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TIMEFRAMES = ["5m", "15m", "1h"]
YF_INTERVAL = {"5m": "5m", "15m": "15m", "1h": "60m"}
YF_PERIOD = {"5m": "30d", "15m": "30d", "1h": "180d"}
HTF = {"5m": "1h", "15m": "1h", "1h": None}   # higher-timeframe trend filter
TRADES_FILE = "trades.csv"
STATE_FILE = "state.json"
MAX_BARS = {"5m": 24, "15m": 16, "1h": 12}      # time exit: 2h / 4h / 12h
CLOSED = ["TP", "SL", "TIME"]
TRADE_COLS = ["id", "tf", "side", "signal_time", "entry", "sl", "tp",
              "status", "exit_time", "exit", "pnl"]


# ----------------------------------------------------------------- data
def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def load_yf(tf: str, symbol: str = "GC=F") -> pd.DataFrame:
    import yfinance as yf  # spot XAUUSD is not offered; GC=F (gold futures) tracks it closely
    df = pd.DataFrame()
    for attempt in range(3):                       # Yahoo sometimes returns empty data; retry
        try:
            df = yf.download(symbol, interval=YF_INTERVAL[tf], period=YF_PERIOD[tf],
                             progress=False, auto_adjust=False)
        except Exception as e:                     # network / rate-limit errors
            print(f"  [{tf}] download error (try {attempt + 1}/3): {e}")
        if df is not None and len(df) > 0:
            break
        time.sleep(2)
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0)          # volume is unused; never drop rows for it
    df = df[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"])
    return to_utc(df.iloc[:-1])   # drop the still-forming candle (no repainting)


def load_csv(tf: str, folder: str) -> pd.DataFrame:
    df = pd.read_csv(f"{folder}/{tf}.csv", parse_dates=["time"], index_col="time")
    return to_utc(df[["open", "high", "low", "close", "volume"]].dropna())


def demo_data(tf: str, n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed + TIMEFRAMES.index(tf))
    step = {"5m": 0.8, "15m": 1.4, "1h": 3.0}[tf]
    drift = np.sin(np.linspace(0, 12, n)) * step * 0.15
    close = 2350 + np.cumsum(rng.normal(drift, step))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, step * 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, step * 0.5, n))
    idx = pd.date_range("2026-01-01", periods=n,
                        freq={"5m": "5min", "15m": "15min", "1h": "1h"}[tf])
    return to_utc(pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                                "volume": rng.integers(100, 1000, n)}, index=idx))


# ----------------------------------------------------------- indicators
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema9"], d["ema21"], d["ema50"] = ema(d.close, 9), ema(d.close, 21), ema(d.close, 50)
    d["rsi"] = rsi(d.close)
    macd = ema(d.close, 12) - ema(d.close, 26)
    d["macd_hist"] = macd - ema(macd, 9)
    d["atr"] = atr(d)
    d["adx"] = adx(d)
    mid, sd = d.close.rolling(20).mean(), d.close.rolling(20).std()
    d["bb_pos"] = (d.close - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    return d


# -------------------------------------------------------------- scoring
def score_row(r) -> float:
    """Score in about [-6, +6]: positive = bullish, negative = bearish."""
    s = 0.0
    s += 1.5 if r.ema9 > r.ema21 else -1.5
    s += 1.5 if r.ema21 > r.ema50 else -1.5
    s += 1.0 if r.macd_hist > 0 else -1.0
    s += 1.0 if r.close > r.ema50 else -1.0
    if r.rsi > 75: s -= 1.0
    elif r.rsi < 25: s += 1.0
    elif r.rsi > 55: s += 0.5
    elif r.rsi < 45: s -= 0.5
    if r.bb_pos > 1.0: s -= 0.5
    elif r.bb_pos < 0.0: s += 0.5
    return s


@dataclass
class Signal:
    tf: str
    action: str
    score: float
    price: float
    sl: float | None
    tp: float | None
    reason: str
    bar_time: pd.Timestamp


def decide(r, htf_bias, threshold, min_adx, rsi_max, rsi_min):
    """Return (action, [reasons]) for one indicator row."""
    sc = score_row(r)
    reasons = [f"score={sc:+.1f}", f"RSI={r.rsi:.0f}", f"ADX={r.adx:.0f}"]
    action = "WAIT"
    if r.adx < min_adx:
        reasons.append("weak trend (ADX low)")
    elif sc >= threshold:
        action = "BUY"
    elif sc <= -threshold:
        action = "SELL"

    if action == "BUY" and r.rsi > rsi_max:
        action = "WAIT"; reasons.append("blocked: RSI overbought")
    elif action == "SELL" and r.rsi < rsi_min:
        action = "WAIT"; reasons.append("blocked: RSI oversold")
    if action == "BUY" and htf_bias < 0:
        action = "WAIT"; reasons.append("blocked: higher TF bearish")
    elif action == "SELL" and htf_bias > 0:
        action = "WAIT"; reasons.append("blocked: higher TF bullish")
    return action, reasons


def make_signal(tf, df, htf_bias=0, threshold=3.5, min_adx=18.0,
                rsi_max=70.0, rsi_min=30.0, sl_atr=1.5, tp_atr=2.5) -> Signal:
    d = add_indicators(df).dropna()
    if d.empty:
        return Signal(tf, "WAIT", 0.0, float("nan"), None, None,
                      f"not enough data ({len(df)} candles, need ~60+)", pd.Timestamp.now(tz="UTC"))
    r = d.iloc[-1]
    action, reasons = decide(r, htf_bias, threshold, min_adx, rsi_max, rsi_min)
    sl = tp = None
    if action == "BUY":
        sl, tp = r.close - sl_atr * r.atr, r.close + tp_atr * r.atr
    elif action == "SELL":
        sl, tp = r.close + sl_atr * r.atr, r.close - tp_atr * r.atr
    return Signal(tf, action, score_row(r), float(r.close), sl, tp,
                  ", ".join(reasons), d.index[-1])


def bias(df: pd.DataFrame) -> int:
    d = add_indicators(df).dropna()
    if d.empty:
        return 0
    s = score_row(d.iloc[-1])
    return 1 if s >= 2 else -1 if s <= -2 else 0


# ------------------------------------------------------------- backtest
def backtest(df, threshold=3.5, min_adx=18.0, rsi_max=70.0, rsi_min=30.0,
             sl_atr=1.5, tp_atr=2.5):
    d = add_indicators(df).dropna()
    trades, pos = [], None
    for i in range(len(d) - 1):
        r, nxt = d.iloc[i], d.iloc[i + 1]
        if pos is None:
            action, _ = decide(r, 0, threshold, min_adx, rsi_max, rsi_min)
            if action != "WAIT":
                side = 1 if action == "BUY" else -1
                entry = nxt.open
                pos = dict(side=side, entry=entry, sl=entry - side * sl_atr * r.atr,
                           tp=entry + side * tp_atr * r.atr)
        else:
            hit_sl = nxt.low <= pos["sl"] if pos["side"] == 1 else nxt.high >= pos["sl"]
            hit_tp = nxt.high >= pos["tp"] if pos["side"] == 1 else nxt.low <= pos["tp"]
            if hit_sl or hit_tp:      # conservative: SL first if both touched
                exit_p = pos["sl"] if hit_sl else pos["tp"]
                trades.append((exit_p - pos["entry"]) * pos["side"])
                pos = None
    return summarize(np.array(trades))


def summarize(t: np.ndarray) -> dict:
    if len(t) == 0:
        return dict(trades=0)
    gross_loss = abs(t[t < 0].sum())
    return dict(trades=len(t), win_rate=round(float((t > 0).mean()) * 100, 1),
                net_points=round(float(t.sum()), 2),
                profit_factor=round(float(t[t > 0].sum() / gross_loss), 2)
                if gross_loss > 0 else float("inf"))


# ------------------------------------------------------------- telegram
def _tg_api(method: str, params: dict | None = None) -> dict | None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        return None
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode() if params else None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  Telegram error: {e}")
        return None


def tg_send(text: str) -> bool:
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat or not os.environ.get("TELEGRAM_TOKEN"):
        print("  Telegram not configured (set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)")
        return False
    res = _tg_api("sendMessage", {"chat_id": chat, "text": text})
    return bool(res and res.get("ok"))


def tg_get_chat_ids():
    """Send any message to your bot first, then run: python gold_signals.py --get-chat-id"""
    res = _tg_api("getUpdates")
    if not res or not res.get("ok"):
        print("Could not reach Telegram. Is TELEGRAM_TOKEN set correctly?")
        return
    ids = {(u.get("message") or {}).get("chat", {}).get("id") for u in res["result"]}
    ids.discard(None)
    print("Your chat ID(s):", ", ".join(map(str, ids)) or "none found - message your bot first")


def stats_line(trades: pd.DataFrame) -> str:
    closed = trades[trades["status"].isin(CLOSED)]
    if closed.empty:
        return "Total: no closed trades yet"
    pnl = pd.to_numeric(closed["pnl"]).to_numpy()
    st = summarize(pnl)
    return f"Total: {st['trades']} trades | win {st['win_rate']}% | net {st['net_points']:+} pts"


# ------------------------------------------------------- paper tracker
def load_trades(path=TRADES_FILE) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, dtype=object)
    return pd.DataFrame(columns=TRADE_COLS, dtype=object)


def update_trades(trades: pd.DataFrame, data: dict, alerts: list | None = None) -> pd.DataFrame:
    """Close OPEN trades on SL / TP touch (candle high/low) or after MAX_BARS candles."""
    for i in trades.index[trades["status"] == "OPEN"]:
        row = trades.loc[i]
        side = 1 if row.side == "BUY" else -1
        entry, sl, tp = float(row.entry), float(row.sl), float(row.tp)
        bars = data[row.tf]
        bars = bars[bars.index > pd.Timestamp(row.signal_time)]
        for n, (ts, b) in enumerate(bars.iterrows(), start=1):
            hit_sl = b.low <= sl if side == 1 else b.high >= sl
            hit_tp = b.high >= tp if side == 1 else b.low <= tp
            status = exit_p = None
            if hit_sl or hit_tp:      # conservative: SL first if both touched
                status, exit_p = ("SL", sl) if hit_sl else ("TP", tp)
            elif n >= MAX_BARS.get(row.tf, 24):
                status, exit_p = "TIME", float(b.close)
            if status:
                pnl_v = (exit_p - entry) * side
                trades.loc[i, ["status", "exit_time", "exit", "pnl"]] = [
                    status, ts.isoformat(), round(exit_p, 2), round(pnl_v, 2)]
                if alerts is not None:
                    head = {"TP": "✅ TP hit", "SL": "❌ SL hit", "TIME": "⏱ Time exit"}[status]
                    alerts.append(f"{head} — XAUUSD {row.tf} {row.side} #{row.id}\n"
                                  f"Entry {entry:.2f} → Exit {exit_p:.2f} ({pnl_v:+.2f} pts)")
                break
    return trades


def open_trade(trades: pd.DataFrame, sig: Signal, alerts: list | None = None) -> pd.DataFrame:
    if sig.action == "WAIT":
        return trades
    same_tf = trades[trades["tf"] == sig.tf]
    if (same_tf["status"] == "OPEN").any():
        return trades                              # one open trade per timeframe
    if (same_tf["signal_time"] == sig.bar_time.isoformat()).any():
        return trades                              # this candle was already traded
    new = dict(id=len(trades) + 1, tf=sig.tf, side=sig.action,
               signal_time=sig.bar_time.isoformat(), entry=round(sig.price, 2),
               sl=round(sig.sl, 2), tp=round(sig.tp, 2), status="OPEN",
               exit_time="", exit="", pnl="")
    print(f"  ➕ New paper trade: {sig.tf} {sig.action} @ {sig.price:.2f}")
    if alerts is not None:
        icon = "🟢" if sig.action == "BUY" else "🔴"
        alerts.append(f"{icon} {sig.action} XAUUSD {sig.tf}\n"
                      f"Price {sig.price:.2f}\nSL {sig.sl:.2f} | TP {sig.tp:.2f}\n"
                      f"{sig.reason}\n(price source: GC=F futures; broker spot may differ)")
    return pd.concat([trades, pd.DataFrame([new], dtype=object)], ignore_index=True)


def open_trade_info(t, data):
    df = data[t.tf]
    if df.empty:
        return None
    side = 1 if t.side == "BUY" else -1
    entry, sl, tp = float(t.entry), float(t.sl), float(t.tp)
    bars = df[df.index > pd.Timestamp(t.signal_time)]
    now = float(df.close.iloc[-1])
    return dict(now=now, pnl=(now - entry) * side, to_tp=(tp - now) * side,
                to_sl=(now - sl) * side,
                hi=float(bars.high.max()) if len(bars) else entry,
                lo=float(bars.low.min()) if len(bars) else entry,
                n=len(bars), max_n=MAX_BARS.get(t.tf, 24))


def status_message(open_: pd.DataFrame, data: dict) -> str:
    lines = ["📊 Open paper trades (price: GC=F)"]
    for _, t in open_.iterrows():
        inf = open_trade_info(t, data)
        if inf:
            lines.append(f"#{t.id} {t.tf} {t.side} entry {t.entry} | now {inf['now']:.2f} "
                         f"({inf['pnl']:+.2f})\nTP in {inf['to_tp']:.2f} | SL in {inf['to_sl']:.2f} "
                         f"| candle {inf['n']}/{inf['max_n']}")
    return "\n".join(lines)


def status_due(hours: float) -> bool:
    try:
        last = json.load(open(STATE_FILE)).get("last_status")
    except Exception:
        last = None
    if last is None:
        return True
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
    return age >= hours * 3600 - 120            # 2-minute tolerance for cron jitter


def mark_status():
    with open(STATE_FILE, "w") as f:
        json.dump({"last_status": datetime.now(timezone.utc).isoformat()}, f)


def print_stats(trades: pd.DataFrame, data: dict | None = None):
    print("\n📒 Paper-trading results")
    closed = trades[trades["status"].isin(CLOSED)]
    open_ = trades[trades["status"] == "OPEN"]
    if closed.empty:
        print("  No closed trades yet.")
    else:
        pnl = pd.to_numeric(closed["pnl"]).to_numpy()
        print(f"  ALL : {summarize(pnl)}")
        for tf in TIMEFRAMES:
            sub = pd.to_numeric(closed[closed["tf"] == tf]["pnl"]).to_numpy()
            if len(sub):
                print(f"  {tf:>3} : {summarize(sub)}")
    if not open_.empty:
        print(f"  Open trades ({len(open_)}):")
        for _, t in open_.iterrows():
            print(f"    #{t.id} {t.tf:>3} {t.side:<4} entry={t.entry} SL={t.sl} TP={t.tp}")
            inf = open_trade_info(t, data) if data else None
            if inf:
                print(f"         now={inf['now']:.2f} ({inf['pnl']:+.2f}) | to TP {inf['to_tp']:.2f} | "
                      f"to SL {inf['to_sl']:.2f} | candle {inf['n']}/{inf['max_n']}")
                print(f"         since entry: low {inf['lo']:.2f} / high {inf['hi']:.2f} (this data feed)")
    print("  (points = USD move per ounce; before spread/commission)")


# ----------------------------------------------------------------- main
def run_once(args):
    data = {}
    for tf in TIMEFRAMES:
        if args.demo:
            data[tf] = demo_data(tf)
        elif args.csv_dir:
            data[tf] = load_csv(tf, args.csv_dir)
        else:
            data[tf] = load_yf(tf)

    counts = {tf: len(data[tf]) for tf in TIMEFRAMES}
    print(f"Candles loaded: {counts}")
    if min(counts.values()) < 60:
        print("⚠ Some timeframes have too little data (Yahoo returned empty/short data). "
              "Check your internet, wait a minute and retry, or use --csv-dir.")
    track = not (args.demo or args.no_track)
    trades = load_trades() if track else None

    if args.stats:
        print_stats(update_trades(trades, data), data) if track else print_stats(load_trades(), data)
        return

    if args.loop:
        os.system("cls" if os.name == "nt" else "clear")
    print(f"Gold (XAU/USD) signals — {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC "
          f"(closed candles only)")
    if not (args.demo or args.csv_dir):
        print("Price source: Yahoo GC=F (gold futures) — can differ from broker spot XAUUSD\n")

    if args.backtest:
        print("Backtest (rule-based, no fees/slippage):")
        for tf in TIMEFRAMES:
            print(f"  {tf:>3}: {backtest(data[tf])}")
        print()

    biases = {tf: bias(data[tf]) for tf in TIMEFRAMES}
    print(f"{'TF':<4} {'SIGNAL':<6} {'PRICE':>9} {'SL':>9} {'TP':>9}  DETAILS")
    signals = []
    for tf in TIMEFRAMES:
        h = HTF[tf]
        sig = make_signal(tf, data[tf], biases[h] if h else 0)
        signals.append(sig)
        sl = f"{sig.sl:9.2f}" if sig.sl else " " * 9
        tp = f"{sig.tp:9.2f}" if sig.tp else " " * 9
        print(f"{sig.tf:<4} {sig.action:<6} {sig.price:9.2f} {sl} {tp}  {sig.reason}")

    if track:
        alerts: list = []
        trades = update_trades(trades, data, alerts)
        for sig in signals:
            trades = open_trade(trades, sig, alerts)
        trades.to_csv(TRADES_FILE, index=False)
        print_stats(trades, data)
        open_now = trades[trades["status"] == "OPEN"]
        if args.telegram and not open_now.empty and not alerts and (
                args.status or (args.status_hours > 0 and status_due(args.status_hours))):
            if tg_send(status_message(open_now, data)):
                mark_status()
        if args.telegram and alerts:
            alerts[-1] += "\n\n" + stats_line(trades)
            for m in alerts:
                tg_send(m + "\n\n⚠ Educational / paper trade — not financial advice.")
    print("\n⚠ Educational only. Always use stop-loss and risk ≤1% per trade.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--stats", action="store_true", help="show paper-trading results only")
    ap.add_argument("--no-track", action="store_true", help="don't write trades.csv")
    ap.add_argument("--loop", type=int, default=0, help="refresh seconds")
    ap.add_argument("--telegram", action="store_true", help="send alerts to Telegram")
    ap.add_argument("--status", action="store_true", help="send open-trade status to Telegram now")
    ap.add_argument("--status-hours", type=float, default=4, help="status report interval (0=off)")
    ap.add_argument("--telegram-test", action="store_true", help="send a test message and exit")
    ap.add_argument("--get-chat-id", action="store_true", help="print your Telegram chat ID and exit")
    a = ap.parse_args()
    if a.get_chat_id:
        tg_get_chat_ids(); raise SystemExit
    if a.telegram_test:
        print("sent" if tg_send("✅ Gold signal bot: test message") else "failed"); raise SystemExit
    while True:
        run_once(a)
        if not a.loop or a.stats:
            break
        time.sleep(a.loop)
