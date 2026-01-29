import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

BASE_URL = "https://api.twelvedata.com/time_series"

# Lookbacks
LOOKBACK_1H_BARS = int(os.getenv("LOOKBACK_1H_BARS", "420"))   # ~105x 4H bars
LOOKBACK_1D_BARS = int(os.getenv("LOOKBACK_1D_BARS", "320"))   # >= 252 for 52W

# Bollinger settings
BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "1.5"))

# "Near band" trigger (percentage)
# Example: 0.003 = within 0.3% of band
NEAR_BAND_PCT = float(os.getenv("NEAR_BAND_PCT", "0.003"))

# RSI confirmation thresholds (simple long-only swing style)
RSI_BUY_MAX = float(os.getenv("RSI_BUY_MAX", "40"))     # allow BUY if RSI <= 40
RSI_SELL_MIN = float(os.getenv("RSI_SELL_MIN", "60"))   # allow TRIM if RSI >= 60

# Chart bars
CHART_BARS = int(os.getenv("CHART_BARS", "140"))

# Twelve Data free tier pacing
TD_MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("TD_MIN_SECONDS_BETWEEN_CALLS", "9.0"))

# Small pause for Telegram sends
SLEEP_BETWEEN_OTHER_CALLS = float(os.getenv("SLEEP_BETWEEN_OTHER_CALLS", "0.3"))

# =========================
# RATE LIMIT (Twelve Data only)
# =========================
_last_td_call_ts = 0.0

def td_throttle():
    global _last_td_call_ts
    now = time.time()
    wait = TD_MIN_SECONDS_BETWEEN_CALLS - (now - _last_td_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_td_call_ts = time.time()

# =========================
# UTIL
# =========================
def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)

def utc_now_str():
    return utc_now().strftime("%Y-%m-%d %H:%M UTC")

def ensure_dirs():
    os.makedirs("charts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

# =========================
# TELEGRAM
# =========================
def tg_send_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()

def tg_send_photo(photo_path: str, caption: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption[:900],
            "disable_web_page_preview": True
        }
        r = requests.post(url, data=data, files=files, timeout=60)
        r.raise_for_status()

# =========================
# TWELVE DATA FETCH (CACHED)
# =========================
_cache = {}  # (symbol, interval) -> DataFrame

def fetch_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    key = (symbol, interval)
    if key in _cache:
        return _cache[key]

    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    td_throttle()

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "apikey": TD_API_KEY,
        "format": "JSON",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"TwelveData HTTP {r.status_code}: {r.text[:250]}")

    data = r.json()

    # Rate/credit handling (retry once after 60s if per-minute cap hit)
    if isinstance(data, dict) and data.get("status") == "error":
        msg = (data.get("message") or "").lower()
        if "run out of api credits" in msg or "current minute" in msg:
            time.sleep(60)
            td_throttle()
            r2 = requests.get(BASE_URL, params=params, timeout=30)
            if r2.status_code != 200:
                raise RuntimeError(f"TwelveData HTTP {r2.status_code}: {r2.text[:250]}")
            data = r2.json()

        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData error: {data.get('message')}")

    values = data.get("values")
    if not values:
        raise RuntimeError(f"TwelveData: no values for {symbol} ({interval}). Keys: {list(data.keys())}")

    df = pd.DataFrame(values).rename(columns={
        "datetime": "t",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    })
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o", "h", "l", "c", "v"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t").set_index("t")
    _cache[key] = df
    return df

def fetch_1h(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1h", LOOKBACK_1H_BARS)

def fetch_1d(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1day", LOOKBACK_1D_BARS)

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # lower-case frequency avoids pandas FutureWarning
    return df.resample(rule).agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }).dropna()

# =========================
# INDICATORS
# =========================
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger(close: pd.Series, period: int, stdev: float):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    return ma, upper, lower

def support_resistance(df: pd.DataFrame, window: int = 60):
    sup = float(df["l"].tail(window).min())
    res = float(df["h"].tail(window).max())
    return sup, res

# =========================
# 52W HIGH
# =========================
def compute_52w(daily_df: pd.DataFrame, last_close: float):
    d = daily_df.tail(252)
    high_52w = float(d["h"].max())
    pct_from = ((last_close - high_52w) / high_52w) * 100.0
    return high_52w, pct_from

# =========================
# ORDER SETTER LOGIC
# =========================
def decide_orders(close, rsi_val, bb_mid, bb_upper, bb_lower):
    """
    Deterministic mean-reversion order setter:
    - BUY LIMIT at lower BB(1.5σ) when price is near lower band and RSI <= RSI_BUY_MAX
    - SELL/TRIM LIMIT at upper BB(1.5σ) when price is near upper band and RSI >= RSI_SELL_MIN
    """
    orders = {
        "buy_limit": float(bb_lower),
        "sell_limit": float(bb_upper),
        "place_buy": False,
        "place_sell": False,
        "setup": "Mean Reversion (BB 1.5σ)",
        "why": ""
    }

    # "Near band" tests
    near_lower = close <= bb_lower * (1.0 + NEAR_BAND_PCT)
    near_upper = close >= bb_upper * (1.0 - NEAR_BAND_PCT)

    # Confirmations
    buy_ok = (rsi_val <= RSI_BUY_MAX)
    sell_ok = (rsi_val >= RSI_SELL_MIN)

    if near_lower and buy_ok:
        orders["place_buy"] = True
        orders["why"] = f"Close near LOWER BB and RSI({rsi_val:.1f}) <= {RSI_BUY_MAX:.0f} (ADD zone)."

    if near_upper and sell_ok:
        orders["place_sell"] = True
        orders["why"] = f"Close near UPPER BB and RSI({rsi_val:.1f}) >= {RSI_SELL_MIN:.0f} (TRIM zone)."

    # If both are true (rare), keep both; user can decide sizing / tranche logic
    if orders["place_buy"] and orders["place_sell"]:
        orders["why"] = "Price is near BOTH bands (very volatile / gap-like). Review manually."

    return orders

# =========================
# CHARTS (only when an order is to be placed)
# =========================
def make_order_chart(symbol: str, df_4h: pd.DataFrame, bb_mid, bb_upper, bb_lower,
                     buy_limit: float, sell_limit: float, out_path: str):
    d = df_4h.tail(CHART_BARS).copy()
    x = d.index
    close = d["c"]

    # compute RSI for plotting
    r = rsi_wilder(close, 14).bfill()

    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, close, label="Close")
    ax1.plot(x, bb_mid.tail(len(d)), label=f"BB Mid ({BB_PERIOD})")
    ax1.plot(x, bb_upper.tail(len(d)), label=f"BB Upper ({BB_STD}σ)")
    ax1.plot(x, bb_lower.tail(len(d)), label=f"BB Lower ({BB_STD}σ)")

    ax1.axhline(buy_limit, linestyle="--")
    ax1.axhline(sell_limit, linestyle="--")
    ax1.set_title(f"{symbol} — 4H | Order Setter (BB {BB_STD}σ)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(x, r, label="RSI(14)")
    ax2.axhline(70, linestyle="--")
    ax2.axhline(30, linestyle="--")
    ax2.set_title("RSI(14) — confirmation only")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# =========================
# MAIN
# =========================
def main():
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    ensure_dirs()
    ts_str = utc_now_str()

    results = []
    failures = []
    actionable_symbols = []

    for symbol in SYMBOLS:
        try:
            df_1h = fetch_1h(symbol)
            df_4h = resample_ohlc(df_1h, "4h")

            if len(df_4h) < max(60, BB_PERIOD + 5):
                raise RuntimeError(f"{symbol} 4H: not enough bars ({len(df_4h)})")

            close = float(df_4h["c"].iloc[-1])
            rsi_val = float(rsi_wilder(df_4h["c"], 14).iloc[-1])

            # BB
            bb_mid, bb_upper, bb_lower = bollinger(df_4h["c"], BB_PERIOD, BB_STD)
            bb_mid_v = float(bb_mid.iloc[-1])
            bb_upper_v = float(bb_upper.iloc[-1])
            bb_lower_v = float(bb_lower.iloc[-1])

            # Structure
            sup, res = support_resistance(df_4h)

            # Distances
            dist_to_sup_pct = ((sup - close) / close) * 100.0
            dist_to_res_pct = ((res - close) / close) * 100.0

            # 52W context
            df_1d = fetch_1d(symbol)
            high_52w, pct_from_52w = compute_52w(df_1d, close)

            # Decide orders
            orders = decide_orders(close, rsi_val, bb_mid_v, bb_upper_v, bb_lower_v)

            # Action label for summary
            if orders["place_buy"] and not orders["place_sell"]:
                action = "PLACE BUY LIMIT"
            elif orders["place_sell"] and not orders["place_buy"]:
                action = "PLACE TRIM LIMIT"
            elif orders["place_buy"] and orders["place_sell"]:
                action = "REVIEW (VOLATILE)"
            else:
                action = "NO ORDER"

            if orders["place_buy"] or orders["place_sell"]:
                actionable_symbols.append(symbol)

            results.append({
                "symbol": symbol,
                "df_4h": df_4h,
                "close": close,
                "rsi": rsi_val,
                "sup": sup,
                "res": res,
                "dist_to_sup_pct": dist_to_sup_pct,
                "dist_to_res_pct": dist_to_res_pct,
                "high_52w": high_52w,
                "pct_from_52w": pct_from_52w,
                "bb_mid": bb_mid,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_mid_v": bb_mid_v,
                "bb_upper_v": bb_upper_v,
                "bb_lower_v": bb_lower_v,
                "orders": orders,
                "action": action
            })

        except Exception as e:
            failures.append(f"{symbol}: {repr(e)}")

    # ===== SUMMARY MESSAGE =====
    lines = [
        "📊 INTRADAY ORDER-SETTER (4H)",
        ts_str,
        "=" * 50,
        f"Rules: BB({BB_PERIOD}, {BB_STD}σ) + RSI(14) confirm | NearBand={NEAR_BAND_PCT*100:.1f}%",
        ""
    ]

    if results:
        for r in results:
            o = r["orders"]
            lines.append(f"🔷 {r['symbol']} | {r['action']}")
            lines.append(f"Close: {r['close']:.2f} | RSI: {r['rsi']:.1f}")
            lines.append(f"BUY LMT: {o['buy_limit']:.2f} | SELL/TRIM LMT: {o['sell_limit']:.2f}")
            lines.append(f"S: {r['sup']:.2f} ({r['dist_to_sup_pct']:.1f}%)  |  R: {r['res']:.2f} ({r['dist_to_res_pct']:.1f}%)")
            lines.append(f"From 52W High: {r['pct_from_52w']:.1f}% (52W: {r['high_52w']:.2f})")
            lines.append(f"Setup: {o['setup']}")
            if o["why"]:
                lines.append(f"Why: {o['why']}")
            lines.append("")
    else:
        lines.append("No results produced.\n")

    if actionable_symbols:
        lines.append("-" * 50)
        lines.append("📌 Orders suggested for: " + ", ".join(actionable_symbols))

    tg_send_message("\n".join(lines))

    # ===== CHARTS only when orders are to be placed =====
    for r in results:
        o = r["orders"]
        if not (o["place_buy"] or o["place_sell"]):
            continue

        symbol = r["symbol"]
        chart_path = f"charts/{symbol}_ORDER_4H.png"

        make_order_chart(
            symbol=symbol,
            df_4h=r["df_4h"],
            bb_mid=r["bb_mid"],
            bb_upper=r["bb_upper"],
            bb_lower=r["bb_lower"],
            buy_limit=o["buy_limit"],
            sell_limit=o["sell_limit"],
            out_path=chart_path
        )

        caption = (
            f"🔷 {symbol} — ORDER SETTER (4H)\n\n"
            f"Action: {r['action']}\n"
            f"Close: {r['close']:.2f} | RSI: {r['rsi']:.1f}\n\n"
            f"BUY LIMIT: {o['buy_limit']:.2f}\n"
            f"SELL/TRIM LIMIT: {o['sell_limit']:.2f}\n\n"
            f"S: {r['sup']:.2f} ({r['dist_to_sup_pct']:.1f}%) | R: {r['res']:.2f} ({r['dist_to_res_pct']:.1f}%)\n"
            f"From 52W High: {r['pct_from_52w']:.1f}% (52W: {r['high_52w']:.2f})\n\n"
            f"Setup: {o['setup']}\n"
            f"Why: {o['why']}"
        )

        tg_send_photo(chart_path, caption)
        time.sleep(SLEEP_BETWEEN_OTHER_CALLS)

    if failures:
        tg_send_message("⚠️ Intraday issues:\n" + "\n".join(failures))

    with open("logs/intraday_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        if failures:
            f.write("\nFailures:\n" + "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
