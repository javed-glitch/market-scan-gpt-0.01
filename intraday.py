import os
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI

# =========================
# CONFIG
# =========================
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ACTION_CONFIDENCE_MIN = int(os.getenv("ACTION_CONFIDENCE_MIN", "70"))

BASE_URL = "https://api.twelvedata.com/time_series"

LOOKBACK_1H_BARS = 360
LOOKBACK_1D_BARS = 320
CHART_BARS = 120

TD_MIN_SECONDS_BETWEEN_CALLS = 9.0
SLEEP_BETWEEN_OTHER_CALLS = 0.3

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# RATE LIMIT
# =========================
_last_td_call = 0.0

def td_throttle():
    global _last_td_call
    wait = TD_MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_td_call)
    if wait > 0:
        time.sleep(wait)
    _last_td_call = time.time()

# =========================
# UTILS
# =========================
def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)

def utc_now_str():
    return utc_now().strftime("%Y-%m-%d %H:%M UTC")

def ensure_dirs():
    os.makedirs("charts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

# =========================
# INVESTOR ACTION
# =========================
def investor_action(bias: str, confidence: int) -> str:
    b = (bias or "Neutral").lower()
    c = int(confidence or 0)

    if b == "buy":
        return "ADD / ACCUMULATE" if c >= 70 else "WATCH / EARLY SETUP"
    if b == "sell":
        if c >= 80:
            return "TRIM / TAKE PROFITS"
        if c >= 70:
            return "HOLD / WAIT (extended)"
        return "IGNORE / NO ACTION"
    return "HOLD / WAIT"

# =========================
# TELEGRAM
# =========================
def tg_send_message(text: str):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        timeout=30,
    )
    r.raise_for_status()

def tg_send_photo(path: str, caption: str):
    with open(path, "rb") as f:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            files={"photo": f},
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:900]},
            timeout=60,
        )
        r.raise_for_status()

# =========================
# DATA FETCH
# =========================
_cache = {}

def fetch_series(symbol, interval, size):
    key = (symbol, interval)
    if key in _cache:
        return _cache[key]

    td_throttle()
    r = requests.get(
        BASE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": size,
            "apikey": TD_API_KEY,
        },
        timeout=30,
    )
    data = r.json()
    if "values" not in data:
        raise RuntimeError(data.get("message", "No data"))

    df = pd.DataFrame(data["values"]).rename(
        columns={"datetime": "t", "open": "o", "high": "h", "low": "l", "close": "c"}
    )
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df[["o", "h", "l", "c"]] = df[["o", "h", "l", "c"]].astype(float)
    df = df.sort_values("t").set_index("t")

    _cache[key] = df
    return df

def fetch_1h(symbol):
    return fetch_series(symbol, "1h", LOOKBACK_1H_BARS)

def fetch_1d(symbol):
    return fetch_series(symbol, "1day", LOOKBACK_1D_BARS)

def resample_ohlc(df, rule):
    return df.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()

# =========================
# INDICATORS
# =========================
def rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    rs = g.ewm(alpha=1/p, adjust=False).mean() / l.ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def macd(close):
    m = ema(close, 12) - ema(close, 26)
    s = ema(m, 9)
    return m, s, m - s

def support_resistance(df):
    return float(df["l"].tail(60).min()), float(df["h"].tail(60).max())

# =========================
# 52W HIGH
# =========================
def compute_52w(df, close):
    hi = float(df.tail(252)["h"].max())
    return hi, (close - hi) / hi * 100

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    ts = utc_now_str()

    results = []
    actionable = []

    for s in SYMBOLS:
        df4 = resample_ohlc(fetch_1h(s), "4H")
        close = float(df4["c"].iloc[-1])
        sup, res = support_resistance(df4)

        dist_s = (sup - close) / close * 100
        dist_r = (res - close) / close * 100

        hi52, from52 = compute_52w(fetch_1d(s), close)

        g = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": f"""
Symbol: {s}
Close: {close}
RSI: {rsi(df4['c']).iloc[-1]:.1f}
MACD computed
Support: {sup}
Resistance: {res}

Return JSON:
bias, confidence, setup_tag, entry_zone, invalidation, definition, why
"""
            }],
            temperature=0.2
        )

        j = json.loads(g.choices[0].message.content)
        bias = j.get("bias", "Neutral")
        conf = int(j.get("confidence", 0))
        action = investor_action(bias, conf)

        results.append((s, close, sup, res, dist_s, dist_r, from52, action, conf, j))

        if bias in ("Buy", "Sell") and conf >= ACTION_CONFIDENCE_MIN:
            actionable.append(s)

    # SUMMARY
    lines = [
        "📊 INTRADAY SCAN (4H)",
        ts,
        "=" * 50,
        ""
    ]

    for s, close, sup, res, ds, dr, f52, action, conf, j in results:
        lines += [
            f"🔷 {s} | {action} | {conf}%",
            f"Close: {close:.2f}",
            f"S: {sup:.2f}  |  R: {res:.2f}",
            f"Δ to S: {ds:.1f}%  |  Δ to R: {dr:.1f}%",
            f"From 52W High: {f52:.1f}%",
            f"Setup: {j.get('setup_tag','')}",
            ""
        ]

    tg_send_message("\n".join(lines))

if __name__ == "__main__":
    main()
