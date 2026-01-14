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

# Charts/details are sent only if Bias is Buy/Sell AND confidence >= this threshold
ACTION_CONFIDENCE_MIN = int(os.getenv("ACTION_CONFIDENCE_MIN", "70"))

BASE_URL = "https://api.twelvedata.com/time_series"

# 1H lookback for building 4H
LOOKBACK_1H_BARS = int(os.getenv("LOOKBACK_1H_BARS", "360"))  # 360h ~ 90x 4H bars
# Daily lookback for 52W high
LOOKBACK_1D_BARS = int(os.getenv("LOOKBACK_1D_BARS", "320"))  # >= 252 needed

CHART_BARS = int(os.getenv("CHART_BARS", "120"))

# FREE TIER RATE LIMITING
TD_MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("TD_MIN_SECONDS_BETWEEN_CALLS", "9.0"))

# Small pauses for GPT/Telegram (not Twelve Data credits)
SLEEP_BETWEEN_OTHER_CALLS = float(os.getenv("SLEEP_BETWEEN_OTHER_CALLS", "0.3"))

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# RATE LIMITER (Twelve Data only)
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

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


# =========================
# INVESTOR ACTION (confidence nuance)
# =========================
def investor_action(trading_bias: str, confidence: int) -> str:
    b = (trading_bias or "Neutral").strip().lower()
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
            "chat_id": TELELEGRAM_CHAT_ID if False else TELEGRAM_CHAT_ID,  # keep stable var
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

    # If we get a credit/rate-limit message, wait and retry once
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
    for col in ["o", "h", "l", "c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t").set_index("t")

    _cache[key] = df
    return df

def fetch_1h(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1h", LOOKBACK_1H_BARS)

def fetch_1d(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1day", LOOKBACK_1D_BARS)

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Use lowercase to avoid FutureWarning ('H' deprecated)
    return df.resample(rule).agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
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

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def support_resistance(df: pd.DataFrame, window: int = 60):
    sup = float(df["l"].tail(window).min())
    res = float(df["h"].tail(window).max())
    return sup, res

def macd_readable(macd_line: pd.Series, sig_line: pd.Series, hist: pd.Series) -> str:
    m0, s0 = float(macd_line.iloc[-1]), float(sig_line.iloc[-1])
    h0 = float(hist.iloc[-1])
    h1 = float(hist.iloc[-2]) if len(hist) >= 2 else h0
    direction = "Bullish" if m0 > s0 else "Bearish" if m0 < s0 else "Neutral"
    slope = "rising" if h0 > h1 else "falling" if h0 < h1 else "flat"
    return f"{direction} (histogram {slope})"


# =========================
# 52W HIGH
# =========================
def compute_52w(daily_df: pd.DataFrame, last_close: float):
    d = daily_df.tail(252)
    high_52w = float(d["h"].max())
    pct_from = ((last_close - high_52w) / high_52w) * 100.0
    lvl_20 = high_52w * 0.80
    lvl_30 = high_52w * 0.70
    lvl_40 = high_52w * 0.60
    return high_52w, pct_from, lvl_20, lvl_30, lvl_40


# =========================
# CHARTS
# =========================
def make_chart(symbol: str, tf: str, df_4h: pd.DataFrame, sup: float, res: float, out_path: str):
    d = df_4h.tail(CHART_BARS).copy()
    close = d["c"]

    r = rsi_wilder(close, 14).bfill()
    macd_line, sig_line, hist = macd(close, 12, 26, 9)
    macd_line = macd_line.bfill()
    sig_line = sig_line.bfill()
    hist = hist.fillna(0)

    x = d.index

    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, close)
    ax1.axhline(sup, linestyle="--")
    ax1.axhline(res, linestyle="--")
    ax1.set_title(f"{symbol} — {tf} | Close + S/R")
    ax1.grid(True, alpha=0.25)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(x, r)
    ax2.axhline(70, linestyle="--")
    ax2.axhline(30, linestyle="--")
    ax2.set_title("RSI(14) — Wilder (TradingView-style)")
    ax2.grid(True, alpha=0.25)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(x, macd_line, label="MACD")
    ax3.plot(x, sig_line, label="Signal")
    ax3.bar(x, hist)
    ax3.set_title("MACD(12,26,9)")
    ax3.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# GPT JSON PARSING (ROBUST)
# =========================
def parse_gpt_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty GPT response")
    t = text.strip()

    # Try direct load
    try:
        return json.loads(t)
    except Exception:
        pass

    # Extract first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end+1]
        return json.loads(candidate)

    raise ValueError("No JSON object found in GPT response")


# =========================
# GPT ANALYSIS
# =========================
def gpt_analyze(symbol: str, tf: str, close: float, rsi_val: float, macd_text: str,
               sup: float, res: float,
               high_52w: float, pct_from_52w: float, lvl_20: float, lvl_30: float, lvl_40: float) -> dict:
    prompt = f"""
You are a trading assistant. Be concise and structured.

Symbol: {symbol}
Timeframe: {tf}
Close: {close:.2f}
RSI(14): {rsi_val:.1f}
MACD: {macd_text}
Support: {sup:.2f}
Resistance: {res:.2f}

52W High: {high_52w:.2f}
% From 52W High: {pct_from_52w:.1f}%
Pullback levels from 52W High:
-20%: {lvl_20:.2f}
-30%: {lvl_30:.2f}
-40%: {lvl_40:.2f}

Return JSON with exactly these keys:
{{
  "bias": "Buy"|"Sell"|"Neutral",
  "entry_zone": "text",
  "invalidation": "text",
  "confidence": 0-100,
  "setup_tag": "text",
  "definition": "text",
  "why": "text"
}}
ONLY output valid JSON. No markdown. No commentary.
"""

    # Try up to 2 times to get valid JSON
    last_err = None
    for _ in range(2):
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            return parse_gpt_json(text)
        except Exception as e:
            last_err = e
            time.sleep(0.3)

    raise RuntimeError(f"GPT JSON parse failed: {repr(last_err)}")


# =========================
# MAIN
# =========================
def main():
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    ensure_dirs()
    ts_str = utc_now_str()
    tf = "4H"

    results = []
    actionable = []
    failures = []

    for symbol in SYMBOLS:
        try:
            df_1h = fetch_1h(symbol)
            df_4h = resample_ohlc(df_1h, "4h")

            if len(df_4h) < 60:
                raise RuntimeError(f"{symbol} 4H: not enough bars ({len(df_4h)})")

            close = float(df_4h["c"].iloc[-1])
            rsi_val = float(rsi_wilder(df_4h["c"], 14).iloc[-1])

            macd_line, sig_line, hist = macd(df_4h["c"], 12, 26, 9)
            macd_text = macd_readable(macd_line, sig_line, hist)

            sup, res = support_resistance(df_4h)

            # % distance to S/R (relative to close)
            dist_to_sup_pct = ((sup - close) / close) * 100.0
            dist_to_res_pct = ((res - close) / close) * 100.0

            df_1d = fetch_1d(symbol)
            high_52w, pct_from, lvl_20, lvl_30, lvl_40 = compute_52w(df_1d, close)

            time.sleep(SLEEP_BETWEEN_OTHER_CALLS)
            g = gpt_analyze(symbol, tf, close, rsi_val, macd_text, sup, res,
                            high_52w, pct_from, lvl_20, lvl_30, lvl_40)

            trading_bias = g.get("bias", "Neutral")
            conf = safe_int(g.get("confidence", 0), 0)
            action = investor_action(trading_bias, conf)

            results.append({
                "symbol": symbol,
                "df_4h": df_4h,  # cached for charting
                "close": close,
                "rsi": rsi_val,
                "macd_text": macd_text,
                "sup": sup,
                "res": res,
                "dist_to_sup_pct": dist_to_sup_pct,
                "dist_to_res_pct": dist_to_res_pct,
                "high_52w": high_52w,
                "pct_from": pct_from,
                "lvl_20": lvl_20,
                "lvl_30": lvl_30,
                "lvl_40": lvl_40,
                "g": g,
                "trading_bias": trading_bias,
                "confidence": conf,
                "investor_action": action
            })

            # Actionable entries = Buy/Sell + >= threshold
            if (trading_bias in ("Buy", "Sell")) and (conf >= ACTION_CONFIDENCE_MIN):
                actionable.append(symbol)

        except Exception as e:
            failures.append(f"{symbol}: {repr(e)}")

    # SUMMARY MESSAGE
    lines = [
        "📊 INTRADAY SCAN (4H)",
        ts_str,
        "=" * 50,
        ""
    ]

    if results:
        for r in results:
            g = r["g"]
            tag = g.get("setup_tag", "")

            lines.append(f"🔷 {r['symbol']} | {r['investor_action']} | {r['confidence']}%")
            lines.append(f"Close: {r['close']:.2f}")
            lines.append(f"S: {r['sup']:.2f}  |  R: {r['res']:.2f}")
            lines.append(f"Δ to S: {r['dist_to_sup_pct']:.1f}%  |  Δ to R: {r['dist_to_res_pct']:.1f}%")
            lines.append(f"From 52W High: {r['pct_from']:.1f}%")
            lines.append(f"Setup: {tag}")
            lines.append("")
    else:
        lines.append("No results produced.")
        lines.append("")

    if actionable:
        lines.append("-" * 50)
        lines.append("📌 Actionable entries detected for: " + ", ".join(actionable))

    tg_send_message("\n".join(lines))

    # CHART + DETAIL only for actionable entries
    for r in results:
        g = r["g"]
        trading_bias = r["trading_bias"]
        conf = r["confidence"]

        if trading_bias not in ("Buy", "Sell"):
            continue
        if conf < ACTION_CONFIDENCE_MIN:
            continue

        symbol = r["symbol"]
        chart_path = f"charts/{symbol}_4H.png"
        make_chart(symbol, "4H", r["df_4h"], r["sup"], r["res"], chart_path)

        caption = (
            f"🔷 {symbol} (4H)\n\n"
            f"Close: {r['close']:.2f}\n"
            f"RSI: {r['rsi']:.1f}\n"
            f"MACD: {r['macd_text']}\n"
            f"Support: {r['sup']:.2f}\n"
            f"Resistance: {r['res']:.2f}\n"
            f"Δ to S: {r['dist_to_sup_pct']:.1f}% | Δ to R: {r['dist_to_res_pct']:.1f}%\n\n"
            f"52W High: {r['high_52w']:.2f}\n"
            f"From 52W High: {r['pct_from']:.1f}%\n"
            f"-20%: {r['lvl_20']:.2f} | -30%: {r['lvl_30']:.2f} | -40%: {r['lvl_40']:.2f}\n\n"
            f"Trading Bias: {trading_bias}\n"
            f"Investor Action: {r['investor_action']}\n"
            f"Confidence: {conf}%\n\n"
            f"Entry Zone: {g.get('entry_zone','')}\n"
            f"Invalidation: {g.get('invalidation','')}\n\n"
            f"Setup Tag: {g.get('setup_tag','')}\n"
            f"Definition: {g.get('definition','')}\n"
            f"Why: {g.get('why','')}"
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
