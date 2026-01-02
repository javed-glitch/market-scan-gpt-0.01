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
ETF_SYMBOLS = {s.strip().upper() for s in os.getenv("ETF_SYMBOLS", "QQQ").split(",") if s.strip()}

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ACTION_CONFIDENCE_MIN = int(os.getenv("ACTION_CONFIDENCE_MIN", "70"))

BASE_URL = "https://api.twelvedata.com/time_series"

# 1H lookback for building 4H (~90x 4H bars)
LOOKBACK_1H_BARS = int(os.getenv("LOOKBACK_1H_BARS", "360"))
# Daily lookback for 52W high
LOOKBACK_1D_BARS = int(os.getenv("LOOKBACK_1D_BARS", "320"))

CHART_BARS = int(os.getenv("CHART_BARS", "120"))
SLEEP = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.8"))

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# UTIL
# =========================
def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)

def utc_now_iso():
    return utc_now().isoformat()

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
    if r.status_code != 200:
        raise RuntimeError(f"Telegram sendMessage error {r.status_code}: {r.text}")

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
        if r.status_code != 200:
            raise RuntimeError(f"Telegram sendPhoto error {r.status_code}: {r.text}")


# =========================
# TWELVE DATA FETCH
# =========================
def fetch_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
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
    return df

def fetch_1h(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1h", LOOKBACK_1H_BARS)

def fetch_1d(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1day", LOOKBACK_1D_BARS)

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
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
def make_chart(symbol: str, tf: str, df: pd.DataFrame, sup: float, res: float, out_path: str):
    d = df.tail(CHART_BARS).copy()
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
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    text = resp.choices[0].message.content.strip()

    # Extract JSON safely if wrapped
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    return json.loads(text)


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
    ts = utc_now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M UTC")

    tf = "4H"
    results = []
    actionable = []
    failures = []

    for symbol in SYMBOLS:
        try:
            time.sleep(SLEEP)
            df_1h = fetch_1h(symbol)
            df_4h = resample_ohlc(df_1h, "4H")

            if len(df_4h) < 60:
                raise RuntimeError(f"{symbol} 4H: not enough bars ({len(df_4h)})")

            close = float(df_4h["c"].iloc[-1])
            rsi_val = float(rsi_wilder(df_4h["c"], 14).iloc[-1])

            macd_line, sig_line, hist = macd(df_4h["c"], 12, 26, 9)
            macd_text = macd_readable(macd_line, sig_line, hist)

            sup, res = support_resistance(df_4h)

            time.sleep(SLEEP)
            df_1d = fetch_1d(symbol)
            high_52w, pct_from, lvl_20, lvl_30, lvl_40 = compute_52w(df_1d, close)

            time.sleep(SLEEP)
            g = gpt_analyze(symbol, tf, close, rsi_val, macd_text, sup, res,
                            high_52w, pct_from, lvl_20, lvl_30, lvl_40)

            bias = g.get("bias", "Neutral")
            conf = int(g.get("confidence", 0))

            results.append({
                "symbol": symbol,
                "close": close,
                "rsi": rsi_val,
                "macd_text": macd_text,
                "sup": sup,
                "res": res,
                "high_52w": high_52w,
                "pct_from": pct_from,
                "lvl_20": lvl_20,
                "lvl_30": lvl_30,
                "lvl_40": lvl_40,
                "g": g
            })

            # Chart + detailed message only if Buy/Sell entry is advised
            if bias in ("Buy", "Sell"):
                actionable.append(symbol)

        except Exception as e:
            failures.append(f"{symbol}: {repr(e)}")

    # 1) ONE SUMMARY MESSAGE
    lines = [
        f"📊 INTRADAY SCAN (4H)",
        f"{ts_str}",
        "=" * 50,
        ""
    ]

    if results:
        for r in results:
            g = r["g"]
            bias = g.get("bias", "Neutral")
            conf = int(g.get("confidence", 0))
            tag = g.get("setup_tag", "")
            lines.append(
                f"🔷 {r['symbol']} | {bias.upper()} | {conf}%"
            )
            lines.append(
                f"Setup: {tag}"
            )
            lines.append(
                f"From 52W High: {r['pct_from']:.1f}%"
            )
            lines.append("")  # spacer
    else:
        lines.append("No results produced.")
        lines.append("")

    if actionable:
        lines.append("-" * 50)
        lines.append("📌 Actionable entries detected for: " + ", ".join(actionable))

    tg_send_message("\n".join(lines))

    # 2) CHART + DETAILED MESSAGE only if Buy/Sell
    for r in results:
        g = r["g"]
        symbol = r["symbol"]
        bias = g.get("bias", "Neutral")
        conf = int(g.get("confidence", 0))

        if bias not in ("Buy", "Sell"):
            continue

        # Optional: enforce confidence threshold for charting (keeps noise down)
        # If you want charts for ALL Buy/Sell regardless of confidence, set ACTION_CONFIDENCE_MIN to 0 in workflow.
        if conf < ACTION_CONFIDENCE_MIN:
            continue

        # Build chart from already-fetched 4H dataframe would be ideal, but we keep it simple/reliable:
        df_4h = resample_ohlc(fetch_1h(symbol), "4H")
        chart_path = f"charts/{symbol}_4H.png"
        make_chart(symbol, "4H", df_4h, r["sup"], r["res"], chart_path)

        caption = (
            f"🔷 {symbol} (4H)\n"
            f"\n"
            f"Close: {r['close']:.2f}\n"
            f"RSI: {r['rsi']:.1f}\n"
            f"MACD: {r['macd_text']}\n"
            f"Support: {r['sup']:.2f}\n"
            f"Resistance: {r['res']:.2f}\n"
            f"\n"
            f"52W High: {r['high_52w']:.2f}\n"
            f"From 52W High: {r['pct_from']:.1f}%\n"
            f"-20%: {r['lvl_20']:.2f} | -30%: {r['lvl_30']:.2f} | -40%: {r['lvl_40']:.2f}\n"
            f"\n"
            f"Bias: {g.get('bias','Neutral')}\n"
            f"Entry Zone: {g.get('entry_zone','')}\n"
            f"Invalidation: {g.get('invalidation','')}\n"
            f"Confidence: {g.get('confidence','')}%\n"
            f"\n"
            f"Setup Tag: {g.get('setup_tag','')}\n"
            f"Definition: {g.get('definition','')}\n"
            f"Why: {g.get('why','')}"
        )

        tg_send_photo(chart_path, caption)
        time.sleep(SLEEP)

    # 3) Failures (if any)
    if failures:
        tg_send_message("⚠️ Intraday issues:\n" + "\n".join(failures))

    # Save logs
    with open("logs/intraday_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        if failures:
            f.write("\nFailures:\n" + "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
