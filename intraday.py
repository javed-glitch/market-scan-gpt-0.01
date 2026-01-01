import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # headless for GitHub Actions
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

BASE_URL = "https://api.twelvedata.com/time_series"
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "260"))  # enough for RSI/MACD stability
CHART_BARS = int(os.getenv("CHART_BARS", "120"))        # how many bars to draw
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.9"))

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# HELPERS
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
            "caption": caption[:900],  # safe under Telegram caption limits
            "disable_web_page_preview": True
        }
        r = requests.post(url, data=data, files=files, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram sendPhoto error {r.status_code}: {r.text}")


# =========================
# TWELVE DATA
# =========================
def fetch_1h(symbol: str) -> pd.DataFrame:
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": str(LOOKBACK_BARS),
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
        raise RuntimeError(f"TwelveData: no values for {symbol}. Response keys: {list(data.keys())}")

    df = pd.DataFrame(values)
    df = df.rename(columns={
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

    df = df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t")
    return df.set_index("t")


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
    }).dropna()
    return out


# =========================
# INDICATORS
# =========================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
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


# =========================
# CHARTS
# =========================
def make_chart(symbol: str, tf: str, df: pd.DataFrame, sup: float, res: float, out_path: str):
    d = df.tail(CHART_BARS).copy()
    close = d["c"]

    r = rsi(close, 14)
    macd_line, sig_line, hist = macd(close, 12, 26, 9)

    # Fill to avoid gaps in plotting (first values may be NaN)
    r = r.bfill()
    macd_line = macd_line.bfill()
    sig_line = sig_line.bfill()
    hist = hist.fillna(0)

    x = d.index

    plt.figure(figsize=(10, 8))

    # Price
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, close)
    ax1.axhline(sup, linestyle="--")
    ax1.axhline(res, linestyle="--")
    ax1.set_title(f"{symbol} — {tf} | Close + S/R")
    ax1.grid(True, alpha=0.25)

    # RSI
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(x, r)
    ax2.axhline(70, linestyle="--")
    ax2.axhline(30, linestyle="--")
    ax2.set_title("RSI(14)")
    ax2.grid(True, alpha=0.25)

    # MACD
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
# GPT ANALYSIS (short + actionable)
# =========================
def gpt_analyze(symbol: str, tf: str, close: float, rsi_val: float, sup: float, res: float, macd_val: float, sig_val: float, hist_val: float) -> str:
    prompt = f"""
You are a trading assistant. Be concise.

Symbol: {symbol}
Timeframe: {tf}
Close: {close:.2f}
RSI(14): {rsi_val:.1f}
Support: {sup:.2f}
Resistance: {res:.2f}
MACD: {macd_val:.4f}
Signal: {sig_val:.4f}
Hist: {hist_val:.4f}

Return 4 bullet lines exactly:
- Bias: Buy/Sell/Neutral
- Entry: <zone>
- Invalidation: <level>
- Confidence: <0-100 integer>
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()


# =========================
# MAIN
# =========================
def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    ensure_dirs()
    ts = utc_now_iso()

    # Header message
    tg_send_message(f"📊 INTRADAY SCAN (2H / 4H)\n{ts} UTC\nSymbols: {', '.join(SYMBOLS)}")

    failures = []
    # If you want fewer images, set this to just ["2H"] later.
    timeframes = ["2H", "4H"]

    for symbol in SYMBOLS:
        try:
            time.sleep(SLEEP_BETWEEN_CALLS)
            df_1h = fetch_1h(symbol)

            # Build TF frames
            df_2h = resample_ohlc(df_1h, "2H")
            df_4h = resample_ohlc(df_1h, "4H")
            tf_map = {"2H": df_2h, "4H": df_4h}

            for tf in timeframes:
                df = tf_map[tf]
                if len(df) < 80:
                    raise RuntimeError(f"{symbol} {tf}: not enough bars ({len(df)})")

                sup, res = support_resistance(df)
                close = float(df["c"].iloc[-1])

                rsi_series = rsi(df["c"], 14)
                rsi_val = float(rsi_series.iloc[-1])

                macd_line, sig_line, hist = macd(df["c"], 12, 26, 9)
                macd_val = float(macd_line.iloc[-1])
                sig_val = float(sig_line.iloc[-1])
                hist_val = float(hist.iloc[-1])

                analysis = gpt_analyze(symbol, tf, close, rsi_val, sup, res, macd_val, sig_val, hist_val)

                chart_path = f"charts/{symbol}_{tf}.png"
                make_chart(symbol, tf, df, sup, res, chart_path)

                caption = (
                    f"{symbol} ({tf})\n"
                    f"Close: {close:.2f} | RSI: {rsi_val:.1f}\n"
                    f"S: {sup:.2f}  R: {res:.2f}\n"
                    f"{analysis}"
                )

                tg_send_photo(chart_path, caption)
                time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            failures.append(f"{symbol}: {repr(e)}")

    # Save a simple log too (optional but useful)
    with open("logs/intraday_last_run.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {ts} UTC\n")
        if failures:
            f.write("Failures:\n" + "\n".join(failures) + "\n")
        else:
            f.write("OK\n")

    if failures:
        tg_send_message("⚠️ Intraday failures:\n" + "\n".join(failures))
        raise RuntimeError("Errors:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
