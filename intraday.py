import os
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI

# =========================
# CONFIG
# =========================
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]
TD_API_KEY = os.environ["TWELVEDATA_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

BASE_URL = "https://api.twelvedata.com/time_series"
LOOKBACK_BARS = 200

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =========================
# UTILITIES
# =========================
def fetch_1h_data(symbol: str) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": LOOKBACK_BARS,
        "apikey": TD_API_KEY,
        "format": "JSON"
    }
    r = requests.get(BASE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df.set_index("datetime")


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def support_resistance(df: pd.DataFrame):
    low = df["low"].tail(50).min()
    high = df["high"].tail(50).max()
    return float(low), float(high)


def analyze_with_gpt(symbol, tf, close, rsi_val, support, resistance):
    prompt = f"""
You are a professional intraday trader.

Symbol: {symbol}
Timeframe: {tf}
Last close: {close:.2f}
RSI: {rsi_val:.1f}
Support: {support:.2f}
Resistance: {resistance:.2f}

Provide:
- Bias (Buy / Sell / Neutral)
- Entry zone
- Invalidation level
- Confidence (0–100)
Keep concise.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    r = requests.post(url, data=payload, timeout=20)
    r.raise_for_status()


# =========================
# MAIN
# =========================
def main():
    now = datetime.now(timezone.utc).isoformat()
    header = f"📊 INTRADAY SCAN (2H / 4H)\n{now}\n\n"
    messages = []
    failures = []

    for symbol in SYMBOLS:
        try:
            df_1h = fetch_1h_data(symbol)

            df_2h = resample(df_1h, "2H")
            df_4h = resample(df_1h, "4H")

            for label, df in [("2H", df_2h), ("4H", df_4h)]:
                close = df["close"].iloc[-1]
                rsi_val = rsi(df["close"])
                support, resistance = support_resistance(df)

                analysis = analyze_with_gpt(
                    symbol, label, close, rsi_val, support, resistance
                )

                messages.append(
                    f"🔹 {symbol} ({label})\n"
                    f"Close: {close:.2f}\n"
                    f"RSI: {rsi_val:.1f}\n"
                    f"Support: {support:.2f}\n"
                    f"Resistance: {resistance:.2f}\n"
                    f"{analysis}\n"
                )

                time.sleep(1)  # be kind to APIs

        except Exception as e:
            failures.append(f"{symbol}: {repr(e)}")

    if not messages:
        raise RuntimeError("No intraday results produced")

    final_msg = header + "\n".join(messages)
    send_telegram(final_msg)

    if failures:
        fail_msg = "⚠️ Failures:\n" + "\n".join(failures)
        send_telegram(fail_msg)


if __name__ == "__main__":
    main()
