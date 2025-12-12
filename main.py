import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from openai import OpenAI

# -----------------------------
# CONFIG (via env vars)
# -----------------------------
PROVIDER = os.getenv("PROVIDER", "finnhub").strip().lower()  # finnhub | alphavantage
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "15m").strip().lower()   # 1m,5m,15m,30m,60m,1h,1d
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "200"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

# -----------------------------
# INDICATORS (local compute)
# -----------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1])

def infer_levels(df: pd.DataFrame):
    # simple swing-based levels from last ~50 bars
    tail = df.tail(50)
    support = float(tail["l"].min())
    resistance = float(tail["h"].max())
    # Add a mid level (optional)
    mid = float((support + resistance) / 2.0)
    return [round(support, 4), round(mid, 4)], [round(resistance, 4)]

# -----------------------------
# DATA FETCH
# -----------------------------
def timeframe_to_finnhub_resolution(tf: str) -> str:
    # Finnhub uses: 1,5,15,30,60,D,W,M
    if tf.endswith("m"):
        return tf.replace("m", "")
    if tf.endswith("h"):
        return str(int(tf.replace("h", "")) * 60)
    if tf in ("1d", "d"):
        return "D"
    raise ValueError(f"Unsupported TIMEFRAME for Finnhub: {tf}")

def timeframe_to_av_interval(tf: str) -> str:
    # Alpha Vantage intraday intervals: 1min, 5min, 15min, 30min, 60min
    if tf.endswith("m"):
        n = int(tf.replace("m", ""))
        if n in (1, 5, 15, 30, 60):
            return f"{n}min"
    if tf.endswith("h"):
        n = int(tf.replace("h", ""))
        if n == 1:
            return "60min"
    raise ValueError(f"Unsupported TIMEFRAME for Alpha Vantage intraday: {tf}")

def fetch_finnhub_candles(symbol: str, tf: str, lookback_bars: int) -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        raise RuntimeError("Missing FINNHUB_API_KEY")
    resolution = timeframe_to_finnhub_resolution(tf)

    # compute from/to in unix seconds with padding
    # rough seconds per bar:
    sec_per_bar = {
        "1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600,
        "D": 86400
    }.get(resolution, 900)

    now = int(time.time())
    frm = now - (lookback_bars * sec_per_bar) - (10 * sec_per_bar)

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": frm,
        "to": now,
        "token": FINNHUB_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub candle fetch failed for {symbol}: {data}")

    df = pd.DataFrame({
        "t": data["t"],
        "o": data["o"],
        "h": data["h"],
        "l": data["l"],
        "c": data["c"],
        "v": data.get("v", [0]*len(data["t"]))
    })
    # Finnhub timestamps are seconds
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df

def fetch_alphavantage_intraday(symbol: str, tf: str) -> pd.DataFrame:
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("Missing ALPHAVANTAGE_API_KEY")
    interval = timeframe_to_av_interval(tf)

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Alpha Vantage returns a key like: "Time Series (15min)"
    series_key = None
    for k in data.keys():
        if "Time Series" in k:
            series_key = k
            break
    if not series_key:
        raise RuntimeError(f"Alpha Vantage response missing time series for {symbol}: {data}")

    rows = []
    for ts, v in data[series_key].items():
        rows.append({
            "t": pd.to_datetime(ts, utc=True),
            "o": float(v["1. open"]),
            "h": float(v["2. high"]),
            "l": float(v["3. low"]),
            "c": float(v["4. close"]),
            "v": float(v["5. volume"]),
        })
    df = pd.DataFrame(rows).sort_values("t")
    return df

# -----------------------------
# GPT CALL (Responses API)
# -----------------------------
SYSTEM_PROMPT = (
    "You are a trading analysis assistant. You do NOT browse the web. "
    "You ONLY use the JSON provided by the user. "
    "Be concise, deterministic, and practical. "
    "Return ONLY valid JSON matching the requested schema. "
    "No financial advice disclaimers. No hype. No emojis."
)

USER_PROMPT_TEMPLATE = """You will be given a market snapshot with RSI(14) and MACD(12,26,9) plus recent candles.
My rule: buy-bias when RSI < 28; sell-bias when RSI > 72; otherwise neutral.
Use MACD to confirm/deny the bias using:
- histogram direction (rising/falling)
- MACD vs signal line relationship

Return ONLY JSON in this schema:
{
  "run_id": string,
  "timestamp_utc": string,
  "symbol": string,
  "timeframe": string,
  "price": number,
  "rsi14": number,
  "macd": { "macd": number, "signal": number, "hist": number },
  "setup": "buy" | "sell" | "neutral",
  "confidence_0_100": number,
  "why": [string, string, string],
  "levels": { "support": [number], "resistance": [number] },
  "invalidation": string,
  "next_action": string
}

Market snapshot JSON:
{SNAPSHOT_JSON}
"""

def call_gpt(snapshot: dict) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    user_prompt = USER_PROMPT_TEMPLATE.replace("{SNAPSHOT_JSON}", json.dumps(snapshot, ensure_ascii=False))
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text.strip()

# -----------------------------
# NOTIFY (optional)
# -----------------------------
def post_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    # Discord expects {"content": "..."}
    r = requests.post(DISCORD_WEBHOOK_URL, json={"content": message[:1900]}, timeout=30)
    r.raise_for_status()

# -----------------------------
# MAIN
# -----------------------------
def run_for_symbol(symbol: str):
    if PROVIDER == "finnhub":
        df = fetch_finnhub_candles(symbol, TIMEFRAME, LOOKBACK_BARS)
    elif PROVIDER == "alphavantage":
        df = fetch_alphavantage_intraday(symbol, TIMEFRAME).tail(LOOKBACK_BARS)
    else:
        raise ValueError("PROVIDER must be finnhub or alphavantage")

    if len(df) < 50:
        raise RuntimeError(f"Not enough candles for {symbol}: got {len(df)}")

    close = df["c"].astype(float)
    rsi14 = compute_rsi(close, 14)
    macd, signal, hist = compute_macd(close, 12, 26, 9)

    supports, resistances = infer_levels(df)

    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    price = float(close.iloc[-1])

    snapshot = {
        "run_id": f"{now_utc}_{symbol}_{TIMEFRAME}",
        "timestamp_utc": now_utc,
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "price": price,
        "rsi14": round(rsi14, 4),
        "macd": {"macd": round(macd, 6), "signal": round(signal, 6), "hist": round(hist, 6)},
        "levels_hint": {"support": supports, "resistance": resistances},
        "candles_tail": df.tail(15).to_dict(orient="records"),
    }

    gpt_json = call_gpt(snapshot)

    # Print for GitHub Actions logs
    print("\n========== GPT OUTPUT ==========")
    print(gpt_json)
    print("================================\n")

    # Optional webhook
    post_discord(f"**{symbol} {TIMEFRAME}**\n```json\n{gpt_json}\n```")

def main():
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        raise RuntimeError("Missing OPENAI_API_KEY")

    errors = []
    for sym in SYMBOLS:
        try:
            run_for_symbol(sym)
        except Exception as e:
            errors.append(f"{sym}: {repr(e)}")

    if errors:
        raise RuntimeError("Errors:\n" + "\n".join(errors))

if __name__ == "__main__":
    main()
