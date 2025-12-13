import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "TSLA").split(",") if s.strip()]
INTRADAY_INTERVAL = os.getenv("INTRADAY_INTERVAL", "60min").strip()  # must be 60min for 2H/4H resample
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LOG_DIR = "logs"

# -----------------------------
# INDICATORS
# -----------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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

def infer_levels(df: pd.DataFrame, bars: int = 80):
    t = df.tail(bars)
    support = float(t["l"].min())
    resistance = float(t["h"].max())
    mid = float((support + resistance) / 2.0)
    return [round(support, 4), round(mid, 4)], [round(resistance, 4)]

# -----------------------------
# ALPHA VANTAGE FETCH
# -----------------------------
def av_get(params: dict) -> dict:
    url = "https://www.alphavantage.co/query"
    params = dict(params)
    params["apikey"] = ALPHAVANTAGE_API_KEY

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Friendly errors (rate limit / premium messages)
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage: {data['Note']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage: {data['Error Message']}")

    return data

def fetch_intraday(symbol: str, interval: str = "60min") -> pd.DataFrame:
    data = av_get({
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "compact",
    })
    # Key like: "Time Series (60min)"
    series_key = None
    for k in data.keys():
        if "Time Series" in k:
            series_key = k
            break
    if not series_key:
        raise RuntimeError(f"Alpha Vantage response missing intraday series for {symbol}: {data}")

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

def fetch_daily(symbol: str) -> pd.DataFrame:
    data = av_get({
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
    })
    key = "Time Series (Daily)"
    if key not in data:
        raise RuntimeError(f"Alpha Vantage response missing daily series for {symbol}: {data}")

    rows = []
    for ts, v in data[key].items():
        rows.append({
            "t": pd.to_datetime(ts, utc=True),
            "o": float(v["1. open"]),
            "h": float(v["2. high"]),
            "l": float(v["3. low"]),
            "c": float(v["4. close"]),
            "v": float(v["6. volume"]),
        })
    return pd.DataFrame(rows).sort_values("t")

def fetch_weekly(symbol: str) -> pd.DataFrame:
    data = av_get({
        "function": "TIME_SERIES_WEEKLY_ADJUSTED",
        "symbol": symbol,
    })
    key = "Weekly Adjusted Time Series"
    if key not in data:
        raise RuntimeError(f"Alpha Vantage response missing weekly series for {symbol}: {data}")

    rows = []
    for ts, v in data[key].items():
        rows.append({
            "t": pd.to_datetime(ts, utc=True),
            "o": float(v["1. open"]),
            "h": float(v["2. high"]),
            "l": float(v["3. low"]),
            "c": float(v["4. close"]),
            "v": float(v["6. volume"]),
        })
    return pd.DataFrame(rows).sort_values("t")

# -----------------------------
# RESAMPLE (60min -> 2H/4H)
# -----------------------------
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("t").sort_index()
    agg = {
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }
    out = d.resample(rule).agg(agg).dropna()
    out = out.reset_index()
    return out

def build_tf_snapshot(df: pd.DataFrame, tf_name: str) -> dict:
    close = df["c"].astype(float)
    rsi = compute_rsi(close, 14)
    macd, signal, hist = compute_macd(close, 12, 26, 9)
    supports, resistances = infer_levels(df)

    return {
        "timeframe": tf_name,
        "price": float(close.iloc[-1]),
        "rsi14": round(rsi, 4),
        "macd": {"macd": round(macd, 6), "signal": round(signal, 6), "hist": round(hist, 6)},
        "levels_hint": {"support": supports, "resistance": resistances},
        "candles_tail": df.tail(15).to_dict(orient="records"),
    }

# -----------------------------
# GPT PROMPT (early signals + S/R + logged)
# -----------------------------
SYSTEM_PROMPT = (
    "You are a trading analysis assistant. You do NOT browse the web. "
    "You ONLY use the JSON provided. "
    "Be concise, deterministic, and practical. "
    "Return ONLY valid JSON matching the requested schema. "
    "No hype. No emojis."
)

USER_PROMPT_TEMPLATE = """Analyze this multi-timeframe market snapshot for early signals.
Timeframes: 2H, 4H, 1D, 1W.

Rules:
- Buy-bias when RSI(14) < 28
- Sell-bias when RSI(14) > 72
- Otherwise neutral
Use MACD(12,26,9) to confirm/deny bias:
- MACD vs Signal relationship
- Histogram direction (rising/falling) based on current vs prior bar if candle data suggests it

Support/Resistance:
- Use provided candles_tail and levels_hint.
- Return 2-3 support levels and 2-3 resistance levels as zone midpoints (numbers).

Output ONLY JSON in this schema:
{
  "run_id": string,
  "timestamp_utc": string,
  "symbol": string,
  "signals": [
    {
      "timeframe": "2H"|"4H"|"1D"|"1W",
      "price": number,
      "rsi14": number,
      "macd": {"macd": number, "signal": number, "hist": number},
      "setup": "buy"|"sell"|"neutral",
      "confidence_0_100": number,
      "support": [number, number, number],
      "resistance": [number, number, number],
      "why": [string, string, string],
      "invalidation": string,
      "next_action": string
    }
  ],
  "overall_bias": "buy"|"sell"|"neutral",
  "watchlist_notes": [string, string]
}

Market snapshot JSON:
{SNAPSHOT_JSON}
"""

def call_gpt(snapshot: dict) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = USER_PROMPT_TEMPLATE.replace("{SNAPSHOT_JSON}", json.dumps(snapshot, ensure_ascii=False))
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text.strip()

# -----------------------------
# MAIN
# -----------------------------
def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def run_symbol(symbol: str):
    # Alpha Vantage free tier has per-minute throttles; be polite between symbols.
    # (helps avoid "Note" responses)
    time.sleep(12)

    intraday = fetch_intraday(symbol, INTRADAY_INTERVAL)
    daily = fetch_daily(symbol)
    weekly = fetch_weekly(symbol)

    # Need enough bars for stable indicators
    if len(intraday) < 80:
        raise RuntimeError(f"Not enough intraday bars for {symbol}: {len(intraday)}")
    if len(daily) < 80:
        raise RuntimeError(f"Not enough daily bars for {symbol}: {len(daily)}")
    if len(weekly) < 40:
        raise RuntimeError(f"Not enough weekly bars for {symbol}: {len(weekly)}")

    tf_2h = resample_ohlcv(intraday, "2H")
    tf_4h = resample_ohlcv(intraday, "4H")

    # Build snapshot bundle for ONE GPT call per symbol
    now = utc_now_iso()
    snapshot = {
        "run_id": f"{now}_{symbol}",
        "timestamp_utc": now,
        "symbol": symbol,
        "data": {
            "2H": build_tf_snapshot(tf_2h.tail(220), "2H"),
            "4H": build_tf_snapshot(tf_4h.tail(220), "4H"),
            "1D": build_tf_snapshot(daily.tail(220), "1D"),
            "1W": build_tf_snapshot(weekly.tail(220), "1W"),
        }
    }

    gpt_out = call_gpt(snapshot)

    # Write logs
    safe_ts = now.replace(":", "-")
    base = f"{LOG_DIR}/{safe_ts}_{symbol}"
    with open(base + "_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with open(base + "_gpt.json", "w", encoding="utf-8") as f:
        f.write(gpt_out)

    print(f"\n===== {symbol} GPT OUTPUT =====")
    print(gpt_out)
    print("==============================\n")

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("Missing ALPHAVANTAGE_API_KEY")
    if INTRADAY_INTERVAL != "60min":
        raise RuntimeError("Set INTRADAY_INTERVAL to 60min for correct 2H/4H resampling.")

    ensure_dirs()

    errors = []
    for sym in SYMBOLS:
        try:
            run_symbol(sym)
        except Exception as e:
            errors.append(f"{sym}: {repr(e)}")

    if errors:
        raise RuntimeError("Errors:\n" + "\n".join(errors))

if __name__ == "__main__":
    main()
