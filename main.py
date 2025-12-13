import os
import json
import time
import io
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()
LOG_DIR = "logs"

# -----------------------------
# UTIL
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# INDICATORS
# -----------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (EMA with alpha=1/period)
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

def infer_levels(df: pd.DataFrame, bars: int = 120):
    t = df.tail(bars)
    support = float(t["l"].min())
    resistance = float(t["h"].max())
    mid = float((support + resistance) / 2.0)
    # Return 2 support candidates (low + mid) and 1 resistance candidate
    return [round(support, 4), round(mid, 4)], [round(resistance, 4)]

# -----------------------------
# STOOQ FETCH (DAILY)
# -----------------------------
def stooq_symbol(symbol: str) -> str:
    # Stooq typically uses .us suffix for US stocks/ETFs
    return f"{symbol.lower()}.us"

def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    s = stooq_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(s)}&i=d"

    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    # Parse CSV safely
    df = pd.read_csv(io.StringIO(raw))

    # Expected columns: Date, Open, High, Low, Close, Volume
    if df.empty or "Date" not in df.columns:
        raise RuntimeError(f"Stooq returned no data for {symbol} ({s}). Response head:\n{raw[:200]}")

    df = df.rename(columns={
        "Date": "t",
        "Open": "o",
        "High": "h",
        "Low": "l",
        "Close": "c",
        "Volume": "v",
    })

    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t")
    return df

def resample_weekly_from_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    # Weekly bars ending Friday (common for equities)
    d = df_daily.set_index("t").sort_index()
    out = d.resample("W-FRI").agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }).dropna()
    return out.reset_index()

def build_tf_snapshot(df: pd.DataFrame, tf_name: str) -> dict:
    close = df["c"].astype(float)
    rsi = compute_rsi(close, 14)
    macd, signal, hist = compute_macd(close, 12, 26, 9)
    supports, resistances = infer_levels(df)

    tail = df.tail(15).copy()
    # IMPORTANT: make timestamps JSON-serializable
    tail["t"] = tail["t"].astype(str)

    return {
        "timeframe": tf_name,
        "price": float(close.iloc[-1]),
        "rsi14": round(rsi, 4),
        "macd": {"macd": round(macd, 6), "signal": round(signal, 6), "hist": round(hist, 6)},
        "levels_hint": {"support": supports, "resistance": resistances},
        "candles_tail": tail.to_dict(orient="records"),
    }

# -----------------------------
# GPT PROMPTS
# -----------------------------
SYSTEM_PROMPT = (
    "You are a trading analysis assistant. You do NOT browse the web. "
    "You ONLY use the JSON provided. Return ONLY valid JSON. "
    "Be concise, deterministic, and practical. No hype. No emojis."
)

USER_PROMPT_TEMPLATE = """Analyze this market snapshot for early signals (daily/weekly only).
Timeframes: 1D, 1W.

Rules:
- Buy-bias when RSI(14) < 28
- Sell-bias when RSI(14) > 72
- Otherwise neutral
Use MACD(12,26,9) to confirm/deny bias.

Support/Resistance:
Use levels_hint and candles_tail. Return 2-3 supports and 2-3 resistances as numbers.

Output ONLY JSON in this schema:
{
  "run_id": string,
  "timestamp_utc": string,
  "symbol": string,
  "signals": [
    {
      "timeframe": "1D"|"1W",
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
    prompt = USER_PROMPT_TEMPLATE.replace(
        "{SNAPSHOT_JSON}",
        json.dumps(snapshot, ensure_ascii=False, default=str)  # extra safety
    )
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text.strip()

# -----------------------------
# RUN
# -----------------------------
def run_symbol(symbol: str):
    # light pause to be polite to Stooq
    time.sleep(0.5)

    daily = fetch_stooq_daily(symbol)
    weekly = resample_weekly_from_daily(daily)

    if len(daily) < 120:
        raise RuntimeError(f"Not enough daily data for {symbol}: {len(daily)} rows")
    if len(weekly) < 60:
        raise RuntimeError(f"Not enough weekly data for {symbol}: {len(weekly)} rows")

    now = utc_now_iso()
    snapshot = {
        "run_id": f"{now}_{symbol}",
        "timestamp_utc": now,
        "symbol": symbol,
        "data": {
            "1D": build_tf_snapshot(daily.tail(260), "1D"),
            "1W": build_tf_snapshot(weekly.tail(260), "1W"),
        }
    }

    gpt_out = call_gpt(snapshot)

    # Write logs
    safe_ts = now.replace(":", "-")
    base = f"{LOG_DIR}/{safe_ts}_{symbol}"
    with open(base + "_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    with open(base + "_gpt.json", "w", encoding="utf-8") as f:
        f.write(gpt_out)

    print(f"\n===== {symbol} GPT OUTPUT =====")
    print(gpt_out)
    print("==============================\n")

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

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
