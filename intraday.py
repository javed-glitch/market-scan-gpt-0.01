import os
import json
import time
import re
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
from openai import OpenAI

LOG_DIR = "logs"

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


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


def infer_levels(df: pd.DataFrame, bars: int = 120):
    t = df.tail(bars)
    support = float(t["l"].min())
    resistance = float(t["h"].max())
    mid = float((support + resistance) / 2.0)
    # return 2 supports + 1 resistance hint (GPT will refine)
    return [round(support, 4), round(mid, 4)], [round(resistance, 4)]


# -----------------------------
# TWELVE DATA FETCH
# -----------------------------
def fetch_twelvedata_1h(symbol: str, outputsize: int = 500) -> pd.DataFrame:
    """
    Fetch 1-hour candles from Twelve Data.
    Returns dataframe with columns: t,o,h,l,c,v (UTC index in 't').
    """
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": str(outputsize),
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Twelve Data HTTP {r.status_code}: {r.text[:250]}")

    data = r.json()

    # Twelve Data errors are often in JSON with "status":"error"
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message')}")

    values = data.get("values")
    if not values or not isinstance(values, list):
        raise RuntimeError(f"Twelve Data: missing values for {symbol}. Response keys: {list(data.keys())}")

    # values newest-first; we sort oldest-first
    df = pd.DataFrame(values)

    # typical keys: datetime, open, high, low, close, volume
    df = df.rename(columns={
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
        else:
            df[col] = np.nan

    df = df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t")
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("t").sort_index()
    out = d.resample(rule).agg({
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

    tail = df.tail(20).copy()
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
# GPT
# -----------------------------
SYSTEM_PROMPT = (
    "You are a trading analysis assistant. "
    "You only use the provided JSON. "
    "Return ONLY valid JSON. No markdown. No extra text."
)

USER_PROMPT_TEMPLATE = """Analyze this intraday snapshot (2H + 4H).
Goal: early signals, support/resistance, and a clear next action.

Rules:
- Buy-bias when RSI(14) < 28
- Sell-bias when RSI(14) > 72
- Otherwise neutral
Use MACD(12,26,9) to confirm/deny.

Output ONLY JSON in this schema:
{
  "run_id": string,
  "timestamp_utc": string,
  "symbol": string,
  "signals": [
    {
      "timeframe": "2H"|"4H",
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

Snapshot JSON:
{SNAPSHOT_JSON}
"""


def call_gpt(snapshot: dict) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = USER_PROMPT_TEMPLATE.replace(
        "{SNAPSHOT_JSON}",
        json.dumps(snapshot, ensure_ascii=False, default=str)
    )
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.output_text or "").strip()


def extract_json_object(text: str) -> dict:
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response", text or "", 0)

    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end+1])

    raise json.JSONDecodeError("No JSON object found", s, 0)


# -----------------------------
# TELEGRAM
# -----------------------------
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    MAX = 3500
    chunks = [text[i:i+MAX] for i in range(0, len(text), MAX)]

    for idx, c in enumerate(chunks, start=1):
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": c,
                "disable_web_page_preview": True
            },
            timeout=20
        )
        if r.status_code != 200:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


# -----------------------------
# SUMMARY
# -----------------------------
def fmt_levels(vals):
    if not vals:
        return "-"
    return ", ".join(str(x) for x in vals[:3])


def build_human_summary(run_ts: str, results: list[dict], failures: list[str]) -> str:
    lines = []
    lines.append(f"[INTRADAY] Market Scan — {run_ts} UTC (2H/4H)")
    lines.append("=" * 60)

    if failures:
        lines.append("Failures:")
        for f in failures:
