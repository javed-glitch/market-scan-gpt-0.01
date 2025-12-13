import os, json, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from openai import OpenAI

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()
LOG_DIR = "logs"

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

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("t").sort_index()
    out = d.resample(rule).agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"}).dropna()
    return out.reset_index()

def finnhub_candles(symbol: str, resolution: str, bars: int) -> pd.DataFrame:
    # Finnhub resolutions: 60, D, W (we use those)
    sec_per_bar = {"60": 3600, "D": 86400, "W": 604800}[resolution]
    now = int(time.time())
    frm = now - (bars * sec_per_bar) - (10 * sec_per_bar)

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {"symbol": symbol, "resolution": resolution, "from": frm, "to": now, "token": FINNHUB_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub candle fetch failed for {symbol} {resolution}: {data}")

    df = pd.DataFrame({"t": data["t"], "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"], "v": data.get("v",[0]*len(data["t"]))})
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df

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

SYSTEM_PROMPT = (
    "You are a trading analysis assistant. You do NOT browse the web. "
    "You ONLY use the JSON provided. Return ONLY valid JSON. No hype. No emojis."
)

USER_PROMPT_TEMPLATE = """Analyze this multi-timeframe market snapshot for early signals.
Timeframes: 2H, 4H, 1D, 1W.

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

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)

def run_symbol(symbol: str):
    time.sleep(2)  # polite spacing

    intraday_60 = finnhub_candles(symbol, "60", 400)
    daily = finnhub_candles(symbol, "D", 400)
    weekly = finnhub_candles(symbol, "W", 260)

    tf_2h = resample_ohlcv(intraday_60, "2H")
    tf_4h = resample_ohlcv(intraday_60, "4H")

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
    if not FINNHUB_API_KEY:
        raise RuntimeError("Missing FINNHUB_API_KEY")

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
