import os
import json
import time
import io
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from openai import OpenAI


# -----------------------------
# CONFIG
# -----------------------------
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "NVDA").split(",") if s.strip()]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()  # default to something widely available
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
    return [round(support, 4), round(mid, 4)], [round(resistance, 4)]


# -----------------------------
# STOOQ FETCH (DAILY)
# -----------------------------
def stooq_symbol(symbol: str) -> str:
    # Stooq typically uses .us for US tickers/ETFs
    return f"{symbol.lower()}.us"

def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    s = stooq_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(s)}&i=d"

    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    df = pd.read_csv(io.StringIO(raw))

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
    d = df_daily.set_index("t").sort_index()
    out = d.resample("W-FRI").agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }).dropna()
    return out.reset_index()


# -----------------------------
# SNAPSHOT BUILD
# -----------------------------
def build_tf_snapshot(df: pd.DataFrame, tf_name: str) -> dict:
    close = df["c"].astype(float)
    rsi = compute_rsi(close, 14)
    macd, signal, hist = compute_macd(close, 12, 26, 9)
    supports, resistances = infer_levels(df)

    tail = df.tail(15).copy()
    tail["t"] = tail["t"].astype(str)  # ensure JSON serializable

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
        json.dumps(snapshot, ensure_ascii=False, default=str)
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
# HUMAN SUMMARY
# -----------------------------
def _fmt_levels(vals) -> str:
    try:
        if not vals:
            return "-"
        return ", ".join(str(x) for x in vals[:3])
    except Exception:
        return "-"

def build_human_summary(run_ts: str, results: list[dict], failures: list[str]) -> str:
    lines = []
    lines.append(f"Market Scan — {run_ts} UTC")
    lines.append("=" * 60)

    if failures:
        lines.append("Failures:")
        for f in failures:
            lines.append(f"  - {f}")
        lines.append("-" * 60)

    for r in results:
        symbol = r.get("symbol", "UNKNOWN")
        overall = str(r.get("overall_bias", "neutral")).upper()
        lines.append(f"\n{symbol} — Overall: {overall}")
        lines.append("-" * 60)

        # sort signals by timeframe order
        tf_order = {"1D": 1, "1W": 2}
        sigs = r.get("signals", []) or []
        sigs = sorted(sigs, key=lambda s: tf_order.get(str(s.get("timeframe", "")), 99))

        for s in sigs:
            tf = s.get("timeframe", "?")
            setup = str(s.get("setup", "neutral")).upper()
            conf = s.get("confidence_0_100", "?")
            price = s.get("price", "?")
            rsi = s.get("rsi14", "?")
            macd = s.get("macd", {})
            sup = s.get("support", [])
            res = s.get("resistance", [])
            why = s.get("why", []) or []
            inv = s.get("invalidation", "")
            nxt = s.get("next_action", "")

            lines.append(f"{tf}: {setup} | conf {conf}/100 | price {price} | RSI {rsi}")
            if isinstance(macd, dict):
                lines.append(f"MACD: {macd.get('macd')} / Signal: {macd.get('signal')} / Hist: {macd.get('hist')}")
            lines.append(f"Support: {_fmt_levels(sup)}")
            lines.append(f"Resist:  {_fmt_levels(res)}")
            if why:
                lines.append(f"Why: {why[0]}")
            if inv:
                lines.append(f"Invalidation: {inv}")
            if nxt:
                lines.append(f"Next: {nxt}")
            lines.append("")

        notes = r.get("watchlist_notes", []) or []
        if notes:
            lines.append("Notes:")
            for n in notes[:5]:
                lines.append(f"  - {n}")

    lines.append("\nEnd.")
    return "\n".join(lines)


# -----------------------------
# RUN
# -----------------------------
def run_symbol(symbol: str):
    time.sleep(0.5)  # be polite

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

    # Write per-symbol logs
    safe_ts = now.replace(":", "-")
    base = f"{LOG_DIR}/{safe_ts}_{symbol}"

    with open(base + "_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

    with open(base + "_gpt.json", "w", encoding="utf-8") as f:
        f.write(gpt_out)

    # Parse GPT output into object for summary
    parsed = json.loads(gpt_out)
    return now, snapshot, parsed


def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    ensure_dirs()

    run_ts = utc_now_iso()

    results: list[dict] = []
    failures: list[str] = []

    for sym in SYMBOLS:
        try:
            _, _, parsed = run_symbol(sym)
            results.append(parsed)
        except Exception as e:
            failures.append(f"{sym}: {repr(e)}")

    # Always write a summary, even if some symbols failed
    summary = build_human_summary(run_ts, results, failures)
    with open(f"{LOG_DIR}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    # Print summary into Actions logs (easy reading)
    print("\n===== HUMAN SUMMARY (logs/summary.txt) =====\n")
    print(summary)
    print("\n===== END SUMMARY =====\n")

    if failures:
        raise RuntimeError("Errors:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
