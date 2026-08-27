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

# Charts are sent only if Bias is Buy/Sell AND confidence >= this threshold
ACTION_CONFIDENCE_MIN = int(os.getenv("ACTION_CONFIDENCE_MIN", "70"))

# Satellite rotation basket — trial dual-lens (momentum vs discount) ranking
# appended to the summary. Purely informational, no effect on Buy/Sell/Size
# Hint logic for these or any other symbol. AMAT added 2026-08-13 for this
# informational ranking only — quant-server's RULES.SATELLITE_TICKERS stays
# at NVDA/TSLA/PLTR/ZETA (grade-waiver/double-down eligibility), so this list
# is deliberately no longer 1:1 with that one.
SATELLITE_TICKERS = [s.strip().upper() for s in os.getenv("SATELLITE_TICKERS", "NVDA,TSLA,PLTR,ZETA,AMAT").split(",") if s.strip()]

# Optional: also forward raw 4H structure to quant-server for Claude to use.
# Additive only — if unset, behaves exactly as before (no GPT/Telegram change).
QUANT_SERVER_URL = os.getenv("QUANT_SERVER_URL", "").strip()
PRICE_CONTEXT_KEY = os.getenv("PRICE_CONTEXT_KEY", "").strip()

BASE_URL = "https://api.twelvedata.com/time_series"

# 1H lookback for building 4H
LOOKBACK_1H_BARS = int(os.getenv("LOOKBACK_1H_BARS", "360"))  # 360h ~ 90x 4H bars
# Daily lookback for 52W high
LOOKBACK_1D_BARS = int(os.getenv("LOOKBACK_1D_BARS", "320"))  # >= 252 needed

CHART_BARS = int(os.getenv("CHART_BARS", "120"))

# FREE TIER RATE LIMITING (Twelve Data)
TD_MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("TD_MIN_SECONDS_BETWEEN_CALLS", "9.0"))

# Small pauses for GPT/Telegram (not Twelve Data credits)
SLEEP_BETWEEN_OTHER_CALLS = float(os.getenv("SLEEP_BETWEEN_OTHER_CALLS", "0.3"))

# ---- VIX/VXN regime overlay ----
# Prefer VXN for tech-heavy basket; fallback to VIX if VXN unavailable
VOL_PREF = os.getenv("VOL_PREF", "VXN").strip().upper()  # "VXN" or "VIX"
# Bands (inclusive lower bounds)
VOL_CALM_MAX = float(os.getenv("VOL_CALM_MAX", "15"))     # < 15
VOL_NORMAL_MAX = float(os.getenv("VOL_NORMAL_MAX", "22")) # 15-22
VOL_FEAR_MAX = float(os.getenv("VOL_FEAR_MAX", "30"))     # 22-30
# Multipliers
MULT_CALM = float(os.getenv("MULT_CALM", "0.75"))
MULT_NORMAL = float(os.getenv("MULT_NORMAL", "1.0"))
MULT_FEAR = float(os.getenv("MULT_FEAR", "1.5"))
MULT_PANIC = float(os.getenv("MULT_PANIC", "2.0"))
# Base tranche for sizing hints (GBP)
BASE_TRANCHE_GBP = float(os.getenv("BASE_TRANCHE_GBP", "1000"))

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

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# =========================
# INVESTOR ACTION (confidence nuance)
# =========================
def investor_action(trading_bias: str, confidence: int) -> str:
    """
    Long-only swing trader framing:
    - BUY = potential add/accumulate zone
    - SELL = potential trim/take-profits zone (not short)
    - NEUTRAL = hold/wait

    SELL has no confidence gate, no RSI/MACD numeric threshold — matches
    quant-server's own /trim command, which hands Claude the full technical
    context and trusts its judgement directly rather than filtering through
    a fixed rule. GPT's bias here already reasons over RSI/MACD/support-
    resistance in the prompt; a second numeric gate on top just second-
    guesses that reasoning without adding real signal.
    """
    b = (trading_bias or "Neutral").strip().lower()
    c = int(confidence or 0)

    if b == "buy":
        return "ADD / ACCUMULATE" if c >= 70 else "WATCH / EARLY SETUP"

    if b == "sell":
        return "TRIM / TAKE PROFITS"

    return "HOLD / WAIT"


# =========================
# POSITION SIZING HINTS
# =========================
def confidence_to_tranche_factor(conf: int) -> float:
    """
    Converts confidence into a fraction of a 'base tranche'.
    Conservative, long-only:
      50-59 -> 0.25
      60-69 -> 0.50
      70-79 -> 0.75
      80+   -> 1.00
      <50   -> 0.00 (no sizing hint)
    """
    c = int(conf or 0)
    if c < 50:
        return 0.0
    if c < 60:
        return 0.25
    if c < 70:
        return 0.50
    if c < 80:
        return 0.75
    return 1.00

def format_gbp(x: float) -> str:
    return f"£{x:,.0f}"

def sizing_hint_text(bias: str, conf: int, vol_mult: float) -> str:
    """
    Returns a short tranche sizing hint string.
    We only show sizing for Buy/Sell biases.
    """
    if (bias or "").strip() not in ("Buy", "Sell"):
        return ""
    base_factor = confidence_to_tranche_factor(conf)
    if base_factor <= 0:
        return "Size Hint: none (low confidence)"
    suggested = BASE_TRANCHE_GBP * vol_mult * base_factor
    mult_txt = f"{vol_mult:.2f}x" if vol_mult is not None else "1.00x"
    return f"Size Hint: {base_factor:.2f} tranche × {mult_txt} ≈ {format_gbp(suggested)}"


# =========================
# TELEGRAM
# =========================
TG_MAX_CHARS = 3800  # stay under Telegram's 4096 hard limit

def _chunk_message(text: str, max_chars: int = TG_MAX_CHARS):
    """Split on line boundaries so a symbol's block never gets cut mid-way.
    Falls back to a hard split only if a single line itself exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]

    chunks, cur = [], ""
    for line in text.split("\n"):
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > max_chars and cur:
            chunks.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        chunks.append(cur)

    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            final.extend(c[i:i + max_chars] for i in range(0, len(c), max_chars))
    return final

def tg_send_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = _chunk_message(text)
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True
        }
        r = requests.post(url, data=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram sendMessage error {r.status_code}: {r.text}")
        if i < len(chunks) - 1:
            time.sleep(SLEEP_BETWEEN_OTHER_CALLS)

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
# QUANT SERVER (PRICE CONTEXT — additive, non-fatal)
# =========================
def send_price_context(symbol, close, rsi_val, macd_line, sig_line, hist, sup, res,
                        high_52w, pct_from, lvl_20, lvl_30, lvl_40, vol_regime, vol_mult,
                        vol_current, vol_avg20, vol_ratio, up_vol, down_vol, atr14,
                        efficiency_ratio, adx14, trend_regime):
    """
    Forwards raw 4H structure (not GPT's interpretation) to quant-server so Claude
    can use it as context. Best-effort only — never raises, never touches
    results/actionable/failures, so it can't affect the existing GPT/Telegram flow.
    """
    if not QUANT_SERVER_URL:
        return
    try:
        payload = {
            "symbol": symbol,
            "close": round(float(close), 4),
            "rsi": round(float(rsi_val), 2),
            "macd": {
                "macd": round(float(macd_line.iloc[-1]), 6),
                "signal": round(float(sig_line.iloc[-1]), 6),
                "hist": round(float(hist.iloc[-1]), 6),
            },
            "support": round(float(sup), 4),
            "resistance": round(float(res), 4),
            "high_52w": round(float(high_52w), 4),
            "pct_from_52w": round(float(pct_from), 2),
            "pullback_levels": {
                "20": round(float(lvl_20), 4),
                "30": round(float(lvl_30), 4),
                "40": round(float(lvl_40), 4),
            },
            "atr14": round(float(atr14), 4),
            "efficiency_ratio": round(float(efficiency_ratio), 4),
            "adx14": round(float(adx14), 2),
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
            "vol_mult": vol_mult,
            "volume": {
                "current": round(float(vol_current), 0),
                "avg20": round(float(vol_avg20), 0),
                "ratio": round(float(vol_ratio), 4) if np.isfinite(vol_ratio) else None,
            },
            "volume_direction": {
                "up_vol": round(float(up_vol), 0),
                "down_vol": round(float(down_vol), 0),
            },
            "timestamp": utc_now().isoformat(),
        }
        headers = {"Content-Type": "application/json"}
        if PRICE_CONTEXT_KEY:
            headers["x-price-context-key"] = PRICE_CONTEXT_KEY

        # QUANT_SERVER_URL is the full endpoint (e.g. https://.../price-context) — no path appended here
        r = requests.post(QUANT_SERVER_URL, json=payload, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"[price-context] {symbol}: HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[price-context] {symbol}: {repr(e)}")


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
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t").set_index("t")

    _cache[key] = df
    return df

def fetch_1h(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1h", LOOKBACK_1H_BARS)

def fetch_1d(symbol: str) -> pd.DataFrame:
    return fetch_series(symbol, "1day", LOOKBACK_1D_BARS)

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Use lowercase 'h' to avoid pandas FutureWarning
    out = df.resample(rule).agg({
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }).dropna(subset=["o", "h", "l", "c"])
    out["v"] = out["v"].fillna(0)
    return out


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

def volume_ratio(vol: pd.Series, period: int = 20) -> tuple[float, float, float]:
    """
    Current bar volume vs its trailing N-bar average (industry-standard
    convention for volume confirmation, e.g. "volume vs 20-bar avg").
    Returns (current_vol, avg_vol, ratio). Ratio is NaN if avg_vol is 0/NaN
    (e.g. too few bars, or a feed that doesn't report volume for this symbol).
    """
    current = float(vol.iloc[-1])
    avg = float(vol.tail(period).mean())
    ratio = (current / avg) if avg > 0 else float("nan")
    return current, avg, ratio

def volume_readable(current: float, avg: float, ratio: float, period: int = 20) -> str:
    if not np.isfinite(ratio):
        return "No volume data"
    tag = "elevated" if ratio >= 1.5 else "below average" if ratio < 0.7 else "normal"
    return f"{current:,.0f} vs {period}-bar avg {avg:,.0f} ({ratio:.2f}x, {tag})"

def volume_direction(close: pd.Series, vol: pd.Series, period: int = 20) -> tuple[float, float]:
    """
    Twelve Data's OHLCV feed has no true buy/sell volume split (that needs
    L2/tick data, which we don't have). Standard proxy when only OHLCV is
    available: classify each of the trailing `period` bars as up/down by
    close vs prior close, and sum volume per side (same logic as OBV).
    Returns (up_vol, down_vol).
    """
    c = close.tail(period + 1)
    v = vol.tail(period + 1)
    delta = c.diff().iloc[1:]
    v_aligned = v.iloc[1:]
    up_vol = float(v_aligned[delta > 0].sum())
    down_vol = float(v_aligned[delta < 0].sum())
    return up_vol, down_vol

def volume_direction_readable(up_vol: float, down_vol: float) -> str:
    if up_vol == 0 and down_vol == 0:
        return "No directional volume data"
    if down_vol == 0:
        return "All up-volume (buying pressure)"
    if up_vol == 0:
        return "All down-volume (selling pressure)"
    ratio = up_vol / down_vol
    if ratio >= 1.15:
        return f"Up-vol {ratio:.1f}x Down-vol (buying pressure)"
    if ratio <= 1 / 1.15:
        return f"Down-vol {(1 / ratio):.1f}x Up-vol (selling pressure)"
    return f"Up-vol/Down-vol balanced ({ratio:.2f}x)"

def volume_direction_simple(up_vol: float, down_vol: float) -> str:
    """
    Plain, Telegram-facing version of volume_direction_readable() — GPT still
    gets the detailed ratio text for its own reasoning; this is just the
    human-readable verdict shown in the caption.
    """
    if up_vol == 0 and down_vol == 0:
        return "❔ No volume data"
    if down_vol == 0:
        return "📈 Buyers in control"
    if up_vol == 0:
        return "📉 Sellers in control"
    ratio = up_vol / down_vol
    if ratio >= 1.15:
        return "📈 Buyers in control"
    if ratio <= 1 / 1.15:
        return "📉 Sellers in control"
    return "⚖️ Neither side in control"


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
# ATR (Average True Range)
# =========================
def compute_atr(daily_df: pd.DataFrame, period: int = 14) -> float:
    """
    Wilder-smoothed ATR(14) from daily OHLC — same smoothing convention as
    rsi_wilder() above. Added 2026-08-26 for quant-server's /rotate AMOUNT,
    which uses it to show a volatility-based expected price range (how big a
    move is typical, NOT a directional forecast) alongside the technical/
    analyst-driven ranking. Deliberately kept separate from RSI/MACD/support-
    resistance, which describe direction/momentum — ATR only describes
    magnitude.
    """
    high, low, close = daily_df["h"], daily_df["l"], daily_df["c"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1).dropna()
    if len(tr) < period:
        raise RuntimeError(f"Not enough daily bars for ATR({period}): {len(tr)}")
    atr = tr.iloc[:period].mean()
    for val in tr.iloc[period:]:
        atr = (atr * (period - 1) + val) / period
    return float(atr)


# =========================
# REGIME (Efficiency Ratio + ADX) — added 2026-08-27
# =========================
def compute_efficiency_ratio(daily_df: pd.DataFrame, period: int = 20) -> float:
    """
    Kaufman's Efficiency Ratio: net price change over `period` days divided
    by the sum of each day's absolute move over the same window. Near 1 =
    price travelled in a straight line (trending); near 0 = lots of daily
    movement with little net progress (choppy/ranging). This is a measure
    of the PRICE'S OWN path efficiency -- unrelated to "market efficiency"
    in the economic/informational sense (Efficient Market Hypothesis is a
    different concept entirely).
    """
    close = daily_df["c"]
    if len(close) < period + 1:
        raise RuntimeError(f"Not enough daily bars for Efficiency Ratio({period}): {len(close)}")
    window = close.tail(period + 1)
    net_change = abs(float(window.iloc[-1]) - float(window.iloc[0]))
    volatility_sum = window.diff().abs().sum()
    if volatility_sum == 0:
        return 0.0
    return float(net_change / volatility_sum)


def compute_adx(daily_df: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index -- the standard trend-strength indicator
    (regardless of direction). >~25 = trending, <~20 = weak/ranging.
    Wilder's original smoothing method, same convention as rsi_wilder()/
    compute_atr() above.
    """
    high, low, close = daily_df["h"], daily_df["l"], daily_df["c"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    combined = pd.concat([tr, plus_dm, minus_dm], axis=1).dropna()
    combined.columns = ["tr", "plus_dm", "minus_dm"]
    if len(combined) < period * 2:
        raise RuntimeError(f"Not enough daily bars for ADX({period}): {len(combined)}")

    def wilder_smooth(s: pd.Series, period: int) -> pd.Series:
        seed = s.iloc[:period].sum()
        out = [seed]
        for val in s.iloc[period:]:
            out.append(out[-1] - (out[-1] / period) + val)
        return pd.Series(out, index=s.index[period - 1:])

    sm_tr = wilder_smooth(combined["tr"], period)
    sm_plus_dm = wilder_smooth(combined["plus_dm"], period)
    sm_minus_dm = wilder_smooth(combined["minus_dm"], period)

    plus_di = 100 * sm_plus_dm / sm_tr
    minus_di = 100 * sm_minus_dm / sm_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    adx_val = dx.iloc[:period].mean()
    for val in dx.iloc[period:]:
        adx_val = (adx_val * (period - 1) + val) / period
    return float(adx_val)


def classify_regime(efficiency_ratio: float, adx_val: float) -> str:
    """
    Combines ER + ADX into a plain label. Thresholds match the values
    validated live against real satellite-ticker data on 2026-08-27
    (AMAT ER 0.10/ADX 14.8 -> CHOPPY; PLTR ER 0.54/ADX 35.0 -> TRENDING).
    """
    if adx_val > 25 and efficiency_ratio > 0.4:
        return "TRENDING"
    if adx_val < 20 and efficiency_ratio < 0.3:
        return "CHOPPY"
    return "MIXED"


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
# VIX/VXN (Regime Overlay)
# =========================
def _fetch_cboe_last_close(symbol: str) -> float:
    """
    Pull latest close from Cboe CSV endpoints.
    Returns float close or raises.
    """
    sym = symbol.upper().strip()
    url_map = {
        "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "VXN": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv",
    }
    if sym not in url_map:
        raise RuntimeError(f"Unsupported vol symbol: {sym}")

    # Read CSV directly via pandas (no extra deps)
    df = pd.read_csv(url_map[sym])
    # Columns usually: DATE, OPEN, HIGH, LOW, CLOSE
    # Be defensive to minor changes:
    close_col = None
    for c in df.columns:
        if str(c).strip().upper() == "CLOSE":
            close_col = c
            break
    if close_col is None:
        raise RuntimeError(f"{sym}: CLOSE column not found in CSV. Columns: {list(df.columns)}")

    # last non-null close
    s = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if s.empty:
        raise RuntimeError(f"{sym}: no close values in CSV")
    return float(s.iloc[-1])

def get_vol_value() -> tuple[str, float]:
    """
    Returns (used_symbol, value).
    Prefers VOL_PREF (default VXN), fallback to the other if fetch fails.
    """
    primary = VOL_PREF if VOL_PREF in ("VIX", "VXN") else "VXN"
    secondary = "VIX" if primary == "VXN" else "VXN"

    try:
        return primary, _fetch_cboe_last_close(primary)
    except Exception:
        try:
            return secondary, _fetch_cboe_last_close(secondary)
        except Exception as e2:
            # If both fail, return unknown marker
            raise RuntimeError(f"Could not fetch VIX/VXN (primary {primary}, secondary {secondary}): {repr(e2)}")

def vol_regime(vol_value: float) -> tuple[str, float]:
    """
    Returns (regime_name, multiplier)
    """
    v = float(vol_value)
    if v < VOL_CALM_MAX:
        return "CALM", MULT_CALM
    if v < VOL_NORMAL_MAX:
        return "NORMAL", MULT_NORMAL
    if v < VOL_FEAR_MAX:
        return "FEAR", MULT_FEAR
    return "PANIC", MULT_PANIC


# =========================
# GPT ANALYSIS
# =========================
def _safe_extract_json(text: str) -> dict:
    """
    Extract JSON object from a model response, robustly.
    """
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    t = text.strip()

    # If the model wrapped it in code fences, strip them
    if t.startswith("```"):
        t = t.strip("`").strip()

    # Find first { ... last }
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end+1]

    return json.loads(t)

def gpt_analyze(symbol: str, tf: str, close: float, rsi_val: float, macd_text: str,
               sup: float, res: float,
               high_52w: float, pct_from_52w: float, lvl_20: float, lvl_30: float, lvl_40: float,
               vol_sym: str, vol_val: float, regime: str, mult: float,
               volume_text: str, volume_dir_text: str) -> dict:
    prompt = f"""
You are a trading assistant for a long-only swing trader. Be concise and structured.

Symbol: {symbol}
Timeframe: {tf}
Close: {close:.2f}
RSI(14): {rsi_val:.1f}
MACD: {macd_text}
Volume (4H bar vs 20-bar avg): {volume_text}
Volume direction (last 20 bars, up-bar vs down-bar volume split): {volume_dir_text}
Support: {sup:.2f}
Resistance: {res:.2f}

52W High: {high_52w:.2f}
% From 52W High: {pct_from_52w:.1f}%
Pullback levels from 52W High:
-20%: {lvl_20:.2f}
-30%: {lvl_30:.2f}
-40%: {lvl_40:.2f}

Market Volatility Regime:
{vol_sym}: {vol_val:.2f}
Regime: {regime}
Tranche Multiplier: {mult:.2f}x

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

Rules:
- "Sell" means trim/take-profits (NOT short).
- Weigh volume as a confirming/disconfirming factor: elevated volume backing
  the RSI/MACD signal strengthens conviction; a move on below-average volume
  is weaker conviction and should be reflected in confidence and "why".
- Use the volume direction split to judge which side is in control: elevated
  volume that aligns with the signal (e.g. Sell bias + selling pressure)
  reinforces conviction; elevated volume against the signal (e.g. Sell bias
  but buying pressure dominant) should lower confidence and be called out
  in "why".
- "setup_tag" must lead with a short plain-English phrase a non-trader would
  understand (2-5 words, no TA jargon), then the technical term in
  parentheses — e.g. "Sellers Losing Steam (Bullish Divergence)" or
  "Riding the Uptrend (Bullish Continuation)". Always include both parts.
- "definition" must be one plain, jargon-free sentence explaining what's
  actually happening in the price action and why it matters for the call —
  written for someone with no technical-analysis background.
- Keep text fields short, one sentence max where possible.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    text = (resp.choices[0].message.content or "").strip()

    # Robust parse + one retry if model returns non-JSON
    try:
        return _safe_extract_json(text)
    except Exception:
        # Retry with a stricter instruction
        resp2 = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": "Return ONLY valid minified JSON for the specified keys. No commentary, no markdown."}
            ],
            temperature=0.0
        )
        text2 = (resp2.choices[0].message.content or "").strip()
        return _safe_extract_json(text2)


# =========================
# SATELLITE ROTATION RANKING (trial — momentum vs discount, informational only)
# =========================
def rank_satellites(results: list, mode: str) -> list:
    """
    Rank-sum ranking across the satellite basket (SATELLITE_TICKERS) using
    RSI, MACD histogram (normalised as % of price so tickers at very
    different price levels are comparable), and % from 52W high.

    Each factor is ranked 1 (best) to N (worst) across the basket, then
    summed — lowest total wins. Two opposite interpretations of the same
    three factors, both purely informational, no effect on any Buy/Sell/
    Size Hint logic:

    mode="momentum": reward strength — higher RSI, more positive/accelerating
      MACD, closer to the 52W high (leadership).
    mode="discount": reward room to recover — lower RSI, more negative MACD,
      further below the 52W high (bigger discount).
    """
    sats = [r for r in results if r["symbol"] in SATELLITE_TICKERS]
    if len(sats) < 2:
        return []

    for r in sats:
        r["_macd_hist_pct"] = (r["macd_hist"] / r["close"] * 100) if r["close"] else 0.0

    reverse = (mode == "momentum")  # higher value = better rank for momentum
    by_rsi = sorted(sats, key=lambda r: r["rsi"], reverse=reverse)
    by_macd = sorted(sats, key=lambda r: r["_macd_hist_pct"], reverse=reverse)
    # pct_from is always <= 0 (negative % below 52W high). Momentum wants
    # closest to 0 (least negative) first; discount wants most negative first.
    by_52w = sorted(sats, key=lambda r: r["pct_from"], reverse=reverse)

    rank_sum = {r["symbol"]: 0 for r in sats}
    for ranked_list in (by_rsi, by_macd, by_52w):
        for i, r in enumerate(ranked_list):
            rank_sum[r["symbol"]] += i + 1

    by_symbol = {r["symbol"]: r for r in sats}
    ordered = sorted(rank_sum.items(), key=lambda kv: kv[1])
    return [(sym, total, by_symbol[sym]) for sym, total in ordered]


def format_satellite_block(momentum_ranking: list, discount_ranking: list) -> list:
    if not momentum_ranking or not discount_ranking:
        return []

    def fmt_line(rank_pos, sym, total, r):
        return (f"{rank_pos}. {sym} (rank sum {total}) — RSI {r['rsi']:.0f}, "
                f"MACD {r['_macd_hist_pct']:+.2f}% of price, "
                f"{r['pct_from']:.1f}% off 52W high")

    out = ["-" * 50, "🎯 Satellite Rotation (trial — momentum vs discount lens)", ""]
    out.append("Momentum lean:")
    for i, (sym, total, r) in enumerate(momentum_ranking, 1):
        out.append(fmt_line(i, sym, total, r))
    out.append("")
    out.append("Discount lean:")
    for i, (sym, total, r) in enumerate(discount_ranking, 1):
        out.append(fmt_line(i, sym, total, r))
    out.append("")
    out.append("_Informational only — no effect on Buy/Sell/Size Hint logic above._")
    return out


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

    # ---- fetch vol regime once per run ----
    vol_sym, vol_val = ("VOL", float("nan"))
    regime, mult = ("UNKNOWN", 1.0)
    vol_err = None
    try:
        vol_sym, vol_val = get_vol_value()
        regime, mult = vol_regime(vol_val)
    except Exception as e:
        vol_err = repr(e)
        # proceed without overlay
        vol_sym, vol_val = ("VOL", float("nan"))
        regime, mult = ("UNKNOWN", 1.0)

    results = []
    actionable_buy = []
    actionable_trim = []
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

            vol_current, vol_avg20, vol_ratio = volume_ratio(df_4h["v"], 20)
            volume_text = volume_readable(vol_current, vol_avg20, vol_ratio, 20)

            up_vol, down_vol = volume_direction(df_4h["c"], df_4h["v"], 20)
            volume_dir_text = volume_direction_readable(up_vol, down_vol)
            volume_dir_simple = volume_direction_simple(up_vol, down_vol)

            df_1d = fetch_1d(symbol)
            high_52w, pct_from, lvl_20, lvl_30, lvl_40 = compute_52w(df_1d, close)
            atr14 = compute_atr(df_1d, 14)
            # Named trend_regime, NOT regime -- that name is already the
            # VIX/VXN volatility regime (FEAR/NORMAL/CALM/PANIC) computed
            # earlier in this function. Trend regime and volatility regime
            # are different dimensions (a market can be calm-and-choppy or
            # calm-and-trending) -- see compute_efficiency_ratio()'s docstring.
            efficiency_ratio = compute_efficiency_ratio(df_1d, 20)
            adx14 = compute_adx(df_1d, 14)
            trend_regime = classify_regime(efficiency_ratio, adx14)

            time.sleep(SLEEP_BETWEEN_OTHER_CALLS)
            g = gpt_analyze(symbol, tf, close, rsi_val, macd_text, sup, res,
                            high_52w, pct_from, lvl_20, lvl_30, lvl_40,
                            vol_sym, vol_val if np.isfinite(vol_val) else 0.0, regime, mult,
                            volume_text, volume_dir_text)

            trading_bias = g.get("bias", "Neutral")
            conf = int(g.get("confidence", 0))
            action = investor_action(trading_bias, conf)

            size_hint = sizing_hint_text(trading_bias, conf, mult)

            results.append({
                "symbol": symbol,
                "df_4h": df_4h,  # cached for charting (no refetch)
                "close": close,
                "rsi": rsi_val,
                "macd_text": macd_text,
                "macd_hist": float(hist.iloc[-1]),
                "volume_text": volume_text,
                "volume_dir_text": volume_dir_text,
                "volume_dir_simple": volume_dir_simple,
                "sup": sup,
                "res": res,
                "high_52w": high_52w,
                "pct_from": pct_from,
                "lvl_20": lvl_20,
                "lvl_30": lvl_30,
                "lvl_40": lvl_40,
                "atr14": atr14,
                "efficiency_ratio": efficiency_ratio,
                "adx14": adx14,
                "trend_regime": trend_regime,
                "g": g,
                "trading_bias": trading_bias,
                "confidence": conf,
                "investor_action": action,
                "size_hint": size_hint
            })

            # Forward raw structure to quant-server (additive, best-effort — see function docstring)
            send_price_context(symbol, close, rsi_val, macd_line, sig_line, hist, sup, res,
                                high_52w, pct_from, lvl_20, lvl_30, lvl_40, regime, mult,
                                vol_current, vol_avg20, vol_ratio, up_vol, down_vol, atr14,
                                efficiency_ratio, adx14, trend_regime)

            # "Entry advised" = Buy with confidence >= threshold, OR any Sell
            # bias at all — trim isn't confidence-gated, matching /trim's
            # "trust the model, no secondary filter" pattern on quant-server.
            if trading_bias == "Buy" and conf >= ACTION_CONFIDENCE_MIN:
                actionable_buy.append(symbol)
            elif trading_bias == "Sell":
                actionable_trim.append(symbol)

        except Exception as e:
            failures.append(f"{symbol}: {repr(e)}")

    # SUMMARY MESSAGE
    header_lines = [
        "📊 INTRADAY SCAN (4H)",
        ts_str,
        "=" * 50,
    ]

    if regime != "UNKNOWN":
        header_lines.append(f"Market Regime: {regime} ({vol_sym} {vol_val:.2f}) | Tranche Mult: {mult:.2f}x")
    else:
        header_lines.append("Market Regime: UNKNOWN (VIX/VXN unavailable) | Tranche Mult: 1.00x")
    header_lines.append("")

    lines = header_lines.copy()

    if results:
        for r in results:
            g = r["g"]
            tag = g.get("setup_tag", "")
            inv = (g.get("invalidation", "") or "").strip()

            lines.append(f"🔷 {r['symbol']} | {r['investor_action']} | {r['confidence']}%")
            lines.append(f"Setup: {tag}")
            lines.append(f"Close: {r['close']:.2f}")
            lines.append(f"S: {r['sup']:.2f} | R: {r['res']:.2f}")
            lines.append(f"From 52W High: {r['pct_from']:.1f}%")
            # Show the fixed pullback levels (explicitly)
            lines.append(f"52W Pullbacks: -20% {r['lvl_20']:.2f} | -30% {r['lvl_30']:.2f} | -40% {r['lvl_40']:.2f}")

            # Show invalidation only when bias is Buy/Sell (so summary stays readable)
            if r["trading_bias"] in ("Buy", "Sell") and inv:
                lines.append(f"Invalidation: {inv}")

            # Show sizing hint only when bias is Buy/Sell
            if r["size_hint"]:
                lines.append(r["size_hint"])

            lines.append("")
    else:
        lines.append("No results produced.\n")

    if actionable_buy or actionable_trim:
        lines.append("-" * 50)
        lines.append("📌 Actionable entries detected:")
        if actionable_buy:
            lines.append("🟢 Buy/Add: " + ", ".join(actionable_buy))
        if actionable_trim:
            lines.append("🔴 Trim: " + ", ".join(actionable_trim))

    momentum_ranking = rank_satellites(results, "momentum")
    discount_ranking = rank_satellites(results, "discount")
    lines.extend(format_satellite_block(momentum_ranking, discount_ranking))

    if vol_err:
        lines.append("-" * 50)
        lines.append(f"⚠️ Vol fetch issue (non-blocking): {vol_err}")

    tg_send_message("\n".join(lines))

    # CHART + DETAIL only for actionable entries — Buy still needs
    # ACTION_CONFIDENCE_MIN, Sell (trim) always gets a chart regardless of
    # confidence, matching /trim's unfiltered "trust the model" pattern.
    for r in results:
        g = r["g"]
        trading_bias = r["trading_bias"]
        conf = r["confidence"]

        if trading_bias not in ("Buy", "Sell"):
            continue
        if trading_bias == "Buy" and conf < ACTION_CONFIDENCE_MIN:
            continue

        symbol = r["symbol"]
        chart_path = f"charts/{symbol}_4H.png"
        make_chart(symbol, "4H", r["df_4h"], r["sup"], r["res"], chart_path)

        caption = (
            f"🔷 {symbol} (4H)\n\n"
            f"Market Regime: {regime} ({vol_sym} {vol_val:.2f}) | Tranche Mult: {mult:.2f}x\n\n"
            f"Close: {r['close']:.2f}\n"
            f"RSI: {r['rsi']:.1f}\n"
            f"MACD: {r['macd_text']}\n"
            f"Volume: {r['volume_text']}\n"
            f"Volume Bias: {r['volume_dir_simple']}\n"
            f"Support: {r['sup']:.2f}\n"
            f"Resistance: {r['res']:.2f}\n\n"
            f"52W High: {r['high_52w']:.2f}\n"
            f"From 52W High: {r['pct_from']:.1f}%\n"
            f"-20%: {r['lvl_20']:.2f} | -30%: {r['lvl_30']:.2f} | -40%: {r['lvl_40']:.2f}\n\n"
            f"Trading Bias: {trading_bias}\n"
            f"Investor Action: {r['investor_action']}\n"
            f"Confidence: {conf}%\n"
            f"{r['size_hint']}\n\n"
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


