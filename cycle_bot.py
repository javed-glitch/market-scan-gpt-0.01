import io
import math
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =========================
# CONFIG
# =========================
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "TSLA,NVDA,PLTR,VUSA").split(",") if s.strip()]
CAPITAL_PER_TICKER = float(os.getenv("CAPITAL_PER_TICKER", "3000"))
TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
BASE_URL = "https://api.twelvedata.com/time_series"
TD_MIN_SECONDS = float(os.getenv("TD_MIN_SECONDS_BETWEEN_CALLS", "11.0"))
MAX_CARDS_PER_PAGE = 6
CARDS_PER_ROW = 3
ROWS_PER_PAGE = 2

_last_td_call = 0.0

YF_SYMBOL_MAP = {
    "VUSA": "VUSA.L",
}

def get_deployed(sym: str) -> float:
    return float(os.getenv(f"DEPLOYED_{sym}", "0"))

# =========================
# COLOURS
# =========================
BG = "#0b0b0d"
CARD = "#17181c"
INNER = "#101115"
INNER2 = "#0c0d10"
T1 = "#f0f0f2"
T2 = "#a8abb3"
T3 = "#5b606b"
BORDER = "#2b2f38"
BORDER_SOFT = "#22252c"

GREEN2 = "#16a085"
AMBER2 = "#d68910"
TEAL = "#1abc9c"

G_FG, G_BG = "#7ee2a8", "#102419"
R_FG, R_BG = "#ff9898", "#261315"
A_FG, A_BG = "#f4c35f", "#2a210d"
B_FG, B_BG = "#9fc7ff", "#101a2b"

STAGE_COLORS = [
    ("#9FE1CB", "#085041"),
    ("#5DCAA5", "#04342C"),
    ("#3266AD", "#E6F1FB"),
    ("#185FA5", "#E6F1FB"),
    ("#7F77DD", "#EEEDFE"),
    ("#AFA9EC", "#26215C"),
    ("#FAC775", "#412402"),
    ("#EF9F27", "#3A2000"),
    ("#F0997B", "#4A1B0C"),
    ("#D85A30", "#FAECE7"),
    ("#E24B4A", "#FCEBEB"),
    ("#A32D2D", "#FCEBEB"),
    ("#888780", "#F1EFE8"),
]
STAGE_NAMES = [
    "Hope", "Optimism", "Belief", "Thrill", "Euphoria", "Complacency",
    "Anxiety", "Denial", "Panic", "Capitulation", "Anger", "Depression", "Disbelief"
]

# =========================
# RATE LIMITER
# =========================
def td_throttle():
    global _last_td_call
    wait = TD_MIN_SECONDS - (time.time() - _last_td_call)
    if wait > 0:
        time.sleep(wait)
    _last_td_call = time.time()

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# =========================
# DATA FETCH
# =========================
def fetch_series_twelvedata(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    td_throttle()
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "apikey": TD_API_KEY,
        "format": "JSON",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    data = r.json()

    if isinstance(data, dict) and data.get("status") == "error":
        msg = (data.get("message") or "").lower()
        if "run out" in msg or "current minute" in msg:
            time.sleep(65)
            td_throttle()
            r = requests.get(BASE_URL, params=params, timeout=30)
            data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData: {data.get('message')}")

    values = data.get("values")
    if not values:
        raise RuntimeError(f"No values for {symbol} ({interval})")

    df = pd.DataFrame(values).rename(
        columns={"datetime": "t", "open": "o", "high": "h", "low": "l", "close": "c"}
    )
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o", "h", "l", "c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t").set_index("t")

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != "" and str(x).lower() != "nan"]
            flat_cols.append("_".join(parts))
        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _find_matching_column(columns, targets):
    lower_map = {str(c).lower(): c for c in columns}
    for target in targets:
        if target.lower() in lower_map:
            return lower_map[target.lower()]
    for c in columns:
        c_low = str(c).lower()
        for target in targets:
            t = target.lower()
            if c_low == t or c_low.startswith(t + "_") or c_low.endswith("_" + t) or t in c_low:
                return c
    return None

def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    df = _flatten_yf_columns(df).reset_index()
    cols = list(df.columns)

    time_col = _find_matching_column(cols, ["Datetime", "Date", "index"])
    open_col = _find_matching_column(cols, ["Open"])
    high_col = _find_matching_column(cols, ["High"])
    low_col = _find_matching_column(cols, ["Low"])
    close_col = _find_matching_column(cols, ["Close", "Adj Close"])

    if not all([time_col, open_col, high_col, low_col, close_col]):
        raise RuntimeError(f"Yahoo Finance missing columns after normalization. Got: {list(df.columns)}")

    df = df.rename(columns={
        time_col: "t",
        open_col: "o",
        high_col: "h",
        low_col: "l",
        close_col: "c",
    })

    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o", "h", "l", "c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["t", "o", "h", "l", "c"]).sort_values("t").set_index("t")

def fetch_series_yahoo(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    yf_symbol = YF_SYMBOL_MAP.get(symbol.upper(), symbol)

    if interval == "1day":
        period = "18mo"
        yf_interval = "1d"
    elif interval == "1h":
        period = "60d"
        yf_interval = "60m"
    else:
        raise RuntimeError(f"Yahoo Finance interval not supported: {interval}")

    df = yf.download(
        yf_symbol,
        period=period,
        interval=yf_interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )

    df = _normalize_yf(df)
    if outputsize and len(df) > outputsize:
        df = df.tail(outputsize)
    return df

def fetch_series(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    if symbol.upper() in YF_SYMBOL_MAP:
        return fetch_series_yahoo(symbol, interval, outputsize)
    return fetch_series_twelvedata(symbol, interval, outputsize)

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()

# =========================
# INDICATORS
# =========================
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd_calc(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ml = ema(close, fast) - ema(close, slow)
    sl = ema(ml, signal)
    hist = ml - sl
    return ml, sl, hist

# =========================
# VXN / VIX
# =========================
def get_vol_value():
    for sym, url in [
        ("VXN", "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"),
        ("VIX", "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"),
    ]:
        try:
            df = pd.read_csv(url)
            col = next((c for c in df.columns if str(c).strip().upper() == "CLOSE"), None)
            if col:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                return sym, float(s.iloc[-1])
        except Exception:
            continue
    return "VOL", None

def vol_regime(val):
    if val is None:
        return "Unknown", 1.0
    if val < 18:
        return "Low", 0.75
    if val < 28:
        return "Normal", 1.0
    if val < 35:
        return "High", 1.25
    return "Extreme", 1.50

def vol_color(name):
    return {
        "Low": G_FG,
        "Normal": A_FG,
        "High": R_FG,
        "Extreme": R_FG,
        "Unknown": T2,
    }.get(name, T2)

# =========================
# CYCLE CLASSIFIER
# =========================
def classify_cycle(rsi_val: float, pct_from_high: float, above_200ma: bool, hist_rising: bool):
    p = pct_from_high

    if above_200ma:
        if rsi_val > 70 and p > -5:
            return "Euphoria", 4
        if rsi_val > 65 and p > -10:
            return "Thrill", 3
        if rsi_val > 60 and p > -15:
            return ("Complacency", 5) if not hist_rising else ("Belief", 2)
        if rsi_val > 55 and p > -20:
            return "Belief", 2
        if rsi_val > 48 and p > -25:
            return "Optimism → Belief", 1
        if rsi_val > 42:
            return "Optimism", 1
        if rsi_val > 36:
            return "Hope", 0
        if p > -10:
            return "Complacency", 5
        if p > -20:
            return "Anxiety", 6
        return "Denial", 7

    if rsi_val < 24 and p < -48:
        return "Depression", 11
    if rsi_val < 28 and p < -40:
        return "Capitulation", 9
    if rsi_val < 32 and p < -32:
        return "Anger", 10
    if rsi_val < 36 and p < -24:
        return "Panic", 8
    if rsi_val < 42 and p < -18:
        return "Denial", 7
    if rsi_val >= 42 and p < -20 and hist_rising:
        return "Disbelief", 12
    return "Anxiety", 6

# =========================
# LEVELS / ACTIONS
# =========================
def compute_levels(close: float, h52: float):
    entry = round(close * 0.97, 2)
    panic_low = round(h52 * 0.65, 2)
    panic_high = round(h52 * 0.75, 2)
    cap_low = round(h52 * 0.50, 2)
    cap_high = round(h52 * 0.60, 2)

    first_trim_low = round(max(close * 1.08, h52 * 0.88), 2)
    first_trim_high = round(max(first_trim_low, h52 * 0.94), 2)

    strong_trim_low = round(max(first_trim_high, h52 * 0.94), 2)
    strong_trim_high = round(h52, 2)

    dist_to_panic = round(((panic_high - close) / close) * 100, 1)

    return {
        "entry": entry,
        "panic_low": panic_low,
        "panic_high": panic_high,
        "cap_low": cap_low,
        "cap_high": cap_high,
        "first_trim_low": first_trim_low,
        "first_trim_high": first_trim_high,
        "strong_trim_low": strong_trim_low,
        "strong_trim_high": strong_trim_high,
        "dist_to_panic": dist_to_panic,
    }

def buy_speed(pct: float):
    p = abs(pct)
    if p < 20:
        return "Slow"
    if p < 30:
        return "Medium"
    if p < 50:
        return "Aggressive"
    return "Max"

def determine_actions(stage_name: str, stage_idx: int, above_200ma: bool, speed: str, vol_mult: float):
    core_total = int(CAPITAL_PER_TICKER * 0.50)
    tactical_total = int(CAPITAL_PER_TICKER * 0.50)

    if any(x in stage_name for x in ["Euphoria", "Thrill"]):
        core_action, core_signal = "HOLD", 0
    elif above_200ma:
        core_action = "BUY dips"
        core_signal = round(min(300, core_total * 0.20) * vol_mult)
    else:
        core_action = "HOLD (below 200MA)"
        core_signal = round(min(150, core_total * 0.10) * vol_mult)

    if any(x in stage_name for x in ["Panic", "Capitulation", "Anger"]) or stage_idx in [8, 9, 10]:
        tactical_action = "BUY"
        base = {"Slow": 300, "Medium": 600}.get(speed, 1000)
        tactical_signal = round(min(base * vol_mult, tactical_total))
    else:
        tactical_action, tactical_signal = "WAIT", 0

    return core_action, core_signal, tactical_action, tactical_signal, core_total, tactical_total

def action_now(r: dict):
    if r["tactical_action"] == "BUY":
        return "TACTICAL BUY", R_FG, R_BG
    if "BUY" in r["core_action"]:
        return "BUY DIPS", G_FG, G_BG
    if "HOLD" in r["core_action"] and r["near_trigger"]:
        return "HOLD / WAIT", A_FG, A_BG
    return "WAIT", B_FG, B_BG

# =========================
# ANALYSIS
# =========================
def analyze_symbol(symbol: str, vol_mult: float):
    df_1h = fetch_series(symbol, "1h", 420)
    df_4h = resample_ohlc(df_1h, "4h")
    if len(df_4h) < 60:
        raise RuntimeError(f"{symbol}: not enough 4H bars")

    df_1d = fetch_series(symbol, "1day", 320)

    close = float(df_4h["c"].iloc[-1])

    rsi_series = rsi_wilder(df_4h["c"], 14)
    rsi_v = float(rsi_series.iloc[-1])

    ml, sl, hist = macd_calc(df_4h["c"])
    hist_rising = float(hist.iloc[-1]) > float(hist.iloc[-2]) if len(hist) >= 2 else False
    macd_dir = "Bullish" if float(ml.iloc[-1]) > float(sl.iloc[-1]) else "Bearish"
    macd_text = f"{macd_dir} ({'hist rising' if hist_rising else 'hist falling'})"

    ma200_series = df_1d["c"].rolling(200).mean()
    ma200 = float(ma200_series.iloc[-1])
    above = close > ma200

    h52 = float(df_1d["h"].tail(252).max())
    pct_h = round(((close - h52) / h52) * 100, 1)

    sup = round(float(df_4h["l"].tail(60).min()), 2)
    res = round(float(df_4h["h"].tail(60).max()), 2)

    stage_name, stage_idx = classify_cycle(rsi_v, pct_h, above, hist_rising)
    levels = compute_levels(close, h52)
    speed = buy_speed(pct_h)
    core_action, core_signal, tactical_action, tactical_signal, core_total, tactical_total = determine_actions(
        stage_name, stage_idx, above, speed, vol_mult
    )

    already_deployed = get_deployed(symbol)
    signal_today = core_signal + tactical_signal
    available_before_signal = max(0, CAPITAL_PER_TICKER - already_deployed)
    available_after_signal = max(0, available_before_signal - signal_today)

    return {
        "symbol": symbol,
        "close": close,
        "high52w": h52,
        "pct_from_high": pct_h,
        "rsi": round(rsi_v, 1),
        "macd_text": macd_text,
        "ma200": round(ma200, 2),
        "above_200ma": above,
        "sup": sup,
        "res": res,
        "lvl20": round(h52 * 0.80, 2),
        "lvl30": round(h52 * 0.70, 2),
        "lvl40": round(h52 * 0.60, 2),
        "stage_name": stage_name,
        "stage_idx": stage_idx,
        "levels": levels,
        "buy_speed": speed,
        "core_action": core_action,
        "core_signal": core_signal,
        "core_total": core_total,
        "tactical_action": tactical_action,
        "tactical_signal": tactical_signal,
        "tactical_total": tactical_total,
        "signal_today": signal_today,
        "near_trigger": levels["dist_to_panic"] > -15,
        "already_deployed": already_deployed,
        "available_before_signal": available_before_signal,
        "available_after_signal": available_after_signal,
    }

# =========================
# DRAW HELPERS
# =========================
def rbox(ax, x, y, w, h, r=0.012, fc=CARD, ec=BORDER, lw=0.7, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
            transform=ax.transAxes, zorder=z, clip_on=False,
        )
    )

def txt(ax, x, y, s, sz=10, c=T1, ha="left", va="center", bold=False, z=5):
    ax.text(
        x, y, str(s),
        fontsize=sz, color=c, ha=ha, va=va,
        fontweight="bold" if bold else "normal",
        transform=ax.transAxes, zorder=z, clip_on=False,
    )

def badge(ax, x, y, text, bg, fg, sz=6.3, ha="center", z=7):
    ax.text(
        x, y, str(text),
        fontsize=sz, color=fg, ha=ha, va="center",
        fontweight="bold", transform=ax.transAxes, zorder=z,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=bg, edgecolor="none"),
        clip_on=False,
    )

def hline(ax, x0, x1, y, color=BORDER_SOFT, lw=0.6):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, transform=ax.transAxes, zorder=3, clip_on=False)

def progress(ax, x, y, w, h, pct, fill, bg=INNER, z=5):
    rbox(ax, x, y, w, h, r=min(h / 2, 0.004), fc=bg, ec="none", lw=0, z=z)
    if pct > 0:
        rbox(ax, x, y, w * max(0, min(1, pct / 100.0)), h, r=min(h / 2, 0.004), fc=fill, ec="none", lw=0, z=z + 1)

def metric_box(ax, x, y, w, h, title, value, sub="", value_color=T1):
    rbox(ax, x, y, w, h, r=0.008, fc=INNER2, ec="none", lw=0)
    txt(ax, x + 0.006, y + h - 0.014, title, sz=5.8, c=T3)
    txt(ax, x + 0.006, y + h - 0.032, value, sz=8.2, c=value_color, bold=True)
    if sub:
        txt(ax, x + 0.006, y + 0.008, sub, sz=5.4, c=T3, va="bottom")

def zone_row(ax, x, y, w, label, value, color):
    txt(ax, x, y, label, sz=6.2, c=T2)
    txt(ax, x + w, y, value, sz=6.5, c=color, ha="right", bold=True)

def confidence_dots(ax, x, y, n_on=7, n_total=10):
    for i in range(n_total):
        c = TEAL if i < n_on else "#31353d"
        ax.add_patch(Circle((x + i * 0.013, y), 0.0044, color=c, transform=ax.transAxes, zorder=6, clip_on=False))

# =========================
# TICKER CARD
# =========================
def draw_ticker_card(ax, x, y, w, h, r):
    stage_bg, stage_fg = STAGE_COLORS[r["stage_idx"]]
    border_col = AMBER2 if r["near_trigger"] else BORDER
    border_lw = 1.1 if r["near_trigger"] else 0.8
    rbox(ax, x, y, w, h, r=0.014, fc=CARD, ec=border_col, lw=border_lw, z=1)

    pad = 0.015
    left = x + pad
    right = x + w - pad
    row = y + h - 0.020

    # Header
    txt(ax, left, row, r["symbol"], sz=14, bold=True)
    badge(ax, right - 0.005, row, r["stage_name"], stage_bg, stage_fg, sz=5.8, ha="right")
    row -= 0.031

    txt(ax, left, row, f"USD{r['close']:.2f}", sz=12.8, bold=True)
    txt(ax, right, row, f"{r['pct_from_high']:.1f}% vs {r['high52w']:.2f}", sz=6.2, c=T2, ha="right")
    row -= 0.028

    # Status badges
    if r["above_200ma"]:
        badge(ax, left + 0.030, row, "↑ Above 200MA", G_BG, G_FG, sz=5.9)
    else:
        badge(ax, left + 0.030, row, "↓ Below 200MA", R_BG, R_FG, sz=5.9)
    if r["near_trigger"]:
        badge(ax, left + 0.145, row, "⚠ Near panic", A_BG, A_FG, sz=5.9)
    row -= 0.024

    # Action row
    action_text, action_fg, action_bg = action_now(r)
    rbox(ax, left, row - 0.030, w - 2 * pad, 0.032, r=0.010, fc=action_bg, ec="none", lw=0)
    txt(ax, x + w / 2, row - 0.014, action_text, sz=10.6, c=action_fg, ha="center", bold=True)
    row -= 0.046

    # Core / tactical
    bw = (w - 2 * pad - 0.010) / 2
    metric_box(ax, left, row - 0.046, bw, 0.048, f"Core £{int(r['core_total'])}", f"£{int(r['core_signal'])}", r["core_action"], value_color=T1)
    metric_box(ax, left + bw + 0.010, row - 0.046, bw, 0.048, f"Tactical £{int(r['tactical_total'])}", f"£{int(r['tactical_signal'])}", f"{r['tactical_action']} · {r['buy_speed']}", value_color=T1)
    progress(ax, left + 0.008, row - 0.041, bw - 0.016, 0.005, (r["core_signal"] / max(1, r["core_total"])) * 100, GREEN2)
    progress(ax, left + bw + 0.018, row - 0.041, bw - 0.016, 0.005, (r["tactical_signal"] / max(1, r["tactical_total"])) * 100, B_FG)
    row -= 0.058

    # Indicators
    small_w = (w - 2 * pad - 0.018) / 4
    ind_h = 0.043
    ind_y = row - ind_h
    indicators = [
        ("RSI", f"{r['rsi']}", ""),
        ("MACD", "Bullish" if "Bullish" in r["macd_text"] else "Bearish",
         "rising" if "rising" in r["macd_text"] else "falling"),
        ("200MA", f"USD{r['ma200']:.2f}", ""),
        ("S/R", f"{r['sup']}/{r['res']}", ""),
    ]
    for i, (title, value, sub) in enumerate(indicators):
        metric_box(ax, left + i * (small_w + 0.006), ind_y, small_w, ind_h, title, value, sub, value_color=T1)
    row -= 0.054

    rbox(ax, left, row - 0.020, w - 2 * pad, 0.021, r=0.006, fc=INNER2, ec="none", lw=0)
    txt(ax, left + 0.006, row - 0.010, f"52W: -20% {r['lvl20']} · -30% {r['lvl30']} · -40% {r['lvl40']}", sz=5.8, c=T2)
    row -= 0.031

    hline(ax, left, right, row)
    row -= 0.014

    txt(ax, left, row, "PSYCHOLOGY PRICE MAP", sz=6.2, c=T3, bold=True)
    row -= 0.017

    zone_row(ax, left, row, w - 2 * pad, "Add zone", f"USD{r['levels']['entry']:.2f}", T1)
    row -= 0.018
    zone_row(ax, left, row, w - 2 * pad, "Panic zone", f"USD{r['levels']['panic_low']:.2f} – {r['levels']['panic_high']:.2f}", R_FG)
    row -= 0.018
    zone_row(ax, left, row, w - 2 * pad, "Capitulation zone", f"USD{r['levels']['cap_low']:.2f} – {r['levels']['cap_high']:.2f}", R_FG)
    row -= 0.018
    zone_row(ax, left, row, w - 2 * pad, "First trim zone", f"USD{r['levels']['first_trim_low']:.2f} – {r['levels']['first_trim_high']:.2f}", G_FG)
    row -= 0.018
    zone_row(ax, left, row, w - 2 * pad, "Strong trim zone", f"USD{r['levels']['strong_trim_low']:.2f} – {r['levels']['strong_trim_high']:.2f}", G_FG)
    row -= 0.018
    zone_row(ax, left, row, w - 2 * pad, "Dist to panic", f"{r['levels']['dist_to_panic']:.1f}%", A_FG if r["near_trigger"] else T2)

# =========================
# PAGE RENDERING
# =========================
def build_dashboard_page(page_results, all_results, vol_sym, vol_val, vol_rname, vol_mult, ts, page_num, total_pages):
    W = 16.0
    H = 10.8

    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    M = 0.030
    usable_w = 1 - 2 * M
    cursor_y = 0.982

    # Header
    txt(ax, M, cursor_y, "MARKET CYCLE DASHBOARD", sz=16, bold=True)
    header_right = f"{ts}   ·   Page {page_num}/{total_pages}" if total_pages > 1 else ts
    txt(ax, 1 - M, cursor_y, header_right, sz=8.0, c=T3, ha="right")
    cursor_y -= 0.038

    # Top strip
    strip_h = 0.075
    rbox(ax, M, cursor_y - strip_h, usable_w, strip_h, r=0.012, fc=CARD, ec=BORDER, lw=0.8)
    strip = [
        (vol_sym, f"{vol_val:.1f}" if vol_val is not None else "N/A", vol_color(vol_rname)),
        ("Regime", vol_rname, vol_color(vol_rname)),
        ("Tactical ×", f"{vol_mult:.2f}×", T1),
        ("Capital", f"£{int(CAPITAL_PER_TICKER)} / ticker", T1),
        ("Structure", "50% Core · 50% Tactical", T1),
    ]
    sw = usable_w / len(strip)
    for i, (label, value, c) in enumerate(strip):
        sx = M + i * sw + 0.014
        txt(ax, sx, cursor_y - 0.021, label, sz=6.2, c=T3)
        txt(ax, sx, cursor_y - 0.050, value, sz=10.0, c=c, bold=True)
    cursor_y -= strip_h + 0.026

    # Cycle strip
    txt(ax, M, cursor_y, "Market cycle position", sz=6.7, c=T3)
    cursor_y -= 0.018

    seg_h = 0.044
    seg_w = usable_w / len(STAGE_NAMES)
    active_idxs = {r["stage_idx"] for r in all_results}
    for i, (name, (bg, fg)) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
        sx = M + i * seg_w
        active = i in active_idxs
        rbox(ax, sx, cursor_y - seg_h, seg_w - 0.001, seg_h, r=0.004, fc=bg, ec="#ffffff" if active else BORDER, lw=1.3 if active else 0.5)
        txt(ax, sx + (seg_w - 0.001) / 2, cursor_y - seg_h / 2, name, sz=5.9, c=fg, ha="center", bold=active)
    cursor_y -= seg_h + 0.028

    # Card grid: fixed 3 x 2
    cols = CARDS_PER_ROW
    rows = ROWS_PER_PAGE
    col_gap = 0.014
    row_gap = 0.028

    card_w = (usable_w - col_gap * (cols - 1)) / cols
    footer_reserved = 0.135
    card_area_top = cursor_y
    card_area_bottom = footer_reserved + 0.035
    total_card_area_h = card_area_top - card_area_bottom
    card_h = (total_card_area_h - row_gap * (rows - 1)) / rows

    for idx, r in enumerate(page_results):
        row = idx // cols
        col = idx % cols
        cx = M + col * (card_w + col_gap)
        cy = card_area_top - (row + 1) * card_h - row * row_gap
        draw_ticker_card(ax, cx, cy, card_w, card_h, r)

    # Footer summary (global across all tickers)
    cursor_y = footer_reserved

    summary_h = 0.095
    rbox(ax, M, cursor_y - summary_h, usable_w, summary_h, r=0.012, fc=CARD, ec=BORDER, lw=0.8)
    txt(ax, M + 0.012, cursor_y - 0.020, "PORTFOLIO SUMMARY", sz=8.0, c=T3)

    total_core = sum(r["core_signal"] for r in all_results)
    total_tactical = sum(r["tactical_signal"] for r in all_results)
    total_capacity = len(all_results) * int(CAPITAL_PER_TICKER)
    tactical_triggers = [r["symbol"] for r in all_results if r["tactical_action"] == "BUY"]

    items = [
        ("Core signals today", f"£{int(total_core)}"),
        ("Tactical signals today", f"£{int(total_tactical)}"),
        ("Total capacity", f"£{int(total_capacity)}"),
        ("Tactical triggers", ", ".join(tactical_triggers) if tactical_triggers else "NONE — WAIT"),
    ]
    sw2 = usable_w / len(items)
    base_y = cursor_y - 0.050
    for i, (label, value) in enumerate(items):
        sx = M + i * sw2 + 0.012
        txt(ax, sx, base_y + 0.017, label, sz=6.2, c=T3)
        txt(ax, sx, base_y - 0.002, value, sz=9.8, c=(G_FG if i == 3 and tactical_triggers else (A_FG if i == 3 else T1)), bold=True)

    confidence_dots(ax, M + 0.016, cursor_y - summary_h + 0.020, n_on=7, n_total=10)
    txt(ax, M + 0.155, cursor_y - summary_h + 0.020, "7/10 confidence", sz=6.5, c=T2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=260, facecolor=BG, bbox_inches=None, pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return buf

def chunk_results(results, size=MAX_CARDS_PER_PAGE):
    for i in range(0, len(results), size):
        yield results[i:i + size]

# =========================
# TELEGRAM
# =========================
def send_dashboard(img_buf, caption, filename="market_cycle_dashboard.png"):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption[:950],
            "disable_web_page_preview": True,
        },
        files={"document": (filename, img_buf.getvalue(), "image/png")},
        timeout=90,
    )
    print("sendDocument status:", r.status_code)
    print("sendDocument response:", r.text[:800])
    r.raise_for_status()

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)]
    for chunk in chunks:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "disable_web_page_preview": True},
            timeout=30,
        )
        print("sendMessage status:", r.status_code)
        print("sendMessage response:", r.text[:800])
        r.raise_for_status()
        time.sleep(0.25)

# =========================
# MAIN
# =========================
def main():
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_CHAT_ID")

    ts = utc_now_str()
    vol_sym, vol_val = get_vol_value()
    vol_rname, vol_mult = vol_regime(vol_val)

    results = []
    failures = []
    for sym in SYMBOLS:
        try:
            results.append(analyze_symbol(sym, vol_mult))
        except Exception as e:
            failures.append(f"{sym}: {repr(e)}")

    if not results:
        raise RuntimeError("All symbols failed:\n" + "\n".join(failures))

    pages = list(chunk_results(results, MAX_CARDS_PER_PAGE))
    total_pages = len(pages)
    tactical_triggers = [r["symbol"] for r in results if r["tactical_action"] == "BUY"]

    for idx, page_results in enumerate(pages, start=1):
        img_buf = build_dashboard_page(
            page_results=page_results,
            all_results=results,
            vol_sym=vol_sym,
            vol_val=vol_val,
            vol_rname=vol_rname,
            vol_mult=vol_mult,
            ts=ts,
            page_num=idx,
            total_pages=total_pages,
        )

        page_symbols = ", ".join([r["symbol"] for r in page_results])
        caption_lines = [
            f"📊 Market Cycle Dashboard — {ts}",
            f"{vol_sym}: {f'{vol_val:.1f}' if vol_val is not None else 'N/A'} | {vol_rname} | ×{vol_mult:.2f}",
            f"Page {idx}/{total_pages} | {page_symbols}",
            ("🚨 Tactical BUY: " + ", ".join(tactical_triggers)) if tactical_triggers else "No tactical triggers",
        ]
        for r in page_results:
            caption_lines.append(
                f"{r['symbol']}: {r['stage_name']} | Action {action_now(r)[0]} | Core £{int(r['core_signal'])} | Tactical £{int(r['tactical_signal'])}"
            )

        send_dashboard(
            img_buf,
            "\n".join(caption_lines),
            filename=f"market_cycle_dashboard_p{idx}.png",
        )
        time.sleep(0.4)

    if failures:
        send_message("⚠️ Cycle bot failures:\n" + "\n".join(failures))

    print(f"Sent dashboard pages: {total_pages}. Tactical triggers: {tactical_triggers}")

if __name__ == "__main__":
    main()
