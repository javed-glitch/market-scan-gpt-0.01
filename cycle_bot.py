import os, time, io, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# =========================
# CONFIG
# =========================
SYMBOLS            = [s.strip().upper() for s in os.getenv("SYMBOLS", "TSLA,NVDA,PLTR").split(",") if s.strip()]
CAPITAL_PER_TICKER = float(os.getenv("CAPITAL_PER_TICKER", "3000"))
TD_API_KEY         = os.getenv("TWELVEDATA_API_KEY", "").strip()
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
BASE_URL           = "https://api.twelvedata.com/time_series"
TD_MIN_SECONDS     = float(os.getenv("TD_MIN_SECONDS_BETWEEN_CALLS", "11.0"))
_last_td_call      = 0.0

# Assumed already-deployed amounts (would come from a state file in production)
def get_deployed(sym):
    return float(os.getenv(f"DEPLOYED_{sym}", "0"))

# =========================
# COLOURS
# =========================
BG       = "#0f0f0f"
CARD     = "#1a1a1a"
INNER    = "#111111"
T1       = "#e8e8e8"
T2       = "#888888"
T3       = "#444444"
BORDER   = "#2a2a2a"
G_FG     = "#7dc87d"; G_BG = "#1a2a1a"
R_FG     = "#f08080"; R_BG = "#2a1a1a"
A_FG     = "#d4a017"; A_BG = "#2a2000"
B_FG     = "#7a9fd4"; B_BG = "#1a1a2a"
TEAL     = "#1d9e75"
ORANGE   = "#EF9F27"

STAGE_COLORS = [
    ("#9FE1CB", "#085041"), ("#5DCAA5", "#04342C"), ("#3266ad", "#e6f1fb"),
    ("#185FA5", "#e6f1fb"), ("#7F77DD", "#EEEDFE"), ("#AFA9EC", "#26215C"),
    ("#FAC775", "#412402"), ("#EF9F27", "#3a2000"), ("#F0997B", "#4A1B0C"),
    ("#D85A30", "#FAECE7"), ("#E24B4A", "#FCEBEB"), ("#A32D2D", "#FCEBEB"),
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
def fetch_series(symbol, interval, outputsize):
    td_throttle()
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "apikey": TD_API_KEY,
        "format": "JSON"
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

def resample_ohlc(df, rule):
    return df.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()

# =========================
# INDICATORS
# =========================
def rsi_wilder(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def macd_calc(close, fast=12, slow=26, signal=9):
    ml = ema(close, fast) - ema(close, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl

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
    return {"Low": G_FG, "Normal": A_FG, "High": R_FG, "Extreme": R_FG}.get(name, T2)

# =========================
# CYCLE CLASSIFIER
# =========================
def classify_cycle(rsi_val, pct_from_high, above_200ma, hist_rising):
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
    else:
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
# LEVELS + ACTIONS
# =========================
def compute_levels(close, h52):
    return {
        "entry": round(close * 0.97, 2),
        "panic_low": round(h52 * 0.65, 2),
        "panic_high": round(h52 * 0.75, 2),
        "cap_low": round(h52 * 0.50, 2),
        "cap_high": round(h52 * 0.60, 2),
        "target_low": round(close * 1.08, 2),
        "target_high": round(min(h52 * 0.96, close * 1.15), 2),
        "dist_to_panic": round(((h52 * 0.75 - close) / close) * 100, 1),
    }

def buy_speed(pct):
    p = abs(pct)
    if p < 20:
        return "Slow"
    if p < 30:
        return "Medium"
    if p < 50:
        return "Aggressive"
    return "Max"

def determine_actions(stage_name, stage_idx, above_200ma, speed, vol_mult):
    ct = int(CAPITAL_PER_TICKER * 0.50)
    tt = int(CAPITAL_PER_TICKER * 0.50)

    if any(x in stage_name for x in ["Euphoria", "Thrill"]):
        ca, cs = "HOLD", 0
    elif above_200ma:
        ca = "BUY dips"
        cs = round(min(300, ct * 0.20) * vol_mult)
    else:
        ca = "HOLD (below 200MA)"
        cs = round(min(150, ct * 0.10) * vol_mult)

    if any(x in stage_name for x in ["Panic", "Capitulation", "Anger"]) or stage_idx in [8, 9, 10]:
        ta = "BUY"
        base = {"Slow": 300, "Medium": 600}.get(speed, 1000)
        ts = round(min(base * vol_mult, tt))
    else:
        ta, ts = "WAIT", 0

    return ca, cs, ta, ts, ct, tt

# =========================
# ANALYZE SYMBOL
# =========================
def analyze_symbol(symbol, vol_mult):
    df_1h = fetch_series(symbol, "1h", 420)
    df_4h = resample_ohlc(df_1h, "4h")
    if len(df_4h) < 60:
        raise RuntimeError(f"{symbol}: not enough 4H bars")

    df_1d = fetch_series(symbol, "1day", 320)

    close = float(df_4h["c"].iloc[-1])
    rsi_v = float(rsi_wilder(df_4h["c"], 14).iloc[-1])
    ml, sl, hist = macd_calc(df_4h["c"])
    hr = float(hist.iloc[-1]) > float(hist.iloc[-2]) if len(hist) >= 2 else False
    macd_dir = "Bullish" if float(ml.iloc[-1]) > float(sl.iloc[-1]) else "Bearish"
    macd_text = f"{macd_dir} ({'rising' if hr else 'falling'})"

    ma200 = float(df_1d["c"].rolling(200).mean().iloc[-1])
    above = close > ma200
    h52 = float(df_1d["h"].tail(252).max())
    pct_h = round(((close - h52) / h52) * 100, 1)
    sup = round(float(df_4h["l"].tail(60).min()), 2)
    res = round(float(df_4h["h"].tail(60).max()), 2)

    sn, si = classify_cycle(rsi_v, pct_h, above, hr)
    levels = compute_levels(close, h52)
    speed = buy_speed(pct_h)
    ca, cs, ta, ts, ct, tt = determine_actions(sn, si, above, speed, vol_mult)
    already_deployed = get_deployed(symbol)

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
        "stage_name": sn,
        "stage_idx": si,
        "levels": levels,
        "buy_speed": speed,
        "core_action": ca,
        "core_signal": cs,
        "core_total": ct,
        "tactical_action": ta,
        "tactical_signal": ts,
        "tactical_total": tt,
        "near_trigger": levels["dist_to_panic"] > -15,
        "already_deployed": already_deployed,
        "available": max(0, CAPITAL_PER_TICKER - already_deployed),
    }

# =========================
# DRAWING HELPERS
# =========================
def rbox(ax, x, y, w, h, r=0.008, fc=CARD, ec=BORDER, lw=0.5, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
        transform=ax.transAxes,
        clip_on=False
    ))

def pbar_draw(ax, x, y, w, h, pct, color, z=4):
    rbox(ax, x, y, w, h, r=0.003, fc=INNER, ec="none", lw=0, z=z)
    if pct > 0:
        rbox(ax, x, y, w * min(pct / 100, 1), h, r=0.003, fc=color, ec="none", lw=0, z=z + 1)

def txt(ax, x, y, s, sz=8, c=T1, ha="left", va="center", bold=False, z=6):
    ax.text(
        x, y, str(s),
        fontsize=sz,
        color=c,
        ha=ha,
        va=va,
        fontweight="bold" if bold else "normal",
        transform=ax.transAxes,
        zorder=z,
        clip_on=False
    )

def bdg(ax, cx, cy, lbl, bg, fg, sz=6.5, z=6):
    ax.text(
        cx, cy, str(lbl),
        fontsize=sz,
        color=fg,
        ha="center",
        va="center",
        fontweight="bold",
        transform=ax.transAxes,
        zorder=z + 1,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=bg, edgecolor="none")
    )

def hline(ax, x0, x1, y, color=BORDER, lw=0.4):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, transform=ax.transAxes, clip_on=False, zorder=3)

# =========================
# BUILD PORTRAIT IMAGE
# =========================
def build_dashboard(results, vol_sym, vol_val, vol_rname, vol_mult, ts):
    import matplotlib as mpl
    mpl.rcParams["text.usetex"] = False

    n = len(results)

    # Increased width for better text clarity
    W = 8.5
    TICKER_H = 5.8
    HEADER_H = 2.2
    FOOTER_H = 1.8
    H = HEADER_H + n * TICKER_H + FOOTER_H

    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    header_frac = HEADER_H / H
    ticker_frac = TICKER_H / H
    footer_frac = FOOTER_H / H

    M = 0.035
    W2 = 1 - 2 * M
    y = 0.995

    # HEADER
    txt(ax, M, y, "MARKET CYCLE DASHBOARD", sz=13, bold=True)
    txt(ax, 1 - M, y, ts, sz=7.5, c=T3, ha="right")
    y -= 0.014

    sh = header_frac * 0.38
    rbox(ax, M, y - sh, W2, sh, fc=CARD, ec=BORDER)
    vol_str = f"{vol_val:.1f}" if vol_val else "N/A"
    vox = [
        (vol_sym, "Value", vol_str, vol_color(vol_rname)),
        ("Regime", "State", vol_rname, vol_color(vol_rname)),
        ("Mult", "Scale", f"{vol_mult:.2f}×", T1),
        ("Capital", "Per ticker", f"£{int(CAPITAL_PER_TICKER)}", T1),
        ("Split", "Core/Tact", "50% Core · 50%", T2),
    ]
    xi = M + 0.012
    sw = W2 / len(vox)
    for lbl, sub, val, col in vox:
        txt(ax, xi, y - sh * 0.22, lbl, sz=7.2, c=T3)
        txt(ax, xi, y - sh * 0.48, sub, sz=6.4, c=T3)
        txt(ax, xi, y - sh * 0.73, val, sz=9.0, c=col, bold=True)
        xi += sw
    y -= sh + 0.015

    txt(ax, M, y, "Cycle position", sz=7, c=T3)
    y -= 0.011
    bh = header_frac * 0.28
    seg_w = W2 / len(STAGE_NAMES)
    active = {r["stage_idx"] for r in results}
    for i, (sn, (bg, fg)) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
        sx = M + i * seg_w
        lw = 2.0 if i in active else 0.4
        ec = "#ffffff" if i in active else BORDER
        rbox(ax, sx, y - bh, seg_w, bh, r=0.003, fc=bg, ec=ec, lw=lw, z=3)
        ax.text(
            sx + seg_w / 2,
            y - bh / 2,
            sn,
            fontsize=6.0,
            color=fg,
            ha="center",
            va="center",
            fontweight="bold" if i in active else "normal",
            transform=ax.transAxes,
            clip_on=True,
            zorder=4
        )
    y -= bh + 0.018

    # TICKER CARDS
    for r in results:
        sc_bg, sc_fg = STAGE_COLORS[r["stage_idx"]]
        bc = "#8a5a00" if r["near_trigger"] else BORDER
        blw = 1.2 if r["near_trigger"] else 0.5
        ch = ticker_frac * 0.96
        cy = y - ch
        rbox(ax, M, cy, W2, ch, r=0.010, fc=CARD, ec=bc, lw=blw)

        lx = M + 0.014
        rx = M + W2 - 0.014
        row = cy + ch - 0.016

        txt(ax, lx, row, r["symbol"], sz=16, bold=True)
        txt(ax, lx + 0.13, row, f"USD{r['close']:.2f}", sz=13, bold=True, c=T1)
        txt(ax, rx, row, f"{r['pct_from_high']:.1f}% from 52W high", sz=8, c=T2, ha="right")
        row -= 0.024

        bdg(ax, lx + 0.050, row, r["stage_name"], sc_bg + "66", sc_bg, sz=8.0)
        tbg = G_BG if r["above_200ma"] else R_BG
        tfg = G_FG if r["above_200ma"] else R_FG
        bdg(ax, lx + 0.185, row, ("↑ Above" if r["above_200ma"] else "↓ Below") + " 200MA", tbg, tfg, sz=7.8)
        if r["near_trigger"]:
            bdg(ax, lx + 0.330, row, "⚠ Near panic", A_BG, A_FG, sz=7.8)
        row -= 0.021

        hline(ax, lx, rx, row)
        row -= 0.011

        ind = [
            ("RSI(14)", str(r["rsi"])),
            ("MACD", r["macd_text"]),
            ("200MA", f"USD{r['ma200']:.2f}"),
            ("S / R", f"USD{r['sup']} / USD{r['res']}")
        ]
        iw = W2 * 0.24
        ix = lx
        for lbl, val in ind:
            rbox(ax, ix, row - 0.036, iw - 0.008, 0.040, r=0.005, fc=INNER, ec="none", lw=0)
            txt(ax, ix + 0.006, row - 0.010, lbl, sz=7.0, c=T3)
            txt(ax, ix + 0.006, row - 0.026, val, sz=8.0, bold=True)
            ix += W2 * 0.245
        row -= 0.049

        rbox(ax, lx, row - 0.022, W2 - 0.028, 0.026, r=0.004, fc=INNER, ec="none", lw=0)
        txt(
            ax, lx + 0.008, row - 0.010,
            f"52W {r['high52w']:.2f}  ·  -20pct {r['lvl20']}  ·  -30pct {r['lvl30']}  ·  -40pct {r['lvl40']}",
            sz=7.2, c=T2
        )
        row -= 0.034

        hline(ax, lx, rx, row)
        row -= 0.011

        txt(ax, lx, row, "CAPITAL POSITION", sz=7, c=T3, bold=True)
        row -= 0.017

        bw = (W2 - 0.028) / 3
        cards = [
            (
                "Already deployed",
                f"£{int(r['already_deployed'])}",
                "Currently in position",
                int(r["already_deployed"] / CAPITAL_PER_TICKER * 100),
                TEAL
            ),
            (
                "Signal today",
                f"£{r['core_signal'] + r['tactical_signal']}",
                f"Core £{r['core_signal']} + Tact £{r['tactical_signal']}",
                int((r["core_signal"] + r["tactical_signal"]) / CAPITAL_PER_TICKER * 100),
                ORANGE
            ),
            (
                "Still available",
                f"£{int(max(0, r['available'] - (r['core_signal'] + r['tactical_signal'])))}",
                "Undeployed capital",
                int(max(0, r["available"] - r["core_signal"] - r["tactical_signal"]) / CAPITAL_PER_TICKER * 100),
                B_FG
            ),
        ]

        for i, (lbl, val, sub, bar_pct, bar_col) in enumerate(cards):
            bx = lx + i * (bw + 0.006)
            rbox(ax, bx, row - 0.064, bw, 0.066, r=0.006, fc=INNER, ec="none", lw=0)
            txt(ax, bx + 0.006, row - 0.014, lbl, sz=7.0, c=T3)
            txt(ax, bx + 0.006, row - 0.031, val, sz=11.0, bold=True, c=T1)
            txt(ax, bx + 0.006, row - 0.046, sub, sz=6.2, c=T3)
            pbar_draw(ax, bx + 0.006, row - 0.058, bw - 0.012, 0.007, bar_pct, bar_col)
        row -= 0.078

        txt(ax, lx, row, "Core:", sz=7.5, c=T2)
        ca_bg = G_BG if "BUY" in r["core_action"] else A_BG
        ca_fg = G_FG if "BUY" in r["core_action"] else A_FG
        bdg(ax, lx + 0.076, row, r["core_action"], ca_bg, ca_fg, sz=7.5)

        txt(ax, lx + 0.24, row, "Tactical:", sz=7.5, c=T2)
        ta_bg = G_BG if r["tactical_action"] == "BUY" else B_BG
        ta_fg = G_FG if r["tactical_action"] == "BUY" else B_FG
        bdg(ax, lx + 0.355, row, f"{r['tactical_action']} · {r['buy_speed']}", ta_bg, ta_fg, sz=7.5)
        row -= 0.020

        hline(ax, lx, rx, row)
        row -= 0.011

        txt(ax, lx, row, "PRICE ZONES", sz=7, c=T3, bold=True)
        row -= 0.017

        zones = [
            ("Entry zone", f"USD{r['levels']['entry']}", T1),
            ("Panic zone", f"USD{r['levels']['panic_low']} – USD{r['levels']['panic_high']}", R_FG),
            ("Capitulation", f"USD{r['levels']['cap_low']} – USD{r['levels']['cap_high']}", R_FG),
            ("Target zone", f"USD{r['levels']['target_low']} – USD{r['levels']['target_high']}", G_FG),
            ("Dist to panic", f"{r['levels']['dist_to_panic']:.1f}%", R_FG if r['levels']['dist_to_panic'] > -20 else T2),
        ]

        mid = len(zones) // 2 + len(zones) % 2
        for col_i, zone_slice in enumerate([zones[:mid], zones[mid:]]):
            zx = lx if col_i == 0 else lx + W2 * 0.50
            zrow = row
            for lbl, val, vc in zone_slice:
                rbox(ax, zx, zrow - 0.022, W2 * 0.46, 0.025, r=0.004, fc=INNER, ec="none", lw=0)
                txt(ax, zx + 0.006, zrow - 0.010, lbl, sz=7.0, c=T2)
                txt(ax, zx + W2 * 0.44, zrow - 0.010, val, sz=7.4, c=vc, ha="right", bold=True)
                zrow -= 0.027

        y -= ch + 0.013

    # FOOTER
    rbox(ax, M, y - footer_frac * 0.92, W2, footer_frac * 0.92, r=0.010, fc=CARD, ec=BORDER)

    fy = y - footer_frac * 0.08
    txt(ax, M + 0.014, fy, "VUAG — Reserve (profit proceeds only)", sz=8, c=T2, bold=True)
    fy -= 0.020

    vuag_bw = W2 / 4
    vuag_data = [
        ("Price", "£95.20", ""),
        ("Profits banked", "£800", "Realised from cycles"),
        ("VUAG allocated", "£320", "50% cap = £400 max"),
        ("Cap remaining", "£80", "Before limit hit"),
    ]
    vxi = M + 0.014
    for lbl, val, sub in vuag_data:
        rbox(ax, vxi, fy - 0.058, vuag_bw - 0.010, 0.061, r=0.005, fc=INNER, ec="none", lw=0)
        txt(ax, vxi + 0.005, fy - 0.014, lbl, sz=7.0, c=T3)
        txt(ax, vxi + 0.005, fy - 0.030, val, sz=10.0, bold=True)
        if sub:
            txt(ax, vxi + 0.005, fy - 0.045, sub, sz=6.0, c=T3)
        pbar_draw(ax, vxi + 0.005, fy - 0.054, vuag_bw - 0.020, 0.006, 80, TEAL)
        vxi += vuag_bw

    txt(
        ax, M + 0.014, fy - 0.068,
        "Add from profit proceeds only · Cap 50% of profits · Hold if above 200MA · Never sell to fund tactical trades",
        sz=6.4, c=T3
    )
    fy -= 0.086

    hline(ax, M + 0.014, M + W2 - 0.014, fy)
    fy -= 0.013

    txt(ax, M + 0.014, fy, "PORTFOLIO SUMMARY", sz=7.5, c=T2, bold=True)
    fy -= 0.020

    total_already = sum(r["already_deployed"] for r in results)
    total_signal = sum(r["core_signal"] + r["tactical_signal"] for r in results)
    total_available = sum(r["available"] for r in results)
    triggers = [r["symbol"] for r in results if r["tactical_action"] == "BUY"]
    cap = len(results) * int(CAPITAL_PER_TICKER)

    s_items = [
        ("Already in positions", f"£{int(total_already)}", "Currently deployed capital", TEAL),
        ("Signal today", f"£{int(total_signal)}", "Suggested to deploy now", ORANGE),
        ("Available capital", f"£{int(total_available)}", "Undeployed, ready to use", B_FG),
        ("Total capacity", f"£{cap}", f"{len(results)} tickers × £{int(CAPITAL_PER_TICKER)}", T2),
    ]
    sw2 = W2 / len(s_items)
    sxi = M + 0.014
    for lbl, val, sub, col in s_items:
        txt(ax, sxi, fy, lbl, sz=6.8, c=T3)
        txt(ax, sxi, fy - 0.016, val, sz=10.2, c=col, bold=True)
        txt(ax, sxi, fy - 0.029, sub, sz=6.0, c=T3)
        sxi += sw2

    fy -= 0.040
    if triggers:
        bdg(ax, M + 0.08, fy, f"🚨 Tactical BUY: {', '.join(triggers)}", R_BG, R_FG, sz=8)
    else:
        bdg(ax, M + 0.11, fy, "No tactical triggers — market not in Panic/Capitulation", INNER, G_FG, sz=7.2)

    dx = M + 0.014
    dy = fy - 0.022
    for i in range(10):
        c2 = plt.Circle(
            (dx + i * 0.020, dy),
            0.006,
            color=TEAL if i < 7 else "#333333",
            transform=ax.transAxes,
            clip_on=False
        )
        ax.add_patch(c2)
    txt(ax, dx + 10 * 0.020 + 0.010, dy, "7/10 confidence", sz=7.2, c=T3, va="center")

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=260,
        bbox_inches=None,
        facecolor=BG,
        pad_inches=0.05
    )
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# TELEGRAM
# =========================
def send_photo(img_buf, caption):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption[:900],
            "disable_web_page_preview": True
        },
        files={
            "document": ("dashboard.png", img_buf.getvalue(), "image/png")
        },
        timeout=60
    )
    print("sendDocument status:", r.status_code)
    print("sendDocument response:", r.text)
    r.raise_for_status()

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [text[i:i + 3800] for i in range(0, len(text), 3800)]:
        r = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "disable_web_page_preview": True
            },
            timeout=30
        )
        print("sendMessage status:", r.status_code)
        print("sendMessage response:", r.text)
        r.raise_for_status()
        time.sleep(0.3)

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

    results, failures = [], []
    for sym in SYMBOLS:
        try:
            results.append(analyze_symbol(sym, vol_mult))
        except Exception as e:
            failures.append(f"{sym}: {repr(e)}")

    if not results:
        raise RuntimeError("All symbols failed:\n" + "\n".join(failures))

    img_buf = build_dashboard(results, vol_sym, vol_val, vol_rname, vol_mult, ts)
    triggers = [r["symbol"] for r in results if r["tactical_action"] == "BUY"]

    cap_lines = [
        f"📊 Market Cycle Dashboard — {ts}",
        f"{vol_sym}: {f'{vol_val:.1f}' if vol_val else 'N/A'} | {vol_rname} | ×{vol_mult:.2f}",
        ("🚨 Tactical BUY: " + ", ".join(triggers)) if triggers else "No tactical triggers",
    ] + [
        f"{r['symbol']}: {r['stage_name']} | Core {r['core_action']} | Tactical {r['tactical_action']}"
        for r in results
    ]

    send_photo(img_buf, "\n".join(cap_lines))

    if failures:
        send_message("⚠️ Cycle bot failures:\n" + "\n".join(failures))

    print(f"Sent. {ts}. Triggers: {triggers}")

if __name__ == "__main__":
    main()
