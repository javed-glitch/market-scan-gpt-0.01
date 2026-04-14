import io
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
CARDS_PER_ROW = 3
ROWS_PER_PAGE = 2
MAX_CARDS_PER_PAGE = CARDS_PER_ROW * ROWS_PER_PAGE  # 6

_last_td_call = 0.0

YF_SYMBOL_MAP = {"VUSA": "VUSA.L"}

def get_deployed(sym: str) -> float:
    return float(os.getenv(f"DEPLOYED_{sym}", "0"))

# =========================
# COLOURS
# =========================
BG      = "#0b0b0d"
CARD    = "#17181c"
INNER2  = "#0c0d10"
T1      = "#f0f0f2"
T2      = "#a8abb3"
T3      = "#5b606b"
BORDER      = "#2b2f38"
BORDER_SOFT = "#22252c"
GREEN2  = "#16a085"
AMBER2  = "#d68910"
TEAL    = "#1abc9c"
G_FG, G_BG = "#7ee2a8", "#102419"
R_FG, R_BG = "#ff9898", "#261315"
A_FG, A_BG = "#f4c35f", "#2a210d"
B_FG, B_BG = "#9fc7ff", "#101a2b"

STAGE_COLORS = [
    ("#9FE1CB","#085041"),("#5DCAA5","#04342C"),("#3266AD","#E6F1FB"),
    ("#185FA5","#E6F1FB"),("#7F77DD","#EEEDFE"),("#AFA9EC","#26215C"),
    ("#FAC775","#412402"),("#EF9F27","#3A2000"),("#F0997B","#4A1B0C"),
    ("#D85A30","#FAECE7"),("#E24B4A","#FCEBEB"),("#A32D2D","#FCEBEB"),
    ("#888780","#F1EFE8"),
]
STAGE_NAMES = [
    "Hope","Optimism","Belief","Thrill","Euphoria","Complacency",
    "Anxiety","Denial","Panic","Capitulation","Anger","Depression","Disbelief"
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
def fetch_series_twelvedata(symbol, interval, outputsize):
    td_throttle()
    params = {"symbol":symbol,"interval":interval,"outputsize":str(outputsize),"apikey":TD_API_KEY,"format":"JSON"}
    r = requests.get(BASE_URL, params=params, timeout=30)
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        msg = (data.get("message") or "").lower()
        if "run out" in msg or "current minute" in msg:
            time.sleep(65); td_throttle()
            r = requests.get(BASE_URL, params=params, timeout=30)
            data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData: {data.get('message')}")
    values = data.get("values")
    if not values:
        raise RuntimeError(f"No values for {symbol} ({interval})")
    df = pd.DataFrame(values).rename(columns={"datetime":"t","open":"o","high":"h","low":"l","close":"c"})
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o","h","l","c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["t","o","h","l","c"]).sort_values("t").set_index("t")

def _flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in col if str(x) not in ("","nan")) for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _find_col(columns, targets):
    lmap = {str(c).lower(): c for c in columns}
    for t in targets:
        if t.lower() in lmap: return lmap[t.lower()]
    for c in columns:
        cl = str(c).lower()
        for t in targets:
            if t.lower() in cl: return c
    return None

def _normalize_yf(df):
    if df is None or df.empty: raise RuntimeError("Yahoo Finance returned no data")
    df = _flatten_yf_columns(df).reset_index()
    cols = list(df.columns)
    t_col = _find_col(cols, ["Datetime","Date","index"])
    o_col = _find_col(cols, ["Open"])
    h_col = _find_col(cols, ["High"])
    l_col = _find_col(cols, ["Low"])
    c_col = _find_col(cols, ["Close","Adj Close"])
    if not all([t_col, o_col, h_col, l_col, c_col]):
        raise RuntimeError(f"Yahoo Finance missing columns. Got: {cols}")
    df = df.rename(columns={t_col:"t",o_col:"o",h_col:"h",l_col:"l",c_col:"c"})
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o","h","l","c"]: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["t","o","h","l","c"]).sort_values("t").set_index("t")

def fetch_series_yahoo(symbol, interval, outputsize):
    yf_sym = YF_SYMBOL_MAP.get(symbol.upper(), symbol)
    period, yf_int = ("18mo","1d") if interval=="1day" else ("60d","60m")
    df = yf.download(yf_sym, period=period, interval=yf_int, auto_adjust=False,
                     progress=False, threads=False, group_by="column")
    df = _normalize_yf(df)
    return df.tail(outputsize) if outputsize and len(df) > outputsize else df

def fetch_series(symbol, interval, outputsize):
    if symbol.upper() in YF_SYMBOL_MAP:
        return fetch_series_yahoo(symbol, interval, outputsize)
    return fetch_series_twelvedata(symbol, interval, outputsize)

def resample_ohlc(df, rule):
    return df.resample(rule).agg({"o":"first","h":"max","l":"min","c":"last"}).dropna()

# =========================
# INDICATORS
# =========================
def rsi_wilder(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100/(1 + ag/al.replace(0, np.nan)))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def macd_calc(close, fast=12, slow=26, signal=9):
    ml = ema(close,fast) - ema(close,slow)
    sl = ema(ml, signal)
    return ml, sl, ml-sl

# =========================
# VXN / VIX
# =========================
def get_vol_value():
    for sym, url in [
        ("VXN","https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"),
        ("VIX","https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"),
    ]:
        try:
            df = pd.read_csv(url)
            col = next((c for c in df.columns if str(c).strip().upper()=="CLOSE"), None)
            if col:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                return sym, float(s.iloc[-1])
        except: continue
    return "VOL", None

def vol_regime(val):
    if val is None: return "Unknown",1.0
    if val < 18:    return "Low",0.75
    if val < 28:    return "Normal",1.0
    if val < 35:    return "High",1.25
    return "Extreme",1.50

def vol_color(name):
    return {"Low":G_FG,"Normal":A_FG,"High":R_FG,"Extreme":R_FG}.get(name, T2)

# =========================
# CYCLE CLASSIFIER
# =========================
def classify_cycle(rsi_val, pct_from_high, above_200ma, hist_rising):
    p = pct_from_high
    if above_200ma:
        if rsi_val>70 and p>-5:   return "Euphoria",4
        if rsi_val>65 and p>-10:  return "Thrill",3
        if rsi_val>60 and p>-15:  return ("Complacency",5) if not hist_rising else ("Belief",2)
        if rsi_val>55 and p>-20:  return "Belief",2
        if rsi_val>48 and p>-25:  return "Optimism → Belief",1
        if rsi_val>42:            return "Optimism",1
        if rsi_val>36:            return "Hope",0
        if p>-10:                 return "Complacency",5
        if p>-20:                 return "Anxiety",6
        return "Denial",7
    if rsi_val<24 and p<-48: return "Depression",11
    if rsi_val<28 and p<-40: return "Capitulation",9
    if rsi_val<32 and p<-32: return "Anger",10
    if rsi_val<36 and p<-24: return "Panic",8
    if rsi_val<42 and p<-18: return "Denial",7
    if rsi_val>=42 and p<-20 and hist_rising: return "Disbelief",12
    return "Anxiety",6

# =========================
# LEVELS / ACTIONS
# =========================
def compute_levels(close, h52):
    entry = round(close*0.97, 2)
    panic_low  = round(h52*0.65, 2)
    panic_high = round(h52*0.75, 2)
    cap_low    = round(h52*0.50, 2)
    cap_high   = round(h52*0.60, 2)
    ft_low  = round(max(close*1.08, h52*0.88), 2)
    ft_high = round(max(ft_low, h52*0.94), 2)
    st_low  = round(max(ft_high, h52*0.94), 2)
    st_high = round(h52, 2)
    dist_to_panic = round(((panic_high - close)/close)*100, 1)
    return {"entry":entry,"panic_low":panic_low,"panic_high":panic_high,
            "cap_low":cap_low,"cap_high":cap_high,
            "first_trim_low":ft_low,"first_trim_high":ft_high,
            "strong_trim_low":st_low,"strong_trim_high":st_high,
            "dist_to_panic":dist_to_panic}

def buy_speed(pct):
    p = abs(pct)
    if p<20: return "Slow"
    if p<30: return "Medium"
    if p<50: return "Aggressive"
    return "Max"

def determine_actions(stage_name, stage_idx, above_200ma, speed, vol_mult):
    ct = int(CAPITAL_PER_TICKER*0.50)
    tt = int(CAPITAL_PER_TICKER*0.50)
    if any(x in stage_name for x in ["Euphoria","Thrill"]):
        ca, cs = "HOLD", 0
    elif above_200ma:
        ca = "BUY dips"; cs = round(min(300,ct*0.20)*vol_mult)
    else:
        ca = "HOLD (below 200MA)"; cs = round(min(150,ct*0.10)*vol_mult)
    if any(x in stage_name for x in ["Panic","Capitulation","Anger"]) or stage_idx in [8,9,10]:
        ta = "BUY"; base = {"Slow":300,"Medium":600}.get(speed,1000)
        ts = round(min(base*vol_mult,tt))
    else:
        ta, ts = "WAIT", 0
    return ca, cs, ta, ts, ct, tt

def action_now(r):
    if r["tactical_action"] == "BUY":      return "TACTICAL BUY", R_FG, R_BG
    if "BUY" in r["core_action"]:          return "BUY DIPS",     G_FG, G_BG
    if "HOLD" in r["core_action"] and r["near_trigger"]: return "HOLD / WAIT", A_FG, A_BG
    return "WAIT", B_FG, B_BG

# =========================
# ANALYSIS
# =========================
def analyze_symbol(symbol, vol_mult):
    df_1h = fetch_series(symbol,"1h",420)
    df_4h = resample_ohlc(df_1h,"4h")
    if len(df_4h)<60: raise RuntimeError(f"{symbol}: not enough 4H bars")
    df_1d = fetch_series(symbol,"1day",320)
    close = float(df_4h["c"].iloc[-1])
    rsi_v = float(rsi_wilder(df_4h["c"],14).iloc[-1])
    ml,sl,hist = macd_calc(df_4h["c"])
    hist_rising = float(hist.iloc[-1])>float(hist.iloc[-2]) if len(hist)>=2 else False
    macd_dir  = "Bullish" if float(ml.iloc[-1])>float(sl.iloc[-1]) else "Bearish"
    macd_text = f"{macd_dir} ({'hist rising' if hist_rising else 'hist falling'})"
    ma200 = float(df_1d["c"].rolling(200).mean().iloc[-1])
    above = close>ma200
    h52   = float(df_1d["h"].tail(252).max())
    pct_h = round(((close-h52)/h52)*100,1)
    sup   = round(float(df_4h["l"].tail(60).min()),2)
    res   = round(float(df_4h["h"].tail(60).max()),2)
    sn,si = classify_cycle(rsi_v,pct_h,above,hist_rising)
    levels= compute_levels(close,h52)
    speed = buy_speed(pct_h)
    ca,cs,ta,ts,ct,tt = determine_actions(sn,si,above,speed,vol_mult)
    deployed = get_deployed(symbol)
    return {
        "symbol":symbol,"close":close,"high52w":h52,"pct_from_high":pct_h,
        "rsi":round(rsi_v,1),"macd_text":macd_text,"ma200":round(ma200,2),
        "above_200ma":above,"sup":sup,"res":res,
        "lvl20":round(h52*0.80,2),"lvl30":round(h52*0.70,2),"lvl40":round(h52*0.60,2),
        "stage_name":sn,"stage_idx":si,"levels":levels,"buy_speed":speed,
        "core_action":ca,"core_signal":cs,"core_total":ct,
        "tactical_action":ta,"tactical_signal":ts,"tactical_total":tt,
        "signal_today":cs+ts,"near_trigger":levels["dist_to_panic"]>-15,
        "already_deployed":deployed,
        "available_before_signal":max(0,CAPITAL_PER_TICKER-deployed),
        "available_after_signal":max(0,CAPITAL_PER_TICKER-deployed-cs-ts),
    }

# =========================
# DRAW PRIMITIVES
# =========================
def rbox(ax, x, y, w, h, r=0.008, fc=CARD, ec=BORDER, lw=0.7, z=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw,edgecolor=ec,facecolor=fc,
        transform=ax.transAxes,zorder=z,clip_on=False))

def txt(ax, x, y, s, sz=9, c=T1, ha="left", va="center", bold=False, z=5):
    ax.text(x,y,str(s),fontsize=sz,color=c,ha=ha,va=va,
            fontweight="bold" if bold else "normal",
            transform=ax.transAxes,zorder=z,clip_on=False)

def badge(ax, x, y, text, bg, fg, sz=6, ha="center", z=7):
    ax.text(x,y,str(text),fontsize=sz,color=fg,ha=ha,va="center",
            fontweight="bold",transform=ax.transAxes,zorder=z,
            bbox=dict(boxstyle="round,pad=0.22",facecolor=bg,edgecolor="none"),
            clip_on=False)

def hline(ax, x0, x1, y, color=BORDER_SOFT, lw=0.5):
    ax.plot([x0,x1],[y,y],color=color,linewidth=lw,
            transform=ax.transAxes,zorder=3,clip_on=False)

def pbar(ax, x, y, w, h, pct, fill, z=5):
    rbox(ax,x,y,w,h,r=0.003,fc=INNER2,ec="none",lw=0,z=z)
    if pct>0:
        rbox(ax,x,y,w*max(0,min(1,pct/100)),h,r=0.003,fc=fill,ec="none",lw=0,z=z+1)

def confidence_dots(ax, x, y, n_on=7, n_total=10):
    for i in range(n_total):
        ax.add_patch(Circle((x+i*0.013,y),0.0042,
            color=TEAL if i<n_on else "#31353d",
            transform=ax.transAxes,zorder=6,clip_on=False))

# =========================
# TICKER CARD  — card-relative coordinates
# ALL internal positions are fractions of the card's own width/height.
# This guarantees content always fits regardless of card size.
# =========================
def draw_ticker_card(ax, cx, cy, cw, ch, r):
    """
    cx, cy  — bottom-left of card in figure-normalised coords
    cw, ch  — card width / height in figure-normalised coords
    All content uses card-relative fractions then converts via X/Y helpers.
    """
    stage_bg, stage_fg = STAGE_COLORS[r["stage_idx"]]
    bc  = AMBER2 if r["near_trigger"] else BORDER
    blw = 1.2   if r["near_trigger"] else 0.8
    rbox(ax, cx, cy, cw, ch, r=0.008, fc=CARD, ec=bc, lw=blw)

    # ── coordinate helpers ──────────────────────────────────────
    PAD = 0.04                      # horizontal padding fraction of cw
    def X(fx): return cx + fx*cw   # card-fraction-x  → figure-normalised
    def Y(fy): return cy + fy*ch   # card-fraction-y  → figure-normalised
    def FW(fw): return fw*cw       # width fraction   → figure-normalised
    def FH(fh): return fh*ch       # height fraction  → figure-normalised
    lx = X(PAD)                    # left content edge
    rx = X(1-PAD)                  # right content edge
    iw = FW(1-2*PAD)               # inner usable width

    # ── layout (all y positions as fractions from card bottom) ──
    # Zone rows occupy bottom 44 %  (6 rows × ~0.063 + title/sep)
    # Indicators + 52W occupy next 18 %
    # Core/Tactical boxes occupy next 13 %
    # Action banner occupies next 9 %
    # Badges occupy 5 %
    # Price row occupies 6 %
    # Header occupies top 5 %

    # ── HEADER (symbol + stage badge) ───────────────────────────
    txt(ax, lx,       Y(0.955), r["symbol"],     sz=12, bold=True)
    badge(ax, rx,     Y(0.955), r["stage_name"], stage_bg, stage_fg, sz=5.5, ha="right")

    # ── PRICE ROW ───────────────────────────────────────────────
    txt(ax, lx,       Y(0.895), f"USD{r['close']:.2f}", sz=11, bold=True)
    txt(ax, rx,       Y(0.895), f"{r['pct_from_high']:.1f}% vs {r['high52w']:.2f}",
        sz=6, c=T2, ha="right")

    # ── STATUS BADGES ───────────────────────────────────────────
    tbg = G_BG if r["above_200ma"] else R_BG
    tfg = G_FG if r["above_200ma"] else R_FG
    badge(ax, X(PAD+0.10), Y(0.840),
          ("↑ Above" if r["above_200ma"] else "↓ Below")+" 200MA", tbg, tfg, sz=5.8)
    if r["near_trigger"]:
        badge(ax, X(PAD+0.36), Y(0.840), "⚠ Near panic", A_BG, A_FG, sz=5.8)

    # ── ACTION BANNER ───────────────────────────────────────────
    bh_action = FH(0.080)
    rbox(ax, lx, Y(0.745), iw, bh_action, r=0.006, fc=action_now(r)[2], ec="none", lw=0)
    txt(ax, X(0.5), Y(0.785), action_now(r)[0], sz=9.5, c=action_now(r)[1], ha="center", bold=True)

    # ── CORE / TACTICAL BOXES ───────────────────────────────────
    bw2   = (iw - FW(0.020)) / 2
    bh2   = FH(0.120)
    box_y = Y(0.615)

    # Core box
    rbox(ax, lx, box_y, bw2, bh2, r=0.005, fc=INNER2, ec="none", lw=0)
    txt(ax, lx+FW(0.02), box_y+bh2-FH(0.026), f"Core  £{int(r['core_total'])}", sz=5.8, c=T3)
    txt(ax, lx+FW(0.02), box_y+bh2-FH(0.060), f"£{int(r['core_signal'])}",      sz=9,   bold=True)
    txt(ax, lx+FW(0.02), box_y+FH(0.020),      r["core_action"],                sz=5.5, c=T3)
    pbar(ax, lx+FW(0.02), box_y+FH(0.008),
         bw2-FW(0.04), FH(0.012),
         (r["core_signal"]/max(1,r["core_total"]))*100, GREEN2)

    # Tactical box
    tx2 = lx+bw2+FW(0.020)
    rbox(ax, tx2, box_y, bw2, bh2, r=0.005, fc=INNER2, ec="none", lw=0)
    txt(ax, tx2+FW(0.02), box_y+bh2-FH(0.026), f"Tactical  £{int(r['tactical_total'])}", sz=5.8, c=T3)
    txt(ax, tx2+FW(0.02), box_y+bh2-FH(0.060), f"£{int(r['tactical_signal'])}",          sz=9,   bold=True)
    txt(ax, tx2+FW(0.02), box_y+FH(0.020),      f"{r['tactical_action']} · {r['buy_speed']}", sz=5.5, c=T3)
    pbar(ax, tx2+FW(0.02), box_y+FH(0.008),
         bw2-FW(0.04), FH(0.012),
         (r["tactical_signal"]/max(1,r["tactical_total"]))*100, B_FG)

    # ── INDICATORS (4 equal boxes) ──────────────────────────────
    ind_w = (iw - FW(0.06)) / 4
    ind_h = FH(0.090)
    ind_y = Y(0.515)
    ind_data = [
        ("RSI",   str(r["rsi"]),  ""),
        ("MACD",  "Bullish" if "Bullish" in r["macd_text"] else "Bearish",
                  "rising"  if "rising"  in r["macd_text"] else "falling"),
        ("200MA", f"USD{r['ma200']:.0f}", ""),
        ("S/R",   f"{r['sup']:.0f}/{r['res']:.0f}", ""),
    ]
    for i,(title,value,sub) in enumerate(ind_data):
        ibx = lx + i*(ind_w+FW(0.020))
        rbox(ax, ibx, ind_y, ind_w, ind_h, r=0.005, fc=INNER2, ec="none", lw=0)
        txt(ax, ibx+FW(0.015), ind_y+ind_h-FH(0.022), title, sz=5.5, c=T3)
        txt(ax, ibx+FW(0.015), ind_y+ind_h-FH(0.056), value, sz=7,   bold=True)
        if sub: txt(ax, ibx+FW(0.015), ind_y+FH(0.012), sub, sz=5.2, c=T3)

    # ── 52W LEVELS BAR ──────────────────────────────────────────
    bar52_h = FH(0.052)
    bar52_y = Y(0.453)
    rbox(ax, lx, bar52_y, iw, bar52_h, r=0.004, fc=INNER2, ec="none", lw=0)
    txt(ax, lx+FW(0.020), Y(0.479),
        f"52W: -20% {r['lvl20']:.0f}  -30% {r['lvl30']:.0f}  -40% {r['lvl40']:.0f}",
        sz=5.8, c=T2)

    # ── SEPARATOR + ZONES TITLE ─────────────────────────────────
    hline(ax, lx, rx, Y(0.438))
    txt(ax, lx, Y(0.420), "PSYCHOLOGY PRICE MAP", sz=6, c=T3, bold=True)

    # ── PRICE ZONES ─────────────────────────────────────────────
    # 6 zone rows, evenly spaced from 0.385 down to 0.065
    zones = [
        ("Add zone",     f"USD{r['levels']['entry']:.2f}",                                                T1),
        ("Panic zone",   f"USD{r['levels']['panic_low']:.2f} – {r['levels']['panic_high']:.2f}",          R_FG),
        ("Capitulation", f"USD{r['levels']['cap_low']:.2f} – {r['levels']['cap_high']:.2f}",              R_FG),
        ("First trim",   f"USD{r['levels']['first_trim_low']:.2f} – {r['levels']['first_trim_high']:.2f}",G_FG),
        ("Strong trim",  f"USD{r['levels']['strong_trim_low']:.2f} – {r['levels']['strong_trim_high']:.2f}",G_FG),
        ("Dist to panic",f"{r['levels']['dist_to_panic']:.1f}%",
                         A_FG if r["near_trigger"] else T2),
    ]
    zone_y_fracs = [0.385, 0.325, 0.265, 0.205, 0.145, 0.085]
    for (lbl,val,col), fy in zip(zones, zone_y_fracs):
        zy = Y(fy)
        txt(ax, lx,  zy, lbl, sz=6.2, c=T2)
        txt(ax, rx,  zy, val, sz=6.5, c=col, ha="right", bold=True)

# =========================
# PAGE BUILDER
# =========================
def build_page(page_results, all_results, vol_sym, vol_val, vol_rname, vol_mult, ts, page_num, total_pages):
    # ── figure sized to guarantee cards are tall enough ──────────
    FIG_W = 16.0   # inches
    FIG_H = 15.0   # inches  (taller = more room per card row)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax  = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); ax.set_facecolor(BG)

    M  = 0.028          # left/right margin (normalised)
    UW = 1 - 2*M        # usable width
    y  = 0.984          # top cursor

    # ── HEADER ──────────────────────────────────────────────────
    txt(ax, M,   y, "MARKET CYCLE DASHBOARD", sz=15, bold=True)
    hdr_right = f"{ts}  ·  Page {page_num}/{total_pages}" if total_pages>1 else ts
    txt(ax, 1-M, y, hdr_right, sz=7.5, c=T3, ha="right")
    y -= 0.030

    # ── VIX / CONFIG STRIP ──────────────────────────────────────
    sh = 0.058
    rbox(ax, M, y-sh, UW, sh, r=0.010, fc=CARD, ec=BORDER, lw=0.8)
    strip = [
        (vol_sym,    f"{vol_val:.1f}" if vol_val is not None else "N/A", vol_color(vol_rname)),
        ("Regime",   vol_rname,                                          vol_color(vol_rname)),
        ("Tactical", f"{vol_mult:.2f}x",                                 T1),
        ("Capital",  f"£{int(CAPITAL_PER_TICKER)} / ticker",             T1),
        ("Structure","50% Core  ·  50% Tactical",                        T1),
    ]
    sw = UW/len(strip)
    for i,(lbl,val,c) in enumerate(strip):
        sx = M + i*sw + 0.012
        txt(ax, sx, y-sh*0.28, lbl, sz=6,   c=T3)
        txt(ax, sx, y-sh*0.68, val, sz=9.5, c=c, bold=True)
    y -= sh + 0.020

    # ── CYCLE BAR ───────────────────────────────────────────────
    txt(ax, M, y, "Market cycle position", sz=6.5, c=T3)
    y -= 0.016
    seg_h = 0.036
    seg_w = UW/len(STAGE_NAMES)
    active = {r["stage_idx"] for r in all_results}
    for i,(name,(bg,fg)) in enumerate(zip(STAGE_NAMES,STAGE_COLORS)):
        sx = M + i*seg_w
        on = i in active
        rbox(ax, sx, y-seg_h, seg_w-0.001, seg_h, r=0.003,
             fc=bg, ec="#ffffff" if on else BORDER, lw=1.4 if on else 0.4)
        txt(ax, sx+(seg_w-0.001)/2, y-seg_h/2, name,
            sz=5.8, c=fg, ha="center", bold=on)
    y -= seg_h + 0.020

    # ── CARD GRID ────────────────────────────────────────────────
    # Remaining space: y (top of card area) down to footer
    FOOTER_H  = 0.085   # normalised height reserved for footer
    FOOTER_GAP= 0.012
    card_area_top    = y
    card_area_bottom = FOOTER_H + FOOTER_GAP
    total_card_h     = card_area_top - card_area_bottom   # e.g. ~0.72

    cols     = CARDS_PER_ROW
    rows     = ROWS_PER_PAGE
    col_gap  = 0.012
    row_gap  = 0.018
    card_w   = (UW - col_gap*(cols-1)) / cols
    card_h   = (total_card_h - row_gap*(rows-1)) / rows   # guaranteed fit

    for idx, r in enumerate(page_results):
        col = idx % cols
        row = idx // cols
        cx  = M + col*(card_w+col_gap)
        # row 0 = top row, row 1 = bottom row
        cy  = card_area_top - (row+1)*card_h - row*row_gap
        draw_ticker_card(ax, cx, cy, card_w, card_h, r)

    # ── FOOTER SUMMARY ──────────────────────────────────────────
    fy = FOOTER_H
    rbox(ax, M, 0, UW, fy, r=0.008, fc=CARD, ec=BORDER, lw=0.8)

    total_core     = sum(r["core_signal"]     for r in all_results)
    total_tactical = sum(r["tactical_signal"] for r in all_results)
    total_capacity = len(all_results)*int(CAPITAL_PER_TICKER)
    triggers       = [r["symbol"] for r in all_results if r["tactical_action"]=="BUY"]

    txt(ax, M+0.012, fy-0.016, "PORTFOLIO SUMMARY", sz=7.5, c=T3, bold=True)

    items = [
        ("Core signals today",     f"£{int(total_core)}",     T1),
        ("Tactical signals today", f"£{int(total_tactical)}", T1),
        ("Total capacity",         f"£{int(total_capacity)}", T1),
        ("Tactical triggers",
         (", ".join(triggers)) if triggers else "NONE — WAIT",
         G_FG if not triggers else R_FG),
    ]
    sw2 = UW/len(items)
    for i,(lbl,val,c) in enumerate(items):
        sx = M + i*sw2 + 0.012
        txt(ax, sx, fy*0.62, lbl, sz=6,   c=T3)
        txt(ax, sx, fy*0.32, val, sz=9.5, c=c, bold=True)

    confidence_dots(ax, M+0.014, fy*0.12, n_on=7, n_total=10)
    txt(ax, M+0.150, fy*0.12, "7/10 confidence", sz=6.5, c=T2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BG,
                bbox_inches=None, pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# CHUNK
# =========================
def chunk_results(results, size=MAX_CARDS_PER_PAGE):
    for i in range(0, len(results), size):
        yield results[i:i+size]

# =========================
# TELEGRAM
# =========================
def send_dashboard(img_buf, caption, filename="dashboard.png"):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
        data={"chat_id":TELEGRAM_CHAT_ID,"caption":caption[:950],"disable_web_page_preview":True},
        files={"document":(filename, img_buf.getvalue(), "image/png")},
        timeout=90,
    )
    r.raise_for_status()

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [text[i:i+3800] for i in range(0,len(text),3800)]:
        requests.post(url, data={"chat_id":TELEGRAM_CHAT_ID,"text":chunk,
                                 "disable_web_page_preview":True}, timeout=30).raise_for_status()
        time.sleep(0.25)

# =========================
# MAIN
# =========================
def main():
    if not TD_API_KEY:       raise RuntimeError("Missing TWELVEDATA_API_KEY")
    if not TELEGRAM_TOKEN:   raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID: raise RuntimeError("Missing TELEGRAM_CHAT_ID")

    ts = utc_now_str()
    vol_sym, vol_val   = get_vol_value()
    vol_rname, vol_mult = vol_regime(vol_val)

    results, failures = [], []
    for sym in SYMBOLS:
        try:    results.append(analyze_symbol(sym, vol_mult))
        except Exception as e: failures.append(f"{sym}: {repr(e)}")

    if not results:
        raise RuntimeError("All symbols failed:\n" + "\n".join(failures))

    pages = list(chunk_results(results, MAX_CARDS_PER_PAGE))
    triggers = [r["symbol"] for r in results if r["tactical_action"]=="BUY"]

    for idx, page_results in enumerate(pages, 1):
        buf = build_page(page_results, results, vol_sym, vol_val,
                         vol_rname, vol_mult, ts, idx, len(pages))
        syms = ", ".join(r["symbol"] for r in page_results)
        cap  = [
            f"📊 Market Cycle Dashboard — {ts}",
            f"{vol_sym}: {f'{vol_val:.1f}' if vol_val else 'N/A'} | {vol_rname} | x{vol_mult:.2f}",
            f"Page {idx}/{len(pages)} | {syms}",
            ("🚨 Tactical BUY: "+", ".join(triggers)) if triggers else "No tactical triggers",
        ]
        for r in page_results:
            cap.append(f"{r['symbol']}: {r['stage_name']} | {action_now(r)[0]} | Core £{int(r['core_signal'])} | Tactical £{int(r['tactical_signal'])}")
        send_dashboard(buf, "\n".join(cap), f"dashboard_p{idx}.png")
        time.sleep(0.4)

    if failures:
        send_message("⚠️ Failures:\n"+"\n".join(failures))

    print(f"Done. Pages: {len(pages)}. Triggers: {triggers}")

if __name__ == "__main__":
    main()
