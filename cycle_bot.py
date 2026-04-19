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
SYMBOLS            = [s.strip().upper() for s in os.getenv("SYMBOLS","TSLA,NVDA,PLTR,VUSA").split(",") if s.strip()]
CAPITAL_PER_TICKER = float(os.getenv("CAPITAL_PER_TICKER","3000"))
TD_API_KEY         = os.getenv("TWELVEDATA_API_KEY","").strip()
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","").strip()
BASE_URL           = "https://api.twelvedata.com/time_series"
TD_MIN_SECONDS     = float(os.getenv("TD_MIN_SECONDS_BETWEEN_CALLS","11.0"))
CARDS_PER_ROW      = 3
ROWS_PER_PAGE      = 2
MAX_CARDS_PER_PAGE = CARDS_PER_ROW * ROWS_PER_PAGE

_last_td_call = 0.0

YF_SYMBOL_MAP = {"VUSA": "VUSA.L"}

CURRENCY_MAP = {"VUSA": "GBP", "VUSA.L": "GBP"}

def get_currency(sym): return CURRENCY_MAP.get(sym.upper(), "USD")
def csym(sym): return "\u00a3" if get_currency(sym) == "GBP" else "USD "
def get_deployed(sym): return float(os.getenv(f"DEPLOYED_{sym}","0"))

# =========================
# COLOURS
# =========================
BG      = "#0b0b0d"
CARD    = "#17181c"
INNER2  = "#0c0d10"
INNER3  = "#13151a"
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
P_FG, P_BG = "#c084fc", "#1e1030"

OV_PERMIT_FG,  OV_PERMIT_BG,  OV_PERMIT_BORDER  = "#7ee2a8","#071a0f","#16a085"
OV_HIGH_FG,    OV_HIGH_BG,    OV_HIGH_BORDER    = "#fbbf24","#1a1000","#d97706"
OV_DOUBLE_FG,  OV_DOUBLE_BG,  OV_DOUBLE_BORDER  = "#34d399","#031208","#059669"
OV_DENY_FG,    OV_DENY_BG,    OV_DENY_BORDER    = "#5b606b","#0d0d10","#2b2f38"

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
    w = TD_MIN_SECONDS - (time.time() - _last_td_call)
    if w > 0: time.sleep(w)
    _last_td_call = time.time()

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# =========================
# DATA FETCH
# =========================
def fetch_series_twelvedata(symbol, interval, outputsize):
    td_throttle()
    params = {"symbol":symbol,"interval":interval,
              "outputsize":str(outputsize),"apikey":TD_API_KEY,"format":"JSON"}
    r = requests.get(BASE_URL, params=params, timeout=30)
    data = r.json()
    if isinstance(data,dict) and data.get("status") == "error":
        msg = (data.get("message") or "").lower()
        if "run out" in msg or "current minute" in msg:
            time.sleep(65); td_throttle()
            r = requests.get(BASE_URL, params=params, timeout=30)
            data = r.json()
        if isinstance(data,dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData: {data.get('message')}")
    values = data.get("values")
    if not values: raise RuntimeError(f"No values for {symbol} ({interval})")
    # DEBUG: log raw keys from first value to confirm volume field name
    if values:
        raw_keys = list(values[0].keys())
        print(f"[DEBUG] {symbol} ({interval}) raw keys: {raw_keys}")
        print(f"[DEBUG] sample value: {values[0]}")
    df = pd.DataFrame(values).rename(
        columns={"datetime":"t","open":"o","high":"h","low":"l","close":"c","volume":"v"})
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o","h","l","c"]: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["t","o","h","l","c"]).sort_values("t").set_index("t")

def _flatten_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in col
                      if str(x) not in ("","nan")) for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _find_col(cols, targets):
    lmap = {str(c).lower():c for c in cols}
    for t in targets:
        if t.lower() in lmap: return lmap[t.lower()]
    for c in cols:
        for t in targets:
            if t.lower() in str(c).lower(): return c
    return None

def _normalize_yf(df):
    if df is None or df.empty: raise RuntimeError("Yahoo Finance returned no data")
    df = _flatten_yf(df).reset_index()
    cols = list(df.columns)
    tc = _find_col(cols,["Datetime","Date","index"])
    oc = _find_col(cols,["Open"]); hc = _find_col(cols,["High"])
    lc = _find_col(cols,["Low"]);  cc = _find_col(cols,["Close","Adj Close"])
    if not all([tc,oc,hc,lc,cc]):
        raise RuntimeError(f"Yahoo Finance missing columns: {cols}")
    df = df.rename(columns={tc:"t",oc:"o",hc:"h",lc:"l",cc:"c"})
    # rename volume column to v if present
    vc = _find_col(list(df.columns), ["Volume"])
    if vc and vc not in ("t","o","h","l","c"):
        df = df.rename(columns={vc:"v"})
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for col in ["o","h","l","c"]: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["t","o","h","l","c"]).sort_values("t").set_index("t")

def fetch_series_yahoo(symbol, interval, outputsize):
    yf_sym = YF_SYMBOL_MAP.get(symbol.upper(), symbol)
    period, yf_int = ("18mo","1d") if interval=="1day" else ("60d","60m")
    df = yf.download(yf_sym, period=period, interval=yf_int,
                     auto_adjust=False, progress=False,
                     threads=False, group_by="column")
    df = _normalize_yf(df)
    return df.tail(outputsize) if outputsize and len(df)>outputsize else df

def fetch_series(symbol, interval, outputsize):
    if symbol.upper() in YF_SYMBOL_MAP:
        return fetch_series_yahoo(symbol, interval, outputsize)
    return fetch_series_twelvedata(symbol, interval, outputsize)

def resample_ohlc(df, rule):
    return df.resample(rule).agg(
        {"o":"first","h":"max","l":"min","c":"last"}).dropna()

def resample_ohlcv(df, rule):
    """Resample including volume sum — required for order block detection."""
    cols = {"o":"first","h":"max","l":"min","c":"last"}
    if "v" in df.columns:
        cols["v"] = "sum"
    return df.resample(rule).agg(cols).dropna()

# =========================
# INDICATORS
# =========================
def rsi_wilder(close, period=14):
    d  = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100/(1 + ag/al.replace(0, np.nan)))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def macd_calc(close, fast=12, slow=26, signal=9):
    ml = ema(close,fast) - ema(close,slow)
    sl = ema(ml,signal)
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
            df  = pd.read_csv(url)
            col = next((c for c in df.columns
                        if str(c).strip().upper()=="CLOSE"), None)
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
    return {"Low":G_FG,"Normal":A_FG,"High":R_FG,
            "Extreme":R_FG}.get(name, T2)

# =========================
# CYCLE CLASSIFIER
# =========================
def classify_cycle(rsi_val, pct_from_high, above_200ma, hist_rising):
    p = pct_from_high
    if above_200ma:
        if rsi_val>70 and p>-5:   return "Euphoria",4
        if rsi_val>65 and p>-10:  return "Thrill",3
        if rsi_val>60 and p>-15:
            return ("Complacency",5) if not hist_rising else ("Belief",2)
        if rsi_val>55 and p>-20:  return "Belief",2
        if rsi_val>48 and p>-25:  return "Optimism",1
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
    ft_lo = round(max(close*1.08, h52*0.88), 2)
    ft_hi = round(max(ft_lo, h52*0.94), 2)
    st_lo = round(max(ft_hi, h52*0.94), 2)
    return {
        "entry":       round(close*0.97,2),
        "panic_lo":    round(h52*0.65,2),
        "panic_hi":    round(h52*0.75,2),
        "cap_lo":      round(h52*0.50,2),
        "cap_hi":      round(h52*0.60,2),
        "ft_lo":       ft_lo,
        "ft_hi":       ft_hi,
        "st_lo":       st_lo,
        "st_hi":       round(h52,2),
        "dist":        round(((h52*0.75-close)/close)*100,1),
    }

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
    if (any(x in stage_name for x in ["Panic","Capitulation","Anger"])
            or stage_idx in [8,9,10]):
        ta = "BUY"
        base = {"Slow":300,"Medium":600}.get(speed,1000)
        ts   = round(min(base*vol_mult,tt))
    else:
        ta, ts = "WAIT", 0
    return ca, cs, ta, ts, ct, tt

def action_now(r):
    if r["tactical_action"] == "BUY":          return "TACTICAL BUY", R_FG, R_BG
    if "BUY" in r["core_action"]:              return "BUY DIPS",     G_FG, G_BG
    if ("HOLD" in r["core_action"]
            and r["near_trigger"]):            return "HOLD / WAIT",  A_FG, A_BG
    return "WAIT", B_FG, B_BG

# =========================
# ORDER BLOCKS
# =========================
def compute_order_blocks(df_4h, df_1d, close, n=4):
    """
    Identifies significant volume zones from 4H and daily candles.
    A block is a candle with >= 2.5x average volume that preceded a
    clear directional move — labelled demand (bullish reversal) or
    supply (bearish reversal).
    Returns top-n blocks sorted by vol_mult descending.
    """
    blocks = []

    for df, tf_label in [(df_4h, "4H"), (df_1d, "1D")]:
        if df is None or len(df) < 30:
            continue

        df = df.copy().tail(120)
        if "v" not in df.columns:
            df["v"] = 1.0

        df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0)
        print(f"[DEBUG] OB {tf_label} volume stats — mean: {df['v'].mean():.0f}, max: {df['v'].max():.0f}, zeros: {(df['v']==0).sum()}")
        avg_vol  = df["v"].rolling(20, min_periods=10).mean()

        for i in range(2, len(df)-2):
            vol    = df["v"].iloc[i]
            avg    = avg_vol.iloc[i]
            if avg <= 0:
                continue
            mult   = vol / avg
            if mult < 2.0:
                continue

            candle = df.iloc[i]
            body   = abs(float(candle["c"]) - float(candle["o"]))
            rng    = float(candle["h"]) - float(candle["l"])
            if rng == 0:
                continue

            # look 2 candles forward to determine direction of move
            next2_close = df["c"].iloc[i+1:i+3].mean()
            if float(candle["c"]) >= float(candle["o"]):
                # bullish candle
                ob_type = "demand"
                lo = round(float(candle["l"]), 2)
                hi = round(float(candle["c"]), 2)
            else:
                # bearish candle
                ob_type = "supply"
                lo = round(float(candle["c"]), 2)
                hi = round(float(candle["h"]), 2)

            # sessions ago (approximate using index position)
            sessions_ago = len(df) - 1 - i

            # skip stale blocks — only keep last 30 sessions
            if sessions_ago > 30:
                continue

            blocks.append({
                "type":         ob_type,
                "lo":           lo,
                "hi":           hi,
                "vol_mult":     round(mult, 1),
                "sessions_ago": sessions_ago,
                "tf":           tf_label,
            })

    if not blocks:
        return []

    # deduplicate overlapping zones (keep highest vol)
    blocks.sort(key=lambda x: x["vol_mult"], reverse=True)
    deduped = []
    for b in blocks:
        overlap = any(
            not (b["hi"] < d["lo"] or b["lo"] > d["hi"])
            for d in deduped
        )
        if not overlap:
            deduped.append(b)
        if len(deduped) >= n:
            break

    # sort for display: supply zones descending (above price),
    # demand zones ascending (below price)
    supply  = sorted([b for b in deduped if b["type"]=="supply"],
                     key=lambda x: x["lo"], reverse=True)
    demand  = sorted([b for b in deduped if b["type"]=="demand"],
                     key=lambda x: x["lo"], reverse=True)

    # interleave so we show nearest blocks first
    result = []
    si, di = 0, 0
    while len(result) < n and (si < len(supply) or di < len(demand)):
        if si < len(supply):
            result.append(supply[si]); si += 1
        if len(result) < n and di < len(demand):
            result.append(demand[di]); di += 1
    return result[:n]

# =========================
# OVERRIDE LOGIC
# =========================
OV_TRIGGER_STAGES = {
    "Panic","Capitulation","Anger","Disbelief","Depression"
}

def compute_override(r, order_blocks):
    """
    Determines the order block override status.
    Rules:
    - DOUBLE_CONFIRM : bot says BUY + demand block ≥3x vol within 5%
    - HIGH_CONVICTION: bot WAIT/HOLD + demand block ≥3x vol
                       + stage in trigger set OR block inside panic/cap zone
                       + vol ≥ 3.5x (higher bar)
    - PERMITTED      : bot WAIT/HOLD + demand block ≥3x vol
                       + block price inside panic or cap zone
    - DENIED         : all other cases
    """
    close     = r["close"]
    lvl       = r["levels"]
    stage     = r["stage_name"]
    bot_act   = action_now(r)[0]

    # find best qualifying demand block
    demand_blocks = [b for b in order_blocks
                     if b["type"]=="demand" and b["vol_mult"]>=3.0]
    demand_blocks.sort(key=lambda x: x["vol_mult"], reverse=True)

    if not demand_blocks:
        return {
            "status":    "DENIED",
            "entry_lo":  0, "entry_hi": 0,
            "size_label":"", "condition":"",
            "vol_mult":  0,
            "reason":    "No demand block meets 3x vol threshold",
        }

    best = demand_blocks[0]
    vol  = best["vol_mult"]
    b_lo = best["lo"]
    b_hi = best["hi"]
    b_mid= (b_lo+b_hi)/2

    # is block inside a key zone?
    in_panic = (lvl["panic_lo"] <= b_mid <= lvl["panic_hi"])
    in_cap   = (lvl["cap_lo"]   <= b_mid <= lvl["cap_hi"])
    in_zone  = in_panic or in_cap

    # proximity: is current price within 5% of block?
    prox_pct = abs((b_mid-close)/close)*100
    near     = prox_pct < 5.0

    # DOUBLE CONFIRM — bot already says BUY + block confirms
    if "BUY" in bot_act and vol >= 3.0:
        return {
            "status":    "DOUBLE_CONFIRM",
            "entry_lo":  b_lo, "entry_hi": b_hi,
            "size_label":f"Core + tactical  £{r['core_signal']+r['tactical_signal']}",
            "condition": f"Bot BUY + {vol:.1f}x vol block aligned",
            "vol_mult":  vol, "reason":"",
        }

    # HIGH CONVICTION — strong block + triggering stage or inside key zone
    stage_trigger = any(s in stage for s in OV_TRIGGER_STAGES)
    if vol >= 3.5 and (stage_trigger or in_zone):
        size_lbl = "Core + partial tactical" if stage_trigger else "Core only"
        condition = (f"{stage.split(' ')[0]} stage  {vol:.1f}x vol"
                     if stage_trigger else f"In {'panic' if in_panic else 'cap'} zone  {vol:.1f}x vol")
        return {
            "status":    "HIGH_CONVICTION",
            "entry_lo":  b_lo, "entry_hi": b_hi,
            "size_label":size_lbl,
            "condition": condition,
            "vol_mult":  vol, "reason":"",
        }

    # PERMITTED — block in key zone, vol ≥ 3x
    if in_zone and vol >= 3.0:
        zone_lbl = "panic zone" if in_panic else "cap zone"
        return {
            "status":    "PERMITTED",
            "entry_lo":  b_lo, "entry_hi": b_hi,
            "size_label":"Core only",
            "condition": f"Block in {zone_lbl}  {vol:.1f}x vol",
            "vol_mult":  vol, "reason":"",
        }

    # DENIED
    reason = (f"Block {vol:.1f}x vol — outside panic/cap zone"
              if vol >= 3.0 else
              f"Best block {vol:.1f}x vol — below 3x threshold")
    return {
        "status":    "DENIED",
        "entry_lo":  0, "entry_hi": 0,
        "size_label":"", "condition":"",
        "vol_mult":  vol, "reason": reason,
    }

# =========================
# FULL SYMBOL ANALYSIS
# =========================
def analyze_symbol(symbol, vol_mult):
    df_1h = fetch_series(symbol,"1h",420)
    df_4h = resample_ohlc(df_1h,"4h")
    if len(df_4h) < 60:
        raise RuntimeError(f"{symbol}: not enough 4H bars")
    df_1d = fetch_series(symbol,"1day",320)

    close = float(df_4h["c"].iloc[-1])
    rsi_v = float(rsi_wilder(df_4h["c"],14).iloc[-1])
    ml,sl,hist = macd_calc(df_4h["c"])
    hr = float(hist.iloc[-1])>float(hist.iloc[-2]) if len(hist)>=2 else False
    macd_dir  = "Bullish" if float(ml.iloc[-1])>float(sl.iloc[-1]) else "Bearish"
    macd_text = f"{macd_dir} ({'hist rising' if hr else 'hist falling'})"
    ma200 = float(df_1d["c"].rolling(200).mean().iloc[-1])
    above = close > ma200
    h52   = float(df_1d["h"].tail(252).max())
    pct_h = round(((close-h52)/h52)*100,1)
    sup   = round(float(df_4h["l"].tail(60).min()),2)
    res   = round(float(df_4h["h"].tail(60).max()),2)

    sn, si  = classify_cycle(rsi_v,pct_h,above,hr)
    levels  = compute_levels(close,h52)
    speed   = buy_speed(pct_h)
    ca,cs,ta,ts,ct,tt = determine_actions(sn,si,above,speed,vol_mult)

    # build 4H dataframe with volume properly summed from 1H candles
    df_1h_v = df_1h.copy()
    if "v" not in df_1h_v.columns:
        df_1h_v["v"] = 1.0
    df_1h_v["v"] = pd.to_numeric(df_1h_v["v"], errors="coerce").fillna(0)
    df_4h_v = resample_ohlcv(df_1h_v, "4h")

    # build daily dataframe with volume
    df_1d_v = df_1d.copy()
    if "v" not in df_1d_v.columns:
        df_1d_v["v"] = 1.0
    df_1d_v["v"] = pd.to_numeric(df_1d_v["v"], errors="coerce").fillna(0)

    order_blocks = compute_order_blocks(df_4h_v, df_1d_v, close, n=4)

    deployed = get_deployed(symbol)
    result = {
        "symbol":    symbol,
        "close":     close,
        "high52w":   h52,
        "pct_from_high": pct_h,
        "rsi":       round(rsi_v,1),
        "macd_text": macd_text,
        "ma200":     round(ma200,2),
        "above_200ma": above,
        "sup": sup, "res": res,
        "lvl20": round(h52*0.80,2),
        "lvl30": round(h52*0.70,2),
        "lvl40": round(h52*0.60,2),
        "stage_name":   sn,
        "stage_idx":    si,
        "levels":       levels,
        "buy_speed":    speed,
        "core_action":  ca, "core_signal":   cs, "core_total":    ct,
        "tactical_action": ta, "tactical_signal": ts, "tactical_total":   tt,
        "signal_today": cs+ts,
        "near_trigger": levels["dist"] > -15,
        "already_deployed": deployed,
        "available":    max(0, CAPITAL_PER_TICKER-deployed),
        "order_blocks": order_blocks,
        "currency":     get_currency(symbol),
        "csym":         csym(symbol),
    }
    print(f"[DEBUG] {symbol} order_blocks found: {len(order_blocks)}")
    for b in order_blocks:
        print(f"[DEBUG]   {b['type']} {b['lo']:.2f}-{b['hi']:.2f} {b['vol_mult']:.1f}x {b['sessions_ago']}d")
    result["override"] = compute_override(result, order_blocks)
    return result

# =========================
# DRAW PRIMITIVES
# =========================
def rbox(ax,x,y,w,h,r=0.008,fc=CARD,ec=BORDER,lw=0.7,z=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw,edgecolor=ec,facecolor=fc,
        transform=ax.transAxes,zorder=z,clip_on=False))

def t(ax,x,y,s,sz=9,c=T1,ha="left",va="center",bold=False,z=5):
    ax.text(x,y,str(s),fontsize=sz,color=c,ha=ha,va=va,
            fontweight="bold" if bold else "normal",
            transform=ax.transAxes,zorder=z,clip_on=False)

def bdg(ax,x,y,s,bg,fg,sz=6.5,ha="center",ec="none",elw=0,z=7):
    ax.text(x,y,str(s),fontsize=sz,color=fg,ha=ha,va="center",
            fontweight="bold",transform=ax.transAxes,zorder=z,
            bbox=dict(boxstyle="round,pad=0.24",facecolor=bg,
                      edgecolor=ec,linewidth=elw),clip_on=False)

def hl(ax,x0,x1,y,color=BORDER_SOFT,lw=0.5):
    ax.plot([x0,x1],[y,y],color=color,linewidth=lw,
            transform=ax.transAxes,zorder=3,clip_on=False)

def pb(ax,x,y,w,h,pct,fill,z=5):
    rbox(ax,x,y,w,h,r=0.002,fc=INNER2,ec="none",lw=0,z=z)
    fw = w*max(0,min(1,pct/100))
    if fw > 0.0001:
        rbox(ax,x,y,fw,h,r=0.002,fc=fill,ec="none",lw=0,z=z+1)

def confidence_dots(ax,x,y,n_on=7,n_total=10):
    for i in range(n_total):
        ax.add_patch(Circle((x+i*0.013,y),0.0042,
            color=TEAL if i<n_on else "#31353d",
            transform=ax.transAxes,zorder=6,clip_on=False))

# =========================
# TICKER CARD
# =========================
def draw_ticker_card(ax, cx, cy, cw, ch, r):
    stage_bg, stage_fg = STAGE_COLORS[r["stage_idx"]]
    bc  = AMBER2 if r["near_trigger"] else BORDER
    rbox(ax,cx,cy,cw,ch,r=0.009,fc=CARD,
         ec=bc,lw=1.3 if r["near_trigger"] else 0.8)

    HP = 0.036
    VP = lambda fy: cy + fy*ch
    LX = cx + HP*cw
    RX = cx + (1-HP)*cw
    IW = cw*(1-2*HP)
    cs = r["csym"]

    # ── HEADER ───────────────────────────────────────────────
    t(ax, LX,        VP(0.962), r["symbol"],           sz=13, bold=True)
    t(ax, LX+cw*0.22,VP(0.962), f"{cs}{r['close']:.2f}", sz=11, bold=True)
    t(ax, RX,        VP(0.962),
      f"{r['pct_from_high']:.1f}% from 52W", sz=6.5, c=T2, ha="right")

    bdg(ax, LX+IW*0.063, VP(0.930), r["stage_name"],   stage_bg, stage_fg, sz=6.2)
    tbg = G_BG if r["above_200ma"] else R_BG
    tfg = G_FG if r["above_200ma"] else R_FG
    bdg(ax, LX+IW*0.258, VP(0.930),
        ("↑ Above" if r["above_200ma"] else "↓ Below")+" 200MA", tbg, tfg, sz=6.2)
    if r["near_trigger"]:
        bdg(ax, LX+IW*0.455, VP(0.930), "⚠ Near panic", A_BG, A_FG, sz=6.2)

    # ── ACTION BANNER ────────────────────────────────────────
    an_txt, an_fg, an_bg = action_now(r)
    rbox(ax, LX, VP(0.874), IW, ch*0.052, r=0.006, fc=an_bg, ec="none", lw=0)
    t(ax, cx+cw*0.5, VP(0.900), an_txt, sz=10, c=an_fg, ha="center", bold=True)

    # ── CORE / TACTICAL ──────────────────────────────────────
    bw = (IW-cw*0.018)/2
    bh = ch*0.090
    for i,(title,sig,tot,bar_c,note) in enumerate([
        (f"Core £{r['core_total']}",
         r["core_signal"], r["core_total"], GREEN2, r["core_action"]),
        (f"Tactical £{r['tactical_total']}",
         r["tactical_signal"], r["tactical_total"], B_FG,
         f"{r['tactical_action']} · {r['buy_speed']}"),
    ]):
        bx = LX + i*(bw+cw*0.018)
        by = VP(0.775)
        rbox(ax,bx,by,bw,bh,r=0.005,fc=INNER2,ec="none",lw=0)
        t(ax,bx+cw*0.014,by+bh-ch*0.021,title,sz=6.5,c=T3)
        t(ax,bx+cw*0.014,by+bh-ch*0.056,f"£{sig}",sz=10,bold=True)
        t(ax,bx+cw*0.014,by+ch*0.012,note,sz=6.0,c=T3)
        pb(ax,bx+cw*0.014,by+ch*0.005,bw-cw*0.028,ch*0.008,(sig/max(1,tot))*100,bar_c)

    # ── INDICATORS ───────────────────────────────────────────
    iw4 = (IW-cw*0.054)/4
    ih  = ch*0.078
    iy  = VP(0.686)
    for i,(title,val,sub) in enumerate([
        ("RSI",  str(r["rsi"]),""),
        ("MACD",
         "Bullish" if "Bullish" in r["macd_text"] else "Bearish",
         "rising"  if "rising"  in r["macd_text"] else "falling"),
        ("200MA", f"{cs}{r['ma200']:.0f}",""),
        ("S/R",   f"{r['sup']:.0f}/{r['res']:.0f}",""),
    ]):
        ibx = LX + i*(iw4+cw*0.018)
        rbox(ax,ibx,iy,iw4,ih,r=0.005,fc=INNER2,ec="none",lw=0)
        t(ax,ibx+cw*0.012,iy+ih-ch*0.018,title,sz=6.2,c=T3)
        t(ax,ibx+cw*0.012,iy+ih-ch*0.050,val,sz=7.8,bold=True)
        if sub: t(ax,ibx+cw*0.012,iy+ch*0.010,sub,sz=5.8,c=T3)

    # ── 52W BAR ──────────────────────────────────────────────
    rbox(ax,LX,VP(0.636),IW,ch*0.044,r=0.004,fc=INNER2,ec="none",lw=0)
    t(ax,LX+cw*0.016,VP(0.658),
      f"52W:  -20% {r['lvl20']:.0f}   -30% {r['lvl30']:.0f}   -40% {r['lvl40']:.0f}",
      sz=6.5,c=T2)

    # ── PSYCHOLOGY PRICE MAP ─────────────────────────────────
    hl(ax,LX,RX,VP(0.624))
    t(ax,LX,VP(0.610),"PSYCHOLOGY PRICE MAP",sz=6.8,c=T3,bold=True)
    for (lbl,val,col),fy in zip([
        ("Add zone",
         f"{cs}{r['levels']['entry']:.2f}", T1),
        ("Panic zone",
         f"{cs}{r['levels']['panic_lo']:.0f} – {cs}{r['levels']['panic_hi']:.0f}", R_FG),
        ("Capitulation",
         f"{cs}{r['levels']['cap_lo']:.0f} – {cs}{r['levels']['cap_hi']:.0f}", R_FG),
        ("First trim",
         f"{cs}{r['levels']['ft_lo']:.0f} – {cs}{r['levels']['ft_hi']:.0f}", G_FG),
        ("Strong trim",
         f"{cs}{r['levels']['st_lo']:.0f} – {cs}{r['levels']['st_hi']:.0f}", G_FG),
        ("Dist to panic",
         f"{r['levels']['dist']:.1f}%",
         A_FG if r["near_trigger"] else T2),
    ],[0.592,0.558,0.524,0.490,0.456,0.424]):
        t(ax,LX, VP(fy),lbl,sz=6.8,c=T2)
        t(ax,RX, VP(fy),val,sz=7.0,c=col,ha="right",bold=True)

    # ── VOLUME ORDER BLOCKS ───────────────────────────────────
    hl(ax,LX,RX,VP(0.408))
    t(ax,LX,VP(0.394),"VOLUME ORDER BLOCKS",sz=6.8,c=T3,bold=True)
    bdg(ax,LX+IW*0.290,VP(0.394),"LIVE  4H + 1D",P_BG,P_FG,sz=5.8)

    for lbl,fx in [("Type",0.00),("Zone",0.21),("Vol",0.60),
                   ("Age",0.74),("Proximity",0.84)]:
        t(ax,LX+IW*fx,VP(0.374),lbl,sz=6.0,c=T3)
    hl(ax,LX,RX,VP(0.364),color=T3,lw=0.3)

    obs = r.get("order_blocks",[])
    for ob,fy in zip(obs,[0.344,0.306,0.268,0.230]):
        is_d   = ob["type"]=="demand"
        ob_fg  = G_FG if is_d else R_FG
        ob_bg  = G_BG if is_d else R_BG
        zmid   = (ob["lo"]+ob["hi"])/2
        prox   = abs((zmid-r["close"])/r["close"])*100
        is_near= prox < 5.0
        row_bg = ("#071a0f" if (is_near and is_d)
                  else "#1a0707" if (is_near and not is_d) else INNER3)
        row_h  = ch*0.036
        rbox(ax,LX,VP(fy)-row_h*0.45,IW,row_h,r=0.004,fc=row_bg,ec="none",lw=0,z=2)
        bdg(ax,LX+IW*0.070,VP(fy),
            "DEMAND" if is_d else "SUPPLY",ob_bg,ob_fg,sz=6.0)
        t(ax,LX+IW*0.210,VP(fy),
          f"{cs}{ob['lo']:.0f} – {cs}{ob['hi']:.0f}",sz=6.8,c=T1,bold=True)
        vol_c = R_FG if ob["vol_mult"]>=3.5 else A_FG if ob["vol_mult"]>=2.5 else T2
        t(ax,LX+IW*0.600,VP(fy),f"{ob['vol_mult']:.1f}x",sz=7.0,c=vol_c,bold=True)
        t(ax,LX+IW*0.740,VP(fy),f"{ob['sessions_ago']}d",sz=6.8,c=T3)
        bar_x = LX+IW*0.840; bar_w = IW*0.120
        bar_h = ch*0.012;    bar_y = VP(fy)-bar_h/2
        pb(ax,bar_x,bar_y,bar_w,bar_h,max(0,100-prox*8),ob_fg,z=4)
        if is_near:
            t(ax,LX+IW*0.968,VP(fy),"<",sz=8.5,c=ob_fg,bold=True)

    # ── OVERRIDE BOX ─────────────────────────────────────────
    ov = r.get("override",{"status":"DENIED","entry_lo":0,"entry_hi":0,
                "size_label":"","condition":"","vol_mult":0,
                "reason":"No order block data"})
    s  = ov["status"]
    ov_fg,ov_bg,ov_ec = {
        "PERMITTED":    (OV_PERMIT_FG, OV_PERMIT_BG, OV_PERMIT_BORDER),
        "HIGH_CONVICTION":(OV_HIGH_FG,OV_HIGH_BG,    OV_HIGH_BORDER),
        "DOUBLE_CONFIRM":(OV_DOUBLE_FG,OV_DOUBLE_BG, OV_DOUBLE_BORDER),
        "DENIED":       (OV_DENY_FG,  OV_DENY_BG,    OV_DENY_BORDER),
    }.get(s,(OV_DENY_FG,OV_DENY_BG,OV_DENY_BORDER))

    status_txt = {
        "PERMITTED":     "LIMIT ORDER PERMITTED",
        "HIGH_CONVICTION":"HIGH CONVICTION LIMIT",
        "DOUBLE_CONFIRM":"DOUBLE CONFIRMATION",
        "DENIED":        "NO OVERRIDE — WAIT",
    }.get(s,"NO OVERRIDE — WAIT")

    ov_h = ch*0.195
    ov_y = VP(0.016)
    hl(ax,LX,RX,VP(0.218))
    rbox(ax,LX,ov_y,IW,ov_h,r=0.008,fc=ov_bg,ec=ov_ec,lw=1.4,z=3)
    rbox(ax,LX,ov_y+ov_h-ch*0.024,IW,ch*0.024,
         r=0.007,fc=ov_ec+"55",ec="none",lw=0,z=4)

    t(ax,LX+IW*0.03,ov_y+ov_h-ch*0.012,
      "ORDER BLOCK OVERRIDE",sz=6.5,c=T3)
    t(ax,RX-IW*0.02,ov_y+ov_h-ch*0.012,
      "3x vol min threshold",sz=6.0,c=T3,ha="right")
    t(ax,LX+IW*0.03,ov_y+ov_h-ch*0.050,
      status_txt,sz=10.5,c=ov_fg,bold=True)

    if s != "DENIED":
        t(ax,LX+IW*0.03,ov_y+ov_h-ch*0.084,
          f"Limit:  {cs}{ov['entry_lo']:.2f} – {cs}{ov['entry_hi']:.2f}",
          sz=8.5,c=T1,bold=True)
        bdg(ax,RX-IW*0.06,ov_y+ov_h-ch*0.084,
            f"{ov['vol_mult']:.1f}x vol",ov_bg,ov_fg,sz=7.0,ec=ov_ec,elw=0.8)
        mb_w = (IW-cw*0.016)/2
        mb_h = ch*0.058
        mb_y = ov_y+ch*0.012
        for i,(lbl,val) in enumerate([
            ("Size",      ov["size_label"]),
            ("Condition", ov["condition"]),
        ]):
            mbx = LX+i*(mb_w+cw*0.016)
            rbox(ax,mbx,mb_y,mb_w,mb_h,r=0.005,fc=INNER2,ec="none",lw=0,z=5)
            t(ax,mbx+cw*0.014,mb_y+mb_h-ch*0.018,lbl,sz=6.2,c=T3)
            t(ax,mbx+cw*0.014,mb_y+ch*0.014,val, sz=7.5,c=ov_fg,bold=True)
    else:
        t(ax,LX+IW*0.03,ov_y+ov_h-ch*0.082,ov["reason"],sz=7.2,c=T2)
        t(ax,LX+IW*0.03,ov_y+ov_h-ch*0.114,
          "Bot trigger required before entry",sz=7.0,c=T3)

# =========================
# PAGE BUILDER
# =========================
def build_page(page_results, all_results, vol_sym, vol_val,
               vol_rname, vol_mult, ts, page_num, total_pages):
    FIG_W, FIG_H = 17.0, 15.5
    fig = plt.figure(figsize=(FIG_W,FIG_H), facecolor=BG)
    ax  = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off"); ax.set_facecolor(BG)

    M  = 0.026
    UW = 1-2*M
    y  = 0.984

    # header
    t(ax,M,y,"MARKET CYCLE DASHBOARD",sz=14,bold=True)
    hdr = f"{ts}  Page {page_num}/{total_pages}" if total_pages>1 else ts
    t(ax,1-M,y,hdr,sz=7.5,c=T3,ha="right")
    y -= 0.026

    # VIX strip
    sh = 0.055
    rbox(ax,M,y-sh,UW,sh,r=0.010,fc=CARD,ec=BORDER,lw=0.8)
    strip = [
        (vol_sym,  f"{vol_val:.1f}" if vol_val else "N/A", vol_color(vol_rname)),
        ("Regime", vol_rname,                               vol_color(vol_rname)),
        ("Tactical",f"{vol_mult:.2f}x",                    T1),
        ("Capital", f"£{int(CAPITAL_PER_TICKER)} / ticker", T1),
        ("Structure","50% Core  ·  50% Tactical",           T1),
    ]
    sw = UW/len(strip)
    for i,(lbl,val,c) in enumerate(strip):
        sx = M+i*sw+0.012
        t(ax,sx,y-sh*0.28,lbl,sz=6.2,c=T3)
        t(ax,sx,y-sh*0.70,val,sz=9.5,c=c,bold=True)
    y -= sh+0.018

    # cycle bar
    t(ax,M,y,"Cycle position",sz=6.5,c=T3)
    y -= 0.013
    seg_h = 0.030
    seg_w = UW/len(STAGE_NAMES)
    active = {r["stage_idx"] for r in all_results}
    for i,(name,(bg,fg)) in enumerate(zip(STAGE_NAMES,STAGE_COLORS)):
        sx = M+i*seg_w
        on = i in active
        rbox(ax,sx,y-seg_h,seg_w-0.001,seg_h,r=0.003,
             fc=bg,ec="#ffffff" if on else BORDER,lw=1.5 if on else 0.4)
        t(ax,sx+(seg_w-0.001)/2,y-seg_h/2,name,
          sz=6.0,c=fg,ha="center",bold=on)
    y -= seg_h+0.016

    # cards
    cols    = CARDS_PER_ROW
    rows    = ROWS_PER_PAGE
    gap_col = 0.012
    gap_row = 0.016
    card_w  = (UW-gap_col*(cols-1))/cols
    FOOTER  = 0.082
    card_h  = (y-FOOTER-gap_row*(rows-1))/rows

    for idx,r in enumerate(page_results):
        col = idx % cols
        row = idx // cols
        cx  = M+col*(card_w+gap_col)
        cy  = y-(row+1)*card_h-row*gap_row
        draw_ticker_card(ax,cx,cy,card_w,card_h,r)

    # footer summary
    rbox(ax,M,0,UW,FOOTER,r=0.008,fc=CARD,ec=BORDER,lw=0.8)
    t(ax,M+0.012,FOOTER-0.018,"PORTFOLIO SUMMARY",sz=7.5,c=T3,bold=True)

    tc     = sum(r["core_signal"]     for r in all_results)
    tt2    = sum(r["tactical_signal"] for r in all_results)
    cap    = len(all_results)*int(CAPITAL_PER_TICKER)
    trigs  = [r["symbol"] for r in all_results if r["tactical_action"]=="BUY"]
    ov_acts= [r["symbol"] for r in all_results
              if r.get("override",{}).get("status") in
              ("PERMITTED","HIGH_CONVICTION","DOUBLE_CONFIRM")]

    items = [
        ("Core signals today",     f"£{int(tc)}",    T1),
        ("Tactical signals today", f"£{int(tt2)}",   T1),
        ("Total capacity",         f"£{int(cap)}",   T1),
        ("OB overrides active",
         ", ".join(ov_acts) if ov_acts else "None",
         G_FG if ov_acts else T3),
        ("Tactical triggers",
         ", ".join(trigs) if trigs else "NONE — WAIT",
         R_FG if trigs else G_FG),
    ]
    sw2 = UW/len(items)
    for i,(lbl,val,c) in enumerate(items):
        sx = M+i*sw2+0.012
        t(ax,sx,FOOTER*0.64,lbl,sz=6.2,c=T3)
        t(ax,sx,FOOTER*0.28,val,sz=9.0,c=c,bold=True)

    confidence_dots(ax,M+0.014,FOOTER*0.10,n_on=7,n_total=10)
    t(ax,M+0.150,FOOTER*0.10,"7/10 confidence",sz=6.5,c=T2)

    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=220,facecolor=BG,
                bbox_inches=None,pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# CHUNK
# =========================
def chunk_results(results, size=MAX_CARDS_PER_PAGE):
    for i in range(0,len(results),size):
        yield results[i:i+size]

# =========================
# TELEGRAM
# =========================
def send_dashboard(img_buf, caption, filename="dashboard.png"):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
        data={"chat_id":TELEGRAM_CHAT_ID,"caption":caption[:950],
              "disable_web_page_preview":True},
        files={"document":(filename,img_buf.getvalue(),"image/png")},
        timeout=90,
    ).raise_for_status()

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in [text[i:i+3800] for i in range(0,len(text),3800)]:
        requests.post(url,data={"chat_id":TELEGRAM_CHAT_ID,"text":chunk,
                                "disable_web_page_preview":True},
                      timeout=30).raise_for_status()
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
        raise RuntimeError("All symbols failed:\n"+"\n".join(failures))

    pages  = list(chunk_results(results, MAX_CARDS_PER_PAGE))
    trigs  = [r["symbol"] for r in results if r["tactical_action"]=="BUY"]
    ov_act = [r["symbol"] for r in results
              if r.get("override",{}).get("status") in
              ("PERMITTED","HIGH_CONVICTION","DOUBLE_CONFIRM")]

    for idx, page_results in enumerate(pages, 1):
        buf = build_page(page_results, results, vol_sym, vol_val,
                         vol_rname, vol_mult, ts, idx, len(pages))
        syms = ", ".join(r["symbol"] for r in page_results)
        cap  = [
            f"Market Cycle Dashboard — {ts}",
            f"{vol_sym}: {f'{vol_val:.1f}' if vol_val else 'N/A'}"
            f" | {vol_rname} | x{vol_mult:.2f}",
            f"Page {idx}/{len(pages)} | {syms}",
            ("Tactical BUY: "+", ".join(trigs)) if trigs else "No tactical triggers",
            ("OB Override active: "+", ".join(ov_act)) if ov_act else "No OB overrides",
        ]
        for r in page_results:
            ov_s = r.get("override",{}).get("status","DENIED")
            cap.append(
                f"{r['symbol']}: {r['stage_name']} | "
                f"{action_now(r)[0]} | OB: {ov_s}"
            )
        send_dashboard(buf, "\n".join(cap), f"dashboard_p{idx}.png")
        time.sleep(0.4)

    if failures:
        send_message("Cycle bot failures:\n"+"\n".join(failures))

    print(f"Done. Pages: {len(pages)}. Triggers: {trigs}. OB overrides: {ov_act}")


if __name__ == "__main__":
    main()
