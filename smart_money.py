"""
Builds smart_money.json for smart-money.html (free Yahoo data only).
Per symbol: last ~20 sessions of 1H bars, unmitigated structure order blocks (1H/2H/4H/1D),
absorption bars (high volume, small range, at the swing extremes), volume at price,
anchored VWAP from the window low, and a plain evidence score. Display only.
"""
import json, os, sys, time
import numpy as np, pandas as pd
import requests
import cycle_bot as cb

BARS = 140                      # ~20 sessions of 1H
MARKET = "SPY"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "smart_money.json")
GRAVITY_URL = "https://javed-glitch.github.io/options-gravity-4-week-forward-scan-v3/data.json"

def _gravity():
    """Options positioning from the gravity board (read only). Empty if unreachable."""
    try:
        return requests.get(GRAVITY_URL, timeout=20).json().get("G", {})
    except Exception as e:
        print(f"smart_money: gravity board unavailable: {e}")
        return {}

def _cycle():
    """Cycle bot stage and action per symbol, from the file the cycle job just wrote."""
    try:
        d = json.load(open(os.path.join(HERE, "cycle_data.json")))["results"]
        rows = d if isinstance(d, list) else list(d.values())
        return {r["symbol"]: r for r in rows}
    except Exception as e:
        print(f"smart_money: cycle data unavailable: {e}")
        return {}

def _cluster(cands, last):
    cands = sorted(cands, key=lambda c: c["lo"])
    out = []
    for c in cands:
        if out and (c["lo"] <= out[-1]["hi"] or (c["lo"] <= out[-1]["hi"] + last*0.006 and max(out[-1]["hi"], c["hi"]) - out[-1]["lo"] <= last*0.04)):
            o = out[-1]
            o["hi"] = max(o["hi"], c["hi"]); o["srcs"].add(c["src"]); o["why"].append(c["why"])
        else:
            out.append({"lo": c["lo"], "hi": c["hi"], "srcs": {c["src"]}, "why": [c["why"]]})
    return out

def _zones(sym, last, flags, blocks, profile, grav, cyc):
    """Merge every footprint into plain buy zones (below price) and sell zones (above price)."""
    buy, sell = [], []
    bw = last*0.005
    for f in flags:
        if f["excess"] < 1.5 or not f["held"]: continue
        if f["kind"] == "ACCUM":
            buy.append({"lo": f["lo"], "hi": f["hi"], "src": "volume", "why": "Heavy buying volume came in here and price held"})
        else:
            sell.append({"lo": f["lo"], "hi": f["hi"], "src": "volume", "why": "Heavy selling volume came in here and price stayed capped"})
    for tf, bl in blocks.items():
        for b in bl:
            c = {"lo": b["lo"], "hi": b["hi"], "src": "turn"}
            if b["type"] == "demand": buy.append({**c, "why": "Price turned back up from here before"})
            else: sell.append({**c, "why": "Price was pushed back down from here before"})
    if profile:
        mx = max(v for _, v in profile)
        for i, (p, v) in enumerate(profile):
            nb = [profile[j][1] for j in range(max(0, i-2), min(len(profile), i+3)) if j != i]
            if v >= 0.55*mx and all(v >= x for x in nb):
                c = {"lo": p-bw, "hi": p+bw, "src": "crowd", "why": "A lot of shares changed hands at this price"}
                (buy if p < last else sell).append(c)
    opt = None
    g = grav.get(sym)
    if g:
        exps = sorted(g)[:2]
        walls_p, walls_c = [], []
        for e in exps:
            r = g[e]
            walls_p += [(w["strike"], w["openInterest"]) for w in (r.get("putWall") or [])]
            walls_c += [(w["strike"], w["openInterest"]) for w in (r.get("callWall") or [])]
        for k, oi in sorted(walls_p, key=lambda x: -x[1])[:3]:
            if last*0.80 <= k < last: buy.append({"lo": k*0.995, "hi": k*1.005, "src": "options", "why": f"Options traders have big bets near ${k:g}"})
        for k, oi in sorted(walls_c, key=lambda x: -x[1])[:3]:
            if last < k <= last*1.25: sell.append({"lo": k*0.995, "hi": k*1.005, "src": "options", "why": f"Options traders have big bets near ${k:g}"})
        r0 = g[exps[0]]
        if r0.get("maxPain"): opt = {"max_pain": r0["maxPain"], "expiry": exps[0]}
    def finish(cands, side):
        cl = _cluster([c for c in cands if (c["hi"] <= last*1.01 if side == "buy" else c["lo"] >= last*0.99)
                       and abs((c["lo"]+c["hi"])/2/last-1) <= 0.25], last)
        res = []
        for z in cl:
            n = len(z["srcs"])
            inside = z["lo"] <= last <= z["hi"]
            dist = 0.0 if inside else min(abs(last-z["lo"]), abs(last-z["hi"]))/last*100
            zlo, zhi = z["lo"], z["hi"]
            if zhi - zlo > last*0.05:             # very wide: keep the 5% slice nearest to price
                if side == "buy": zlo = zhi - last*0.05
                else: zhi = zlo + last*0.05
            res.append({"lo": round(zlo, 2), "hi": round(zhi, 2), "n": n,
                        "strength": "Strong" if n >= 3 else "Medium" if n == 2 else "Light",
                        "why": list(dict.fromkeys(z["why"]))[:4], "dist_pct": round(dist, 1), "inside": inside})
        res.sort(key=lambda z: z["dist_pct"])
        return res[:3]
    c = cyc.get(sym) or {}
    cycle = {"stage": c.get("stage_name"), "core": c.get("core_action"), "tactical": c.get("tactical_action")} if c else None
    return {"buy": finish(buy, "buy"), "sell": finish(sell, "sell")}, opt, cycle


def _rvol(df):
    """Volume vs the median of the same hour over the prior 15 sessions."""
    out = pd.Series(np.nan, index=df.index)
    slot = df.index.hour
    for h in np.unique(slot):
        g = df["v"][slot == h]
        out[g.index] = g / g.shift().rolling(15, min_periods=8).median()
    return out

def _blocks(df1h, df1d, close):
    d2 = cb.resample_ohlcv(df1h, "2h")
    d4 = cb.resample_ohlcv(df1h, "4h")
    res = {}
    for tf, df, size, hrs in (("1H", df1h, 5, 1), ("2H", d2, 5, 2), ("4H", d4, 5, 4), ("1D", df1d, 5, 24)):
        try:
            bl = cb._smc_blocks_one(df, size, tf, hrs)
        except Exception:
            bl = []
        keep = []
        for b in bl:
            if b["type"] == "demand" and close < b["lo"]: continue
            if b["type"] == "supply" and close > b["hi"]: continue
            b["inside"] = b["lo"] <= close <= b["hi"]
            keep.append(b)
        res[tf] = keep[-6:]
    return res

def analyse(sym, mkt_rvol, grav, cyc):
    df = cb.fetch_series_yahoo(sym, "1h", 0)
    d1 = cb.fetch_series_yahoo(sym, "1day", 0)
    for d in (df, d1):
        d["v"] = pd.to_numeric(d["v"], errors="coerce").fillna(0) if "v" in d.columns else 0.0
    if len(df) < 150: return None
    last = float(df["c"].iloc[-1])
    pc = df["c"].shift()
    tr = pd.concat([df["h"]-df["l"], (df["h"]-pc).abs(), (df["l"]-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    rv = _rvol(df)
    mk = mkt_rvol.reindex(df.index)
    w = df.tail(BARS)
    lo_w, hi_w = float(w["l"].min()), float(w["h"].max())
    n = len(df)
    flags = []
    for k in range(n-BARS, n-1):
        r = df.iloc[k]
        if not (rv.iloc[k] >= 2.0 and (r["h"]-r["l"]) <= 1.3*atr.iloc[k]): continue
        m = mk.iloc[k]
        excess = float(rv.iloc[k]/m) if m and m > 0 else float(rv.iloc[k])
        pos = (r["c"]-lo_w)/(hi_w-lo_w) if hi_w > lo_w else 0.5
        after = df.iloc[k+1:]
        if pos <= 0.35:
            kind, held, moved = "ACCUM", bool(after["c"].min() >= r["l"]), bool(last > r["h"])
        elif pos >= 0.65:
            kind, held, moved = "DISTR", bool(after["c"].max() <= r["h"]), bool(last < r["l"])
        else:
            continue
        flags.append({"kind": kind, "ts": int(df.index[k].timestamp()*1000), "lo": round(float(r["l"]), 2),
                      "hi": round(float(r["h"]), 2), "rvol": round(float(rv.iloc[k]), 1),
                      "excess": round(excess, 1), "held": held, "moved": moved})
    # volume at price
    bw = max(last*0.005, 0.01)
    price = (w["h"]+w["l"]+w["c"])/3
    bins = np.floor(price/bw)*bw
    vp = w["v"].groupby(bins.values).sum().sort_index()
    poc = float(vp.idxmax()+bw/2) if vp.sum() > 0 else last
    cum, va, tot = 0, [], vp.sum()
    for p in vp.sort_values(ascending=False).index:
        va.append(p); cum += vp[p]
        if cum >= 0.7*tot: break
    # anchored VWAP from the window low
    j = int(np.argmin(w["l"].values))
    seg = w.iloc[j:]
    tp = (seg["h"]+seg["l"]+seg["c"])/3
    avwap = float((tp*seg["v"]).sum()/seg["v"].sum()) if seg["v"].sum() > 0 else last
    # evidence score (market-wide bars do not count)
    real = [f for f in flags if f["excess"] >= 1.5]
    acc = [f for f in real if f["kind"] == "ACCUM" and f["held"]]
    dis = [f for f in real if f["kind"] == "DISTR" and f["held"]]
    score, why = 0, []
    if acc: score += 1; why.append(f"{len(acc)} absorption bar(s) at the lows that held")
    if any(f["moved"] for f in acc): score += 1; why.append("price now above the absorption zone")
    if last > avwap: score += 1; why.append("above anchored VWAP")
    else: why.append("below anchored VWAP")
    if dis: score -= 1; why.append(f"{len(dis)} distribution bar(s) at the highs")
    if not real: why.append("no stock-specific high-volume bars")
    verdict = "ACCUMULATION" if score >= 3 else "BUILDING" if score == 2 else "NEUTRAL" if score == 1 else "WEAK"
    flow = {"ACCUMULATION": "IN", "BUILDING": "SOME", "NEUTRAL": "QUIET", "WEAK": "NONE"}[verdict]
    if dis and score <= 0: flow = "OUT"
    blocks = _blocks(df, d1, last)
    prof = [[round(float(p+bw/2), 2), int(v)] for p, v in vp.items()]
    zones, opt, cycle = _zones(sym, last, flags, blocks, prof, grav, cyc)
    bars = [[int(t.timestamp()*1000), round(float(r.o), 2), round(float(r.h), 2), round(float(r.l), 2),
             round(float(r.c), 2), int(r.v)] for t, r in w.iterrows()]
    return {
        "symbol": sym, "last": round(last, 2), "ccy": cb.csym(sym) if hasattr(cb, "csym") else "$",
        "bars": bars, "blocks": blocks, "zones": zones, "options": opt, "cycle": cycle, "flow": flow,
        "absorption": flags[-8:], "poc": round(poc, 2),
        "va_lo": round(float(min(va)), 2), "va_hi": round(float(max(va)+bw), 2),
        "profile": prof,
        "avwap": round(avwap, 2), "avwap_ts": int(w.index[j].timestamp()*1000),
        "score": score, "verdict": verdict, "why": why,
    }

def main():
    t0 = time.time()
    mdf = cb.fetch_series_yahoo(MARKET, "1h", 0)
    mdf["v"] = pd.to_numeric(mdf["v"], errors="coerce").fillna(0)
    mkt = _rvol(mdf)
    grav, cyc = _gravity(), _cycle()
    out = {"generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "market": MARKET, "results": []}
    for s in cb.SYMBOLS:
        if s in cb.ETF_PAGE_SYMBOLS:      # London ETFs: no US-hours market filter, not the use case
            continue
        try:
            r = analyse(s, mkt, grav, cyc)
            if r: out["results"].append(r)
        except Exception as e:
            print(f"smart_money: {s} skipped: {e}")
    order = {"ACCUMULATION": 0, "BUILDING": 1, "NEUTRAL": 2, "WEAK": 3}
    out["results"].sort(key=lambda r: (order[r["verdict"]], -r["score"]))
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"smart_money.json: {len(out['results'])} symbols, {os.path.getsize(OUT)//1024} KB, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
