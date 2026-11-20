#!/usr/bin/env python3
"""cycle_state.py: the weekly pacing schedule for the Market Cycle Dashboard (added 20 Nov 2026).

Reads cycle_data.json (written by cycle_bot.py) and keeps cycle_state.json: for each symbol that is in an ADD or
DOUBLE DOWN zone, when its episode started, which weeks it has been active, one tranche per week (and one extra if the
zone deepens from ADD to DOUBLE DOWN inside a week), the planned total against the core and tactical caps, and what is
due this week.

It knows NOTHING about what was bought. It is a schedule derived from market data only (the same signal the page
already shows), so it exposes no holdings. "Acted" marks live elsewhere (a private store, see the page).

Rules (owner, 20 Nov 2026):
- one tranche per calendar week (Monday to Sunday, UK time) while the symbol stays in the zone;
- an episode ends when the symbol has been out of ADD / DOUBLE DOWN for 2 runs in a row;
- planned totals stop at the caps (core_total, tactical_total per symbol): the due amount shrinks, then "capped".

Safe by design: it never edits cycle_data.json, and any failure leaves the previous cycle_state.json untouched.
Usage: python cycle_state.py [cycle_data.json] [cycle_state.json]
"""
import datetime as dt
import json
import os
import sys
from zoneinfo import ZoneInfo

ZONES = ("ADD", "DOUBLE DOWN")
OUT_RUNS = 2          # runs out of zone before an episode ends
LONDON = ZoneInfo("Europe/London")


def week_id(d):
    y, w, _ = d.isocalendar()
    return "%d-W%02d" % (y, w)


def week_start(d):
    return d - dt.timedelta(days=d.weekday())


def load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def step(prev, results, now_utc):
    """Pure function: previous state + today's results -> new state."""
    today = now_utc.astimezone(LONDON).date()
    wk = week_id(today)
    old = (prev or {}).get("symbols", {})
    new = {}
    for r in results:
        sym = r.get("symbol")
        zone = r.get("cycle_zone")
        if not sym:
            continue
        p = old.get(sym)
        in_zone = zone in ZONES
        if not in_zone:
            if p and p.get("outRuns", 0) + 1 < OUT_RUNS:
                q = dict(p)
                q["outRuns"] = p.get("outRuns", 0) + 1   # still held open: one miss allowed
                q["zone"], q["stage"] = zone, r.get("stage_name")
                q["status"] = "paused"
                new[sym] = q
            continue
        core_sig = int(r.get("core_signal") or 0)
        tac_sig = int(r.get("tactical_signal") or 0)
        cap_core = int(r.get("core_total") or 0)
        cap_tac = int(r.get("tactical_total") or 0)
        e = dict(p) if p else {"episodeStart": today.isoformat(), "tranches": []}
        e.update({"zone": zone, "stage": r.get("stage_name"), "outRuns": 0, "lastSeen": today.isoformat(),
                  "capCore": cap_core, "capTactical": cap_tac})
        tr = {(t["week"], t["zone"]): t for t in e.get("tranches", [])}
        key = (wk, zone)
        # the amount for this week's tranche follows today's signal until the week ends
        tr[key] = {"week": wk, "zone": zone, "core": core_sig, "tactical": tac_sig}
        e["tranches"] = sorted(tr.values(), key=lambda t: (t["week"], t["zone"]))
        # planned so far = every tranche from earlier weeks / zones + this week's, each limited by the caps
        pc = pt = 0
        due_core = due_tac = 0
        for t in e["tranches"]:
            c = max(0, min(t["core"], cap_core - pc))
            a = max(0, min(t["tactical"], cap_tac - pt))
            t["plannedCore"], t["plannedTactical"] = c, a
            pc += c
            pt += a
            if t["week"] == wk:
                due_core += c
                due_tac += a
        e["plannedCore"], e["plannedTactical"] = pc, pt
        e["dueThisWeek"] = {"core": due_core, "tactical": due_tac, "total": due_core + due_tac}
        e["weeksActive"] = len({t["week"] for t in e["tranches"]})
        capped = pc >= cap_core and pt >= cap_tac and (cap_core + cap_tac) > 0
        e["status"] = "capped" if (capped and e["dueThisWeek"]["total"] == 0) else "due"
        new[sym] = e
    return {"version": 1, "asOf": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "week": wk,
            "weekStart": week_start(today).isoformat(),
            "note": "Schedule only, derived from the market scan. Tracking began 2026-11-20; earlier weeks are not counted.",
            "symbols": new}


def main(argv):
    src = argv[1] if len(argv) > 1 else "cycle_data.json"
    dst = argv[2] if len(argv) > 2 else "cycle_state.json"
    data = load(src, None)
    if not data or not isinstance(data.get("results"), list) or not data["results"]:
        print("cycle_state: no usable results in", src, "- leaving", dst, "unchanged")
        return 0
    prev = load(dst, {})
    state = step(prev, data["results"], dt.datetime.now(dt.timezone.utc))
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, dst)
    n = len(state["symbols"])
    print("cycle_state: %d symbols in an ADD / DOUBLE DOWN episode, week %s" % (n, state["week"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as ex:            # never break the daily job
        print("cycle_state: skipped (%s)" % ex)
        sys.exit(0)
