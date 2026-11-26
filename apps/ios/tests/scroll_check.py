#!/usr/bin/env python3
"""The chat view's jump checks on a UITEST harness recording (MovedStore/UITest.swift).

usage: scroll_check.py uitest-<scenario>.json [...] [--json summary.json]

Checks, by scenario (each sample is one display frame of the chat's scroll view):
- follow, keyboard (and every sample of any scenario while following, untouched): the distance from the bottom stays at
  its resting value within 1 pt.
- hold: after the user's drag ends, the tracked row (the top visible row) keeps its on-screen y within 1 pt while the
  reply below it changes (thinking collapse, tool card, sources, caption, keyboard, draft).
- hold, history: while a finger is down, the visible content moves with the finger (row y step = pan step within 2 pt);
  while the scroll decelerates, it moves one way (no step against the motion over 1 pt).
- open: per chat opened or switched, the first frame's distance from the bottom and the frames that move after it.
"""
import json
import statistics
import sys


def load(path):
    d = json.load(open(path))
    f = {k: i for i, k in enumerate(d["fields"])}
    s = [x for x in d["samples"] if len(x) == len(d["fields"])]
    return d, f, s


def near_event(events, t):
    best = None
    for et, name in events:
        if et <= t + 0.05:
            best = (et, name)
    return f"{best[1]} (+{t - best[0]:.2f} s)" if best else "-"


def resting_dist(f, s):
    """The distance from the bottom at rest while following (the list's bottom padding below the marker)."""
    vals = [round(x[f["dist"]]) for x in s if x[f["follow"]] and not x[f["tracking"]] and not x[f["decelerating"]]
            and fits(f, x)]
    return statistics.mode(vals) if vals else 0


def fits(f, x):
    """The content is taller than the visible area (there is a bottom to pin to)."""
    return x[f["content_h"]] + x[f["inset_top"]] + x[f["inset_bottom"]] > x[f["container_h"]] + 1


def follow_check(d, f, s, t_from=0.0, t_to=1e9):
    base = resting_dist(f, s)
    bad = []
    n = 0
    for x in s:
        t = x[f["t"]]
        if t < t_from or t > t_to or not x[f["follow"]] or x[f["tracking"]] or x[f["decelerating"]] or not fits(f, x):
            continue
        n += 1
        dev = x[f["dist"]] - base
        if abs(dev) > 1:
            bad.append((abs(dev), t, dev))
    # contiguous episodes off the bottom (gaps under 0.1 s join)
    eps = []
    for dev_abs, t, dev in sorted(bad, key=lambda b: b[1]):
        if eps and t - eps[-1]["end"] < 0.1:
            e = eps[-1]; e["end"] = t; e["frames"] += 1
            if dev_abs > abs(e["max_dev"]): e["max_dev"] = round(dev, 1)
        else:
            eps.append({"start": t, "end": t, "frames": 1, "max_dev": round(dev, 1), "at": near_event(d["events"], t)})
    for e in eps:
        e["seconds"] = round(e["end"] - e["start"] + 0.017, 2)
    eps.sort(key=lambda e: -abs(e["max_dev"]))
    return {"samples": n, "resting_dist": base, "over_1pt": len(bad), "max_pt": round(max(b[0] for b in bad), 1) if bad else 0.0,
            "episodes": len(eps), "worst_episodes": eps[:10]}


def finger_check(d, f, s, t_from=0.0, t_to=1e9):
    """While tracking: the top visible row's step equals the pan's step (same row in both frames)."""
    bad = []
    n = 0
    for a, b in zip(s, s[1:]):
        t = b[f["t"]]
        if t < t_from or t > t_to or not (a[f["tracking"]] and b[f["tracking"]]):
            continue
        if a[f["top_row"]] is None or a[f["top_row"]] != b[f["top_row"]]:
            continue
        if a[f["pan_y"]] == 0 or b[f["pan_y"]] == 0:  # the pan's first frames (its translation reads 0, then jumps)
            continue
        n += 1
        mis = (b[f["top_row_y"]] - a[f["top_row_y"]]) - (b[f["pan_y"]] - a[f["pan_y"]])
        if abs(mis) > 2:
            bad.append((abs(mis), t, mis))
    bad.sort(reverse=True)
    return {"frames": n, "over_2pt": len(bad), "max_pt": round(bad[0][0], 1) if bad else 0.0,
            "worst": [{"t": t, "mismatch": round(m, 1), "at": near_event(d["events"], t)} for _, t, m in bad[:6]]}


def decel_check(d, f, s, t_from=0.0, t_to=1e9):
    """While decelerating: the visible content moves one way."""
    bad = []
    n = 0
    run = []
    def close(run):
        if len(run) < 3:
            return
        total = sum(v for _, v in run)
        sign = 1 if total >= 0 else -1
        for t, v in run:
            if v * sign < -1:
                bad.append((abs(v), t, v))
    for a, b in zip(s, s[1:]):
        t = b[f["t"]]
        if t < t_from or t > t_to or not (b[f["decelerating"]] and not b[f["tracking"]]):
            close(run); run = []
            continue
        if a[f["top_row"]] is None or a[f["top_row"]] != b[f["top_row"]]:
            continue
        n += 1
        run.append((t, b[f["top_row_y"]] - a[f["top_row_y"]]))
    close(run)
    bad.sort(reverse=True)
    return {"frames": n, "against_motion_over_1pt": len(bad), "max_pt": round(bad[0][0], 1) if bad else 0.0,
            "worst": [{"t": t, "step": round(v, 1), "at": near_event(d["events"], t)} for _, t, v in bad[:6]]}


def hold_check(d, f, s):
    start = next((t for t, e in d["events"] if e.startswith("hold start")), None)
    if start is None:
        return {"error": "no hold"}
    end = next((t for t, e in d["events"] if e == "script done"), 1e9)
    still = [x for x in s if start <= x[f["t"]] <= end and not x[f["tracking"]] and not x[f["decelerating"]]
             and x[f["tracked_y"]] is not None]
    if not still:
        return {"error": "tracked row never on screen"}
    y0 = still[0][f["tracked_y"]]
    moves = []
    for a, b in zip(still, still[1:]):
        dy = b[f["tracked_y"]] - a[f["tracked_y"]]
        if abs(dy) > 1:
            moves.append((abs(dy), b[f["t"]], dy))
    drift = max(abs(x[f["tracked_y"]] - y0) for x in still)
    moves.sort(reverse=True)
    touched_again = any(x[f["tracking"]] for x in s if start <= x[f["t"]] <= end)
    return {"samples": len(still), "frames_moved_over_1pt": len(moves), "max_step_pt": round(moves[0][0], 1) if moves else 0.0,
            "max_drift_pt": round(drift, 1), "touched_again": touched_again,
            "follow_during_hold": any(x[f["follow"]] for x in still),
            "worst": [{"t": t, "step": round(dy, 1), "at": near_event(d["events"], t)} for _, t, dy in moves[:8]]}


def open_check(d, f, s):
    out = []
    marks = [(t, e) for t, e in d["events"] if e.split(" ")[0] in ("switch", "push")]
    ends = [t for t, e in d["events"] if e.split(" ")[0] in ("switch", "push", "back", "touch", "script")] + [1e9]
    base = resting_dist(f, s)
    for t0, name in marks:
        t1 = min(t for t in ends if t > t0 + 0.01)
        win = [x for x in s if t0 <= x[f["t"]] < t1 and x[f["content_h"]] > 0]
        if not win:
            out.append({"event": name, "error": "no frames"})
            continue
        conv = win[-1][f["conv"]]
        win = [x for x in win if x[f["conv"]] == conv]
        first = win[0]
        moved = [x for a, x in zip(win, win[1:]) if abs(x[f["offset"]] - a[f["offset"]]) > 0.5]
        travel = sum(abs(x[f["offset"]] - a[f["offset"]]) for a, x in zip(win, win[1:]))
        out.append({"event": name, "first_frame_s": round(first[f["t"]] - t0, 3),
                    "first_frame_dist": round(first[f["dist"]] - base, 1), "frames_moving_after_first": len(moved),
                    "offset_travel_after_first_pt": round(travel, 1),
                    "last_move_s": round(moved[-1][f["t"]] - t0, 3) if moved else 0.0,
                    "final_dist": round(win[-1][f["dist"]] - base, 1)})
    return out


def hitches(f, s, t_from=0.0, t_to=1e9):
    ts = [x[f["t"]] for x in s if t_from <= x[f["t"]] <= t_to]
    dts = [b - a for a, b in zip(ts, ts[1:])]
    if not dts:
        return {}
    med = statistics.median(dts)
    return {"frames": len(dts), "frame_s": round(med, 4), "over_1.5x": sum(dt > 1.5 * med for dt in dts),
            "longest_s": round(max(dts), 3)}


def check(path):
    d, f, s = load(path)
    sc = d["scenario"]
    r = {"scenario": sc, "os": d.get("os"), "legacy_follow_path": d.get("legacy"), "frames": len(s)}
    end = next((t for t, e in d["events"] if e == "script done"), 1e9)
    if sc in ("follow", "keyboard"):
        r["follow"] = follow_check(d, f, s, t_to=end)
    if sc == "hold":
        start = next((t for t, e in d["events"] if e.startswith("hold start")), 1e9)
        r["follow_before_drag"] = follow_check(d, f, s, t_to=next((t for t, e in d["events"] if e.startswith("touch 1")), 1e9))
        r["drag_finger"] = finger_check(d, f, s, t_to=start)
        r["hold"] = hold_check(d, f, s)
    if sc == "history":
        r["scroll_finger"] = finger_check(d, f, s)
        r["scroll_decel"] = decel_check(d, f, s)
        first = next((t for t, e in d["events"] if e.startswith("touch 1")), 0)
        r["content_h_changes_while_scrolling"] = sum(
            1 for a, b in zip(s, s[1:]) if b[f["t"]] >= first and abs(b[f["content_h"]] - a[f["content_h"]]) > 0.5)
        r["hitches_while_scrolling"] = hitches(f, s, first, end)
        r["open"] = open_check(d, f, s) if any(e.startswith("switch") for _, e in d["events"]) else None
    if sc == "open":
        r["open"] = open_check(d, f, s)
        r["hitches"] = hitches(f, s, 0, end)
    return r


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = [check(p) for p in paths]
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    print(json.dumps(out, indent=1))
