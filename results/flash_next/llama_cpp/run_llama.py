#!/usr/bin/env python3
"""Run one llama.cpp command with a time limit and a memory guard; log everything; print the timing lines.
usage: run_llama.py NAME TIMEOUT_S -- cmd args...
Guard: the command is killed if swap grows by more than SWAP_MB (default 6000) over its starting value or free disk
falls under DISK_GB (default 15). The first unguarded run (-ngl 0, load mode auto) took swap from 2.8 to 40 GB."""
import os, re, shutil, signal, subprocess, sys, time
name, limit = sys.argv[1], int(sys.argv[2]); cmd = sys.argv[sys.argv.index("--") + 1:]
SWAP_MB, DISK_GB = float(os.environ.get("SWAP_MB", 6000)), float(os.environ.get("DISK_GB", 15))
log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", name + ".log")
def swap():
    return float(re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout).group(1))
def rss(pid):
    o = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(o) / 1e6 if o else 0.0
s0, t0, why, peak_swap, peak_rss = swap(), time.time(), None, 0.0, 0.0
with open(log, "w") as f:
    f.write("CMD: " + " ".join(cmd) + "\n"); f.flush()
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    while p.poll() is None:
        time.sleep(2)
        s, d = swap(), shutil.disk_usage("/").free / 1e9
        peak_swap, peak_rss = max(peak_swap, s - s0), max(peak_rss, rss(p.pid))
        if s - s0 > SWAP_MB: why = f"swap grew {s - s0:.0f} MB"
        elif d < DISK_GB: why = f"disk free {d:.1f} GB"
        elif time.time() - t0 > limit: why = "time limit"
        if why:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); break
print(f"{name}: rc={p.returncode} killed_by_guard={why} wall={time.time() - t0:.0f}s swap_start={s0:.0f}MB peak_swap_growth={peak_swap:.0f}MB peak_rss={peak_rss:.1f}GB log={log}")
lines = open(log, errors="replace").read().splitlines()
print(f"  ({len(lines)} log lines)")
for l in lines:
    if any(k in l for k in ("eval time", "load time", "total time", "tokens per second", "rror", "failed", "offload", "n_threads", "mmap", "load_mode", "lazy", "ple", "Metal", "fit", "system_info", "buffer size", "| ")):
        print("  " + l[:240])
