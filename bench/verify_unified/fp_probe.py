#!/usr/bin/env python3
"""Footprint of the model worker around one prompt: before it, its peak while the prompt runs, and S seconds after the
first token (the worker idle by then), sampled every 20 ms with proc_pid_rusage (phys_footprint, the figure iOS limits).
Optionally `vmmap --summary` of the live worker before the request and after the wait.
  fp_probe.py OUT.json IDS_FILE MAX_NEW WAIT_S VMMAP_PREFIX|- -- WORKER --model DIR [flags]"""
import ctypes, json, subprocess, sys, threading, time

out, idsf, max_new, wait_s, vmp = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
cmd = sys.argv[sys.argv.index("--") + 1:]


class RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size", "phys_footprint",
        "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance",
        "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
        "billed_system_time", "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint")]


LIB = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def fp(pid):
    ru = RU()
    return ru.phys_footprint / 1e9 if LIB.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(4), ctypes.byref(ru)) == 0 else 0.0


p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
ready = p.stdout.readline()
assert ready.startswith("READY"), ready
time.sleep(3)
samples, stop = [], [False]


def sampler():
    while not stop[0]:
        samples.append((time.time(), fp(p.pid)))
        time.sleep(0.02)


def vmmap(tag):
    if vmp != "-":
        open(f"{vmp}_{tag}.txt", "w").write(subprocess.run(["vmmap", "--summary", str(p.pid)], capture_output=True, text=True).stdout)


before = fp(p.pid)
vmmap("before")
threading.Thread(target=sampler, daemon=True).start()
ids = [int(x) for x in open(idsf).read().replace(",", " ").split()]
t0 = time.time()
p.stdin.write(f"R 1 {max_new} 0 1 0 1 {len(ids)} 0 0 0\n{' '.join(map(str, ids))}\n\n\n[]\n")
p.stdin.flush()
t_first, end, b = None, None, None
for line in p.stdout:
    if line.startswith("T ") and t_first is None:
        t_first = time.time()
    elif line.startswith("B "):
        b = json.loads(line.split(" ", 2)[2])
    elif line.startswith("E "):
        end = json.loads(line.split(" ", 3)[3]) if len(line.split(" ", 3)) > 3 else {}
        break
t_end = time.time()
time.sleep(max(0.0, t_first + wait_s - time.time()))
after = fp(p.pid)
vmmap("after")
stop[0] = True
pre = [f for t, f in samples if t <= t_first]
res = {"cmd": cmd, "prompt": len(ids), "before_gb": round(before, 3), "prompt_peak_gb": round(max(pre) if pre else 0, 3),
       "peak_gb": round(max(f for _, f in samples), 3), f"after_{wait_s:g}s_gb": round(after, 3), "ttft_s": round(t_first - t0, 2),
       "request_s": round(t_end - t0, 2), "B": b, "mem": (end or {}).get("mem")}
p.stdin.write("Q\n")
p.stdin.flush()
p.wait(60)
json.dump(res, open(out, "w"))
print(json.dumps({k: v for k, v in res.items() if k not in ("cmd", "B")}))
