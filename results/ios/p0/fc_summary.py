"""Summaries of the phone's -fileCacheTest JSONs (phone/filecache-*.json): python3 fc_summary.py FILE..."""
import json, sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    r = d.get("result", d)
    ab, aa = d.get("app_before", {}), d.get("app_after", {})
    print(f"\n### {p.split('/')[-1]}: variant {r.get('variant')}")
    print(f"- {ab.get('os')} {ab.get('machine')}, battery {ab.get('battery_level')} {ab.get('battery_state')}, thermal {ab.get('thermal')} -> {aa.get('thermal')}, "
          f"low power {ab.get('low_power')}")
    print(f"- mapped {r.get('files_mapped')} of {r.get('files_given')} files, {r.get('mapped_gb', 0):.2f} GB ({r.get('mapping')}); map error: '{r.get('map_error')}'; "
          f"resident at start {r.get('resident_gb_at_start', 0):.2f} GB")
    print(f"- footprint start {r.get('footprint_start', 0) / 1e9:.3f} GB, end {r.get('footprint_end', 0) / 1e9:.3f}; available start "
          f"{r.get('available_start', 0) / 1e9:.2f} GB, end {r.get('available_end', 0) / 1e9:.2f}; warnings {r.get('memory_warnings')}, pressure events {r.get('pressure_events')}")
    if "dirty_gb" in r:
        print(f"- dirty {r['dirty_gb']:.2f} GB held: footprint {r['footprint_after_dirty'] / 1e9:.2f} GB, available {r['available_after_dirty'] / 1e9:.2f} GB")
    cols = r.get("samples_columns", [])
    S = [dict(zip(cols, s)) for s in r.get("samples", [])]
    if S:
        res = [s["resident_gb"] for s in S]
        fp = [s["footprint"] for s in S]
        av = [s["available"] for s in S]
        print(f"- per-second samples {len(S)}: resident max {max(res):.2f} GB, last {res[-1]:.2f}; footprint {min(fp) / 1e9:.3f}-{max(fp) / 1e9:.3f} GB; "
              f"available {min(av) / 1e9:.2f}-{max(av) / 1e9:.2f} GB; early-set resident max {max(s['early_set_resident_gb'] for s in S):.2f} -> last {S[-1]['early_set_resident_gb']:.2f}")
        ph = {}
        for s in S:
            ph.setdefault(s["phase"], []).append(s)
        for k, v in ph.items():
            c = [x["cold_gbps_per_thread"] for x in v if x["cold_gbps_per_thread"] > 0]
            w = [x["warm_gbps_per_thread"] for x in v if x["warm_gbps_per_thread"] > 0]
            print(f"  - {k}: {len(v)} s, resident {v[0]['resident_gb']:.2f} -> {v[-1]['resident_gb']:.2f} GB, touched {v[-1]['touched_gb']:.2f} GB, "
                  f"cold {sum(c) / max(1, len(c)):.3f} / warm {sum(w) / max(1, len(w)):.2f} GB/s per thread, thermal {v[-1]['thermal']}")
    for k in ("sweep", "decode_random_all", "retouch_resident", "retouch_after_return"):
        if k in r:
            print(f"- {k}: " + ", ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}" for a, b in r[k].items()))
    for k in ("before_background_touched_set_resident_gb", "after_return_touched_set_resident_gb", "before_background_resident_gb",
              "after_return_resident_gb", "background_seconds", "backgrounded"):
        if k in r:
            print(f"- {k}: {r[k]}")
    if "single" in r:
        print("- single-record cost (median wall ms; mean user / sys ms; faults, page-ins per record; resident fraction after):")
        for path, v in r["single"].items():
            if isinstance(v, dict) and "cold" in v:
                for kk in ("cold", "second_access"):
                    x = v[kk]
                    print(f"  - {path} {kk}: {x['wall_ms_median']:.2f} ms, user {x['user_ms_mean']:.2f} sys {x['sys_ms_mean']:.2f}, faults {x['faults_per_record']:.1f}, "
                          f"pageins {x['pageins_per_record']:.1f}, resident after {x['resident_frac_after']:.2f}, {x['gbps_single_thread']:.2f} GB/s")
            else:
                x = v
                print(f"  - {path}: {x['wall_ms_median']:.2f} ms, user {x['user_ms_mean']:.2f} sys {x['sys_ms_mean']:.2f}, faults {x['faults_per_record']:.1f}, {x['gbps_single_thread']:.2f} GB/s")
    for w in r.get("willneed", []):
        print(f"  - willneed k={w['in_flight']}: advise {w['advise_ms']:.2f} ms, advise+touch {w['advise_then_touch_gbps']:.2f} GB/s, advise+wait {w['advise_then_wait_gbps']:.2f} GB/s (frac {w['wait_resident_frac']:.2f})")
    for t in r.get("throughput", []):
        print(f"  - throughput {t['path']} x{t['threads']}: {t['gbps']:.2f} GB/s, user {t['cpu_user_s_per_gb']:.3f} sys {t['cpu_sys_s_per_gb']:.3f} s/GB, faults/rec {t['faults_per_record']:.1f}")
    if "interference" in r:
        I = r["interference"]
        print(f"- interference: compute {I['compute']} on {I['compute_threads']} threads ({I['compute_qos']}), {I['readers']} readers, {I['window_s']} s windows")
        for w in I["windows"]:
            print(f"  - {w['window']:34s} compute {w['compute_gbps']:.2f} GB/s, reads {w['read_gbps']:.2f} GB/s, sys {w['task_cpu_sys_s']:.2f} s, user {w['task_cpu_user_s']:.2f} s, "
                  f"faults {w['faults']}, pageins {w['pageins']}, fp {w['footprint_end'] / 1e9:.2f}, avail {w['available_end'] / 1e9:.2f}, hot {w['hot_set_resident_gb']:.2f} GB, {w['thermal']}")
