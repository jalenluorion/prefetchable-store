#!/bin/zsh
# FILE (adaptive budget) vs MLOCK on the Mac, for a quiet machine (the engine session runs it). Resumable: a run whose
# log already holds a GUARD line is skipped. Every run goes through the engine's guard (bench/guard.py), one at a time,
# 400 tokens, lossless, greedy, ctx 16384, i16, the common store, decode naming early with mlp_k2_h512 (m 8).
#
#   main      per prompt (pitt, code), two repetitions of the interleaved order
#             MLOCK 3.3 | FILE adaptive | MLOCK 5.0 | FILE adaptive   (FILE: start and ceiling 8 GB, floor 1 GB)
#   pressure  (optional) FILE adaptive on pitt while a helper holds 0 / 2 / 4 / 6 GB of dirty memory
#   core      (optional) FILE adaptive + a locked core of the hottest records, 0.5 and 1.0 GB, pitt and code
#
#   zsh results/ios/p1/file_vs_mlock.sh [main] [pressure] [core]       (default: main)
# Refuses to start while a model process (dmeng, dmink, dmserve, dm_gate, inklm_check, PhoneStoreApp) runs or the
# engine's QUIET file is fresher than 12 minutes. Outputs: results/ios/p1/fvm/<run>.{json,log,ids,state.json} and
# results/ios/p1/fvm/summary.md (rewritten after every run). Ids are checked against the references on the way.
cd ${0:A:h}/../../..
WT=$PWD; DM=$(cd $WT/../../../.. && pwd); ENG=$DM/engine
MODEL=$DM/local/q36/model; RT=$MODEL/routers/q36_router_full_k2_h512.dmr
O=$WT/results/ios/p1/fvm; mkdir -p $O
typeset -A IDS REF
IDS[pitt]=$ENG/results/stage2/pitt_ids.txt; REF[pitt]=$ENG/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids
IDS[code]=$ENG/results/stage3/code_ids.txt; REF[code]=$ENG/results/stage4/runs/dc_off_code.ids
SERIES=("$@"); [ ${#SERIES} -eq 0 ] && SERIES=(main)
PAT='/dmeng( |$)|/dmink( |$)|dmserve( |$)|/dm_gate( |$)|/inklm_check( |$)|PhoneStoreApp'

models() { pgrep -fl "$PAT" | awk '$1 ~ /^[0-9]+$/ && $2 !~ /(zsh|bash|sh|python3?|pgrep|time|tail|grep)$/' | head -3; }
fresh_quiet() { local q=$ENG/bench/QUIET; [ -f $q ] && [ $(( $(date +%s) - $(stat -f %m $q) )) -lt 720 ]; }
if [ -n "$(models)" ] || fresh_quiet; then
  echo "refusing to start: $( [ -n "$(models)" ] && echo "model process running: $(models | cut -c1-120)" || echo "QUIET is fresh")"; exit 2
fi
[ -x $WT/build/dmeng ] || { echo "build first: cmake -S . -B build && cmake --build build -j3 --target dmeng"; exit 2; }

state() {  # the machine at the start of a run: load, memory in use by everything else, file cache
  python3 - "$1" <<'PY'
import json, re, subprocess, sys
sh = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout
vm = sh("vm_stat"); g = lambda k: int(re.search(k + r":\s+(\d+)", vm).group(1)) * 16384 / 1e9
total = int(sh("sysctl", "-n", "hw.memsize")) / 1e9
used = g("Pages active") + g("Pages wired down") + g("Pages occupied by compressor")
json.dump({"loadavg": sh("sysctl", "-n", "vm.loadavg").strip(), "mem_total_gb": round(total, 2), "active_wired_compressed_gb": round(used, 2),
           "file_backed_gb": round(g("File-backed pages"), 2), "free_gb": round(g("Pages free"), 2), "swap": sh("sysctl", "-n", "vm.swapusage").strip(),
           "power": sh("pmset", "-g", "batt").splitlines()[0] if sh("pmset", "-g", "batt") else "?"}, open(sys.argv[1], "w"))
PY
}
run() {  # name prompt extra-args...
  local n=$1 p=$2; shift 2
  if [ -f $O/$n.log ] && grep -q '^GUARD' $O/$n.log; then echo "skip $n (done)"; return; fi
  while fresh_quiet; do sleep 60; done
  state $O/$n.state.json
  POLL_S=5 $ENG/.venv/bin/python $ENG/bench/guard.py $O/$n.log 900 -- $WT/build/dmeng --model $MODEL --prompt-ids $(cat ${IDS[$p]}) \
    --n 400 --ctx 16384 --temp 0 --prec i16 --json $O/$n.json --out-ids $O/$n.ids --prefetch router --routers $RT --m 8 "$@" > /dev/null
  cmp -s <(tr -d '\n ' < $O/$n.ids) <(tr -d '\n ' < ${REF[$p]}) && echo "$n: ids identical" || echo "$n: IDS DIFFER"
  summary
}
summary() {
  python3 - $O > $O/summary.md <<'PY'
import glob, json, os, re, sys
O = sys.argv[1]
print("| run | tok/s | compute ms | wait ms | MB read/tok | MB issued/tok | peak footprint GB | file cache GB (end) | budget GB (end) | refaults/tok | others' mem GB / load at start | others' CPU % | ids |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for j in sorted(glob.glob(os.path.join(O, "*.json"))):
    if j.endswith(".state.json"): continue
    n = os.path.basename(j)[:-5]
    try:
        d = json.load(open(j)); s = json.load(open(os.path.join(O, n + ".state.json")))
        g = json.loads(re.search(r"GUARD (.*)", open(os.path.join(O, n + ".log")).read()).group(1))
    except Exception:
        continue
    f = d.get("file", {})
    ref = {"pitt": "r_r8_pitt_gb3.3_r1.ids", "code": "dc_off_code.ids"}["pitt" if "pitt" in n else "code"]
    print(f"| {n} | {d['tok_s']:.2f} | {d['compute_ms_per_token']:.1f} | {d['wait_ms_per_token']:.1f} | {d.get('mb_preads_per_token', 0):.1f} | "
          f"{d['mb_read_per_token']:.1f} | {d['footprint_peak_gb']:.2f} | {f.get('file_cache_resident_gb_at_end', '')} | {f.get('budget_gb_at_end', '')} | "
          f"{f.get('refaults_per_token', '')} | {s['active_wired_compressed_gb']} / {s['loadavg']} | {g.get('others_cpu_pct_mean')} | see log |")
PY
}
hold() {  # GB: a helper holding that much dirty memory until killed; prints its pid
  python3 -c "
import os, sys, time
n = int(float(sys.argv[1]) * 1e9); b = bytearray(n)
for i in range(0, n, 16384): b[i] = 1
open(sys.argv[2], 'w').write('held')
time.sleep(3600)" $1 $O/.held &
  local pid=$!
  until [ -f $O/.held ]; do sleep 1; done; rm -f $O/.held
  echo $pid
}
FA=(--mem file --cache-gb 8 --file-adapt 1 --file-min-gb 1)
for s in $SERIES; do
  case $s in
    main)
      for r in 1 2; do
        for p in pitt code; do
          run ${p}_mlock3.3_r$r $p --mem mlock --cache-gb 3.3
          run ${p}_fileadapt_a_r$r $p $FA
          run ${p}_mlock5.0_r$r $p --mem mlock --cache-gb 5.0
          run ${p}_fileadapt_b_r$r $p $FA
        done
      done ;;
    pressure)
      for gb in 0 2 4 6; do
        n=pitt_fileadapt_dirty${gb}gb
        if [ -f $O/$n.log ] && grep -q '^GUARD' $O/$n.log; then echo "skip $n (done)"; continue; fi
        hp=""; [ $gb -gt 0 ] && hp=$(hold $gb)
        run $n pitt $FA
        [ -n "$hp" ] && kill $hp 2>/dev/null
      done ;;
    core)
      for c in 0.5 1.0; do
        for p in pitt code; do run ${p}_fileadapt_core${c} $p $FA --file-core-gb $c; done
      done ;;
  esac
done
echo "summary: $O/summary.md"
