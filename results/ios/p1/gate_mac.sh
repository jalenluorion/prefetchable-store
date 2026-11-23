#!/bin/zsh
# P1 gates on the Mac (lossless, greedy, --ctx 16384, --prec i16, the 8-bit resident, the common store): Mem::MLOCK vs
# Mem::FILE at the same policy budget (3.3 GB) and Mem::FILE with 8 GB of bookkeeping, through the dmeng tool; then the
# same gate through the C API only (tests/c_api/dm_gate.c). Every run under the engine's guard (bench/guard.py, as the
# engine session's filecache runs: POLL_S=2, 900 s, default swap / footprint / disk limits), one at a time. Decode names
# early with the trained MLP lag-2 router (mlp_k2_h512 = routers/q36_router_full_k2_h512.dmr, m 8), as the references.
#   zsh results/ios/p1/gate_mac.sh [tool] [capi]          (from the repository root; default both)
cd ${0:A:h}/../../..
WT=$PWD
DM=$(cd $WT/../../../.. && pwd)          # deep-memory/
ENG=$DM/engine
MODEL=$DM/local/q36/model
RT=$MODEL/routers/q36_router_full_k2_h512.dmr
O=$WT/results/ios/p1/runs; mkdir -p $O
typeset -A IDS REF
IDS[pitt]=$ENG/results/stage2/pitt_ids.txt; REF[pitt]=$ENG/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids
IDS[code]=$ENG/results/stage3/code_ids.txt; REF[code]=$ENG/results/stage4/runs/dc_off_code.ids
STEPS=("$@"); [ ${#STEPS} -eq 0 ] && STEPS=(tool capi)
fb() { vm_stat | awk '/File-backed pages/{gsub("\\.","",$3); printf "%.2f", $3*16384/1e9}'; }
quiet() { local q=$ENG/bench/QUIET; while [ -f $q ] && [ $(( $(date +%s) - $(stat -f %m $q) )) -lt 720 ]; do sleep 60; done; }
cmpids() {  # name ids ref
  if cmp -s <(tr -d '\n ' < $2) <(tr -d '\n ' < $3); then echo "$1: IDENTICAL to ${3:t}"
  else python3 -c "
import sys
a=[int(x) for x in open('$2').read().replace(chr(10),'').split(',') if x]; b=[int(x) for x in open('$3').read().replace(chr(10),'').split(',') if x]
d=next((i for i in range(min(len(a),len(b))) if a[i]!=b[i]),min(len(a),len(b)))
print('$1: DIFFER (%d vs %d ids), first difference at position %d' % (len(a),len(b),d))"; fi
}
run() {  # name prompt -- command...
  local n=$1 p=$2; shift 3
  quiet
  local before=$(fb)
  POLL_S=2 $ENG/.venv/bin/python $ENG/bench/guard.py $O/$n.log 900 -- "$@" > /dev/null
  echo "$n file-backed GB (vm_stat) before/after: $before/$(fb)" >> $O/filebacked.txt
  tail -1 $O/$n.log | cut -c1-160
  cmpids $n $O/$n.ids ${REF[$p]} | tee -a $O/gates.txt
}
for s in $STEPS; do
  if [ $s = tool ]; then
    for p in pitt code; do
      for cfg in mlock:3.3 file:3.3 file:8.0; do
        mem=${cfg%%:*} gb=${cfg##*:}
        run tool_${p}_${mem}_gb$gb $p -- $WT/build/dmeng --model $MODEL --prompt-ids $(cat ${IDS[$p]}) --n 400 --cache-gb $gb --ctx 16384 \
          --temp 0 --prec i16 --mem $mem --json $O/tool_${p}_${mem}_gb$gb.json --out-ids $O/tool_${p}_${mem}_gb$gb.ids \
          --prefetch router --routers $RT --m 8
      done
    done
  elif [ $s = capi ]; then
    for p in pitt code; do
      run capi_${p}_file_gb3.3 $p -- $WT/build/dm_gate --model $MODEL --prompt-ids ${IDS[$p]} --n 400 --out $O/capi_${p}_file_gb3.3.ids \
        --json $O/capi_${p}_file_gb3.3.json --mem file --cache-gb 3.3 --routers-id mlp_k2_h512 --threads 8 --readers 4
    done
  fi
done
