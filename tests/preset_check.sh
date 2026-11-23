#!/bin/zsh
# Precedence of the settings (include/dmeng/presets.h): an explicit option > the profile > the manifest's defaults > the
# built-in values, in all three front ends. On the tiny Qwen3.6-geometry model (a named profile on a folder without a
# manifest takes the built-in profile of its config.json family), each cell runs dmeng, the C API (dm_gate) and the
# server worker (dmserve's READY line) and checks the "resolved" settings each one writes in its JSON.
#   tests/preset_check.sh Q36_TINY [INK_TINY]   (Q36_TINY: the folder holding model/ and routers_full_k4_h512.dmr; INK_TINY:
#   the tiny Inkling folder, for Inkling's default router; INK_LIN: its linear_k2 engine file)
# DMENG_BIN / DMSERVE_BIN / DMGATE_BIN: other binaries (default build/). Prints PRESET CHECK PASSED / FAILED.
set -u
R=${0:A:h}/..
Q=${1:?Q36_TINY}
DE=${DMENG_BIN:-$R/build/dmeng}; DS=${DMSERVE_BIN:-$R/build/dmserve}; DG=${DMGATE_BIN:-${DE:h}/dm_gate}
T=$(mktemp -d -t preset_check)
trap 'rm -rf $T' EXIT
M=$T/model; mkdir -p $M/routers/lin
for f in $Q/model/*; do ln -s $f $M/${f:t}; done
ln -s $Q/routers_full_k4_h512.dmr $M/routers/lin/q36_router_lin_k2.dmr  # the built-in profile's linear_k2 (a tiny stand-in)
RT=$Q/routers_full_k4_h512.dmr
print -l 1 2 3 4 5 6 7 8 > $T/ids.txt
COMMON=(--threads 2)
# cell NAME PROFILE EXPECT(json) -- CLI flags (dmeng / dmserve spelling); the C API spelling comes from capi_flags
cells=()
cell() { cells+=("$1|$2|$3|${(j: :)@[4,-1]}"); }
#     name           profile      expected resolved settings (subset)                                                flags
cell  default        mac          '{"mode":"lossless","names":{"rule":"top","m":8},"cache_auto":true,"eviction":"auto","drafts":0,"prefill_gpu":{"projections":0.4,"experts":0.4},"run_ahead":{"picks":8,"depth":1,"budget":0,"max_tok":0,"sat_skip":false}}'
cell  mode_fast      mac          '{"mode":"approx:drop"}'                                                             --mode fast
cell  small_default  phone-small  '{"mode":"approx:drop","names":{"rule":"confidence"},"run_ahead":{"picks":0},"readers":8}'
cell  small_exact    phone-small  '{"mode":"lossless","run_ahead":{"picks":8,"depth":1,"budget":0.2,"max_tok":0,"sat_skip":true},"readers":8}'                                                                --mode exact
cell  small_top_m    phone-small  '{"names":{"rule":"top","m":4}}'                                                     --m 4
cell  names_off      mac          '{"routers":"","names":null}'                                                        --routers none
cell  router_file    phone-small  '{"routers":"RT","names":{"rule":"top","m":8}}'                                      --routers RT
cell  cache_size     mac          '{"cache_gb":0.05,"cache_auto":false}'                                               --cache-gb 0.05
cell  evict          mac          '{"eviction":"freq"}'                                                                --evict freq
cell  drafts         mac          '{"drafts":1}'                                                                       --drafts 1
cell  gpu_share      mac          '{"prefill_gpu":{"projections":0.2,"experts":0.4}}'                                  --gpu 0.2
cell  gpu_off        mac          '{"prefill_gpu":{"projections":0,"experts":0}}'                                      --gpu-off
cell  run_ahead_off  mac          '{"run_ahead":{"picks":0}}'                                                          --ra-off
cell  run_ahead_4    phone-small  '{"run_ahead":{"picks":4,"depth":1,"budget":0.2}}'                                              --run-ahead 4
cell  ra_limits      mac          '{"run_ahead":{"picks":8,"budget":0.5,"max_tok":3,"sat_skip":true}}'                --run-ahead-budget 0.5 --run-ahead-max 3 --run-ahead-sat-skip 1
cell  readers        phone        '{"readers":2}'                                                                     --readers 2
fail=0
for c in $cells; do
  name=${c%%|*}; rest=${c#*|}; prof=${rest%%|*}; rest=${rest#*|}; exp=${rest%%|*}; flags=(${(s: :)${rest#*|}})
  exp=${exp//RT/$RT}
  cli=(); capi=(); srv=()
  for ((i = 1; i <= $#flags; i++)); do
    f=$flags[i]; v=${flags[i+1]:-}
    v=${v//RT/$RT}
    case $f in
      --mode|--m|--routers|--cache-gb|--evict) cli+=($f $v); capi+=($f $v); srv+=($f $v); ((i++)) ;;
      --drafts) cli+=($f $v); srv+=($f $v); ((i++)) ;;  # the C API has no drafting
      --gpu) cli+=(--prefill-gpu-frac $v); capi+=(--gpu-frac $v); srv+=(--gpu-frac $v); ((i++)) ;;
      --readers|--run-ahead|--run-ahead-budget|--run-ahead-max|--run-ahead-sat-skip) cli+=($f $v); capi+=($f $v); srv+=($f $v); ((i++)) ;;
      --ra-off) cli+=(--run-ahead 0); capi+=(--run-ahead -1); srv+=(--run-ahead 0) ;;
      --gpu-off) cli+=(--prefill-gpu off); capi+=(--gpu-frac -1 --gpu-experts -1); srv+=(--prefill-gpu off) ;;
    esac
  done
  [[ " ${capi[*]} " == *" --cache-gb "* ]] || capi+=(--cache-gb auto)  # dm_gate's own default is 3.3 GB
  $DE --model $M --profile $prof --prompt-file $T/ids.txt --n 4 $COMMON $cli --json $T/$name.cli.json > $T/$name.cli.out 2>&1
  $DG --model $M --profile $prof --prompt-ids $T/ids.txt --n 4 $COMMON $capi --json $T/$name.capi.json > $T/$name.capi.out 2>&1
  $DS --model $M --profile $prof $COMMON $srv < /dev/null 2> $T/$name.srv.err | grep -m1 '^READY ' | cut -c7- > $T/$name.srv.json
  for fe in cli capi srv; do
    [ $fe = capi ] && [[ " ${flags[*]} " == *" --drafts "* ]] && continue
    python3 - $T/$name.$fe.json "$exp" "$name $fe" <<'PY' || fail=1
import json, sys
path, exp, tag = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
try:
    j = json.load(open(path))
    got = j["resolved"] if "resolved" in j else j["stats"]["resolved"]  # dm_gate: dm_stats_json under "stats"
except Exception as e:
    print(f"FAIL {tag}: no resolved settings ({e})"); sys.exit(1)
def sub(e, g, k=""):
    if isinstance(e, dict):
        if not isinstance(g, dict):
            return [f"{k}: {g!r} != {e!r}"]
        return [x for kk, v in e.items() for x in sub(v, g.get(kk), f"{k}.{kk}")]
    if isinstance(e, float) or isinstance(g, float):
        return [] if g is not None and abs(float(g) - float(e)) < 1e-6 else [f"{k}: {g!r} != {e!r}"]
    return [] if g == e else [f"{k}: {g!r} != {e!r}"]
bad = sub(exp, got)
print(("FAIL " if bad else "ok   ") + tag + ("  " + "; ".join(bad) if bad else ""))
sys.exit(1 if bad else 0)
PY
  done
done
# Inkling's default router (optional second argument INK_TINY: the tiny Inkling folder; the default router resolved from
# the built-in profile, with the real model's linear_k2 engine file linked in as a stand-in): linear_k2, confidence rule
if [ -n "${2:-}" ] && [ -f ${INK_LIN:-$Q/../../../inkling/model/routers/ink_r_main/ink_router_lin_k2_b8.dmr} ]; then
  IM=$T/ink; mkdir -p $IM/routers/ink_r_main
  for f in $2/*; do ln -s $f $IM/${f:t}; done
  ln -s ${INK_LIN:-$Q/../../../inkling/model/routers/ink_r_main/ink_router_lin_k2_b8.dmr} $IM/routers/ink_r_main/ink_router_lin_k2_b8.dmr
  for prof in mac phone phone-small; do
    $DE --model $IM --profile $prof --print-defaults 2> /dev/null | python3 -c "
import json,sys; d=json.load(sys.stdin); ok=d['routers']=='linear_k2' and d['names']['rule']=='confidence'
print(('ok   ' if ok else 'FAIL ')+'ink default router $prof: '+d['routers']+' '+d['names']['rule']); sys.exit(0 if ok else 1)" || fail=1
  done
fi
(( fail )) && { echo "PRESET CHECK FAILED"; exit 1; } || echo "PRESET CHECK PASSED"
