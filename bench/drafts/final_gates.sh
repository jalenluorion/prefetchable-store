#!/bin/zsh
# The gates the integration tree's campaign does not cover (each through engine/bench/guard.py, sequential):
#   per model, one server-worker tool-call request (the rewrite turn, its tool-call fragments sent) drafting off vs on
#   (--draft lookup --early 4, routers on): identical ids, skeleton drafts offered; and the 35B's fixed-draft stress
#   (dmeng --draft fixed: the whole draft every step, rolled back part way) against the server's off run of the same prompt.
R=${0:A:h}/../..; DM=${R:A}/../../../..; DM=${DM:A}; ENG=$DM/engine; G=$ENG/bench/guard.py; O=$R/results/drafts/runs; P=$R/results/drafts/prompts
Q=$DM/local/q36/model; I=$DM/local/inkling/model
QW=(--model $Q --cache-gb 5 --mem mlock --ctx 16384 --routers $Q/routers/q36_router_full_k2_h512.dmr --m 8)
IW=(--model $I --resident $ENG/build_ink/resident_real_b8.bin --cache-gb 6 --mem mlock --ctx 16384 --routers $I/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr)
for m in q36 ink; do
  if [ $m = q36 ]; then W=($QW); else W=($IW); fi
  for d in off on; do
    [ -s $O/srv_${m}_tool_$d.json ] && continue
    D=(); [ $d = on ] && D=(--draft lookup --early 4)
    echo "$(date +%T) $m server $d $(pmset -g batt | tail -1 | cut -c1-60)"
    python3 $G $O/srv_${m}_tool_$d.log 2400 -- python3 $R/bench/drafts/worker_drive.py $O/srv_${m}_tool_$d.json $P/${m}_server_tool1.json -- $R/build/dmserve $W $D
    sleep 20
  done
done
if [ ! -s $O/q36_toolrw_fixed.ids ]; then
  echo "$(date +%T) q36 fixed $(pmset -g batt | tail -1 | cut -c1-60)"
  python3 $G $O/q36_toolrw_fixed.log 2400 -- $R/build/dmeng --model $Q --prefill-chunk 512 --prompt-file $P/q36_toolrw_ids.txt --n 200 --cache-gb 5 \
    --ctx 16384 --temp 0 --mem mlock --stop-eos 248046 --prefetch router --routers $Q/routers/q36_router_full_k2_h512.dmr --m 8 --draft fixed \
    --json $O/q36_toolrw_fixed.json --out-ids $O/q36_toolrw_fixed.ids
fi
python3 - $O <<'PY'
import json, sys
O = sys.argv[1]
for m in ("q36", "ink"):
    a, b = [json.load(open(f"{O}/srv_{m}_tool_{d}.json"))[0] for d in ("off", "on")]
    dr = b["E"]["json"].get("draft", {})
    print(f"SERVER {m}: off {len(a['gen'])} / on {len(b['gen'])} tokens, ids {'IDENTICAL' if a['gen'] == b['gen'] else 'DIFFER'}; "
          f"tokens/step {dr.get('tokens_per_step')}, verifies {dr.get('verifies_by_source')}, accepted {dr.get('accepted_by_source')}")
off = json.load(open(f"{O}/srv_q36_tool_off.json"))[0]["gen"]
fx = [int(x) for x in open(f"{O}/q36_toolrw_fixed.ids").read().strip().split(",")]
d = json.load(open(f"{O}/q36_toolrw_fixed.json"))["draft"]
print(f"FIXED q36: {len(fx)} tokens, ids {'IDENTICAL' if fx == off[:len(fx)] else 'DIFFER'} to the server's off run; verifies {d['verifies']}, "
      f"drafted {d['drafted']}, accepted {d['accepted']}, steps_by_positions {d['steps_by_positions']}")
PY
