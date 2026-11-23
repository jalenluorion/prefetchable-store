#!/bin/bash
# verification batch: every policy per cache size, 35B (pitt, 400 tokens) and Inkling (pitt, 160 tokens), one guarded
# model run at a time; each run records its routing and its router rows for the simulator
W=${CP_WORK:-.}
QR=${DEEP_MEMORY}/local/q36/model/routers/q36_router_full_k2_h512.dmr
IR=${DEEP_MEMORY}/local/inkling/model/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr
QV="--evict-policy $W/q36_d_evict3_policy.json --routers $QR --router-stats"
IV="--evict-policy $W/ink_d_evict3_policy.json --routers $IR --router-stats"
q() { local n=$1 gb=$2; shift 2; N=400 $W/run_q36.sh $n $gb pitt --evict-trace $n.trace "$@"; echo "done $n $(date +%T)"; sleep 3; }
i() { local n=$1 gb=$2; shift 2; N=160 $W/run_ink.sh $n $gb pitt --evict-trace $n.trace "$@"; echo "done $n $(date +%T)"; sleep 3; }
q q_def_0.8 0.8 --evict freq
q q_hzc_0.8 0.8 --evict hazard --layer-caps uniform
q q_def_1.27 1.27402 --evict freq
q q_hzc_1.27 1.27402 --evict hazard --layer-caps uniform
q q_def_2.0 2.0 --evict freq
q q_hzc_2.0 2.0 --evict hazard --layer-caps uniform
q q_def_3.3 3.3 --evict freq
q q_hz_3.3 3.3 --evict hazard
q q_def_5.0 5.0 --evict freq
q q_hz_5.0 5.0 --evict hazard
q q_v3_5.0 5.0 $QV
q q_def_6.6 6.6 --evict freq
q q_hz_6.6 6.6 --evict hazard
q q_v3_6.6 6.6 $QV
i i_def_4 4 --evict freq
i i_hz_4 4 --evict hazard
i i_v3_4 4 $IV
i i_def_6 6 --evict freq
i i_hz_6 6 --evict hazard
i i_v3_6 6 $IV
echo BATCH1_DONE
