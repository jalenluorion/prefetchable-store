#!/bin/zsh
# serve2 runs, in order, each through slot.sh (resumable: a step whose log ends with a good GUARD line is skipped).
cd ${0:A:h}/../..
export SKIP_DONE=1 FOOT_GB=${FOOT_GB:-12.5}
R=results/serving2; mkdir -p $R
Q36=../../../q36/model; INK=../../../inkling/model
[ $# -eq 0 ] && set -- mem35 live35 liveink
for s in "$@"; do
  case $s in
  mem35)  # auto size, forced shrink to the minimum and growth back: same tokens, footprint at each size
    bench/serving/slot.sh $R/memwatch_q36.log 1200 ../../.venv/bin/python bench/serving/memwatch_check.py $R/memwatch_q36.jsonl \
      bench/prompts/qwen3_5_moe/code.ids 48 -- build/dmserve --model $Q36 --cache-gb auto --mem mlock --chunk 512 --span 8192 \
      --routers $Q36/routers/lin/q36_router_lin_k2.dmr --m 8 ;;
  memink)
    STOP=200006 bench/serving/slot.sh $R/memwatch_ink.log 1200 ../../.venv/bin/python bench/serving/memwatch_check.py $R/memwatch_ink.jsonl \
      bench/prompts/inkling2_loop/code.ids 48 -- build/dmserve --model $INK --cache-gb auto --mem mlock --chunk 512 --span 8192 \
      --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --xtok $INK/routers/ink_x_main/ink_router_xtok_s47_t0-5_b8.dmr --rd-prio 1 ;;
  live35)
    SERVE=1 bench/serving/slot.sh $R/live_q36.log 3600 bench/serving/live_profile.sh $Q36 qwen3.6-35b-a3b bench/serving/opencode_q36.json $R/live_q36 ;;
  liveink)
    SERVE=1 bench/serving/slot.sh $R/live_ink.log 3600 bench/serving/live_profile.sh $INK inkling-2-mini bench/serving/opencode_inkling.json $R/live_ink ;;
  esac
done
