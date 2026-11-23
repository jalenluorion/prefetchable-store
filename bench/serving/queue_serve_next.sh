#!/bin/zsh
# The runs behind results/serving/SERVING-NEXT.md, in order, each through slot.sh (power rule, QUIET, PRIORITY,
# guard.py). Resumable: a step whose log ends with a successful GUARD line is skipped. usage: queue_serve_next.sh [STEP...]
cd ${0:A:h}/../..
export SKIP_DONE=1
R=results/serving
Q36=../../../q36/model; INK=../../../inkling/model
REP=${REP:-../../../../engine/results/agentic/replay}
W35=(--cache-gb 4.6 --mem mlock --chunk 512 --span 8192 --stream 3 --disk $Q36/.dmeng/prefix --ahead-span 512)
mkdir -p $R/check $R/ahead $R/hotset $R/inkling
steps=(${@:-liveink check35 tok35 hot35 checkink})
for s in $steps; do
  case $s in
  check35)  # the exactness check on a state produced by an idle-time prefill (cancelled once, resumed), 35B
    bench/serving/slot.sh $R/check/check_ahead_q36.log 3600 build/dmserve --model $Q36 --cache-gb 4.6 --mem mlock --ctx 4096 --chunk 512 --span 8192 \
      --ahead-span 512 --check-ahead $R/check/ahead_A.txt $R/check/ahead_B.txt --check-ahead-len 2267 --gen 16 || { echo check35 failed; exit 1; } ;;
  ahead35)  # new user turn after idle gaps, idle-time prefill off / on / off / on (forced replay of s2_clean)
    for c in off1:0 on1:1 off2:0 on2:1; do
      SERVE=1 TIMED=1 bench/serving/slot.sh $R/ahead/${c%%:*}.log 1500 bench/serving/ahead_replay.sh ${c##*:} $R/ahead/${c%%:*} $REP/s2_clean 7 5,15,30 $W35
    done ;;
  tok35)  # correctness of the idle-time prefill in serving: generated tokens off vs on (fixed seed), gaps 5 s (cancelled mid-way) and 30 s
    for c in toff:0 ton:1; do
      SERVE=1 bench/serving/slot.sh $R/ahead/${c%%:*}.log 1800 bench/serving/ahead_replay.sh ${c##*:} $R/ahead/${c%%:*} $REP/s2_clean 7 5,30 $W35
    done
    ../../.venv/bin/python bench/serving/ahead_table.py $R/ahead toff ton > $R/ahead/tokens_off_on.json ;;
  hot35)  # correctness of the saved hot set: the first request after a restart, without / with it (fixed seed)
    SERVE=1 bench/serving/slot.sh $R/hotset/q36.log 1800 bench/serving/hotset_restart.sh $R/hotset/q36 $REP/s2_clean 2 $Q36/.dmeng/hotset.bin \
      "save off on" $W35 ;;
  hotset35)
    SERVE=1 TIMED=1 bench/serving/slot.sh $R/hotset/q36.log 1500 bench/serving/hotset_restart.sh $R/hotset/q36 $REP/s2_clean 2 $Q36/.dmeng/hotset.bin \
      "save off on off on" $W35 ;;
  checkink)  # the server's batched prefill vs token by token on Inkling (1,100 replay-like ids), and the idle-time path
    python3 -c "import random; random.seed(1100); print(','.join(str(random.randrange(1000, 100000)) for _ in range(1100)))" > $R/check/ids_1100.txt
    bench/serving/slot.sh $R/check/check_ink.log 3600 build/dmserve --model $INK --cache-gb 6 --mem mlock --ctx 2048 --chunk 512 --span 8192 \
      --check $R/check/ids_1100.txt --gen 16 || { echo checkink failed; exit 1; } ;;
  liveink)  # stock OpenCode against Inkling
    SERVE=1 bench/serving/slot.sh $R/inkling/live1.log 3600 env LIVE_REPO_BASE=${LIVE_REPO_BASE:-/tmp} bench/serving/live_opencode.sh $INK inkling-2-mini \
      bench/serving/opencode_inkling.json $R/inkling/live1 --cache-gb 6 --mem mlock --chunk 512 --span 8192 --disk $INK/.dmeng/prefix --disk-every 1024 ;;
  esac
done
