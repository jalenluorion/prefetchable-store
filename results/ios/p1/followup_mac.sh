#!/bin/zsh
# P1 follow-ups on the Mac (correctness only, no timing): through the C API on the phone-layout folder PHONE (the
# common layout WITHOUT the 4.9 GB bf16 resident.safetensors: links to store/, .dmeng/ and the small files, plus
# embeddings.safetensors, byte-identical to the phone's), each run through the engine's guard, one at a time.
#   A: code prompt, FILE, adaptive budget from 8 GB, a cancelled cold prefill first, then the budget ceiling lowered to
#      2 GB at token 300 (dm_set_cache_bytes): ids identical, cancel consistent, the adaptive budget moves
#   B: pitt prompt, MLOCK 3.3 GB, budget lowered to 1.5 GB at token 200 (slab memory given back): ids identical
#   PHONE=DIR zsh results/ios/p1/followup_mac.sh [A] [B]
cd ${0:A:h}/../../..
WT=$PWD; DM=$(cd $WT/../../../.. && pwd); ENG=$DM/engine
O=$WT/results/ios/p1/followup; mkdir -p $O
STEPS=("$@"); [ ${#STEPS} -eq 0 ] && STEPS=(A B)
models() { pgrep -fl "/dmeng( |$)|/dmink( |$)|dmserve( |$)|/dm_gate( |$)|PhoneStoreApp|/inklm_check( |$)" | grep -v pgrep; }
quiet() { local q=$ENG/bench/QUIET; while [ -f $q ] && [ $(( $(date +%s) - $(stat -f %m $q) )) -lt 720 ]; do sleep 60; done; }
cmpids() { cmp -s <(tr -d '\n ' < $2) <(tr -d '\n ' < $3) && echo "$1: IDENTICAL to ${3:t}" || echo "$1: DIFFER from ${3:t}"; }
for s in $STEPS; do
  quiet
  m=$(models); [ -n "$m" ] && echo "before $s, model processes running (the guard waits): $m" | cut -c1-200
  if [ $s = A ]; then
    POLL_S=2 $ENG/.venv/bin/python $ENG/bench/guard.py $O/A_code_file_adapt.log 900 -- $WT/build/dm_gate --model $PHONE \
      --prompt-ids $ENG/results/stage3/code_ids.txt --n 400 --out $O/A_code_file_adapt.ids --json $O/A_code_file_adapt.json --mem file \
      --cache-gb 8 --file-adapt 1 --file-min-gb 1 --routers-id mlp_k2_h512 --cancel-test --resize-at 300 --resize-gb 2.0 > /dev/null
    cmpids A $O/A_code_file_adapt.ids $ENG/results/stage4/runs/dc_off_code.ids | tee -a $O/gates.txt
  else
    POLL_S=2 $ENG/.venv/bin/python $ENG/bench/guard.py $O/B_pitt_mlock_resize.log 900 -- $WT/build/dm_gate --model $PHONE \
      --prompt-ids $ENG/results/stage2/pitt_ids.txt --n 400 --out $O/B_pitt_mlock_resize.ids --json $O/B_pitt_mlock_resize.json --mem mlock \
      --cache-gb 3.3 --routers-id mlp_k2_h512 --resize-at 200 --resize-gb 1.5 > /dev/null
    cmpids B $O/B_pitt_mlock_resize.ids $ENG/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids | tee -a $O/gates.txt
  fi
  grep -h "overlap\|GUARD" $O/*_$( [ $s = A ] && echo code_file_adapt || echo pitt_mlock_resize).log | cut -c1-120
done
