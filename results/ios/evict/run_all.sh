#!/bin/bash
# three processes at most (shared Mac); each writes its own JSON per group; merge with tables.py
cd "$(dirname "$0")"
PY=<work>/engine/.venv/bin/python
S=0.8,1.0,1.27,1.6,2.0,2.5,3.3,5.0
p1() { for g in base hl caps look look2; do $PY evict_sim.py $g pitt,coding,agent $S res_${g}_main.json; done; $PY evict_sim.py tier pitt,coding,agent 1.27 res_tier_main.json; }
p2() { $PY evict_sim.py base code_text,prose_text $S res_base_tf.json; $PY evict_sim.py pf agent,code_text,prose_text,pitt,coding $S res_pf.json;
       for g in caps hl look; do $PY evict_sim.py $g code_text,prose_text $S res_${g}_tf.json; done; }
p3() { for g in base caps hl; do $PY evict_sim.py $g ph_chat,ph_wiki,ph_code $S res_${g}_ph.json; done;
       $PY evict_sim.py tier code_text,prose_text,ph_chat 1.27 res_tier_other.json; }
p1 > log_p1.txt 2>&1 &
p2 > log_p2.txt 2>&1 &
p3 > log_p3.txt 2>&1 &
wait
