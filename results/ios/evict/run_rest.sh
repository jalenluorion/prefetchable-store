#!/bin/bash
# the secondary texts and the named-early / tier jobs at four sizes (P1 of run_all.sh keeps the main texts at eight)
cd "$(dirname "$0")"
PY=<work>/engine/.venv/bin/python
S=0.8,1.27,2.0,3.3
p2() { $PY evict_sim.py base code_text,prose_text $S res_base_tf.json; $PY evict_sim.py pf agent,code_text,prose_text,pitt,coding $S res_pf.json;
       for g in caps look; do $PY evict_sim.py $g code_text,prose_text $S res_${g}_tf.json; done; }
p3() { for g in base caps; do $PY evict_sim.py $g ph_chat,ph_wiki,ph_code $S res_${g}_ph.json; done;
       $PY evict_sim.py tier code_text,prose_text,ph_chat 1.27 res_tier_other.json; $PY evict_sim.py hl code_text,prose_text,ph_chat $S res_hl_other.json; }
p2 > log_p2.txt 2>&1 &
p3 > log_p3.txt 2>&1 &
wait
