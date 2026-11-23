#!/bin/bash
cd "$(dirname "$0")"
PY=<work>/engine/.venv/bin/python
S4=0.8,1.27,2.0,3.3
( $PY evict_sim.py look agent $S4 res_look_main.json; $PY evict_sim.py hl agent $S4 res_hl_main.json ) > log_a.txt 2>&1 &
$PY evict_sim.py look2 pitt,coding,agent,code_text $S4 res_look2_main.json > log_b.txt 2>&1 &
$PY evict_sim.py tier pitt,coding,agent 1.27 res_tier_main.json > log_c.txt 2>&1 &
wait
