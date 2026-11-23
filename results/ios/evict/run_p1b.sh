#!/bin/bash
cd "$(dirname "$0")"
PY=<work>/engine/.venv/bin/python
S8=0.8,1.0,1.27,1.6,2.0,2.5,3.3,5.0; S4=0.8,1.27,2.0,3.3
$PY evict_sim.py caps pitt,coding $S8 res_caps_main.json
$PY evict_sim.py caps agent $S4 res_caps_main.json
$PY evict_sim.py look pitt,coding $S8 res_look_main.json
$PY evict_sim.py look agent $S4 res_look_main.json
$PY evict_sim.py look2 pitt,coding,agent $S4 res_look2_main.json
$PY evict_sim.py tier pitt,coding,agent 1.27 res_tier_main.json
$PY evict_sim.py hl agent $S4 res_hl_main.json
