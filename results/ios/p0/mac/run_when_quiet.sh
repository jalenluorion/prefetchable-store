#!/bin/bash
# Run the Mac calibration bench only when no other engine/bench/compile is running and the total CPU is low:
# 3 checks in a row 20 s apart, at most 90 min of waiting. QUIET respected.
cd "$(dirname "$0")"
Q=<work>/engine/bench/QUIET
ok=0
for i in $(seq 1 270); do
  busy=$(ps -Ao pcpu,comm | awk '/dmeng|dmserve|mlx|python|clang|swift-frontend|xcodebuild|phone-store-run/ && !/awk/ {s+=$1} END {print s+0}')
  tot=$(ps -Ao pcpu | awk 'NR>1 {s+=$1} END {print s+0}')
  quiet=1; [ -e $Q ] && [ $(( $(date +%s) - $(stat -f %m $Q) )) -lt 720 ] && quiet=0
  if [ $quiet = 1 ] && awk "BEGIN{exit !($busy < 20 && $tot < 250)}"; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 3 ] && break
  sleep 20
done
[ $ok -ge 3 ] || { echo "never quiet in 90 min"; exit 1; }
TS=$(date +%Y%m%d-%H%M)
(date; pmset -g batt | head -1; ps -Ao pcpu,comm -r | head -6) > kernelbench-mac-$TS.env.txt
./mac_bench "cell=1,threads=1:4:8,qos=ui,mb=512,sustain=0,batch=1,bgthreads=0,metal=1" > kernelbench-mac-$TS.json 2> kernelbench-mac-$TS.log
(date; ps -Ao pcpu,comm -r | head -6) >> kernelbench-mac-$TS.env.txt
echo "done kernelbench-mac-$TS.json"
