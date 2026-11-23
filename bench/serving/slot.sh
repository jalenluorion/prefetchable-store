#!/bin/zsh
# Run one model command in the machine's model slot, by the rules of engine/bench: power (>= 30%, or >= 20% and
# charging), no fresh QUIET of another job (< 12 min), one model process at a time (engine/bench/guard.py waits for
# it), PRIORITY touched while a serving test runs (SERVE=1) and QUIET held for timed runs (TIMED=1), both removed after.
#   slot.sh LOG TIMEOUT_S cmd...      (skips the run if LOG already ends with a GUARD line of rc 0 and SKIP_DONE=1)
ENG=${ENG:-${0:A:h}/../../../../../../engine}
LOG=$1; TO=$2; shift 2
if [ -n "$SKIP_DONE" ] && [ -f $LOG ] && grep -q '^GUARD {"rc": 0' $LOG; then echo "skip $LOG"; exit 0; fi
for attempt in 1 2 3 4 5 6; do
while true; do
  b=$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)
  if { pmset -g batt | grep -q "; charging" && [ $b -ge 20 ]; } || [ $b -ge 30 ]; then :; else sleep 60; continue; fi
  Q=$ENG/bench/QUIET
  if [ -f $Q ] && [ $(( $(date +%s) - $(stat -f %m $Q) )) -lt 720 ]; then sleep 30; continue; fi
  busy=$(ps -axo comm= | awk -F/ '{print $NF}' | grep -E '^(dmeng|dmink|dmserve|inklm[^ ]*|bmoe-cli|llama-.*|dmeng_.*|dm_gate)$' | head -1)
  [ -z "$busy" ] && break  # a lost race shows as a failed memory lock and is retried below
  sleep 2
done
[ -n "$TIMED" ] && echo "serve-next $(date +%H:%M:%S)" > $ENG/bench/QUIET
[ -n "$SERVE" ] && touch $ENG/bench/PRIORITY
echo "slot: $(pmset -g batt | tail -1 | tr -s '\t' ' ')" > $LOG.power
POLL_S=2 python3 $ENG/bench/guard.py $LOG $TO -- "$@"
rc=$?
[ -n "$TIMED" ] && rm -f $ENG/bench/QUIET
[ -n "$SERVE" ] && rm -f $ENG/bench/PRIORITY
# lost a race for the slot (another queue's model started at the same moment): the lock of our memory failed; again
if [ $rc -ne 0 ] && grep -q "mlock failed" $LOG; then mv $LOG $LOG.mlock$attempt; sleep 60; continue; fi
exit $rc
done
exit 1
