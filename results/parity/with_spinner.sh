#!/bin/zsh
# runs "$@" with a co-runner process of DM_SPIN spinning threads beside it (none if DM_SPIN is 0 or unset)
B=${0:A:h}/../../bin-rr
S=""
if [ "${DM_SPIN:-0}" -gt 0 ]; then $B/spinner $DM_SPIN & S=$!; fi
"$@"
rc=$?
[ -n "$S" ] && kill $S
exit $rc
