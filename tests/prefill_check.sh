#!/bin/zsh
# Permanent test: a batched prefill of N tokens leaves byte-identical logits, recurrent / conv state and KV to N single
# decode steps, and the greedy continuation is the same (tools/dmserve.cpp --check). Exit 0 only if all hold.
#   tests/prefill_check.sh MODEL_DIR [IDS_FILE] [extra dmserve flags]      (DMSERVE_BIN: another worker binary)
# Without IDS_FILE: 1,100 pseudo-random token ids (fixed seed), long enough to cross the 512-position switch of the
# attention kernels; sub-chunks of 256 so the span is cut several times.
set -e
R=${0:A:h}/..
M=$1; shift
if [ -n "$1" ] && [ -f "$1" ]; then IDS=$1; shift; else
  IDS=$(mktemp -t prefill_ids); python3 -c "import random; random.seed(1100); print(','.join(str(random.randrange(1000, 100000)) for _ in range(1100)))" > $IDS
fi
${DMSERVE_BIN:-$R/build/dmserve} --model $M --cache-gb 5 --mem mlock --ctx 2048 --chunk 256 --span 8192 --check $IDS --gen 16 "$@" | tee /dev/stderr | \
  python3 -c "import json,sys; r=json.loads(sys.stdin.read().strip().splitlines()[-1]); ok=r['logit_bits_differ']==0 and r['fixed_state_bytes_differ']==0 and r['kv_bytes_differ']==0 and r['greedy_identical']; print('PREFILL CHECK', 'PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)"
