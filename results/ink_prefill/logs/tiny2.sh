#!/bin/zsh
# per-application sinks inside multi-sub-chunk spans, cancel inside a span: check-ahead with checkpoints every 48 positions
for sp in "16 256 128" "64 512 256"; do set -- ${=sp}
  DM/local/flash/github/moved-store-ink-prefill/build/dmserve --model DM/local/inkling/_tiny/model --resident DM/engine/build_ink/resident_tiny_b8.bin --cache-gb 0.01 --mem plain --ctx 2048 --chunk $1 --span $2 --ahead-span $3 --ckpt-every 48 --check-ahead tiny/ah_A.txt tiny/ah_B.txt --check-ahead-len 600 --gen 16 --threads 2 --readers 2 > tiny/ah2_$1.txt 2>&1
  python3 -c "import json; r=json.loads(open('tiny/ah2_$1.txt').read().strip().splitlines()[-1]); w=r['with_ahead']; print('chunk $1 span $2:', 'PASSED' if r['passed'] else 'FAILED', 'b_reused', w['b_reused'], 'b_source', w['b_source'], 'cancelled', w['ahead_cancelled_run'].get('cancelled'), 'computed', w['ahead_cancelled_run'].get('computed'), 'req_path reused', r['request_path']['b_reused'])"
done
