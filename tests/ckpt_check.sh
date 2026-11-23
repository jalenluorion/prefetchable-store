#!/bin/zsh
# The prefix cache's checkpoints on both tiny models (tools/dmserve.cpp --check-ckpt): a long request (1,200 ids), the
# identical request, then one diverging at 60% of its length, through one prefix cache with a checkpoint every 256
# positions and the checkpoint memory set to 0 bytes (so only the minimum count, --ckpt-min, keeps them: the rule that
# lets Inkling's 173 MB states be kept at all). Each request's logits, fixed state, KV and greedy reply must equal a cold
# token-by-token computation bit for bit, and the second and third must restore from a checkpoint.
#   tests/ckpt_check.sh Q36_TINY_MODEL INK_TINY_MODEL INK_TINY_RESIDENT_B8      (DMSERVE_BIN: another worker)
set -u
R=${0:A:h}/..
W=${DMSERVE_BIN:-$R/build/dmserve}
T=$(mktemp -d -t ckpt_check)
trap 'rm -rf $T' EXIT
python3 -c "import random; random.seed(5); print(','.join(str(random.randrange(1000, 30000)) for _ in range(1200)))" > $T/ids.txt
fail=0
for m in q36 ink; do
  if [ $m = q36 ]; then MA=(--model $1); else MA=(--model $2 --resident $3 --mem plain --res-mem plain); fi
  $W $MA --cache-gb 0.05 --ctx 2048 --chunk 128 --span 8192 --ckpt-every 256 --ckpt-gb 0 --check-ckpt $T/ids.txt --gen 8 --threads 4 --readers 2 \
    2> $T/$m.err | tail -1 > $T/$m.json
  python3 - $T/$m.json $m <<'PY' || fail=1
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"FAIL {sys.argv[2]}: no result ({e})"); sys.exit(1)
rows = ", ".join(f"{k} {r[k]['source']} reused {r[k]['reused']} computed {r[k]['computed']} bits {r[k]['cmp']['logit_bits_differ']}/"
                 f"{r[k]['cmp']['fixed_state_bytes_differ']}/{r[k]['cmp']['kv_bytes_differ']}" for k in ("long", "identical", "diverging"))
print(("ok   " if r["passed"] else "FAIL ") + sys.argv[2] + ": " + rows)
sys.exit(0 if r["passed"] else 1)
PY
done
(( fail )) && { echo "CKPT CHECK FAILED"; exit 1; } || echo "CKPT CHECK PASSED"
