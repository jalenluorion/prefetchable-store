#!/bin/zsh
# Permanent test of draft and verify on the tiny models (seconds; no real model): exit 0 only if all hold.
#  1. dmeng --verify-check: 150 greedy single steps, then the same positions through verify passes of 2-8 positions (every
#     third with a wrong last draft, rolled back): every kept position's logits and the final state bit-identical
#     (Qwen3.6 tiny at i16 / i8 / f16 and through the split-K attention; Inkling tiny, 8-bit and bf16 resident, i16 / f32;
#     both with the screened head (--fast-head 1: the passes screen every position as the single steps do) and the
#     previous pass (--verify-old 1))
#  2. dmeng --draft fixed | lookup gives the ids of plain decoding, greedy and sampled (seed 7, top-k 20): drafts come from
#     tool-call fragments made of plain decoding's own output with every 6th token changed (random tiny models do not
#     repeat themselves, so a lookup alone would draft nothing), so passes are kept whole, cut, or rejected; 1 and 7
#     drafts per pass; --draft fixed fails if nothing was verified (the controller, lookup, may decline every pass)
#  3. the server worker (build/dmserve) gives the same ids with --draft lookup / fixed as without, over requests sharing a
#     prefix, one with tool-call fragments, and the same prompt asked again (the lookup copies the earlier answer from
#     the session's archive) greedy and sampled (bench/drafts/worker_drive.py)
#   tests/draft_check.sh Q36_TINY_MODEL INK_TINY_MODEL INK_TINY_RESIDENT_B0 INK_TINY_RESIDENT_B8
set -u
R=${0:A:h}/..
B=${DMENG_BIN:-$R/build/dmeng}; W=${DMSERVE_BIN:-$R/build/dmserve}
QT=$1; IT=$2; I0=$3; I8=$4
T=$(mktemp -d -t draft_check)
fail=0
P=$(python3 -c "import random; random.seed(3); print(','.join(str(random.randrange(500)) for _ in range(60)))")
Q=(--threads 2 --readers 2 --mem plain --res-mem plain)
vc() { local name=$1; shift; "$@" --verify-check 150 > /dev/null 2> $T/vc.err; local r=$?; local l=$(grep -o 'verify_check .*' $T/vc.err)
  if [[ $r -eq 0 && $l == *'"ok":true'* ]]; then echo "PASS verify-check $name"; else echo "FAIL verify-check $name: $l"; fail=1; fi; }
for p in i16 i8 f16; do vc "q36 $p" $B --model $QT --prompt-ids $P --n 8 --cache-gb 0.002 --prec $p $Q; done
DMENG_SPLIT_MIN=40 vc "q36 split-K" $B --model $QT --prompt-ids $P --n 8 --cache-gb 0.002 $Q
vc "q36 screened head" $B --model $QT --prompt-ids $P --n 8 --cache-gb 0.002 --fast-head 1 $Q
vc "q36 previous pass" $B --model $QT --prompt-ids $P --n 8 --cache-gb 0.002 --verify-old 1 $Q
for r in $I0 $I8; do for p in i16 f32; do vc "ink ${r:t} $p" $B --model $IT --resident $r --ctx 1024 --prompt-ids $P --n 8 --cache-gb 0.01 --prec $p $Q; done; done
vc "ink screened head" $B --model $IT --resident $I8 --ctx 1024 --prompt-ids $P --n 8 --cache-gb 0.01 --fast-head 1 $Q
vc "ink previous pass" $B --model $IT --resident $I8 --ctx 1024 --prompt-ids $P --n 8 --cache-gb 0.01 --verify-old 1 $Q
# drafts from fragments of plain decoding's own output (every 6th token changed)
for m in q36 ink; do
  if [[ $m == q36 ]]; then M=(--model $QT --cache-gb 0.01); else M=(--model $IT --resident $I8 --ctx 1024 --cache-gb 0.01); fi
  for sm in greedy sampled; do
    S=(); [[ $sm == sampled ]] && S=(--temp 1 --top-k 20 --top-p 0.95 --seed 7)
    $B $M --prompt-ids $P --n 100 $Q $S --out-ids $T/$m.$sm.none.txt > /dev/null 2>&1
    python3 -c "p='$P'.split(',')[-6:]; g=open('$T/$m.$sm.none.txt').read().strip().split(','); print(' '.join(p+[t if i%6!=5 else str((int(t)+1)%500) for i,t in enumerate(g)]))" > $T/frag.txt
    for d in fixed lookup; do for dm in 1 7; do
      $B $M --prompt-ids $P --n 100 $Q $S --draft $d --draft-max $dm --skeleton $T/frag.txt --out-ids $T/$m.$sm.$d.txt --json $T/d.json > /dev/null 2>&1
      nv=$(python3 -c "import json; print(json.load(open('$T/d.json'))['draft']['verifies'])" 2>/dev/null)
      if cmp -s $T/$m.$sm.none.txt $T/$m.$sm.$d.txt && [[ ${nv:-0} -gt 0 || $d == lookup ]]; then echo "PASS ids $m $sm --draft $d --draft-max $dm ($nv passes)"; else echo "FAIL ids $m $sm --draft $d --draft-max $dm (${nv:-?} passes)"; fail=1; fi
    done; done
  done
done
python3 -c "
import json,random
random.seed(5); a=[random.randrange(500) for _ in range(50)]
json.dump([{'ids':a,'max_new':80},{'ids':[7,8,9]+a[:20],'max_new':60,'continue':True},{'ids':a[:30],'max_new':40,'skel':[a[10:30],[1,2,3,4,5]]},
           {'ids':a,'max_new':80},{'ids':a,'max_new':80,'temp':1,'top_p':0.95,'top_k':20,'seed':3},{'ids':a,'max_new':80,'temp':1,'top_p':0.95,'top_k':20,'seed':3}],open('$T/reqs.json','w'))"
for d in none fixed lookup; do python3 $R/bench/drafts/worker_drive.py $T/w.$d.json $T/reqs.json -- $W --model $QT --cache-gb 0.002 --ctx 2048 $Q --draft $d > /dev/null 2>&1; done
for d in fixed lookup; do
  if python3 -c "import json,sys; a=json.load(open('$T/w.none.json')); b=json.load(open('$T/w.$d.json')); sys.exit(0 if [x['gen'] for x in a]==[x['gen'] for x in b] and a and sum(x['E']['json']['draft'].get('accepted',0) for x in b)>0 else 1)"; then echo "PASS server ids --draft $d"; else echo "FAIL server ids --draft $d"; fail=1; fi
done
echo "DRAFT CHECK $([ $fail = 0 ] && echo PASSED || echo FAILED)"
exit $fail
