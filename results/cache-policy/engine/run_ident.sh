#!/bin/bash
# old scan vs incremental: eviction logs identical, policy ms/token
W=${CP_WORK:-.}
ENG=${DEEP_MEMORY}/engine
mkdir -p $W/logs
for cell in "pitt results/stage2/runs/record_400.bin" "coding results/stage3/runs/record_code_400.bin" "agent results/stage3c/runs/record_tf_agent.bin"; do
  set -- $cell
  for slots in 452 720 1130 1864 2825 3729; do
    for caps in pooled uniform; do
      extra=$([ $caps = uniform ] && echo uniform || true)
      tag=${1}_${slots}_${caps}
      o=$($W/bin/evict_check_old $ENG/$2 $slots hazard $W/logs/old_$tag.txt $extra)
      n=$($W/bin/evict_check_new $ENG/$2 $slots hazard $W/logs/new_$tag.txt $extra)
      if cmp -s $W/logs/old_$tag.txt $W/logs/new_$tag.txt; then r=identical; else r=DIFFER; fi
      echo "$tag $r $(tail -1 $W/logs/new_$tag.txt) | old: $(echo $o | sed 's/.*picks, //') | new: $(echo $n | sed 's/.*picks, //')"
    done
  done
done
