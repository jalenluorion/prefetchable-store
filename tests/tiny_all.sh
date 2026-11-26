#!/bin/zsh
# One command for the engine's checks on tiny random models of all three families (minutes; no real model). Each family's
# tiny model is built by its generator in this repository (tests/<family>/make_tiny.py: a synthetic release in the real
# release's layout, the repo's converter, a router set as the weight repo publishes it, the manifest), the engine-ready
# files are derived (ms_derive) and listed with their sha256 in the manifest, and tests/family_check.py runs the SAME
# checks on each: reference, invariance, prefill, checkpoint, server, drafting, presets. A check a family cannot run is
# reported "not supported for this family", with the engine's refusal checked. Prints the family x check matrix.
#
#   tests/tiny_all.sh [OUT]          (OUT default: build/tiny_all; configure and build the repo first)
#   TORCH_PY   a python with torch, transformers, safetensors, numpy, tokenizers  (the 35B's and Flash-Next's generators and
#              references; default python3)
#   MLX_PY     a python with mlx, mlx-lm with the Inkling 2 model (mlx-lm PR 2086), numpy  (Inkling's; default python3)
#   BUILD      the build folder (default build/); FAMILIES a subset ("qwen3_5_moe inkling2_loop qwen4_exp")
#   BASELINE   an earlier OUT: each family's fingerprint (tests/fingerprint.py, the logits and ids of eleven fixed runs)
#              must be identical to the one there (a refactor that must not move a bit)
# Exit 0 only when no check fails.
set -u
R=${0:A:h:h}
B=${BUILD:-$R/build}
O=${1:-$B/tiny_all}
TP=${TORCH_PY:-python3}
MP=${MLX_PY:-python3}
mkdir -p $O
typeset -A PY
PY=(qwen3_5_moe $TP inkling2_loop $MP qwen4_exp $TP)
fail=0
# the engine's fallback for keys a folder's manifest leaves out is generated from the published manifests: it must be current
python3 $R/python/convert/published_defaults.py --check || fail=1
# the spare tier's mechanics on a small store of known bytes (no model: tests/spare_tier/spare_tier_check.cpp)
$B/spare_tier_check $O > $O/spare_tier_check.txt 2>&1 || { echo "FAIL spare_tier_check (see $O/spare_tier_check.txt)"; fail=1; }
tail -1 $O/spare_tier_check.txt
for f in ${=FAMILIES:-qwen3_5_moe inkling2_loop qwen4_exp}; do
  py=${PY[$f]}
  log=$O/$f.log
  : > $log
  if ! $py $R/tests/$f/make_tiny.py $O/$f >> $log 2>&1; then
    echo "FAIL generator $f (see $log)"; fail=1; continue
  fi
  # the engine-ready files: derived once by the engine's own code, then listed with their sha256 in the manifest (the
  # engine checks every one it uses against it)
  $B/ms_derive resident $O/$f/model >> $log 2>&1 || { echo "FAIL ms_derive resident $f (see $log)"; fail=1; continue; }
  for id in $(python3 -c "import json, sys; print(' '.join(r['id'] for r in json.load(open(sys.argv[1]))['routers']))" $O/$f/model/moved-store.json); do
    $B/ms_derive router $O/$f/model $id >> $log 2>&1 || { echo "FAIL ms_derive router $id $f (see $log)"; fail=1; }
  done
  (cd $O/$f/model/.dmeng && shasum -a 256 *.bin) > $O/$f.derived.sha256
  python3 $R/tests/tiny_manifest.py $O/$f/model --derived-sha $O/$f.derived.sha256 >> $log 2>&1 || { echo "FAIL manifest $f"; fail=1; continue; }
  echo "== $f"
  python3 $R/tests/family_check.py $O/$f $O/$f.check --bin-dir $B --ref-python $py || fail=1
  if [ -n "${BASELINE:-}" ]; then
    python3 $R/tests/fingerprint.py $O/$f/model $O/$f.fingerprint.json --bin-dir $B --against $BASELINE/$f.fingerprint.json > $O/$f.fingerprint.txt
    r=$?; tail -1 $O/$f.fingerprint.txt; [ $r = 0 ] || fail=1
  else
    python3 $R/tests/fingerprint.py $O/$f/model $O/$f.fingerprint.json --bin-dir $B > /dev/null || fail=1
  fi
done
python3 - $O ${=FAMILIES:-qwen3_5_moe inkling2_loop qwen4_exp} <<'PY'
import json, os, sys
out, fams = sys.argv[1], sys.argv[2:]
rows = {f: json.load(open(os.path.join(out, f + ".check", "family_check.json")))["checks"] for f in fams if os.path.exists(os.path.join(out, f + ".check", "family_check.json"))}
checks = []
for r in rows.values():
    checks += [c for c in r if c not in checks]
w = max([len(f) for f in rows] + [6])
print("\n" + "check".ljust(12) + "".join(f.ljust(max(w, 30) + 2) for f in rows))
for c in checks:
    print(c.ljust(12) + "".join(rows[f].get(c, {}).get("status", "-").ljust(max(w, 30) + 2) for f in rows))
PY
echo "TINY ALL $([ $fail = 0 ] && echo PASSED || echo FAILED)"
exit $fail
