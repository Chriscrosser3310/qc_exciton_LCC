#!/usr/bin/env bash
# Compare which factors are constrained to the symmetric subspace, at identical penalty
# settings. Upstream can only do all-or-nothing, because it symmetrises the shared X_ao
# before splitting into occupied and virtual. Selecting a subset is legitimate: the two
# spaces are separately invariant under the crystal group, verified to 2e-14.
#
#   (none)     no symmetry constraint at all -- what jsun3's shipped opt files used
#   xo,w       occupied factor and W constrained, virtual factor free
#   xo,xv,w    everything constrained -- upstream's --symm
#
# Usage:  ./run_symm_variants.sh [kmesh] [nsteps_factor]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results logs

KMESH="${1:-2}"
NSTEPS="${2:-1}"
COMMON=(--kmesh "$KMESH" --c-isdf 5 --c-ref 20 --norm-const 0.1 --power 2
        --base-lr 1e-2 --nsteps-factor "$NSTEPS" --xv-norm op --save)

for PARTS in "" "xo,w" "xo,xv,w"; do
  LABEL="${PARTS:-none}"
  LOG="logs/symm_${KMESH}x${KMESH}x${KMESH}_$(echo "$LABEL" | tr ',' '-').log"
  echo "=== kmesh=${KMESH}  symmetrised=${LABEL}  -> ${LOG}"
  if [ -z "$PARTS" ]; then
    python optimize.py "${COMMON[@]}" > "$LOG" 2>&1
  else
    python optimize.py "${COMMON[@]}" --symm-parts "$PARTS" > "$LOG" 2>&1
  fi
  grep -E 'final (rel_error|norm_loss)' "$LOG"
done

echo "=== evaluating"
python evaluate.py results/*.chk --out results/optimized_norms.csv
python make_results.py
