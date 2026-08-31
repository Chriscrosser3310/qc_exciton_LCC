#!/usr/bin/env bash
# Optimize diamond THC collocation matrices at 2x2x2 and 3x3x3, under two penalties:
#   (a) the upstream norm  -- ||X^v||_op        (baseline, reproduces jsun3's convention)
#   (b) the modified norm  -- ||X^v||_{2->inf}
# then evaluate all five alpha norms on the results.
#
# Usage:  ./run_diamond.sh [nsteps_factor]
# Runtime on CPU with nsteps_factor=1: ~5 min per 2x2x2 run, ~14 min per 3x3x3 run.
set -euo pipefail
cd "$(dirname "$0")"

NSTEPS="${1:-1}"
C_ISDF=5
C_REF=20
COMMON=(--c-isdf "$C_ISDF" --c-ref "$C_REF" --norm-const 0.1 --power 2
        --base-lr 1e-2 --nsteps-factor "$NSTEPS" --save)

mkdir -p results logs

for KMESH in 2 3; do
  for XV in op 2inf; do
    LOG="logs/opt_${KMESH}x${KMESH}x${KMESH}_xv${XV}.log"
    echo "=== kmesh=${KMESH}  xv-norm=${XV}  -> ${LOG}"
    python optimize.py --kmesh "$KMESH" --xv-norm "$XV" "${COMMON[@]}" > "$LOG" 2>&1
    tail -5 "$LOG"
  done
done

echo "=== evaluating norms on all produced checkpoints"
python evaluate.py results/*.chk --out results/optimized_norms.csv
