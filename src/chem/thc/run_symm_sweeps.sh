#!/usr/bin/env bash
# Trace the accuracy/norm trade-off for the 2->inf X^v penalty under three symmetry
# restrictions, so the three can be compared at matched rel_error. Constraint sets nest
# as F(xo,xv,w) subset F(xo,w) subset F(none), so alpha at matched accuracy must satisfy
# none <= xo,w <= xo,xv,w. A violation means the optimizer did not converge, not that
# more symmetry helps.
set -euo pipefail
cd "$(dirname "$0")"
K="${1:-2}"; N="${2:-1}"; shift 2 || true
CONSTS=("${@:-0.1 0.4 1.6}")
                      ./sweep_norm_const.sh "$K" 2inf "$N" 0.1 0.4 1.6
SYMM_PARTS=xo,w       ./sweep_norm_const.sh "$K" 2inf "$N" 0.1 0.4 1.6
SYMM_PARTS=xo,xv,w    ./sweep_norm_const.sh "$K" 2inf "$N" 0.1 0.4 1.6
python evaluate.py data/ISDFov_bareGDF_symm_"$K"x"$K"x"$K"_c5.chk results/*.chk --out results/optimized_norms.csv
python make_results.py
