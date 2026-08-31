#!/usr/bin/env bash
# Sweep the penalty weight for one choice of X^v norm, optionally restricting some
# factors to the space-group symmetric subspace.
#
# Why this exists: the 2->inf norm is numerically smaller than the operator norm, so at
# a fixed --norm-const the penalty term carries less weight and the optimizer buys fit
# accuracy instead. Comparing two norms at equal --norm-const therefore conflates
# "different norm" with "weaker penalty". Sweeping recovers the accuracy/norm trade-off
# curve for each setting, and comparisons are then read at matched rel_error.
#
# Usage:  ./sweep_norm_const.sh <kmesh> <xv-norm> [nsteps_factor] [const ...]
#         SYMM_PARTS=xo,w ./sweep_norm_const.sh 2 2inf 1 0.1 0.4 1.6
#
# SYMM_PARTS is a comma-separated subset of xo,xv,w passed to --symm-parts. Empty means
# no symmetry restriction.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results logs

KMESH="${1:?usage: sweep_norm_const.sh <kmesh> <xv-norm> [nsteps] [const ...]}"
XV="${2:?}"
NSTEPS="${3:-1}"
if [ "$#" -ge 3 ]; then shift 3; else shift 2; fi
CONSTS=("$@")
if [ "${#CONSTS[@]}" -eq 0 ]; then CONSTS=(0.025 0.1 0.4 1.6 6.4); fi

PARTS="${SYMM_PARTS:-}"
STAG=$(printf '%s' "${PARTS:-none}" | tr ',' '-')

for CONST in "${CONSTS[@]}"; do
  LOG="logs/sweep_${KMESH}x${KMESH}x${KMESH}_xv${XV}_symm${STAG}_c${CONST}.log"
  echo "=== kmesh=${KMESH} xv-norm=${XV} symm=${PARTS:-none} norm-const=${CONST} -> ${LOG}"
  if [ -n "$PARTS" ]; then
    python optimize.py --kmesh "$KMESH" --xv-norm "$XV" --symm-parts "$PARTS" \
        --c-isdf 5 --c-ref 20 --norm-const "$CONST" --power 2 \
        --base-lr 1e-2 --nsteps-factor "$NSTEPS" --save > "$LOG" 2>&1
  else
    python optimize.py --kmesh "$KMESH" --xv-norm "$XV" \
        --c-isdf 5 --c-ref 20 --norm-const "$CONST" --power 2 \
        --base-lr 1e-2 --nsteps-factor "$NSTEPS" --save > "$LOG" 2>&1
  fi
  grep -E 'final (rel_error|norm_loss)' "$LOG"
done
