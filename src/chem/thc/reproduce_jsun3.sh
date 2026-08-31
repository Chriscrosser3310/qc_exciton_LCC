#!/usr/bin/env bash
# Reproduce jsun3's ISDFov_opt_bareGDF_2x2x2_c5_cref12_norm0.1p2.chk from scratch.
#
# The two inputs their optimizer consumed -- the NON-symmetry-adapted
# ISDFov_bareGDF_2x2x2_c5.chk and _c12.chk -- were deleted from the source tree, so
# this regenerates them from GDF_2x2x2.chk first, then runs their optimize_X_ov.py
# unmodified.
#
# Two things this had to work around, both discovered the hard way:
#
#   1. optimize_X_common.get_device() imports a `gpu_register` module that is not in
#      the source tree. A CPU stub is written below.
#   2. The shipped c5 and c12 references were built by DIFFERENT versions of the
#      generator. Their logs record reg0 = 1e-9 for c5 but reg0 = 1e-10 for c12, and
#      the current generate_isdf_gdf.py hardcodes 1e-9 with no flag. Using the current
#      script for both silently produces a c12 reference with reg = 5.12e-7 instead of
#      their 4.096e-7. vendor/generate_isdf_gdf_reg0.py is their script with reg0
#      exposed as -reg0 (default unchanged).
#
# Requires read access to jsun3's tree for GDF_2x2x2.chk, DFT_2x2x2.pkl and the
# `fft` ISDF package.
#
# Usage:  ./reproduce_jsun3.sh [workdir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JSUN3=/resnick/groups/changroup/members/jsun3/xprize/THC_general
WORK="${1:-$(mktemp -d)}"
DATA="$WORK/data_diamond_gth-cc-pvdz_final"

echo "workdir: $WORK"
mkdir -p "$DATA"
cp -n "$JSUN3/data_diamond_gth-cc-pvdz_final/DFT_2x2x2.pkl" "$DATA/"
cp -n "$JSUN3/data_diamond_gth-cc-pvdz_final/GDF_2x2x2.chk" "$DATA/"

cat > "$WORK/gpu_register.py" <<'PY'
import torch
def acquire_gpu():
    return torch.device("cpu"), None
PY

cd "$WORK"
export PYTHONPATH="$WORK:$JSUN3"

# get_data_dir() resolves relative to the cwd, which is why we cd here.
echo "=== regenerating the initial guess (c=5, reg0=1e-9 as in their log)"
python "$HERE/vendor/generate_isdf_gdf_reg0.py" diamond 2 2 2 gth-cc-pvdz 5 1.5 \
    -suffix final -reg0 1e-9 --ov --save 2>&1 | grep -E 'rel_error =|^reg |X norm|W norm'

echo "=== regenerating the reference (c=12, reg0=1e-10 as in their log)"
python "$HERE/vendor/generate_isdf_gdf_reg0.py" diamond 2 2 2 gth-cc-pvdz 12 1.5 \
    -suffix final -reg0 1e-10 --ov --save 2>&1 | grep -E 'rel_error =|^reg |X norm|W norm'

# base_lr and nsteps_factor are NOT recorded in their filenames or run header, so these
# two values are a guess. use_Fnorm is likewise unrecorded; omitting it selects the
# operator norm, which is their default branch.
echo "=== running jsun3's optimize_X_ov.py, unmodified"
python "$JSUN3/optimize_X_ov.py" diamond 2 gth-cc-pvdz 5 12 0.1 1e-2 1 \
    -power 2 -suffix final --save 2>&1 | tee "$WORK/optimize.log" | grep -E '^final '

echo "=== comparing against the shipped checkpoint"
python "$HERE/compare_reproduction.py" \
    --jsun3 "$JSUN3/data_diamond_gth-cc-pvdz_final" \
    --repro "$DATA"
