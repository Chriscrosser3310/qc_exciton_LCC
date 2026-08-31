# Vendored data

Copied read-only on 2026-08-25 from

    /resnick/groups/changroup/members/jsun3/xprize/THC_general/data_diamond_gth-cc-pvdz_final/

Diamond primitive cell, `gth-cc-pvdz`, n_ao = 26, n_occ = 4, n_vir = 22.

| file | role | size |
|---|---|---|
| `DFT_2x2x2_symm.pkl` | mean field; source of `mo_coeff` and `cell` | 944 KB |
| `DFT_3x3x3_symm.pkl` | mean field | 1.2 MB |
| `ISDFov_bareGDF_symm_2x2x2_c5.chk` | optimizer initial guess, M = 136 | 2.8 MB |
| `ISDFov_bareGDF_symm_2x2x2_c20.chk` | optimizer reference, M = 520 | 36 MB |
| `ISDFov_bareGDF_symm_3x3x3_c5.chk` | optimizer initial guess | 9.5 MB |
| `ISDFov_bareGDF_symm_3x3x3_c20.chk` | optimizer reference | 134 MB |
| `diamond_prim.xyz`, `diamond_prim.lattice` | geometry, for provenance | <1 KB |

## Why the symmetry-adapted checkpoints, and not the ones jsun3 actually used

`optimize_X_ov.py` resolves its inputs by naming convention:

    init_chk = ISDFov_{screen_tag}GDF{symm_tag}_{klabel}_c{c_isdf}.chk
    ref_chk  = ISDFov_{screen_tag}GDF{symm_tag}_{klabel}_c{c_ref}.chk

jsun3's shipped `ISDFov_opt_*` checkpoints carry no `_symm` tag and are named
`cref12`, so they were produced from `ISDFov_bareGDF_<mesh>_c5.chk` and
`ISDFov_bareGDF_<mesh>_c12.chk` -- the **non-symmetry-adapted** ov family.

**Those files no longer exist.** The diamond directory now holds only the `_symm`
ov family (c = 5, 6, 8, 10, 16, 20), and there is no saved `opt_X_ov_*.pt` optimizer
state either. So jsun3's optimized checkpoints cannot be reproduced or re-derived
from what survives.

The substitute used here is the symmetry-adapted family: **init c = 5** (M = 136
rather than their 130) and **reference c = 20** (the highest available, rather than
their 12). Consequences:

- Runs here are **not** bit-comparable to `ISDFov_opt_bareGDF_*_cref12_norm0.1p2.chk`.
- They *are* internally consistent, so the operator-norm and `2->inf` runs are
  directly comparable to each other, which is the actual question.
- The c = 20 reference is more accurate than a c = 12 one would be, so `rel_error`
  here is measured against a harder target and will read higher than jsun3's.

## One setting that is unrecoverable

`optimize_X_ov.py` has a `--use_Fnorm` flag switching the central-tensor penalty
between the operator norm and the Frobenius norm. It is neither printed by
`print_header()` nor encoded in the output filename, so **which one produced the
existing `ISDFov_opt_*` files cannot be determined** from any surviving artifact.
Runs here default to the operator norm (`--w-norm op`, upstream's default branch).
