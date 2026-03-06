# Conversation Log - 2026-03-06

## Scope
Continued development and benchmarking of exciton block-encoding workflows (Qualtran-based), with focus on QSVT-style resource tables, sparse/dense data-loading behavior, and scaling interpretation.

## Key Requests and Actions
- Expanded QSVT table generation to larger sweeps and added CLI configurability in `experiments/generate_qsvt_exciton_lc_eq_drc_table.py`.
- Added and ran fixed-locality QSVT benchmark script with independent `L_c` and `d`:
  - `experiments/generate_qsvt_exciton_lc_d_table_fixed_locality.py`
  - Produced tables for:
    - fixed-locality (`R_c=3`) case
    - corrected `R_c=0` case
    - random-nonzero variant (oracle-compatible magnitude projection)
- Adjusted table formatting and ordering:
  - Removed `R_c` column for fixed-locality table.
  - Sorted from small to large (`L_c`, then `d`).
- Added separate QROM scaling notebook:
  - `experiments/qrom_t_count_scaling.ipynb`
  - Includes random-data QROM sweeps and power-law fit plotting.
- Investigated observed scaling exponents by examining implementation details:
  - highlighted role of tensor masks/cutoffs in `benchmark_tensors.py`
  - highlighted QROM break-early behavior in Qualtran.
- Added separate dense-diagonal W/V benchmark script and output table:
  - `experiments/generate_qsvt_exciton_dense_diag_wv_table.py`
  - `docs/qsvt_exciton_table_m2_d2_dense_diag_wv.tex`
  - Caption explicitly states `R_c=R_loc=0` dense-diagonal setup.

## Notable Outputs
- `docs/qsvt_exciton_table_m2_d2_fixed_locality_rc0_lc_d.tex`
- `docs/qsvt_exciton_table_m2_d2_fixed_locality_rc0_lc_d_random_nonzero.tex`
- `docs/qsvt_exciton_table_m2_d2_dense_diag_wv.tex`
- Additional table variants produced for larger sweeps under related settings.

## Notes
- Some large sweeps (`L_c` upper bound very high) hit runtime limits in resource counting; reduced bounds were used to complete tables.
- Current entry-oracle implementation requires values in `[0,1]`; signed random inputs were converted to magnitudes for compatible runs.
