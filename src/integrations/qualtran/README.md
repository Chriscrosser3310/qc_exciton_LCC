# `src/integrations/qualtran/` — the BSE walk operator

Everything here is reachable from `bse_walk_operator.py` / `bse_block_encoding.py`.
The reachable set was computed by walking the actual Qualtran call graph over all four
option combinations (`ex_density_fitting` x `exchange_central`), controlled and
uncontrolled — not by reading imports. Everything else is in `archive/` (see its README).

## Entry points

| file | what |
|---|---|
| `bse_walk_operator.py` | `W = (2Pi - I) U_A` — what QPE runs on. One reflection on top of the block encoding. |
| `bse_block_encoding.py` | `U_A` — the five templates, the LCU SELECT, the routing. All the physics. Contains `FockTemplate`, `ExchangeTemplate`, `ExchangeDensityFittingTemplate`, `DirectTemplate`. |

## Primitives, with the class actually reached

| file | class(es) in the call graph |
|---|---|
| `eigendecomposition_block_encoding.py` | `EigendecompositionBlockEncoding` |
| `block_isometry_column_synthesis_QROAM.py` | `ColumnIsometryRectangularBlockEncoding`, `BlockIsometryColumnSynthesisQROAM` |
| `berry_isometry_synthesis_QROAM.py` | `BerryIsometrySynthesisQROAM` (density-fitting exchange only) |
| `block_unitary_interferometer_QROAM.py` | `BlockUnitaryInterferometerSynthesisQROAM`, `BlockInterferometerPhaseLayerQROAM`, `BlockInterferometerFinalPhasesQROAM` |
| `load_all_state_preparation_QROAM.py` | `LoadAllStatePreparationQROAM` |
| `diagonal_kernel_block_encoding.py` | `DiagonalCoulombKernelBlockEncoding` — split out of `direct_Coulomb_block_encoding.py`, 2026-08-13 |
| `real_rotation_layers_QROAM.py` | `RealMultiControlledRotationQROAM` — the other two only under `real_data=True` |
| `classical_matrix_block_encoding_QROAM.py` | `BlockDiagonalClassicalMatrixBlockEncoding`, `DirectHermitianBlockEncoding` (Frobenius central) |
| `block_state_preparation_QROAM.py` | `BlockStatePreparationViaQROAMRotations`, `BlockPRGAViaPhaseGradientQROAM` (inside Frobenius) |

## Function-only helpers (no classes, so they never appear in the graph)

| file | provides |
|---|---|
| `toffoli_cost.py` | `toffoli_count` = `n_ccz + n_t/4`. **Use this, not `n_ccz`** — `n_ccz` alone omits the QROAM select-swap network. |
| `qroam_block_sizes.py` | exact (brute-force, measured) QROAM block-size optimiser |
| `range_safe_qroam.py` | the P-13 range-safety surcharge `delta = w - popcount(L-1)` |
| `utils.py` | shared small helpers |

## Import-time dependencies only

Never instantiated, but imported for helper functions or for options that are off by
default. They cannot be archived without editing code.

| file | why it is needed |
|---|---|
| `state_prep_QROAM.py` | `_to_tuple_or_none`, `_cap_log_block_sizes`, `RotationTree` — imported by 6 live modules |
| `three_phase_layer_state_prep_QROAM.py` | `ThreePhaseLayerStatePreparation`, the unselected `three_phase_layer_prep` option of the Frobenius central |

## Reporting / verification

| file | what |
|---|---|
| `bse_cost_report.py` | the diamond table, `[2,2,2]`..`[6,6,6]` |
| `primitive_usage_report.py` | how many times each `P-nn` primitive is used per template |
| `compare_primitives_to_model.py` | differential harness against `XPRIZE_paper/script`'s analytic model |

## The 2026-08-13 reorganization

`direct_Coulomb_block_encoding.py` held two unrelated things: the live
`DiagonalCoulombKernelBlockEncoding` and the dead `DirectCoulombBlockEncoding` (the v1
direct template — `DirectTemplate` composes the isometries and this diagonal itself).
One file forced the dead class, **and the two reflection modules it alone imported**, to
stay live. Split; content unchanged apart from a provenance note and a pruned import block:

* `diagonal_kernel_block_encoding.py` — the live half;
* `archive/direct_Coulomb_block_encoding.py` — the original, intact;
* `archive/rectangular_block_encoding_reflection.py`, `archive/block_unitary_reflection_QROAM.py`
  — freed by the split; nothing live imports them.

Net **22 -> 20** live modules, and two of the four "import-time only" entries are gone.

## Remaining conditionally-live code

`RealPhaseLayerQROAM` and `RealSignLayerQROAM` in `real_rotation_layers_QROAM.py` are
reached only under `real_data=True`, which is not the default (the THC factors are complex
at general `k`). Conditionally live, not dead — `RealMultiControlledRotationQROAM` in the
same file *is* used unconditionally.
