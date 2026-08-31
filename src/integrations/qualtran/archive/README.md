# Archived qualtran modules — not in the BSE walk operator

Everything here is outside the dependency closure of `bse_walk_operator.py` /
`bse_block_encoding.py`, computed by transitive import analysis rather than by hand.
Archived, not deleted — several are the direct ancestors of what replaced them.

## Superseded by the Sec.-2 construction

| module | replaced by | why |
|---|---|---|
| `bse_block_encoding_v1.py`, `bse_walk_operator_v1.py` | `../bse_block_encoding.py`, `../bse_walk_operator.py` | the `<psi\| S P A D P S \|psi>` form; not the manuscript's |
| `fock_block_encoding.py` | `FockTemplate` + `EigendecompositionBlockEncoding` | used SVD (`U S V`); Sec. 2 specifies eigendecomposition (`U D U^dag`), which is also an involution for free |
| `exchange_Coulomb_block_encoding.py` | `ExchangeTemplate` | isometries on both sides; Sec. 2 uses controlled state preparation on the virtual side |
| `svd_block_encoding_interferometer.py` | `EigendecompositionBlockEncoding` | `U S V` is not self-inverse and needs both interferometers controlled |
| `antisymmetric_projector_block_encoding.py`, `particle_number_counter.py` | — | only the v1 sandwich used them; the manuscript imposes antisymmetry on the input state |

## Alternative syntheses not selected

`interferometer_isometry_QROAM.py`, `block_interferometer_isometry_QROAM.py`,
`recursive_csd_synthesis_QROAM.py`, `unitary_reflection_QROAM.py` — other isometry /
unitary syntheses. The construction uses Iten column-by-column (THC templates) and
Berry §III.B (density-fitting exchange); the crossover between those two is measured in
`context/primitives.md` P-07.

## Reporting / analysis

`model_resource_counts.py`, `physical_resource_counts.py`, `data_loading_comparison.py`.

## Future work, deliberately parked

**`symmetry_adaptation_QROAM.py`** — the space-group symmetry-adaptation transform `Q`.
Not dead code: it is the intended route to making the block-dependent data flat in
`N_k`, which is the one lever that would change the density-fitting comparison
(`N_k^2` address). It was never wired into any walk operator. See
`../../../chem/fftisdf/symmetry_glossary.tex` and `symmetry_blocks_report.tex`.

## Dependents

`../../../../tests/archive/` and `../../../../scripts/archive/` hold the tests and
scripts that import these modules and would otherwise break. Note some of the archived
tests *also* covered modules that are still live — `test_three_phase_layer_toggle.py`
(6 live modules), `test_block_unitary_reflection_*` (3), `test_interferometer_isometry.py`,
`test_recursive_csd_synthesis.py` — so live-module coverage is reduced by this sweep.

## 2026-08-13 — the direct-Coulomb split

`direct_Coulomb_block_encoding.py` (here, intact) held two unrelated classes:

* `DiagonalCoulombKernelBlockEncoding` — **live**, now in `../diagonal_kernel_block_encoding.py`;
* `DirectCoulombBlockEncoding` — the v1 direct template, never reached by the Sec.-2
  construction (`DirectTemplate` composes the isometries and the diagonal itself).

Splitting them freed two more modules that only the dead half imported:
`rectangular_block_encoding_reflection.py` (the unselected `outer_synthesis="reflection"`
option) and `block_unitary_reflection_QROAM.py` (its `optimal_reflection_*` helpers).

Coverage lost: `tests/archive/test_direct_Coulomb_block_encoding.py` exercised **both**
classes (9 references to the dead one, 6 to the live diagonal), so it had to move whole —
the live diagonal is no longer directly unit-tested, only through the walk operator.
Likewise `scripts/archive/block_isometry_column_synthesis_report.py` reported on the
*live* column synthesis but imported the reflection module for comparison.
