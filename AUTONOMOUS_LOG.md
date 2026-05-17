=== claude cycle started: Sat May 16 07:32:41 PM PDT 2026 ===

## Cycle 1 — 2026-05-16

**Task selected:** Add a unit-test file for the analytic interferometer
resource model in `src/integrations/qualtran/model_resource_counts.py`,
which previously had no test coverage despite being used by
`scripts/block_unitary_resource_report.py` and by the report figures in
`docs/`.

**Major changes:**
- New `tests/test_model_resource_counts.py` covering:
  - `ceil_log2` (boundary values and rejection of non-positive input)
  - `assert_power_of_two` (acceptance + rejection)
  - `block_unitary_interferometer_toffoli` with three hand-checked
    minimal cases (K=1,N=2 / K=2,N=2 / K=1,N=4 at λ=1, b=2)
  - `block_unitary_interferometer_qubits` with two hand-checked cases
  - `block_unitary_interferometer_count` input validation
    (block_dim, lambda_1, lambda_2 must be powers of two) and consistency
    of its derived fields
  - `optimize_block_unitary_interferometer` produces toffoli/qubit
    Pareto-consistent outputs, rejects unknown objectives, and only
    returns power-of-two lambdas
  - **Cross-check:** the analytic
    `block_unitary_interferometer_count` and the Bloq-side
    `estimate_interferometer_resources` agree (toffoli + qubits) for
    matched parameters — this is a useful invariant since the two were
    independently authored.
- The test file also runs as a script (`python tests/test_model_resource_counts.py`)
  for environments without pytest installed (current py312 env has no
  pytest).

**Files changed:**
- Added: `tests/test_model_resource_counts.py`
- Added: `AUTONOMOUS_LOG.md` (this file)

**Tests/checks run:**
- `PYTHONPATH=src python tests/test_model_resource_counts.py` → 15/15 passed.

**Achieved goal:** Locked in an analytic-vs-Bloq equivalence test for
the block-unitary interferometer resource model. This protects future
"Improve constant factors in quantum algorithms" work (GOALS.md) from
silent drift between the two formulas the project tracks.

**Next recommended task:** Add similar unit-tests for
`BlockUnitarySynthesisQROAM` against `UnitarySynthesisQROAM` in the
single-block limit (n_blocks=1 should equal the un-blocked variant up
to register layout), which would harden the regression captured in the
existing `project_block_unitary_report.md` memory.
=== claude cycle ended: Sat May 16 07:38:54 PM PDT 2026 ===
=== claude cycle started: Sat May 16 07:39:54 PM PDT 2026 ===

## Cycle 2 — 2026-05-16

**Task selected:** The previously recommended next task — add unit tests
for `BlockUnitarySynthesisQROAM` against the un-blocked
`UnitarySynthesisQROAM` in the single-block (`n_blocks=1`) limit, and
pin down structural invariants of the block variant.

**Major changes:**
- New `tests/test_block_unitary_synthesis_equivalence.py` covering:
  - **Single-block Toffoli equivalence** at N=2, 4, 8 between
    `BlockUnitarySynthesisQROAM(block_unitaries=U[None])` and
    `UnitarySynthesisQROAM(unitary=U)` using `QECGatesCost`. (Toffoli
    counts match exactly; `and_bloq` and `measurement` counts can drift
    by O(1) at N=8 because the block variant's data tensor is 2D and
    its QROAM block-size optimizer picks a different point on the
    Pareto curve — this is noted in a code comment, not enforced as
    equality.)
  - Signature shape: n_blocks=1 ⇒ no `block` register slot;
    n_blocks>1 ⇒ a `block` register of width `ceil_log2(n_blocks)`.
  - Isometry support: `block_unitaries` of shape `(B, N, K)` with
    `K < N` ⇒ exactly `K` reflections in the call graph.
  - Data-free `from_shape` raises `DecomposeTypeError` on decompose.
  - `from_shape` with `n_reflections=K` preserves K through the bloq.
  - Symbolic `from_shape(n_blocks=Nb, n_rows=N, ...)` retains the
    symbols on the shape, bitsizes, and reflection count.
  - `__attrs_post_init__` rejects non-orthonormal block data.
  - `BlockPrepareHouseholderStateQROAM.adjoint()` toggles `uncompute`
    and is an involution.
  - `reflection(k)` carries the correct `basis_index` and the
    column-k coefficients from the supplied block unitary.
- The test file also runs as a script (no pytest required) and uses
  `importorskip("qualtran")` so it skips cleanly if qualtran is absent.

**Files changed:**
- Added: `tests/test_block_unitary_synthesis_equivalence.py` (12 tests)

**Tests/checks run:**
- `PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py`
  → 12/12 passed.
- `PYTHONPATH=src python tests/test_model_resource_counts.py`
  → 15/15 passed (unaffected, confirming no regression).

**Achieved goal:** Locked in a single-block equivalence Toffoli-cost
contract between `BlockUnitarySynthesisQROAM` and `UnitarySynthesisQROAM`.
This guards "Improve constant factors in quantum algorithms" (GOALS.md):
when the block variant is later tuned to amortize QROAM data loading
across blocks, the n_blocks=1 limit must remain faithful to the
established un-blocked Sec. 4 construction of arXiv:1812.00954.

**Next recommended task:** Extend the analytic model in
`integrations/qualtran/model_resource_counts.py` (or its peer for the
synthesis bloq, if applicable) to predict the Toffoli count of
`BlockUnitarySynthesisQROAM` directly, and add a cross-check test
analogous to `test_estimator_matches_closed_form` so that the
synthesis bloq has the same analytic-vs-Bloq guard as the
interferometer bloq.
=== claude cycle ended: Sat May 16 07:48:00 PM PDT 2026 ===
