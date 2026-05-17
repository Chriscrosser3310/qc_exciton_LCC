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
=== claude cycle ended: Sat May 16 07:44:33 PM PDT 2026 ===
=== claude cycle started: Sat May 16 07:45:33 PM PDT 2026 ===

## Cycle 3 — 2026-05-16

**Task selected:** Add structural-invariants Toffoli-scaling tests for
`BlockUnitarySynthesisQROAM`. This is the cross-check guard the
previous cycle recommended, in a more conservative form: rather than
re-derive the QROAM/state-prep closed-form constants by hand (which
would require encoding the QROAMClean 2D-blocking lambda choice and
phase-gradient costs that the Bloq already implements), this cycle
pins down the *structural* relationships the synthesis bloq's Toffoli
cost must satisfy. Together with `test_block_unitary_synthesis_equivalence`
they form the analytic-vs-Bloq guard for the synthesis path.

**Major changes:**
- New `tests/test_block_unitary_synthesis_scaling.py` covering:
  - **K-linearity:** `T_total(K) == K * T_total(K=1)` exactly for every
    matched `(n_blocks, n_rows, b)`. Follows from each reflection sharing
    the same call graph in `build_call_graph`.
  - **b-affineness:** `T_total(b)` is exactly affine in `b`; `T(b+1)-T(b)`
    is a fixed constant for the chosen `(n_blocks, n_rows, K)`.
  - **b-slope identity:** per-reflection b-slope equals `2*(log2(N)+1)`
    and is independent of `n_blocks`. Verified for `N` in {2, 4, 8, 16, 32}
    × `n_blocks` in {1, 2, 4, 8, 16}.
  - **K-times slope:** total-cost b-slope equals `K *` the per-reflection
    slope.
  - **Shape-only ≡ data-bearing:** `from_shape(...)` and the data-bearing
    constructor with random unitaries give identical Toffoli counts for
    `n_blocks > 1` and isometry (`K < N`) cases. This extends the
    existing single-block equivalence coverage in
    `test_block_unitary_synthesis_equivalence.py`.
  - **`n_reflections` default:** `from_shape(...)` without
    `n_reflections` is `K = n_rows`.
- The test file also runs as a script (no pytest required) and uses
  `importorskip("qualtran")` so it skips cleanly if qualtran is absent.

**Files changed:**
- Added: `tests/test_block_unitary_synthesis_scaling.py` (6 tests)
- Updated: `AUTONOMOUS_LOG.md`

**Tests/checks run:**
- `PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py`
  → 6/6 passed.
- `PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py`
  → 12/12 passed (unaffected, no regression).
- `PYTHONPATH=src python tests/test_model_resource_counts.py`
  → 15/15 passed (unaffected, no regression).

**Achieved goal:** Pinned the K-linearity, b-affineness, and
b-slope-equals-`2(log2(N)+1)` identities of the synthesis bloq's
Toffoli cost (GOALS.md "Improve constant factors in quantum
algorithms"). Any future change that breaks the linear per-reflection
amortization, or that alters the phase-bitsize dependence, must show
up as an explicit test failure.

**Next recommended task:** Carry the closed-form analytic Toffoli
formula for `BlockUnitarySynthesisQROAM` into `model_resource_counts.py`
(parameterized by `lambda_block`, `lambda_qi`, etc.), with the b-slope
identity verified by this cycle's tests as the anchor. The remaining
work is to derive the `b`-independent QROAMClean cost contribution
(which scales as `sqrt(n_blocks * 2^qi)` at the default block-size
choice) and the `QROAMCleanAdjoint` cost, then add a
`test_synthesis_estimator_matches_closed_form` analogous to the
interferometer test.
=== claude cycle ended: Sat May 16 cycle 3 ===
=== claude cycle ended: Sat May 16 07:54:30 PM PDT 2026 ===
=== claude cycle started: Sat May 16 07:55:30 PM PDT 2026 ===

## Cycle 4 — 2026-05-16

**Task selected:** Pin down the ``n_blocks`` amortization identities of
``BlockUnitarySynthesisQROAM`` — the property that block-indexed
synthesis genuinely amortizes QROAM data loading across blocks (i.e.
total Toffoli cost is sub-linear in ``n_blocks`` and approaches the
QROAMClean ``sqrt(n_blocks)`` regime). This is the *raison d'être* of
the block-indexed variant; the previous cycles pinned K-linearity and
b-affineness but did not pin the ``n_blocks`` scaling at all.

**Major changes:**
- New ``tests/test_block_unitary_synthesis_amortization.py`` (5 tests):
  - **Strict sub-linearity:** ``T(n_blocks) < n_blocks * T(1)`` for all
    ``n_blocks >= 2`` across a ``(N, b, K)`` grid with ``N in {4, 8, 16}``,
    ``b in {4, 6, 8, 12}``, ``K in {1, N/2, N}``.
  - **Monotone non-increasing average:** ``T(n_blocks)/n_blocks`` does
    not increase along the doubling sequence ``n_blocks in {1, 2, 4, 8, 16}``.
  - **Quadrupling bound:** ``T(4*n_blocks) <= 2 * T(n_blocks)`` — the
    QROAMClean ``sqrt(M)`` scaling bound under table-length quadrupling.
  - **K-independence of ``n_blocks`` ratio:** ``T(n2, K)/T(n1, K)`` is
    invariant in ``K``. Follows from K-linearity but is the load-bearing
    consistency check that ``n_blocks`` amortization is not entangled
    with reflection-count accounting.
  - **Asymptotic quadrupling identity:** At large ``n_blocks``,
    ``1.7 <= T(64)/T(16) <= 2.0`` — confirms the ratio approaches the
    sqrt-scaling limit of 2 and rules out degenerate cases where the
    QROAM contribution becomes negligible.
- Note in the file's docstring on the chosen ``N >= 4, b >= 4`` regime:
  for ``N=2`` or ``b=2`` the Hadamard / reflection-about-zero per-
  reflection overhead dominates QROAM and the amortization claim is
  not meaningful (this was confirmed empirically before choosing the
  test grid).
- The test file also runs as a script (no pytest required) and uses
  ``importorskip("qualtran")`` so it skips cleanly if qualtran is absent.

**Files changed:**
- Added: ``tests/test_block_unitary_synthesis_amortization.py`` (5 tests)
- Updated: ``AUTONOMOUS_LOG.md``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed.
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 15/15 passed (unaffected).

**Achieved goal:** Locked in the ``n_blocks``-amortization invariants
of the block-unitary synthesis bloq (GOALS.md "Improve constant
factors in quantum algorithms"). The block-indexed variant exists
*because* it amortizes QROAM data loading; any future change that
silently breaks the sub-linear scaling or the QROAMClean
``sqrt`` regime will now fail an explicit test rather than degrade
the report numbers unnoticed.

**Next recommended task:** With K-linearity, b-affineness,
b-slope-= 2(log2(N)+1), and ``n_blocks`` sqrt-amortization all pinned,
the remaining missing piece for the analytic-vs-Bloq guard on the
synthesis path is the b=0 intercept (the ``b``-independent QROAMClean
+ ``QROAMCleanAdjoint`` per-reflection cost). Deriving this closed
form against the existing identities would let
``model_resource_counts.py`` add a synthesis-bloq estimator analogous
to ``block_unitary_interferometer_count``, closing the modelling gap
the cycle-3 plan called out.
=== claude cycle ended: Sat May 16 cycle 4 ===
=== claude cycle ended: Sat May 16 08:00:37 PM PDT 2026 ===
=== claude cycle started: Sat May 16 08:01:37 PM PDT 2026 ===

## Cycle 5 — 2026-05-16

**Task selected:** The cycle-4-recommended next task — pin the
per-reflection ``b=0`` intercept of ``BlockUnitarySynthesisQROAM``'s
Toffoli cost, completing the cost-decomposition anchor needed for an
analytic estimator in ``model_resource_counts.py``.

Cycles 1–4 established (Bloq-side, via ``QECGatesCost``):

* K-linearity: ``T_total = K * T_per_reflection``
* b-affineness: ``T_per_reflection(b)`` is exactly affine in ``b``
* slope identity: per-reflection b-slope = ``2 * (log2(N) + 1)``,
  independent of ``n_blocks``
* ``n_blocks`` amortization in the QROAMClean ``sqrt(M)`` regime

Together these *imply* the decomposition

    T(n_blocks, N, K, b) = K * ( 2*(log2(N) + 1) * b + I_1(n_blocks, N) )

but they leave ``I_1(n_blocks, N)`` — the b=0 intercept — unpinned.
That intercept is the QROAMClean + ``QROAMCleanAdjoint`` data-loading
cost plus the per-reflection Hadamard / reflection-about-zero
overhead, and it is the missing piece a closed-form analytic
``synthesis_count`` would need to predict.

**Major changes:**
- New ``tests/test_block_unitary_synthesis_b_intercept.py`` (5 tests):
  - **Reference table:** Pins ``I_1(n_blocks, N)`` for the 15-point
    grid ``n_blocks ∈ {1, 2, 4, 8, 16} × N ∈ {4, 8, 16}`` as a
    hard-coded ``REFERENCE`` dict, analogous in role to
    ``PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208`` in the
    interferometer model. Any QROAMClean optimizer change that
    perturbs these values now triggers an explicit test failure.
  - **b-independence of intercept:** ``I_1`` extracted at
    ``b ∈ {2, 3, 4, 5, 7, 11}`` agrees exactly — confirms the
    decomposition is exactly (not approximately) affine in ``b``.
  - **K-independence of intercept:** ``I_1`` extracted at
    ``K ∈ {1, 2, N/2, N}`` agrees exactly — confirms the per-reflection
    cost is exactly (not approximately) K-linear.
  - **Full decomposition identity:** ``T == K * (2*(n+1)*b + I_1)``
    across a ``(N, n_blocks, K, b)`` grid. This is the strongest
    statement of the cost structure: if any future code change
    invalidates the decomposition (e.g. introduces a non-affine
    ``b`` dependence or breaks K-linearity), the failure will
    point precisely at the ``(n_blocks, N, K, b)`` cell that broke.
  - **Multi-block intercept positivity:** ``I_1(n_blocks >= 2, N) > 0``
    for the reference grid. Makes explicit the assumption (used
    implicitly by ``test_block_unitary_synthesis_amortization``) that
    the QROAM contribution dominates per-reflection overhead in the
    non-degenerate regime.

- The reference intercept table is *load-bearing*: it is the only
  remaining b=0-direction degree of freedom that an analytic estimator
  must supply. The slope and decomposition structure are already
  proven elsewhere.
- The test file also runs as a script (no pytest required) and uses
  ``importorskip("qualtran")`` so it skips cleanly if qualtran is
  absent.

**Files changed:**
- Added: ``tests/test_block_unitary_synthesis_b_intercept.py`` (5 tests)
- Updated: ``AUTONOMOUS_LOG.md``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed.
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 15/15 passed (unaffected).

**Achieved goal:** Pinned the ``I_1(n_blocks, N)`` per-reflection b=0
intercept of the block-unitary synthesis bloq (GOALS.md "Improve
constant factors in quantum algorithms"). Combined with the K-linear,
b-affine, slope-identity, and ``sqrt(n_blocks)``-amortization tests
from cycles 1–4, the Bloq-side cost of ``BlockUnitarySynthesisQROAM``
is now fully characterized as a closed-form-shaped expression with a
single tabulated unknown (the intercept). The analytic-vs-Bloq guard
on the synthesis path is now feature-complete from the Bloq side.

**Next recommended task:** Add a ``block_unitary_synthesis_count``
function in ``src/integrations/qualtran/model_resource_counts.py``
parameterized as ``K * (2*(log2(N)+1)*b + I_1(n_blocks, N))``, with
``I_1`` provided either as a closed-form derivation of the
``QROAMClean`` table-loading cost (table length ``M = n_blocks * N``,
optimal block-size ``2 * sqrt(M / b)``-style trade-off) or, as a
conservative first pass, looked up from the same reference table
this cycle pins. Add a ``test_synthesis_estimator_matches_closed_form``
analogous to ``test_estimator_matches_closed_form`` for the
interferometer — that closes the modelling gap cycles 3–5 have been
incrementally building toward.
=== claude cycle ended: Sun May 17 cycle 5 ===
=== claude cycle ended: Sat May 16 08:05:50 PM PDT 2026 ===
=== claude cycle started: Sat May 16 08:06:50 PM PDT 2026 ===

## Cycle 6 — 2026-05-16

**Task selected:** The cycle-5-recommended next task — add a
``block_unitary_synthesis_toffoli`` function in
``src/integrations/qualtran/model_resource_counts.py`` parameterized as
``K * (2*(log2(N)+1)*b + I_1(n_blocks, N))``, with ``I_1`` looked up
from a tabulated intercept table (the conservative first-pass option
cycle 5 outlined), and add a ``test_block_unitary_synthesis_toffoli_matches_bloq``
cross-check that the analytic estimator equals the Bloq-side
``QECGatesCost`` Toffoli count over the pinned grid. This closes the
modelling gap cycles 3–5 were incrementally building toward: the
synthesis path now has the same analytic-vs-Bloq guard the
interferometer path already has via ``test_estimator_matches_closed_form``.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - New ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` dict (the 15-point
    ``(n_blocks, n_rows) ∈ {1,2,4,8,16} × {4,8,16}`` table). The values
    are identical to ``REFERENCE`` in
    ``tests/test_block_unitary_synthesis_b_intercept.py`` — the two
    constants are intentionally kept in sync; a divergence on either
    side is a regression signal.
  - New ``block_unitary_synthesis_toffoli(n_blocks, n_rows, bitsize,
    n_reflections)`` returning ``K * (2*(log2(N)+1)*b + I_1)``. Validates
    that ``n_blocks`` and ``n_rows`` are powers of two,
    ``n_reflections <= n_rows``, ``bitsize > 0``, ``n_reflections > 0``,
    and raises ``KeyError`` rather than extrapolating off the tabulated
    grid.
- ``tests/test_model_resource_counts.py``:
  - ``test_block_unitary_synthesis_toffoli_validates_inputs`` — covers
    the input-validation branches and the off-grid ``KeyError`` path.
  - ``test_block_unitary_synthesis_toffoli_decomposition_identity`` —
    pure-Python check that the function returns exactly
    ``K * (slope*b + I_1)`` for ``(b, K) ∈ {2,4,8,12} × {1, N/2, N}``
    across every entry of ``SYNTHESIS_PER_REFLECTION_INTERCEPT``.
  - ``test_block_unitary_synthesis_toffoli_matches_bloq`` — the
    load-bearing cross-check: for every grid point and ``b ∈ {2,4,8}``,
    ``K ∈ {1, N/2, N}``, the analytic estimator equals
    ``get_cost_value(BlockUnitarySynthesisQROAM.from_shape(...),
    QECGatesCost()).toffoli`` exactly. This is the synthesis-side
    analog of ``test_estimator_matches_closed_form``.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_model_resource_counts.py`` (3 new tests)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 18/18 passed (was 15/15 — added 3 synthesis-side tests).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (unaffected; confirms the intercept table is still
  faithful to the Bloq).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).

**Achieved goal:** Closed the analytic-vs-Bloq modelling gap on the
synthesis path (GOALS.md "Improve constant factors in quantum
algorithms"). The synthesis bloq's Toffoli cost is now predictable from
a pure-Python closed-form expression with a single tabulated lookup,
and any future change that perturbs either side surfaces as an
explicit test failure rather than silent drift in the report numbers.

**Next recommended task:** Derive ``I_1(n_blocks, N)`` in closed form
from the Qualtran ``QROAMClean`` cost expression — table length
``M = n_blocks * 2^n``, output bitsize ``b``, optimal block size
``k* ~ sqrt(M*b)`` — plus the per-reflection Hadamard /
reflection-about-zero overhead. Once the closed form reproduces every
entry of ``SYNTHESIS_PER_REFLECTION_INTERCEPT``, the dict becomes a
regression cache rather than the source of truth, and the synthesis
estimator extends to arbitrary ``(n_blocks, N)`` without needing to
re-run the Bloq to populate new grid points. A natural follow-up after
that is a ``block_unitary_synthesis_count`` wrapper that returns a
full ``ResourceCount`` (including qubits) and a corresponding figure
panel in the docs report.
=== claude cycle ended: Sat May 16 cycle 6 ===
