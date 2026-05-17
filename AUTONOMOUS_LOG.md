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
=== claude cycle ended: Sat May 16 08:10:00 PM PDT 2026 ===
=== claude cycle started: Sat May 16 08:11:00 PM PDT 2026 ===

## Cycle 7 — 2026-05-16

**Task selected:** Extend ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` (and its
mirror ``REFERENCE`` table in
``tests/test_block_unitary_synthesis_b_intercept.py``) from the 15-point
``(n_blocks, N) ∈ {1,2,4,8,16} × {4,8,16}`` grid to the 49-point
``{1,2,4,8,16,32,64} × {4,8,16,32,64,128,256}`` grid.

The previous cycles' decomposition

    T(n_blocks, N, K, b) = K * ( 2*(log2(N)+1)*b + I_1(n_blocks, N) )

reduces the analytic estimator to a single ``I_1`` lookup, and prior
cycles deferred a closed-form derivation of ``I_1``. Until that closed
form is derived, the practical bottleneck is that the existing 15-point
table cannot predict the synthesis cost at the parameters the actual
report uses (default ``N=256``, ``n_blocks=k^3``). This cycle extends
the tabulated regime to cover those parameters directly. The
closed-form derivation remains the longer-term goal but is no longer
the gating step for using the estimator in report-shaped runs.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` extended from 15 → 49 entries
    covering ``n_blocks ∈ {1,2,4,8,16,32,64}`` and ``N ∈ {4,8,16,32,64,128,256}``.
    Values computed from ``BlockUnitarySynthesisQROAM.from_shape(...)``
    via ``QECGatesCost`` at ``K=1, b=4`` and the established b-affineness
    / K-linearity identities — i.e. the same extraction procedure the
    existing ``_intercept`` helper in
    ``tests/test_block_unitary_synthesis_b_intercept.py`` uses.
- ``tests/test_block_unitary_synthesis_b_intercept.py``:
  - ``REFERENCE`` extended to the same 49-point grid. The two constants
    stay synced; their equality remains an invariant a future
    auto-generation step could enforce.
- ``tests/test_model_resource_counts.py``:
  - ``test_block_unitary_synthesis_toffoli_validates_inputs`` updated to
    use ``(n_blocks=128, n_rows=4)`` as the off-grid ``KeyError`` probe,
    since ``(32, 4)`` is now a populated entry.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_block_unitary_synthesis_b_intercept.py``
- Modified: ``tests/test_model_resource_counts.py``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (the full-decomposition test now ranges over the 49-point
  grid × 5 b-values × 4 K-values).
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 18/18 passed (including ``test_block_unitary_synthesis_toffoli_matches_bloq``
  which now cross-checks the analytic count against the Bloq over the
  full 49-point grid).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).

**Achieved goal:** The analytic synthesis-bloq Toffoli estimator can
now predict resources at the parameters the actual model-resource-counts
report uses (default ``N=256``, ``n_blocks`` up to 64) without needing
to re-run the Bloq (GOALS.md "Improve constant factors in quantum
algorithms" / "Any potential improvements on the final Toffoli
complexity/qubit counts/scaling"). This removes the practical KeyError
blocker that would have prevented adding a synthesis-bloq panel to the
report PDF generator.

**Next recommended task:** With the 49-point grid in place, add a
``block_unitary_synthesis_count`` wrapper (analogous to
``block_unitary_interferometer_count``) that returns a full
``ResourceCount`` including qubits, then plumb it into ``_plot_report``
to add a synthesis-bloq panel in the report PDF. The qubit count needs
deriving — base = ``ceil_log2(n_blocks) + log2(N) + b + 1`` (block +
system + phase gradient + reflection ancilla) plus the QROAMClean
workspace which depends on the optimal block size at the chosen
``M = n_blocks * N``. The Bloq's ``log_block_sizes`` property exposes
the optimizer's choice and could be queried for the per-row workspace.
=== claude cycle ended: Sat May 16 cycle 7 ===
=== claude cycle ended: Sat May 16 08:56:00 PM PDT 2026 ===
=== claude cycle started: Sat May 16 08:57:00 PM PDT 2026 ===

## Cycle 8 — 2026-05-16

**Task selected:** The cycle-7-recommended next task — begin the
``block_unitary_synthesis_count`` wrapper by adding the analytic
**signature qubit count** for ``BlockUnitarySynthesisQROAM``. This
captures the persistent (input/output) register width
``ceil_log2(n_blocks) + 1 + log2(n_rows) + bitsize`` exactly. The
remaining piece — the transient QROAMClean workspace, which depends on
the chosen ``log_block_sizes`` — is intentionally deferred to a later
cycle since (a) it requires tabulating or deriving a non-trivial
optimizer choice and (b) the signature qubit count is independently
useful as a lower bound and as the base term of the eventual full
``ResourceCount`` wrapper.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - New ``block_unitary_synthesis_signature_qubits(n_blocks, n_rows,
    bitsize) -> int`` returning
    ``ceil_log2(n_blocks) + 1 + log2(n_rows) + bitsize``. Validates
    ``n_rows`` power-of-two, ``n_blocks > 0``, ``bitsize > 0``. Note:
    unlike the Toffoli helper this does **not** require ``n_blocks`` to
    be a power of two — ``ceil_log2`` already handles arbitrary
    positive ``n_blocks`` and the bloq's ``block_bitsize`` follows
    ``bit_length(n_blocks - 1)``.
  - Docstring explicitly flags it as a lower bound (no QROAM workspace).
- ``tests/test_model_resource_counts.py`` (3 new tests):
  - ``test_block_unitary_synthesis_signature_qubits_validates_inputs``
    — covers non-power-of-two ``n_rows`` rejection and the positivity
    checks.
  - ``test_block_unitary_synthesis_signature_qubits_formula`` — three
    hand-checked points including a non-power-of-two ``n_blocks=3``
    case.
  - ``test_block_unitary_synthesis_signature_qubits_matches_bloq`` —
    the load-bearing cross-check: for every entry of
    ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` and ``b ∈ {2, 8, 32}``, the
    analytic count equals ``bloq.signature.n_qubits()`` exactly. This
    is the signature-qubit-side analog of
    ``test_block_unitary_synthesis_toffoli_matches_bloq``.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_model_resource_counts.py`` (3 new tests, 18→21)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 21/21 passed (was 18/18 — added 3 signature-qubit tests).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed (unaffected).

**Achieved goal:** Pinned the analytic signature qubit count of
``BlockUnitarySynthesisQROAM`` against the bloq's
``signature.n_qubits()`` over the full 49-point ``(n_blocks, N) ∈
{1,2,4,8,16,32,64} × {4,...,256}`` grid × three bitsizes (GOALS.md
"Improve constant factors in quantum algorithms" / "Any potential
improvements on the final Toffoli complexity/qubit counts/scaling").
This closes the qubit-side base-term half of the eventual analytic
``block_unitary_synthesis_count`` wrapper; the only remaining piece is
the transient QROAMClean workspace contribution.

**Next recommended task:** Tabulate the QROAMClean workspace
contribution (peak transient qubits) for
``BlockUnitarySynthesisQROAM`` at the
``log_block_sizes`` choice the optimizer picks for each
``(n_blocks, N, bitsize)`` grid point. The cleanest path is a Bloq-side
extraction analogous to the ``I_1`` intercept extraction: run
``QubitCount`` (or ``get_qubit_counts`` from
``integrations.qualtran.utils``) on
``BlockUnitarySynthesisQROAM.from_shape(...)`` at fixed
``log_block_sizes``, subtract ``signature.n_qubits()``, and pin the
resulting workspace table. Once that's in place,
``block_unitary_synthesis_count`` can return a full
``ResourceCount``-shaped record (with ``log_block_sizes`` replacing
``lambda_1, lambda_2`` — perhaps via a new ``SynthesisResourceCount``
dataclass to keep the field semantics honest).
=== claude cycle ended: Sat May 16 cycle 8 ===
=== claude cycle ended: Sat May 16 09:00:39 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:01:39 PM PDT 2026 ===

## Cycle 9 — 2026-05-16

**Task selected:** Bundle the two existing analytic synthesis estimators
(``block_unitary_synthesis_toffoli`` and
``block_unitary_synthesis_signature_qubits``, pinned in cycles 5–8) into
a single ``block_unitary_synthesis_count`` wrapper returning a
``SynthesisResourceCount`` record. This is the natural composition of
the two helpers and mirrors the ``block_unitary_interferometer_count``
API shape that the interferometer side already exposes.

**Pivot from cycle-8 recommendation:** Cycle 8 recommended next adding a
QROAMClean workspace table by running Qualtran's ``QubitCount`` on
``BlockUnitarySynthesisQROAM`` and subtracting the signature qubits.
A direct probe revealed an upstream blocker: ``QubitCount`` raises
``RuntimeError: tuple index out of range`` from
``BlockPRGAViaPhaseGradientQROAM`` for many ``(n_blocks, N)`` cells
(e.g. ``(2, 16)``, ``(4, 32)``, ``(8, 64)``), with sporadic non-monotone
support. This means a clean workspace table cannot be tabulated this
cycle without first investigating the upstream bug. Rather than block
on that, this cycle takes the smaller composition step: it locks in the
``ResourceCount``-shaped API the next-cycle work would need anyway, and
documents the workspace gap explicitly in the ``SynthesisResourceCount``
docstring so future work has a clear named place to extend.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - New ``SynthesisResourceCount`` frozen dataclass with fields
    ``toffoli``, ``signature_qubits``, ``n_blocks``, ``n_rows``,
    ``bitsize``, ``n_reflections``. Its docstring explicitly flags
    ``signature_qubits`` as a strict lower bound and names the missing
    transient QROAMClean workspace contribution.
  - New ``block_unitary_synthesis_count(n_blocks, n_rows, bitsize,
    n_reflections) -> SynthesisResourceCount`` wrapper composing the
    two existing analytic helpers. Validation flows through unchanged.
- ``tests/test_model_resource_counts.py`` (2 new tests, 21→23):
  - ``test_block_unitary_synthesis_count_composes_existing_estimators``
    — for three representative grid points, checks that the wrapper's
    fields exactly equal the underlying helpers and that the parameter
    fields round-trip.
  - ``test_block_unitary_synthesis_count_propagates_validation`` —
    confirms that the wrapper inherits ``ValueError`` on bad parameters
    and ``KeyError`` on off-grid intercepts from the underlying helpers.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_model_resource_counts.py`` (2 new tests)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 23/23 passed (was 21/21).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).

**Achieved goal:** Exposed a single composed analytic
``block_unitary_synthesis_count`` entry point with a stable record-shaped
return type, in line with the existing
``block_unitary_interferometer_count`` API (GOALS.md "Improve constant
factors in quantum algorithms" / "Any potential improvements on the
final Toffoli complexity/qubit counts/scaling"). The wrapper is the
shape a future ``_plot_report`` synthesis panel can pull from without
having to wire two separate calls.

**Next recommended task:** Investigate the upstream
``BlockPRGAViaPhaseGradientQROAM`` ``QubitCount`` ``tuple index out of
range`` failure that blocked the original cycle-8 next task. Repro at
``n_blocks=2, N=16, b=4, K=1``. The fix likely lies in the bloq's
``build_call_graph`` or signature shape — once ``QubitCount`` succeeds
across the 49-point grid, ``SynthesisResourceCount`` can be extended
with a ``workspace_qubits`` field tabulated against the bloq's
``QubitCount - signature.n_qubits()``, completing the qubit-side
analytic-vs-Bloq guard. If the upstream fix turns out to be
non-trivial, a smaller fallback cycle would be adding a synthesis-bloq
panel to the report PDF generator using only the lower-bound
``signature_qubits`` and noting the workspace gap in the report
summary page.
=== claude cycle ended: Sat May 16 cycle 9 ===
=== claude cycle ended: Sat May 16 09:09:35 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:10:35 PM PDT 2026 ===

## Cycle 10 — 2026-05-16

**Task selected:** The cycle-9-recommended next task — investigate the
upstream ``QubitCount`` failure that blocked the original cycle-8 plan
to tabulate transient QROAMClean workspace qubits for
``BlockUnitarySynthesisQROAM``. Cycle 9 reported failures at
``(n_blocks=2, n_rows=16)``, ``(4, 32)``, ``(8, 64)`` with sporadic
non-monotone support. A clean repro this cycle showed those grid
points actually work today; the *real* failure mode is at any
``n_blocks=1`` cell, with the error ``KeyError: 'block'`` from inside
``BlockPrepareHouseholderStateQROAM.build_composite_bloq``.

**Root cause:** ``block_bitsize = bit_length(n_blocks - 1)`` evaluates
to ``0`` when ``n_blocks=1``. ``Signature.build(...)`` omits registers
of bitsize 0, so the bloq's signature has *no* ``block`` slot.
``QubitCount`` follows the bloq's ``build_composite_bloq`` path (rather
than the ``build_call_graph`` path that ``QECGatesCost`` uses), and
that decomposition unconditionally called ``soqs.pop("block")`` and
then ``bb.add_d(self.state_prep, block=block, ...)``. Both fail when
``block`` isn't in the signature. Note that the downstream
``BlockStatePreparationViaQROAMRotations.signature`` already correctly
omits ``block`` when ``n_blocks=1``, so the fix is fully local to
``BlockPrepareHouseholderStateQROAM``.

**Major changes:**
- ``src/integrations/qualtran/block_unitary_synthesis_QROAM.py``:
  - ``BlockPrepareHouseholderStateQROAM.build_composite_bloq`` now
    pops ``block`` with a ``None`` default rather than required-key
    semantics. The output dict adds the ``block`` soquet back only if
    one was present.
  - ``_apply_controlled_state_prep`` accepts ``block: Optional[Soquet]``
    and forwards it to ``self.state_prep`` only when non-``None`` (via
    a kwarg unpack), so the data-free single-block path now matches
    the state-prep signature exactly.
  - ``BlockHouseholderReflectionQROAM.build_composite_bloq`` was
    unaffected — it threads soqs as ``**soqs`` and never refers to the
    ``block`` key by name.
- ``tests/test_block_unitary_synthesis_qubit_count.py`` (5 new tests):
  - **Regression guard:** ``test_qubit_count_succeeds_for_single_block``
    explicitly pins the ``n_blocks=1`` case that previously raised.
  - **Full-grid coverage:** ``test_qubit_count_works_over_full_intercept_grid``
    iterates the same 49-point grid covered by
    ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` and confirms ``QubitCount``
    succeeds and returns at least the signature lower bound. Cross-checks
    the bloq's signature width against the analytic
    ``block_unitary_synthesis_signature_qubits`` for every grid point.
  - **Qubit-side single-block equivalence:**
    ``test_qubit_count_single_block_matches_unblocked`` — qubit-side
    analog of the existing Toffoli single-block equivalence test;
    ``BlockUnitarySynthesisQROAM(n_blocks=1)`` and
    ``UnitarySynthesisQROAM`` agree on ``QubitCount`` at ``N=2,4,8``.
  - **Monotonicity invariants:**
    ``test_workspace_monotone_in_n_blocks`` and
    ``test_workspace_monotone_in_n_rows`` pin the structural
    requirement that QROAMClean workspace (= ``QubitCount`` minus
    signature) is monotone non-decreasing in both ``n_blocks`` and
    ``n_rows`` along the doubling sequence — a sanity check on future
    QROAMClean optimizer changes.

**Files changed:**
- Modified: ``src/integrations/qualtran/block_unitary_synthesis_QROAM.py``
- Added:    ``tests/test_block_unitary_synthesis_qubit_count.py`` (5 tests)
- Modified: ``AUTONOMOUS_LOG.md``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_qubit_count.py``
  → 5/5 passed.
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected — confirms the existing Toffoli
  single-block equivalence still holds after the build_composite_bloq
  refactor).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (unaffected — the intercept extraction at K=1 only
  exercises the call-graph path, not the new decompose path).
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 23/23 passed (unaffected).

**Achieved goal:** Unblocked Qualtran's ``QubitCount`` across the full
analytic 49-point ``(n_blocks, n_rows)`` grid for
``BlockUnitarySynthesisQROAM`` (GOALS.md "Any potential improvements
on the final Toffoli complexity/qubit counts/scaling"). The
``SynthesisResourceCount.workspace_qubits`` field that cycle 9 had to
defer can now be populated from the bloq directly. The signature path
remains the load-bearing lower bound; the upstream blocker is gone.

**Next recommended task:** With ``QubitCount`` now working over the
full grid, populate the ``SynthesisResourceCount.workspace_qubits``
field by tabulating
``QubitCount(BlockUnitarySynthesisQROAM.from_shape(...))`` minus
``signature.n_qubits()`` across the same 49-point grid (parameterized
by ``bitsize`` too, since the QROAMClean optimizer's choice depends on
``bitsize``). Add the resulting table to ``model_resource_counts.py``
alongside ``SYNTHESIS_PER_REFLECTION_INTERCEPT``, extend
``block_unitary_synthesis_count`` to return a ``workspace_qubits``
field, and add a ``test_block_unitary_synthesis_workspace_matches_bloq``
cross-check analogous to ``test_block_unitary_synthesis_toffoli_matches_bloq``.
That closes the qubit-side analytic-vs-Bloq guard on the synthesis
path, mirroring what the Toffoli side already has.
=== claude cycle ended: Sat May 16 cycle 10 ===
=== claude cycle ended: Sat May 16 09:18:26 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:19:26 PM PDT 2026 ===

## Cycle 11 — 2026-05-16

**Task selected:** The cycle-10-recommended next task — populate the
``SynthesisResourceCount.workspace_qubits`` field by tabulating
``QubitCount(BlockUnitarySynthesisQROAM.from_shape(...)) -
signature.n_qubits()`` across the same 49-point
``(n_blocks, n_rows) ∈ {1,2,4,8,16,32,64} × {4,8,16,32,64,128,256}`` grid,
parameterized by ``bitsize ∈ {2, 4, 8, 16, 32}``. With this in place the
analytic synthesis estimator can predict total qubits (signature +
workspace) and the qubit-side analytic-vs-Bloq guard mirrors the Toffoli
side that ``test_block_unitary_synthesis_toffoli_matches_bloq`` already
provides.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - New ``SYNTHESIS_WORKSPACE_QUBITS: dict[tuple[int, int, int], int]``
    tabulating workspace qubits across the 245-point
    ``49 × 5`` grid. Values extracted with ``n_reflections=1``; the
    workspace is the *peak* over the reflection loop (not the sum) and
    is independent of ``n_reflections``.
  - New ``block_unitary_synthesis_workspace_qubits(n_blocks, n_rows,
    bitsize) -> int`` lookup helper; off-grid raises ``KeyError`` with
    a clear message rather than silently extrapolating.
  - ``SynthesisResourceCount`` extended with two new fields:
    ``workspace_qubits`` and ``total_qubits``. The docstring is updated
    to describe ``total_qubits = signature_qubits + workspace_qubits``
    as the bloq's ``QubitCount`` value (no longer just a lower bound).
  - ``block_unitary_synthesis_count`` now also calls the workspace
    helper and populates ``workspace_qubits`` / ``total_qubits``.

- ``tests/test_model_resource_counts.py`` (3 new tests, 23→26):
  - ``test_block_unitary_synthesis_workspace_qubits_validates_inputs``
    covers non-power-of-two ``n_rows`` / ``n_blocks``, non-positive
    ``bitsize``, and off-grid ``KeyError`` paths (both off-grid
    ``bitsize`` and off-grid ``n_blocks``).
  - ``test_block_unitary_synthesis_workspace_table_consistent`` pins
    that every cell of the 245-point grid is present and positive.
  - ``test_block_unitary_synthesis_count_total_qubits_matches_bloq`` is
    the load-bearing cross-check: for every ``(n_blocks, n_rows,
    bitsize)`` in the table, ``block_unitary_synthesis_count(...).
    total_qubits == get_cost_value(bloq, QubitCount())``. This is the
    qubit-side analog of ``test_block_unitary_synthesis_toffoli_matches_bloq``.
  - ``test_block_unitary_synthesis_count_composes_existing_estimators``
    extended to also assert ``workspace_qubits`` and ``total_qubits``
    fields.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_model_resource_counts.py``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 26/26 passed (was 23/23).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_qubit_count.py``
  → 5/5 passed (unaffected; the workspace monotonicity invariants this
  file pins are preserved by the new tabulated values).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).

**Achieved goal:** Closed the qubit-side analytic-vs-Bloq guard on the
synthesis path. ``block_unitary_synthesis_count`` now returns a fully
populated ``SynthesisResourceCount`` whose ``total_qubits`` field equals
the bloq's ``QubitCount`` exactly across the 245-point tabulated grid
(GOALS.md "Improve constant factors in quantum algorithms" / "Any
potential improvements on the final Toffoli complexity/qubit counts/
scaling"). The synthesis estimator can now be plumbed into a report PDF
panel without needing to re-run the bloq for qubit counts.

**Next recommended task:** Add a synthesis-bloq panel to the report PDF
generator (``_plot_report`` in ``model_resource_counts.py``). With the
analytic estimator now fully populated for both Toffoli and total qubits,
the panel can compare ``block_unitary_synthesis_count(...)`` curves
across ``n_blocks`` at fixed ``N=256, bitsize=32`` (the report defaults)
in the same plot style as the existing interferometer panels. The
remaining longer-horizon task is to *derive* the workspace table in
closed form from QROAMClean's optimizer (so the dict becomes a
regression cache rather than the source of truth) — but that work is no
longer gating any report-side use of the estimator.
=== claude cycle ended: Sat May 16 cycle 11 ===
=== claude cycle ended: Sat May 16 09:24:50 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:25:50 PM PDT 2026 ===

## Cycle 12 — 2026-05-16

**Task selected:** The cycle-11-recommended next task — add a synthesis-bloq
panel to ``_plot_report`` in ``model_resource_counts.py``. With the analytic
``block_unitary_synthesis_count`` estimator fully populated for Toffoli and
total qubits over the tabulated grid, the report PDF can now show synthesis
curves without re-running the bloq.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - New ``SYNTHESIS_PANEL_N_BLOCKS = (1, 2, 4, 8, 16, 32, 64)`` covering the
    full tabulated intercept grid for n_blocks.
  - New ``_synthesis_panel_records(n_rows, bitsize, n_reflections,
    n_blocks_seq)`` helper that sweeps ``block_unitary_synthesis_count`` over
    the power-of-two ``n_blocks`` sequence.
  - ``_plot_report`` extended with two new plot pages
    (``plot_synthesis("toffoli")``, ``plot_synthesis("qubits")``) and a
    ``synthesis_table_page`` page; signature now accepts
    ``synthesis_n_blocks`` and ``synthesis_n_reflections`` kwargs
    (defaults: tabulated grid and ``K = block_dim``).
  - ``summary_page`` updated to describe the new synthesis panel: which
    parameters were swept and the ``total_qubits = signature + workspace``
    decomposition.
- ``tests/test_model_resource_counts.py`` (3 new tests, 26→29):
  - ``test_synthesis_panel_records_match_block_unitary_synthesis_count``
    pins that the helper is a thin sweep over the underlying analytic
    estimator (no aggregation, no transformation).
  - ``test_synthesis_panel_records_use_tabulated_grid`` confirms every
    default ``(n_blocks, n_rows, bitsize=32)`` panel point is in both the
    intercept and workspace tables, ruling out a silent KeyError.
  - ``test_plot_report_generates_pdf`` smoke-tests that the full PDF
    pipeline (including the new synthesis pages) writes a non-empty file
    at the default block_dim/bitsize parameters.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_model_resource_counts.py`` (3 new tests)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 29/29 passed (was 26/26).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_qubit_count.py``
  → 5/5 passed (unaffected; confirms the workspace-monotonicity invariants
  the new panel relies on are preserved).

**Achieved goal:** The analytic synthesis estimator
(``block_unitary_synthesis_count``) is now plumbed into the report PDF
generator (GOALS.md "Improve constant factors in quantum algorithms" /
"Any potential improvements on the final Toffoli complexity/qubit
counts/scaling"). Toffoli and total-qubit curves for
``BlockUnitarySynthesisQROAM`` can be regenerated from a pure-Python
closed-form expression with tabulated workspace, without needing to
re-run the bloq — closing the report-side use case cycles 5–11 were
incrementally building toward.

**Next recommended task:** Derive ``SYNTHESIS_WORKSPACE_QUBITS`` (or at
least ``SYNTHESIS_PER_REFLECTION_INTERCEPT``) in closed form from
QROAMClean's optimizer expression. Once the closed form reproduces every
tabulated entry, the two dicts become a regression cache rather than the
source of truth, and the synthesis estimator extends to arbitrary
``(n_blocks, n_rows, bitsize)`` without needing to re-run the bloq to
populate new grid points. The b-affineness, K-linearity, and
``log_block_sizes``-monotonicity tests from cycles 3–10 are the anchors a
closed-form derivation must satisfy.
=== claude cycle ended: Sat May 16 cycle 12 ===
=== claude cycle ended: Sat May 16 09:29:20 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:30:20 PM PDT 2026 ===

## Cycle 13 — 2026-05-16

**Task selected:** A small consistency guard the prior cycles flagged
but never implemented. Cycle 7's log noted that
``REFERENCE`` (in ``tests/test_block_unitary_synthesis_b_intercept.py``)
and ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` (in
``src/integrations/qualtran/model_resource_counts.py``) "stay synced;
their equality remains an invariant a future auto-generation step could
enforce." This cycle implements that enforcement in its simplest form
— a direct equality test — so drift between the production lookup and
the Bloq-derived reference table surfaces as a loud failure rather
than silent disagreement between the analytic estimator and the bloq.

Deferred the larger cycle-12 recommendation (deriving
``SYNTHESIS_WORKSPACE_QUBITS`` in closed form from QROAMClean's
optimizer) as it requires non-trivial new work; the small consistency
guard is a faster way to harden the existing tabulated infrastructure.

**Major changes:**
- ``tests/test_block_unitary_synthesis_b_intercept.py``:
  - Import ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` from
    ``integrations.qualtran.model_resource_counts``.
  - New ``test_reference_table_matches_module_table`` asserts the two
    49-entry tables are equal dicts. With ``test_reference_intercept_table_matches``
    (which re-derives ``REFERENCE`` from the Bloq) this transitively
    pins the production lookup against the bloq's current
    ``QECGatesCost`` output.

**Files changed:**
- Modified: ``tests/test_block_unitary_synthesis_b_intercept.py``
  (+1 test, 5→6)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 6/6 passed.
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 29/29 passed (unaffected).

**Achieved goal:** Closed the auto-generation gap the cycle-7 log
flagged: the two intercept tables are now structurally guaranteed
equal, so future grid extensions (the natural next step toward
covering ``n_blocks > 64`` or ``N > 256``) cannot accidentally update
one without the other (GOALS.md "Improve constant factors in quantum
algorithms" / "Any potential improvements on the final Toffoli
complexity/qubit counts/scaling").

**Next recommended task:** Resume the cycle-12 recommendation —
derive ``SYNTHESIS_WORKSPACE_QUBITS`` (or
``SYNTHESIS_PER_REFLECTION_INTERCEPT``) in closed form from
QROAMClean's optimizer expression (table length
``M = n_blocks * n_rows``, output bitsize ``b``, optimal block size
``k* ~ sqrt(M*b)``-style trade-off) plus the reflection-about-zero /
Hadamard per-reflection overhead. The 245-point workspace table and
49-point intercept table together form a comprehensive ground-truth
fixture against which any candidate closed form can be checked, and
the new equality guard ensures the production estimator stays in
sync with whichever side ends up being the source of truth.
=== claude cycle ended: Sat May 16 cycle 13 ===
=== claude cycle ended: Sat May 16 09:32:58 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:33:58 PM PDT 2026 ===

## Cycle 14 — 2026-05-16

**Task selected:** Add a fast pure-Python structural-invariant
monotonicity test on ``SYNTHESIS_WORKSPACE_QUBITS``. Cycle 10 added
Bloq-driven monotonicity tests
(``test_workspace_monotone_in_n_blocks`` and
``test_workspace_monotone_in_n_rows`` in
``tests/test_block_unitary_synthesis_qubit_count.py``) but each only
sweeps a single slice — ``n_rows=16, bitsize=4`` and
``n_blocks=4, bitsize=4`` respectively. Silent corruption of a value at,
say, ``(8, 256, 16)`` would not be caught by the existing tests until
``test_block_unitary_synthesis_count_total_qubits_matches_bloq`` re-ran
the full Bloq grid. This cycle adds a fast pure-dict check that locks
in the same monotonicity invariants across every slice.

Deferred (again) the cycle-12 / cycle-13-recommended derivation of
``SYNTHESIS_WORKSPACE_QUBITS`` in closed form from QROAMClean's
optimizer — that remains a larger task; this cycle hardens the existing
tabulated infrastructure with a single small test.

**Major changes:**
- ``tests/test_model_resource_counts.py``:
  - ``test_block_unitary_synthesis_workspace_table_monotone`` (1 new
    test, 29→30):
    * non-decreasing in ``n_blocks`` along ``(1,2,4,8,16,32,64)`` at
      every fixed ``(n_rows, bitsize)`` (35 slices)
    * non-decreasing in ``n_rows`` along ``(4,8,16,32,64,128,256)``
      at every fixed ``(n_blocks, bitsize)`` (35 slices)
  - Docstring explicitly notes that monotonicity in ``bitsize`` is
    **not** an invariant (verified empirically: 16 of the 245 entries
    show non-monotone bitsize trends, driven by the QROAMClean
    optimizer's discrete block-size choices — e.g.,
    ``(64, 256, 4)=139`` vs ``(64, 256, 8)=77``). This documents the
    edge case for future authors who might be tempted to add an
    over-broad invariant.

**Files changed:**
- Modified: ``tests/test_model_resource_counts.py`` (+1 test)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 30/30 passed (was 29/29).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_qubit_count.py``
  → 5/5 passed (unaffected; the cycle-10 single-slice monotonicity
  tests are preserved).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 6/6 passed (unaffected).

**Achieved goal:** Locked in ``n_blocks`` and ``n_rows`` monotonicity
of ``SYNTHESIS_WORKSPACE_QUBITS`` across every (n_rows, bitsize) and
(n_blocks, bitsize) slice — extending the cycle-10 single-slice
Bloq-based coverage to the full 70-slice structural invariant
(GOALS.md "Improve constant factors in quantum algorithms" / "Any
potential improvements on the final Toffoli complexity/qubit
counts/scaling"). Any single-entry corruption surfaces immediately
without needing to re-run the Bloq.

**Next recommended task:** Resume the long-standing recommendation from
cycles 12/13 — derive ``SYNTHESIS_WORKSPACE_QUBITS`` (or
``SYNTHESIS_PER_REFLECTION_INTERCEPT``) in closed form from
QROAMClean's optimizer expression. The empirical observation pinned
this cycle — that workspace is non-monotone in ``bitsize`` due to the
optimizer's discrete block-size choice — is a useful constraint a
candidate closed form must reproduce, alongside the now-confirmed
``n_blocks``/``n_rows`` monotonicity.
=== claude cycle ended: Sat May 16 cycle 14 ===
=== claude cycle ended: Sat May 16 09:38:21 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:39:21 PM PDT 2026 ===

## Cycle 15 — 2026-05-16

**Task selected:** Add a report-visible power-law fit for the synthesis
panel (Toffoli and total_qubits vs ``n_blocks``) in ``_plot_report``,
and pin its alpha as sub-linear (alpha < 1). Cycles 4 and 5 already
proved sub-linear ``n_blocks`` amortization at the bloq-internal level
via ``test_block_unitary_synthesis_amortization`` (``T(n_blocks) <
n_blocks * T(1)``), but that property is a per-reflection-cost
statement and is invisible to a reader of the report PDF. The
interferometer side already exposes a ``power_law_fit`` table on the
summary page; the synthesis side did not. This cycle closes that
asymmetry with a small targeted change.

Deferred (again) the long-standing recommendation to derive
``SYNTHESIS_WORKSPACE_QUBITS`` or
``SYNTHESIS_PER_REFLECTION_INTERCEPT`` in closed form from
QROAMClean's optimizer — that remains a larger task and is not
gating any report-side use of the estimator.

**Major changes:**
- ``src/integrations/qualtran/model_resource_counts.py``:
  - ``_plot_report`` now computes ``synth_fits`` (a dict of
    ``synth_t``, ``synth_q`` power-law fits via ``power_law_fit``
    over ``synthesis_n_blocks``).
  - ``plot_synthesis`` overlays the fit line ``c * n_blocks^alpha``
    on both the Toffoli and total-qubit plots, labelled with alpha.
  - ``summary_page`` reports the two synthesis alphas and coefficients
    next to the existing interferometer fits, with a one-line note that
    ``alpha < 1`` is the report-visible signature of QROAM
    amortization.
- ``tests/test_model_resource_counts.py``:
  - Import ``power_law_fit`` from the module.
  - New ``test_synthesis_panel_power_law_sublinear`` confirms
    ``alpha_t ∈ [0, 1)`` and ``alpha_q ∈ [0, 1)`` for the canonical
    report parameters (N=256, b=32, K=block_dim=256). Empirically
    ``alpha_t ≈ 0.275`` and ``alpha_q ≈ 0.024`` — both well below 1,
    consistent with the slope-times-b term dominating the synthesis
    Toffoli at these parameters and ``total_qubits`` being essentially
    flat in ``n_blocks``.

**Files changed:**
- Modified: ``src/integrations/qualtran/model_resource_counts.py``
- Modified: ``tests/test_model_resource_counts.py`` (+1 test, 30→31)

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 31/31 passed (was 30/30).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_scaling.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_amortization.py``
  → 5/5 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_equivalence.py``
  → 12/12 passed (unaffected).
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_qubit_count.py``
  → 5/5 passed (unaffected).

**Achieved goal:** The QROAMClean ``n_blocks``-amortization claim
(GOALS.md "Improve constant factors in quantum algorithms" / "Any
potential improvements on the final Toffoli complexity/qubit
counts/scaling") is now both (a) plotted on the synthesis panel as a
visible fit line and (b) summarized in the report's text page with
alpha values. A regression that broke sub-linear scaling would now
fail a fast pure-Python test rather than silently changing the report
curve.

**Next recommended task:** Resume the cycle-12+ long-standing
recommendation — derive ``SYNTHESIS_WORKSPACE_QUBITS`` (or
``SYNTHESIS_PER_REFLECTION_INTERCEPT``) in closed form from
QROAMClean's optimizer expression. The empirical fits this cycle
pinned (``alpha_t ≈ 0.275`` at N=256, b=32) are a concrete asymptotic
constraint any candidate closed form must reproduce, alongside the
``n_blocks``/``n_rows`` monotonicity and ``b=0`` intercept tables.
A natural intermediate step before that is to add a small
``scripts/regenerate_synthesis_tables.py`` helper that extracts the
two tabulated dicts directly from the bloq, so re-tabulation under a
QROAMClean upstream change becomes a one-command operation rather
than a manual edit.
=== claude cycle ended: Sat May 16 cycle 15 ===
=== claude cycle ended: Sat May 16 09:45:37 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:46:37 PM PDT 2026 ===

## Cycle 16 — 2026-05-16

**Inbox:** AUTONOMOUS_INBOX.md (2026-05-16) — "send me a pdf report
each time, with math equations rendered as latex". A LaTeX-rendered
PDF deliverable for this cycle is at
``docs/cycle_reports/cycle16_report.pdf`` (math via matplotlib
mathtext; no external LaTeX install required). An attempt to email it
via the existing ``email_report`` SMTP path was denied by the harness'
auto-mode classifier as unauthorized external communication; the user
can pick up the file locally or add an explicit Bash permission rule
for ``sendmail`` if they want future cycles to email automatically.

**Task selected:** Resume cycle 15's recommended next task in its
*intermediate* form — add a small ``scripts/regenerate_synthesis_tables.py``
helper that extracts ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` and
``SYNTHESIS_WORKSPACE_QUBITS`` directly from
``BlockUnitarySynthesisQROAM`` so re-tabulation under a QROAMClean
upstream change is a one-command operation rather than a manual edit.
This is the safer intermediate step before deriving the dicts in
closed form. The full closed-form derivation remains the open follow-up.

**Major changes:**
- New ``scripts/regenerate_synthesis_tables.py``:
  - ``extract_intercept(n_blocks, n_rows)`` subtracts the
    ``2*(log2(N)+1)*b_ref`` slope contribution from
    ``QECGatesCost(bloq).toffoli`` to recover the per-reflection
    ``b=0`` intercept ``I_1``.
  - ``extract_workspace(n_blocks, n_rows, bitsize)`` subtracts
    ``signature.n_qubits()`` from ``QubitCount(bloq)``.
  - ``regenerate_intercept_table`` / ``regenerate_workspace_table``
    sweep the same 49-point and 245-point grids the shipped dicts
    cover.
  - ``--check`` (default) diffs the regenerated dicts against the
    shipped tables and exits non-zero on drift; ``--print`` emits
    Python source ready to paste back into ``model_resource_counts.py``;
    ``--no-workspace`` skips the slower workspace sweep.
- New ``tests/test_regenerate_synthesis_tables.py`` (4 tests):
  - ``extract_intercept`` matches the shipped lookup on a 4-point
    corner sample.
  - ``extract_workspace`` matches the shipped lookup on a 3-point
    corner sample.
  - ``_format_intercept_table`` / ``_format_workspace_table`` emit
    ``ast.literal_eval``-roundtrippable Python source.
- New ``scripts/_cycle16_report.py`` (private helper for this cycle's
  PDF deliverable; not part of the shipped report pipeline).
- New ``docs/cycle_reports/cycle16_report.pdf`` (the cycle deliverable
  per the inbox instruction).

**Files changed:**
- Added: ``scripts/regenerate_synthesis_tables.py``
- Added: ``scripts/_cycle16_report.py``
- Added: ``tests/test_regenerate_synthesis_tables.py``
- Added: ``docs/cycle_reports/cycle16_report.pdf``
- Modified: ``AUTONOMOUS_LOG.md``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_regenerate_synthesis_tables.py``
  → 4/4 passed.
- ``PYTHONPATH=src python tests/test_block_unitary_synthesis_b_intercept.py``
  → 6/6 passed (unaffected).
- ``PYTHONPATH=src python tests/test_model_resource_counts.py``
  → 31/31 passed (unaffected).
- ``PYTHONPATH=src python scripts/regenerate_synthesis_tables.py``
  (full intercept + workspace sweep) → ``SYNTHESIS_PER_REFLECTION_INTERCEPT:
  OK (49 entries match)``; ``SYNTHESIS_WORKSPACE_QUBITS: OK (245
  entries match)``.

**Achieved goal:** Closed cycle-12+'s long-standing intermediate
recommendation. The two tabulated dicts the analytic synthesis
estimator depends on are now regenerable from the Bloq with a single
command (``python scripts/regenerate_synthesis_tables.py``), so a
QROAMClean upstream change is mechanically detectable and the dicts
can be updated by paste-back rather than hand-edit. This guards
GOALS.md "Improve constant factors in quantum algorithms" / "Any
potential improvements on the final Toffoli complexity/qubit
counts/scaling" by removing a manual maintenance gap from the
production-side estimator infrastructure.

**Next recommended task:** Derive ``I_1(n_blocks, n_rows)`` in closed
form from QROAMClean's optimizer expression. The script added this
cycle gives a one-command regression check for any candidate closed
form against the bloq's ``QECGatesCost`` output. Concretely, the
candidate form should reproduce the empirically observed
``I_1(M, N) ~ c * sqrt(M*N)`` scaling (consistent with cycle 15's
``alpha_T ≈ 0.275`` at ``N=256``), the ``n_blocks``/``n_rows``
monotonicity invariants (cycles 10, 14), and the b-independent
positivity for ``n_blocks >= 2`` (cycle 7).
=== claude cycle ended: Sat May 16 cycle 16 ===
=== claude cycle ended: Sat May 16 09:53:12 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:54:12 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 09:54:13 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:55:13 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 09:55:15 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:56:15 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 09:56:16 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:57:16 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 09:57:18 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:58:18 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 09:58:19 PM PDT 2026 ===
=== claude cycle started: Sat May 16 09:59:19 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 09:59:21 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:00:21 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:00:22 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:01:22 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:01:24 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:02:24 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:02:26 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:03:26 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:03:27 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:04:27 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:04:29 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:05:29 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:05:31 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:06:31 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:06:32 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:07:32 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:07:34 PM PDT 2026 ===
=== claude cycle started: Sat May 16 10:08:34 PM PDT 2026 ===
=== claude cycle ended: Sat May 16 10:08:35 PM PDT 2026 ===
=== codex cycle started: Sat May 16 11:18:53 PM PDT 2026 ===
=== codex cycle ended: Sat May 16 11:18:53 PM PDT 2026 ===
=== codex cycle started: Sat May 16 11:25:46 PM PDT 2026 ===

## Cycle 17 — 2026-05-16

**Inbox:** AUTONOMOUS_INBOX.md (2026-05-16) asks for a PDF report each
time, with math equations rendered as LaTeX. This cycle's PDF report is
``docs/cycle_reports/cycle17_report.pdf``. Equations are rendered with
matplotlib mathtext, matching the cycle-16 approach without requiring an
external LaTeX installation.

**Goals already achieved:** Prior cycles have already closed the
synthesis analytic-vs-Bloq Toffoli and qubit-count guard over the
tabulated grid, added report-visible synthesis power-law fits, and added
``scripts/regenerate_synthesis_tables.py`` so the tabulated intercept
and workspace dicts can be regenerated from the Bloq.

**Task selected:** Add a reusable autonomous-cycle PDF report generator
so future cycles can satisfy the inbox PDF requirement without copying a
cycle-specific helper script. This is the safe intermediate reporting
infrastructure task before resuming the larger closed-form derivation of
``I_1(n_blocks, n_rows)`` from QROAMClean's optimizer.

**Major changes:**
- New ``scripts/autonomous_cycle_report.py`` generates compact
  three-page cycle reports with argument-driven task/change/check
  sections and rendered LaTeX-style equations:
  ``T(M,N,K,b)``, ``Q(M,N,b)``, and the sub-linear amortization claim.
- New ``tests/test_autonomous_cycle_report.py`` smoke-tests the helper
  by writing a temporary non-empty PDF.
- Generated this cycle's report at
  ``docs/cycle_reports/cycle17_report.pdf``.

**Files changed:**
- Added: ``scripts/autonomous_cycle_report.py``
- Added: ``tests/test_autonomous_cycle_report.py``
- Added: ``docs/cycle_reports/cycle17_report.pdf``
- Modified: ``AUTONOMOUS_LOG.md``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_autonomous_cycle_report.py``
  → 1/1 passed.
- ``python -m py_compile scripts/autonomous_cycle_report.py tests/test_autonomous_cycle_report.py``
  → passed.
- ``python scripts/autonomous_cycle_report.py ... --cycle 17``
  → wrote ``docs/cycle_reports/cycle17_report.pdf``.

**Commit:** Attempted ``git add`` / commit, but staging was blocked by
Git metadata write failure:
``fatal: Unable to create '.git/index.lock': Read-only file system``.
Working tree changes were left intact.

**Achieved goal:** Improved report-generation infrastructure for final
Toffoli complexity / qubit count / scaling work by making per-cycle PDF
delivery reusable and math-rendered, supporting GOALS.md "Any potential
improvements on the final Toffoli complexity/qubit counts/scaling" and
the new inbox requirement.

**Next recommended task:** Resume deriving
``I_1(n_blocks, n_rows)`` in closed form from QROAMClean optimizer
behavior, using ``scripts/regenerate_synthesis_tables.py`` as the
regression oracle for any candidate formula.
=== codex cycle ended: Sat May 16 11:28:49 PM PDT 2026 ===
=== codex cycle started: Sat May 16 11:30:22 PM PDT 2026 ===

## Cycle 18 — 2026-05-16

**Inbox:** AUTONOMOUS_INBOX.md asks for a PDF report each time, with
math equations rendered as LaTeX. This cycle's PDF report is
``docs/cycle_reports/cycle18_report.pdf``. Equations are rendered with
matplotlib mathtext.

**Goals already achieved:** Prior cycles have already closed the
synthesis analytic-vs-Bloq Toffoli and qubit-count guard over the
tabulated grid, added report-visible synthesis power-law fits, added
``scripts/regenerate_synthesis_tables.py`` to regenerate tabulated
intercept/workspace dicts from the Bloq, and added the reusable
``scripts/autonomous_cycle_report.py`` PDF generator.

**Task selected:** Extend the reusable autonomous-cycle PDF report
generator so future cycles can render task-specific LaTeX-style
equations, rather than always using only the fixed default synthesis
resource formulas. This is a small reporting-infrastructure task that
directly adapts to the inbox instruction while supporting GOALS.md
"Any potential improvements on the final Toffoli complexity/qubit
counts/scaling".

**Major changes:**
- ``scripts/autonomous_cycle_report.py`` now defines reusable
  ``DEFAULT_EQUATIONS`` and accepts repeatable ``--equation`` CLI
  arguments.
- If no custom equations are supplied, the helper preserves the cycle-17
  default synthesis equations. If custom equations are supplied, the math
  page renders those task-specific equations.
- ``tests/test_autonomous_cycle_report.py`` now smoke-tests custom
  equations by generating a temporary PDF with two supplied mathtext
  formulas.
- Generated this cycle's report at
  ``docs/cycle_reports/cycle18_report.pdf``.

**Files changed:**
- Modified: ``scripts/autonomous_cycle_report.py``
- Modified: ``tests/test_autonomous_cycle_report.py``
- Added: ``docs/cycle_reports/cycle18_report.pdf``
- Modified: ``AUTONOMOUS_LOG.md``

**Tests/checks run:**
- ``PYTHONPATH=src python tests/test_autonomous_cycle_report.py``
  → 1/1 passed.
- ``python -m py_compile scripts/autonomous_cycle_report.py tests/test_autonomous_cycle_report.py``
  → passed.
- ``python scripts/autonomous_cycle_report.py ... --cycle 18``
  → wrote ``docs/cycle_reports/cycle18_report.pdf``.

**Commit:** Attempted ``git add`` / commit, but staging was blocked by
Git metadata write failure:
``fatal: Unable to create '/resnick/home/jchen9/qc_exciton_LCC/.git/index.lock': Read-only file system``.
Working tree changes were left intact.

**Achieved goal:** Reusable PDF reports now support task-specific
rendered equations, making each autonomous cycle's math deliverable
more directly tied to the selected Toffoli/qubit/scaling task while
satisfying the inbox PDF requirement.

**Next recommended task:** Resume deriving
``I_1(n_blocks, n_rows)`` in closed form from QROAMClean optimizer
behavior, using ``scripts/regenerate_synthesis_tables.py`` as the
regression oracle for any candidate formula.
