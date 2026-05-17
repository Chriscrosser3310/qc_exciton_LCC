"""Per-reflection ``b=0`` intercept table for ``BlockUnitarySynthesisQROAM``.

Cycles 1-4 pinned three structural identities of the synthesis bloq's
Toffoli cost: K-linearity, b-affineness, and a per-reflection b-slope of
exactly ``2*(log2(N) + 1)``. Together those identities imply the
decomposition

    T(n_blocks, N, K, b) = K * ( 2*(log2(N) + 1) * b + I_1(n_blocks, N) )

where ``I_1(n_blocks, N)`` is the per-reflection ``b=0`` intercept and
captures the QROAMClean + ``QROAMCleanAdjoint`` data-loading cost plus
the reflection-about-zero / Hadamard overhead of a single Householder
reflection. ``I_1`` is what an analytic estimator in
``model_resource_counts.py`` needs to predict — the b-slope is already
known from the existing scaling tests.

This file:

* pins ``I_1(n_blocks, N)`` for a small reference grid (the ``REFERENCE``
  table below) as a regression guard, analogous to
  ``PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208`` for the
  interferometer model;
* verifies that ``I_1`` is K-independent and b-independent (i.e. that
  the decomposition above is exact across the grid);
* verifies that for the non-degenerate multi-block regime
  (``n_blocks >= 2``) the intercept is strictly positive — a sanity
  check that the QROAM contribution dominates the per-reflection
  overhead, which is the assumption under which the
  ``test_block_unitary_synthesis_amortization`` claims are meaningful.

This complements the existing
``tests/test_block_unitary_synthesis_*`` files and forms the final
Bloq-side anchor for a future ``synthesis_count`` analytic estimator.
"""

from __future__ import annotations

import math
import sys


try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for envs without pytest

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__}, got nothing")
            return issubclass(exc_type, self.exc)

    class _PytestShim:
        @staticmethod
        def raises(exc):
            return _Raises(exc)

        @staticmethod
        def importorskip(name):
            return __import__(name)

    pytest = _PytestShim()  # type: ignore[assignment]
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

qualtran = pytest.importorskip("qualtran")
_ = qualtran

from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_unitary_synthesis_QROAM import (
    BlockUnitarySynthesisQROAM,
)


# Reference per-reflection b=0 intercepts I_1(n_blocks, N).
# Computed once against the current Qualtran QROAMClean optimizer; any
# upstream change that perturbs these by more than a constant must be
# investigated rather than silently absorbed into the report numbers.
# Keyed by (n_blocks, N).
REFERENCE: dict[tuple[int, int], int] = {
    (1, 4): -6, (2, 4): 4,  (4, 4): 12, (8, 4): 32, (16, 4): 48,
    (1, 8): -2, (2, 8): 16, (4, 8): 28, (8, 8): 64, (16, 8): 88,
    (1, 16): 6, (2, 16): 32, (4, 16): 52, (8, 16): 104, (16, 16): 144,
}


def _toffoli(n_blocks: int, N: int, b: int, K: int) -> int:
    bloq = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=n_blocks, n_rows=N, phase_bitsize=b, n_reflections=K
    )
    return get_cost_value(bloq, QECGatesCost()).toffoli


def _intercept(n_blocks: int, N: int, *, K: int = 1, b_ref: int = 4) -> int:
    """Extract I_1 by subtracting K * slope * b_ref from T(n_blocks, N, K, b_ref).

    Slope is known to be exactly ``2*(log2(N)+1)`` per the scaling test
    ``test_toffoli_b_slope_equals_2_n_plus_1``. Returns the per-reflection
    intercept (divides out K).
    """
    n = int(math.log2(N))
    slope_per_reflection = 2 * (n + 1)
    t = _toffoli(n_blocks, N, b_ref, K)
    return (t - K * slope_per_reflection * b_ref) // K


def test_reference_intercept_table_matches():
    """Pin per-reflection ``I_1(n_blocks, N)`` for the reference grid."""
    for (n_blocks, N), expected in REFERENCE.items():
        got = _intercept(n_blocks, N)
        assert got == expected, (n_blocks, N, got, expected)


def test_intercept_is_b_independent():
    """``I_1`` extracted at different ``b`` agrees exactly (decomposition is affine)."""
    for (n_blocks, N) in REFERENCE:
        ref = _intercept(n_blocks, N, b_ref=4)
        for b in (2, 3, 5, 7, 11):
            got = _intercept(n_blocks, N, b_ref=b)
            assert got == ref, (n_blocks, N, b, got, ref)


def test_intercept_is_K_independent():
    """``I_1`` extracted at different ``K`` agrees exactly (decomposition is K-linear)."""
    for (n_blocks, N) in REFERENCE:
        ref = _intercept(n_blocks, N, K=1)
        for K in (2, max(1, N // 2), N):
            got = _intercept(n_blocks, N, K=K)
            assert got == ref, (n_blocks, N, K, got, ref)


def test_full_decomposition_holds():
    """``T = K * (2*(n+1)*b + I_1(n_blocks, N))`` exactly across the grid."""
    for (n_blocks, N), I_1 in REFERENCE.items():
        n = int(math.log2(N))
        slope = 2 * (n + 1)
        for b in (2, 4, 6, 8, 12):
            for K in (1, 2, max(1, N // 2), N):
                expected = K * (slope * b + I_1)
                got = _toffoli(n_blocks, N, b, K)
                assert got == expected, (n_blocks, N, K, b, got, expected)


def test_intercept_positive_for_multiblock():
    """For ``n_blocks >= 2`` the per-reflection intercept is strictly positive.

    The amortization test file assumes the QROAM contribution dominates
    the per-reflection Hadamard / reflection-about-zero overhead. A
    negative intercept would mean the slope-times-b extrapolation
    *exceeds* the actual cost at b=0, which only happens in the
    degenerate single-block regime where there is no real QROAM table to
    load. Pinning positivity at ``n_blocks >= 2`` keeps that assumption
    explicit.
    """
    for (n_blocks, N), I_1 in REFERENCE.items():
        if n_blocks >= 2:
            assert I_1 > 0, (n_blocks, N, I_1)


if __name__ == "__main__":
    failed = 0
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
