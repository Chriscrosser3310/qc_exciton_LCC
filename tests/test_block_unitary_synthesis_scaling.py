"""Scaling-invariants for ``BlockUnitarySynthesisQROAM`` Toffoli costs.

These tests pin structural identities that hold for the Bloq-side Toffoli
count ``QECGatesCost`` of ``BlockUnitarySynthesisQROAM``:

* The total Toffoli count is exactly linear in the number of reflections
  ``K = n_reflections`` — each reflection has the same call graph and they
  appear in the call graph once each.
* The total Toffoli count is affine in ``phase_bitsize = b``, with slope
  ``K * 2 * (n + 1)`` where ``n = log2(n_rows)``. The slope is independent
  of ``n_blocks``.
* The shape-only resource path
  (``BlockUnitarySynthesisQROAM.from_shape(...)``) and the data-bearing
  constructor with random unitaries agree on the Toffoli cost across the
  full ``(n_blocks, n_rows, K=n_reflections, b)`` grid considered here.

This complements ``test_block_unitary_synthesis_equivalence.py``, which
covers the single-block (``n_blocks=1``) reduction to
``UnitarySynthesisQROAM``. Together, the two test files form the
analytic-vs-Bloq guard for the synthesis bloq, analogous to
``test_estimator_matches_closed_form`` for the interferometer bloq.
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

import numpy as np

qualtran = pytest.importorskip("qualtran")
_ = qualtran

from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_unitary_synthesis_QROAM import (
    BlockUnitarySynthesisQROAM,
)


def _toffoli_from_shape(n_blocks: int, n_rows: int, b: int, K: int) -> int:
    bloq = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=n_blocks, n_rows=n_rows, phase_bitsize=b, n_reflections=K
    )
    return get_cost_value(bloq, QECGatesCost()).toffoli


def _random_unitary(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))


def test_toffoli_linear_in_n_reflections():
    """T_total(K) = K * T_total(K=1) for matched (n_blocks, n_rows, b)."""
    for n_blocks in (1, 2, 4):
        for N in (4, 8):
            for b in (4, 6):
                t1 = _toffoli_from_shape(n_blocks, N, b, K=1)
                for K in (2, N // 2, N):
                    tK = _toffoli_from_shape(n_blocks, N, b, K=K)
                    assert tK == K * t1, (n_blocks, N, b, K, tK, t1)


def test_toffoli_affine_in_phase_bitsize():
    """T(b) is exactly affine in b: T(b+1) - T(b) is constant in b."""
    for n_blocks in (1, 2, 4):
        for N in (4, 8):
            for K in (1, N):
                # Compute T for a strip of b values; differences should be constant.
                counts = [_toffoli_from_shape(n_blocks, N, b, K) for b in range(2, 8)]
                diffs = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
                assert len(set(diffs)) == 1, (n_blocks, N, K, counts, diffs)


def test_toffoli_b_slope_equals_2_n_plus_1():
    """Per-reflection b-slope is exactly ``2 * (log2(N) + 1)``, independent of n_blocks."""
    for N in (2, 4, 8, 16, 32):
        n = int(math.log2(N))
        expected = 2 * (n + 1)
        for n_blocks in (1, 2, 4, 8, 16):
            t_b4 = _toffoli_from_shape(n_blocks, N, b=4, K=1)
            t_b5 = _toffoli_from_shape(n_blocks, N, b=5, K=1)
            slope = t_b5 - t_b4
            assert slope == expected, (n_blocks, N, slope, expected)


def test_toffoli_b_slope_is_K_times_per_reflection():
    """Total-cost b-slope is exactly K * per-reflection b-slope."""
    for n_blocks in (1, 4):
        for N in (4, 8):
            n = int(math.log2(N))
            per_refl_slope = 2 * (n + 1)
            for K in (1, 2, N):
                t_b4 = _toffoli_from_shape(n_blocks, N, b=4, K=K)
                t_b5 = _toffoli_from_shape(n_blocks, N, b=5, K=K)
                assert t_b5 - t_b4 == K * per_refl_slope, (n_blocks, N, K, t_b5 - t_b4)


def test_shape_only_matches_data_bearing_multi_block():
    """For (n_blocks > 1, K ≤ N), shape-only and data-bearing Toffoli costs match.

    The existing equivalence file covers the n_blocks=1 reduction to the
    un-blocked variant; this extends the guard to genuinely multi-block
    cases and to isometry columns (K < N).
    """
    cases = [
        (2, 4, 4, 1),
        (2, 4, 6, 2),
        (3, 4, 4, 4),
        (4, 4, 6, 2),
        (2, 8, 6, 1),
        (3, 8, 4, 4),
        (4, 8, 6, 8),
    ]
    for n_blocks, N, b, K in cases:
        blocks_data = np.stack(
            [_random_unitary(N, seed=1000 + i + 7 * n_blocks)[:, :K] for i in range(n_blocks)]
        )
        data_bloq = BlockUnitarySynthesisQROAM(
            block_unitaries=blocks_data, phase_bitsize=b
        )
        t_data = get_cost_value(data_bloq, QECGatesCost()).toffoli
        t_shape = _toffoli_from_shape(n_blocks, N, b, K)
        assert t_data == t_shape, (n_blocks, N, b, K, t_data, t_shape)


def test_n_reflections_default_is_n_rows():
    """from_shape with no n_reflections uses K = N (square full unitary)."""
    bloq_default = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=2, n_rows=8, phase_bitsize=6
    )
    bloq_explicit = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=2, n_rows=8, phase_bitsize=6, n_reflections=8
    )
    t_default = get_cost_value(bloq_default, QECGatesCost()).toffoli
    t_explicit = get_cost_value(bloq_explicit, QECGatesCost()).toffoli
    assert t_default == t_explicit


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
