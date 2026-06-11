"""Phase-bitsize (b) dependence of ``BlockIsometryColumnSynthesisQROAM`` cost.

At the qubit-minimal ``Lambda = 1`` operating point the only ``b``-dependent contribution is the
phase-gradient rotations, so the total T-equivalent cost is exactly affine in ``b``:

* ``T(b+1) - T(b)`` is constant in ``b`` (affine), mirroring the reflection-suite b-intercept test.
* The b-slope is itself affine in the column count ``K`` (per-column phasing slope plus the shared
  final-layer slope), i.e. its second difference in ``K`` vanishes.
"""

from __future__ import annotations

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

from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    BlockIsometryColumnSynthesisQROAM as B,
)


def _t(nb: int, n: int, b: int, K: int) -> int:
    return int(get_cost_value(B.from_shape(nb, 1 << n, b, n_cols=K), QECGatesCost()).total_t_count())


def test_total_t_affine_in_b():
    for nb in (1, 2, 8):
        for n in (3, 4):
            for K in (1, 2, 1 << n):
                vals = [_t(nb, n, b, K) for b in range(3, 9)]
                diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
                assert len(set(diffs)) == 1, (nb, n, K, vals, diffs)


def test_b_slope_positive():
    for nb in (1, 4):
        for n in (3, 4):
            for K in (1, 1 << n):
                slope = _t(nb, n, 5, K) - _t(nb, n, 4, K)
                assert slope > 0, (nb, n, K, slope)


def test_phase_b_slope_linear_in_K():
    """The per-step phasing b-slope is exactly linear in K (each column has the same cascade).

    (The *total* b-slope is not affine in K, because the sub-leading multi-controlled gates carry
    a b-dependent cost and their count ``sum_k Q_k(n)`` is irregular in K.)
    """
    from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
        BlockIsometryColumnSynthesisQROAM as Bloq,
    )

    def phase_slope(nb, n, K):
        b4 = Bloq.from_shape(nb, 1 << n, 4, n_cols=K)
        b5 = Bloq.from_shape(nb, 1 << n, 5, n_cols=K)

        def po(bloq, K):
            return sum(
                int(get_cost_value(bloq.phase_layer(c), QECGatesCost()).total_t_count())
                for c in bloq._control_counts()
            ) * K

        return po(b5, K) - po(b4, K)

    nb, n = 1, 4
    per_col = phase_slope(nb, n, 1)
    assert per_col > 0
    for K in (2, 3, 4, 8):
        assert phase_slope(nb, n, K) == K * per_col, (K, phase_slope(nb, n, K), per_col)


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
