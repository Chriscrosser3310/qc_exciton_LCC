"""Scaling invariants for ``BlockIsometryColumnSynthesisQROAM`` Toffoli/T costs.

Pins the structural identities of the column-by-column construction:

* The per-step phase-layer cost is exactly linear in the number of synthesized columns ``K``
  (each column applies the same multiplexed-gate cascade).
* The total cost is monotone non-decreasing in ``K`` (the shared final phase layer and the
  multi-controlled gates make it slightly super/sub-linear, so only monotonicity is pinned).
* ``optimal_T`` never increases the total T cost (it picks the per-layer Toffoli optimum).
* The multi-controlled-gate term is sub-leading: its share of the cost decreases as ``n`` grows.
* The exact number of multiplexed-gate QROAM entries per column is ``n_blocks * (2^n - 2)``
  (Iten cascade with ``C^u_0 = I``).
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
    num_mcgs,
)


def _t(bloq) -> int:
    """Total T-equivalent cost (robust whether or not the QROAM emits raw T gates)."""
    return int(get_cost_value(bloq, QECGatesCost()).total_t_count())


def _phase_only_t(nb: int, n: int, b: int, K: int) -> int:
    bloq = B.from_shape(nb, 1 << n, b, n_cols=K)
    return sum(_t(bloq.phase_layer(c)) for c in bloq._control_counts()) * K


def test_phase_layers_exactly_linear_in_K():
    for nb in (1, 4):
        for n in (3, 4, 5):
            base = _phase_only_t(nb, n, 8, 1)
            for K in (2, 4, 1 << n):
                assert _phase_only_t(nb, n, 8, K) == K * base, (nb, n, K)


def test_total_monotone_in_K():
    for nb in (1, 8):
        for n in (3, 4):
            vals = [_t(B.from_shape(nb, 1 << n, 8, n_cols=K)) for K in range(1, (1 << n) + 1)]
            assert all(vals[i + 1] >= vals[i] for i in range(len(vals) - 1)), (nb, n, vals)


def test_optimal_T_never_worse():
    for (nb, n, K, b) in [(8, 5, 32, 10), (4, 6, 64, 12), (1, 6, 64, 10), (2, 5, 16, 8)]:
        t_lambda1 = _t(B.from_shape(nb, 1 << n, b, n_cols=K))
        t_opt = _t(B.from_shape(nb, 1 << n, b, n_cols=K, optimal_T=True))
        assert t_opt <= t_lambda1, (nb, n, K, b, t_opt, t_lambda1)


def test_mcg_term_subleading_in_n():
    """The multi-controlled-gate share of the cost decreases as n grows (full unitary K = 2^n).

    The Iten ``C_{n-1}(U)`` correction scales like ``O(2^m n b)`` against the ``O(2^{m+n})``
    multiplexed-gate cost, so its share decays like ``n b / 2^n`` for ``n >= 5``.
    """
    shares = []
    for n in (5, 6, 7, 8):
        bloq = B.from_shape(1, 1 << n, 10, n_cols=1 << n)
        phase = _phase_only_t(1, n, 10, 1 << n)
        mcg = _t(bloq.mcg) * num_mcgs(1 << n, 1 << n)
        shares.append(mcg / phase)
    assert all(shares[i + 1] < shares[i] for i in range(len(shares) - 1)), shares
    assert shares[-1] < shares[0], shares


def test_phase_entries_per_column_exact():
    """Sum of QROAM table entries across the per-column phase layers is n_blocks * (2^n - 2)."""
    for nb in (1, 2, 8):
        for n in (2, 3, 4, 5):
            bloq = B.from_shape(nb, 1 << n, 8, n_cols=1)
            entries = 0
            for c in bloq._control_counts():
                shape = bloq.phase_layer(c).qroam_data_shape  # (n_blocks, 2^c) or (2^c,)
                prod = 1
                for d in shape:
                    prod *= int(d)
                entries += prod
            assert entries == nb * ((1 << n) - 2), (nb, n, entries)


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
