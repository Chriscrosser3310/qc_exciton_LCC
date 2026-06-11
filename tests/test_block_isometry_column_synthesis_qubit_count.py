"""Qubit-count tests for ``BlockIsometryColumnSynthesisQROAM``.

Pins:
  1. ``QubitCount`` succeeds (the bloq decomposes) across an ``(n_blocks, n_rows)`` grid,
     including ``n_blocks = 1`` where the ``block`` register is absent.
  2. The reported qubit count is at least the signature width
     ``block_bitsize + n + phase_bitsize``.
  3. ``optimal_T`` (wider QROAM batching) never uses fewer qubits than the ``Lambda = 1`` model.
  4. The QROAM workspace ``QubitCount - signature`` is monotone non-decreasing in ``n_rows``.
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

from qualtran.resource_counting import QubitCount, get_cost_value

from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    BlockIsometryColumnSynthesisQROAM as B,
)


def _qubits(bloq) -> int:
    return int(get_cost_value(bloq, QubitCount()))


def test_qubit_count_succeeds_over_grid():
    for nb in (1, 2, 4, 8):
        for n in (2, 3, 4, 5):
            for opt in (False, True):
                bloq = B.from_shape(nb, 1 << n, 8, n_cols=1 << n, optimal_T=opt)
                qc = _qubits(bloq)
                sig = bloq.signature.n_qubits()
                assert qc >= sig, (nb, n, opt, qc, sig)
                assert sig == bloq.block_bitsize + n + 8


def test_optimal_T_uses_at_least_lambda1_qubits():
    for (nb, n, K, b) in [(8, 5, 32, 10), (1, 6, 64, 10), (4, 5, 16, 8)]:
        q1 = _qubits(B.from_shape(nb, 1 << n, b, n_cols=K))
        q2 = _qubits(B.from_shape(nb, 1 << n, b, n_cols=K, optimal_T=True))
        assert q2 >= q1, (nb, n, K, b, q1, q2)


def test_workspace_monotone_in_n_rows():
    nb, b = 4, 8
    workspaces = []
    for n in (2, 3, 4, 5, 6, 7):
        bloq = B.from_shape(nb, 1 << n, b, n_cols=1 << n, optimal_T=True)
        workspaces.append(_qubits(bloq) - bloq.signature.n_qubits())
    for prev, nxt in zip(workspaces, workspaces[1:]):
        assert nxt >= prev, workspaces


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
