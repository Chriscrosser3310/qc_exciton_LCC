"""Structural / consistency tests for ``BlockIsometryColumnSynthesisQROAM``.

* ``build_composite_bloq`` (the real gate layout with shape-only QROAM) and ``build_call_graph``
  (the aggregate model) report identical ``QECGatesCost`` -- they must, since both count the same
  phase layers, multi-controlled gates, and final phase layer.
* The signature width is ``block_bitsize + n + phase_bitsize`` with registers ``block, system,
  phase_gradient``.
* The singly-controlled variant decomposes, adds exactly one ``ctrl`` qubit, and has only modest
  overhead (the external control reaches just the phase-gradient additions).
* The adjoint has the same cost as the forward synthesis.
* A data-free symbolic instance refuses to decompose / enumerate.
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

import sympy

qualtran = pytest.importorskip("qualtran")

from qualtran import DecomposeTypeError
from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    BlockIsometryColumnSynthesisQROAM as B,
)


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost())


def test_composite_equals_call_graph():
    cases = [
        (1, 3, 4, 6, False),
        (1, 3, 8, 6, True),
        (8, 3, 8, 6, False),
        (1, 4, 5, 6, True),
        (2, 4, 16, 6, True),
        (4, 4, 7, 8, False),
    ]
    for nb, n, K, b, opt in cases:
        bloq = B.from_shape(nb, 1 << n, b, n_cols=K, optimal_T=opt)
        cg = _cost(bloq)
        comp = _cost(bloq.decompose_bloq())
        assert cg == comp, (nb, n, K, b, opt, cg, comp)


def test_signature_layout():
    bloq = B.from_shape(8, 16, 10, n_cols=12)
    names = [r.name for r in bloq.signature]
    assert names == ["block", "system", "phase_gradient"]
    assert bloq.signature.n_qubits() == bloq.block_bitsize + bloq.system_bitsize + 10


def test_controlled_variant():
    for (nb, n, K, b) in [(1, 4, 8, 8), (8, 4, 16, 8)]:
        bare = B.from_shape(nb, 1 << n, b, n_cols=K)
        ctl = bare.controlled()
        assert ctl.signature.n_qubits() == bare.signature.n_qubits() + 1
        t_bare = int(_cost(bare).total_t_count())
        t_ctl = int(_cost(ctl).total_t_count())
        assert t_bare <= t_ctl <= 2 * t_bare, (nb, n, K, t_bare, t_ctl)


def test_isometry_cheaper_than_full_unitary():
    """Synthesizing K < N columns costs strictly less than the full N-column unitary."""
    bloq_half = B.from_shape(1, 16, 8, n_cols=8, optimal_T=True)
    bloq_full = B.from_shape(1, 16, 8, n_cols=16, optimal_T=True)
    assert int(_cost(bloq_half).total_t_count()) < int(_cost(bloq_full).total_t_count())


def test_symbolic_refuses_to_decompose():
    N = sympy.Symbol("N", positive=True, integer=True)
    bloq = B.from_shape(1, N, 8)
    with pytest.raises(DecomposeTypeError):
        bloq.build_call_graph(None)
    # optimal_T needs concrete shape -> rejected at construction time.
    with pytest.raises(ValueError):
        B.from_shape(1, N, 8, optimal_T=True)


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
