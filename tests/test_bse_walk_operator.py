"""Tests for ``BSEWalkOperator`` -- the qubitization walk ``W = (2 Pi - I) U_A`` for BSE.

Verifies that the walk operator wraps the *Hermitian-unitary* BSE block encoding ``U_A``
(``BSEBlockEncoding`` in ``hermitian`` mode), that its call graph is exactly ``{U_A, reflect}``
(mirroring Qualtran's ``QubitizationWalkOperator``), that the reflection is a cheap reflection
about ``|0>`` on the full ancilla register, that the exchange term becomes Hermitian-unitary via
the Frobenius-norm central while the other terms are untouched, and that the controlled walk
(reflection-only control) composes.  All counts are data-free (Qualtran resource walk).
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

from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare

from integrations.qualtran.bse_block_encoding import BSEBlockEncoding
from integrations.qualtran.bse_walk_operator import BSEWalkOperator, _ControlledBSEWalkOperator
from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
    DirectHermitianBlockEncoding,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

M, N_O, N_V, N_IP, N_K, B = 3, 4, 22, 208, 216, 32


def _mk(**kw):
    base = dict(m=M, N_o=N_O, N_v=N_V, N_IP=N_IP, N_k=N_K, phase_bitsize=B)
    base.update(kw)
    return BSEWalkOperator(**base)


def test_interface_matches_inner():
    w = _mk(optimal_T=True)
    ua = w.block_encoding
    assert isinstance(ua, BSEBlockEncoding) and ua.hermitian is True
    # walk acts in place on the same registers as U_A
    assert {r.name for r in w.signature} == {"system", "ancilla", "resource"}
    assert w.system_bitsize == ua.system_bitsize
    assert w.ancilla_bitsize == ua.ancilla_bitsize
    assert w.resource_bitsize == ua.resource_bitsize
    # Lambda = sum_i |c_i| alpha_i (data-free: the inner block encoding's alpha)
    assert w.Lambda == ua.alpha


def test_call_graph_is_be_plus_reflection():
    w = _mk(optimal_T=True)
    cg = w.build_call_graph(None)
    assert cg == {w.block_encoding: 1, w.reflect: 1}
    assert isinstance(w.reflect, ReflectionUsingPrepare)
    # reflection acts on (a register the size of) the full ancilla, which carries the LCU bits
    assert w.reflect.signature.n_qubits() == w.ancilla_bitsize


def test_hermitian_terms_only_exchange_changes():
    w = _mk(optimal_T=True)
    ua = w.block_encoding
    plain = BSEBlockEncoding(m=M, N_o=N_O, N_v=N_V, N_IP=N_IP, N_k=N_K, phase_bitsize=B,
                             optimal_T=True)
    # exchange term becomes Hermitian-unitary via the Frobenius-norm central
    assert ua.exchange.hermitian_fro_central is True
    assert isinstance(ua.exchange.C_inner, DirectHermitianBlockEncoding)
    assert isinstance(ua.exchange.C_inner.inner, BlockDiagonalClassicalMatrixBlockEncoding)
    # the other (already-involutive) terms are unchanged from the plain encoding
    assert ua.fock_occ == plain.fock_occ and ua.fock_virt == plain.fock_virt
    assert ua.direct == plain.direct and ua.central_W == plain.central_W
    # one extra (Hermitian-flag) ancilla in the exchange term only
    assert ua.exchange.ancilla_bitsize == plain.exchange.ancilla_bitsize + 1


def test_walk_costs_finite_and_dominated_by_ua():
    for opt in (True, False):
        w = _mk(optimal_T=opt)
        tw = get_Toffoli_counts(w)
        tua = get_Toffoli_counts(w.block_encoding)
        tref = get_Toffoli_counts(w.reflect)
        assert tw > 0 and get_qubit_counts(w) > 0
        # W = reflect . U_A ; the reflection is a cheap O(ancilla) tail next to U_A
        assert tw == tua + tref
        assert tref < tua


def test_controlled_walk_reflection_only():
    w = _mk(optimal_T=True)
    cw = w.controlled()
    assert isinstance(cw, _ControlledBSEWalkOperator)
    assert "ctrl" in [r.name for r in cw.signature]
    t_unc, t_ctrl = get_Toffoli_counts(w), get_Toffoli_counts(cw)
    # control reaches only the (cheap) reflection -> negligible overhead, <= 1 extra qubit-ish
    assert t_ctrl >= t_unc
    assert (t_ctrl - t_unc) < 0.001 * t_unc
    assert get_qubit_counts(cw) <= get_qubit_counts(w) + 2


def test_m1_corner():
    # m = 1: no antisymmetrizer in U_A, but the walk still assembles.
    w = _mk(m=1, optimal_T=True)
    assert get_Toffoli_counts(w) > 0
    assert w.block_encoding.hermitian is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All BSE walk-operator tests passed.")
