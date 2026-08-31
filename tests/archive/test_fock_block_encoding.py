"""Tests for ``FockBlockEncoding`` (SVD-interferometer block Fock encoding).

Encodes ``sum_k |k><k| (x) F_k`` with each ``F_k`` an ``N x N`` contraction, acting on an
``N_k x N_IP`` system register (``N <= N_IP``), synthesized data-free via
``SVDBlockEncodingInterferometer`` on a ``next_pow2(N)``-dim register, with a
``LessThanConstant`` comparator cutting the input off to the first ``N`` entries.

Verified: the standard BlockEncoding interface (system = N_k x N_IP, ancilla = SVD + flag,
alpha = 1); the call graph (comparator x2 at threshold N + one SVD); the ``restrict_input``
toggle; the SVD block dimension = next_pow2(N); ``N <= N_IP`` validation; and a
single-qubit controlled variant that controls only the SVD and keeps the cutoff.
"""

from __future__ import annotations

import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover

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

from qualtran.bloqs.arithmetic import LessThanConstant
from qualtran.symbolics import bit_length

from integrations.qualtran.fock_block_encoding import (
    FockBlockEncoding,
    _ControlledFockBlockEncoding,
)
from integrations.qualtran.svd_block_encoding_interferometer import (
    SVDBlockEncodingInterferometer,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

N_K, N_IP, N, B = 216, 208, 16, 32


def _mk(**kw):
    return FockBlockEncoding(N_k=N_K, N_IP=N_IP, N=N, phase_bitsize=B, **kw)


def _comparators(cg):
    return {int(b.less_than_val): int(c) for b, c in cg.items()
            if isinstance(b, LessThanConstant)}


def test_interface():
    f = _mk(optimal_T=True)
    assert {r.name for r in f.signature} == {"system", "ancilla", "resource"}
    # system = N_k x N_IP register: momentum + N_IP-sized matrix register.
    assert f.system_bitsize == bit_length(N_K - 1) + bit_length(N_IP - 1)
    # ancilla = SVD block-encoding ancilla (1) + cutoff flag (1).
    assert f.ancilla_bitsize == f.svd.ancilla_bitsize + 1
    assert f.resource_bitsize == B
    assert f.alpha == 1.0
    assert abs(f.epsilon - 2.0 ** -B) < 1e-18


def test_call_graph():
    f = _mk(optimal_T=True)
    cg = f.build_call_graph(None)
    # comparator cuts off input < N, computed + uncomputed.
    assert _comparators(cg) == {N: 2}
    # exactly one SVD interferometer, sized to next_pow2(N).
    svds = [b for b in cg if isinstance(b, SVDBlockEncodingInterferometer)]
    assert len(svds) == 1 and cg[svds[0]] == 1
    assert svds[0].n_rows == f.n_rows_inner == 16


def test_restrict_input_toggle():
    on = _mk(optimal_T=True, restrict_input=True)
    off = _mk(optimal_T=True, restrict_input=False)
    assert on.ancilla_bitsize == off.ancilla_bitsize + 1
    assert on.system_bitsize == off.system_bitsize  # cutoff is ancilla-only
    assert _comparators(off.build_call_graph(None)) == {}
    assert get_Toffoli_counts(on) > get_Toffoli_counts(off)


def test_controlled_variant():
    f = _mk(optimal_T=True)
    cf = f.controlled()
    assert isinstance(cf, _ControlledFockBlockEncoding)
    cg = cf.build_call_graph(None)
    # cutoff comparator preserved; the SVD is the controlled piece.
    assert _comparators(cg) == {N: 2}
    assert any(isinstance(b, type(f.svd.controlled())) for b in cg)
    # control adds a qubit line; cost stays finite and at least the uncontrolled cost.
    assert get_Toffoli_counts(cf) >= get_Toffoli_counts(f)


def test_block_dimension_padding():
    for n, want in [(1, 4), (2, 4), (4, 4), (5, 8), (16, 16), (208, 256)]:
        f = FockBlockEncoding(N_k=N_K, N_IP=N_IP, N=n, phase_bitsize=B)
        assert f.n_rows_inner == want
        assert get_Toffoli_counts(f) > 0 and get_qubit_counts(f) > 0


def test_N_must_not_exceed_N_IP():
    with pytest.raises(ValueError):
        FockBlockEncoding(N_k=4, N_IP=8, N=16, phase_bitsize=B)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Fock block-encoding tests passed.")
