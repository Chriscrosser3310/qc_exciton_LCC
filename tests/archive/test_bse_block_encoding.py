"""Tests for ``BSEBlockEncoding`` -- the data-free BSE block encoding <psi|S P A D P S|psi>.

Verifies the register layout (2m two-component registers of N_k x (N_IP+1)), the
BlockEncoding interface, the presence of every named structural piece in the call graph
(the five A terms incl. the combined term's shared central W; the particle-number
counters; the particle-number-controlled antisymmetrizers; the diagonal D; the |psi>
prepare), the m=1 corner (no antisymmetrizer), the controlled variant, and validation.
All counts are data-free (Qualtran resource walk); no real data is populated.
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

from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
from qualtran.symbolics import bit_length

from integrations.qualtran.antisymmetric_projector_block_encoding import (
    AntisymmetricProjectorBlockEncoding,
)
from integrations.qualtran.bse_block_encoding import BSEBlockEncoding, _ControlledBSEBlockEncoding
from integrations.qualtran.direct_Coulomb_block_encoding import (
    DiagonalCoulombKernelBlockEncoding,
    DirectCoulombBlockEncoding,
)
from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
from integrations.qualtran.fock_block_encoding import FockBlockEncoding
from integrations.qualtran.particle_number_counter import ParticleNumberCounter
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

M, N_O, N_V, N_IP, N_K, B = 3, 4, 22, 208, 216, 32


def _mk(**kw):
    base = dict(m=M, N_o=N_O, N_v=N_V, N_IP=N_IP, N_k=N_K, phase_bitsize=B)
    base.update(kw)
    return BSEBlockEncoding(**base)


def _count(cg, cls):
    return sum(c for b, c in cg.items() if isinstance(b, cls))


def test_register_layout():
    b = _mk(optimal_T=True)
    # each register = momentum (N_k) + orbital (N_IP+1); 2m registers.
    assert b.reg_bitsize == bit_length(N_K - 1) + bit_length(N_IP)  # 8 + 8
    assert b.n_registers == 2 * M
    assert b.system_bitsize == 2 * M * b.reg_bitsize
    assert b.orbital_dim == N_IP + 1
    assert b.pn_bitsize == bit_length(M)


def test_interface():
    b = _mk(optimal_T=True)
    assert {r.name for r in b.signature} == {"system", "ancilla", "resource"}
    assert b.resource_bitsize == B
    assert abs(b.epsilon - 2.0 ** -B) < 1e-18
    # alpha = LCU sum of the five term alphas.
    assert b.alpha == float(
        b.fock_occ.alpha + b.fock_virt.alpha + b.exchange.alpha
        + b.direct.alpha + b.combined_term_alpha
    )


def test_call_graph_has_all_pieces():
    b = _mk(optimal_T=True)
    cg = b.build_call_graph(None)
    # the five A terms (uncontrolled in the base call graph)
    assert _count(cg, FockBlockEncoding) == 2          # occupied + virtual
    assert _count(cg, ExchangeCoulombBlockEncoding) == 1
    assert _count(cg, DirectCoulombBlockEncoding) == 1
    assert _count(cg, DiagonalCoulombKernelBlockEncoding) == 1   # combined-term shared W
    # particle-number counters: one per partition, each on both sandwich sides.
    assert _count(cg, ParticleNumberCounter) == 4
    # diagonal D forward QROAM present.
    assert _count(cg, QROAMClean) >= 1


def test_antisymmetrizers_scale_with_m():
    # m = 1: nothing to antisymmetrize.
    cg1 = _mk(m=1, optimal_T=True).build_call_graph(None)
    assert all("Antisym" not in type(b).__name__ and
               "Antisym" not in type(getattr(b, "subbloq", b)).__name__ for b in cg1)
    # m >= 2: controlled antisymmetrizers appear for k = 2..m (occupied and virtual).
    b4 = _mk(m=4, optimal_T=True)
    cg4 = b4.build_call_graph(None)
    anti = sum(c for bl, c in cg4.items()
               if isinstance(getattr(bl, "subbloq", None), AntisymmetricProjectorBlockEncoding))
    assert anti > 0
    # antisymmetrizer subsystem size matches one full system register.
    a = b4.antisymmetrizer(3, virtual=False)
    assert a.subsystem_bitsize == int(b4.reg_bitsize)


def test_controlled_variant():
    b = _mk(optimal_T=True)
    cb = b.controlled()
    assert isinstance(cb, _ControlledBSEBlockEncoding)
    t_unc, t_ctrl = get_Toffoli_counts(b), get_Toffoli_counts(cb)
    assert t_ctrl >= t_unc
    # control only touches the operator-defining pieces -> modest overhead.
    assert (t_ctrl - t_unc) < 0.05 * t_unc
    assert get_qubit_counts(cb) <= get_qubit_counts(b) + 2


def test_counts_finite():
    for opt in (True, False):
        b = _mk(optimal_T=opt)
        assert get_Toffoli_counts(b) > 0
        assert get_qubit_counts(b) > 0


def test_validation():
    with pytest.raises(ValueError):
        _mk(N_o=300)          # N_o > N_IP
    with pytest.raises(ValueError):
        _mk(N_v=300)          # N_v > N_IP
    with pytest.raises(ValueError):
        _mk(m=0)              # m < 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All BSE block-encoding tests passed.")
