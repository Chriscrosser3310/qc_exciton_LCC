"""Tests for ``DirectCoulombBlockEncoding`` (direct/Coulomb THC orientation).

These verify, for the data-free direct-Coulomb block encoding of arXiv:2601.16379 Eq. (9):

* the standard ``BlockEncoding`` interface (registers, alpha, epsilon, signal state);
* call-graph structure: two ``X`` reflection block encodings (forward + adjoint), the
  ``|Q>`` uniform LCU prepare/unprepare, two momentum modular additions, and exactly one
  central diagonal Coulomb-kernel block encoding;
* the central :class:`DiagonalCoulombKernelBlockEncoding` realizes the QROAM -> R_y ->
  QROAM^dagger pattern and exposes tunable tradeoff parameters (``optimal_T`` lowers the
  Toffoli count by orders of magnitude relative to the un-blocked default);
* a single-qubit controlled variant whose Toffoli/qubit overhead over the uncontrolled
  bloq is negligible (only the diagonal kernel and the two cheap additions gain a control).
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

from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

from integrations.qualtran.direct_Coulomb_block_encoding import (
    DiagonalCoulombKernelBlockEncoding,
    DirectCoulombBlockEncoding,
    _ControlledDirectCoulombBlockEncoding,
    optimal_diag_log_block_sizes,
)
from integrations.qualtran.rectangular_block_encoding_reflection import (
    ReflectionRectangularBlockEncoding,
)


def _t_count(bloq) -> int:
    c = get_cost_value(bloq, QECGatesCost())
    return int(c.total_t_count())


def _toffoli(bloq) -> int:
    return int(get_cost_value(bloq, QECGatesCost()).toffoli)


def _qubits(bloq) -> int:
    return int(get_cost_value(bloq, QubitCount()))


# --------------------------------------------------------------------------- #
# BlockEncoding interface
# --------------------------------------------------------------------------- #


def test_block_encoding_interface():
    b = DirectCoulombBlockEncoding(N_up=4, N_down=22, N_IP=208, N_k=8, phase_bitsize=32)
    names = {r.name for r in b.signature}
    assert names == {"system", "ancilla", "resource"}
    assert b.resource_bitsize == 32
    # system = k_mu + sys_mu + k_lambda + sys_lambda + Q
    k = b.k_bitsize
    assert b.system_bitsize == k + 8 + k + 8 + k  # n_rows_up = n_rows_down = 256 -> 8 bits
    # alpha = n_up^2 * n_down^2 * N_k
    assert b.alpha == float(b.n_rows_up) ** 2 * float(b.n_rows_down) ** 2 * 8.0
    assert abs(b.epsilon - 2.0 ** -32) < 1e-18


# --------------------------------------------------------------------------- #
# Call-graph structure
# --------------------------------------------------------------------------- #


def test_call_graph_structure():
    b = DirectCoulombBlockEncoding(N_up=4, N_down=22, N_IP=208, N_k=8, phase_bitsize=32)
    cg = b.build_call_graph(None)
    # Two X reflection block encodings, each appearing twice (forward + adjoint).
    assert cg[b.B_mu] == 2
    assert cg[b.B_lam] == 2
    assert isinstance(b.B_mu, ReflectionRectangularBlockEncoding)
    # |Q> uniform LCU prepare + unprepare.
    assert cg[b.uniform_prep] == 2
    # Two momentum modular additions (k_mu += Q, k_lambda += Q).
    assert cg[b.mod_add] == 2
    # Exactly one central diagonal Coulomb kernel.
    assert cg[b.C_diag] == 1
    assert isinstance(b.C_diag, DiagonalCoulombKernelBlockEncoding)


def test_single_block_drops_momentum_bookkeeping():
    b = DirectCoulombBlockEncoding(N_up=4, N_down=22, N_IP=208, N_k=1, phase_bitsize=32)
    cg = b.build_call_graph(None)
    assert b.uniform_prep not in cg
    assert b.mod_add not in cg
    assert cg[b.C_diag] == 1
    # diagonal kernel for a single block drops the Q axis -> 2-D QROAM
    assert b.C_diag.diag_data_shape == (208, 208)


# --------------------------------------------------------------------------- #
# Diagonal Coulomb kernel
# --------------------------------------------------------------------------- #


def test_diagonal_kernel_structure():
    d = DiagonalCoulombKernelBlockEncoding(N_k=8, N_IP=208, phase_bitsize=32)
    assert d.diag_data_shape == (8, 208, 208)
    assert d.alpha == 1.0
    assert d.ancilla_bitsize == 1
    cg = d.build_call_graph(None)
    # QROAM forward + rotation + QROAM adjoint.
    assert cg[d.diag_qroam] == 1
    assert cg[d.diag_qroam_adjoint] == 1
    assert cg[d.ctrl_phase_grad_add] == 1


def test_optimal_T_lowers_diagonal_cost():
    default = DiagonalCoulombKernelBlockEncoding(N_k=216, N_IP=208, phase_bitsize=32)
    opt = DiagonalCoulombKernelBlockEncoding(
        N_k=216, N_IP=208, phase_bitsize=32, optimal_T=True
    )
    # Blocking the QROAM cuts the Toffoli count by orders of magnitude.
    assert _toffoli(opt) < _toffoli(default) / 100


def test_optimal_diag_log_block_sizes_respects_caps():
    shape = (216, 208, 208)
    fwd = optimal_diag_log_block_sizes(shape, 32, adjoint=False)
    adj = optimal_diag_log_block_sizes(shape, 32, adjoint=True)
    caps = (7, 7, 7)  # floor(log2(216))=7, floor(log2(208))=7
    assert all(0 <= f <= c for f, c in zip(fwd, caps))
    assert all(0 <= a <= c for a, c in zip(adj, caps))
    # Adjoint optimum (sqrt(M)) is at least as large as the forward (sqrt(M/b)).
    assert sum(adj) >= sum(fwd)


# --------------------------------------------------------------------------- #
# Controlled variant: negligible overhead
# --------------------------------------------------------------------------- #


def test_controlled_overhead_negligible():
    b = DirectCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=208, N_k=216, phase_bitsize=32, optimal_T=True
    )
    cb = b.controlled()
    assert isinstance(cb, _ControlledDirectCoulombBlockEncoding)
    t_unc, t_ctrl = _toffoli(b), _toffoli(cb)
    # Only the central kernel + two cheap additions gain a control.
    assert t_ctrl >= t_unc
    assert (t_ctrl - t_unc) < 0.01 * t_unc
    # No extra system/ancilla qubits beyond the single control line.
    assert _qubits(cb) <= _qubits(b) + 1


def test_resource_counts_are_finite_and_positive():
    for nk in (1, 8, 216):
        b = DirectCoulombBlockEncoding(
            N_up=4, N_down=22, N_IP=208, N_k=nk, phase_bitsize=32, optimal_T=True
        )
        assert _toffoli(b) > 0
        assert _qubits(b) > 0
        assert _t_count(b) >= _toffoli(b)  # T-count includes the Toffoli T-contribution


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All direct-Coulomb block-encoding tests passed.")
