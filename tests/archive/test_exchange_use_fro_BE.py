"""Tests for the ``use_fro_BE`` option on ``ExchangeCoulombBlockEncoding``.

When ``use_fro_BE=True`` the central W^q tensor (just W, not the X matrices) is encoded
by the Clader-Frobenius block-matrix scheme
(:class:`BlockDiagonalClassicalMatrixBlockEncoding`) instead of the default SVD
interferometer ("unitary synthesis").  These tests verify:

* the central encoder swaps to the Frobenius block-matrix BE (and only that piece changes);
* the outer X reflection block encodings and momentum bookkeeping are unchanged;
* the controlled variant promotes only the central Frobenius BE (negligible overhead);
* counts are finite/positive across N_k including the single-block (N_k = 1) edge case.
"""

from __future__ import annotations

import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for envs without pytest

    class _PytestShim:
        @staticmethod
        def importorskip(name):
            return __import__(name)

    pytest = _PytestShim()  # type: ignore[assignment]
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

qualtran = pytest.importorskip("qualtran")

from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
)
from integrations.qualtran.exchange_Coulomb_block_encoding import (
    ExchangeCoulombBlockEncoding,
)
from integrations.qualtran.svd_block_encoding_interferometer import (
    SVDBlockEncodingInterferometer,
)


def _toffoli(bloq) -> int:
    return int(get_cost_value(bloq, QECGatesCost()).toffoli)


def _qubits(bloq) -> int:
    return int(get_cost_value(bloq, QubitCount()))


def _mk(use_fro_BE, N_k=27, optimal_T=True):
    return ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=256, N_k=N_k, phase_bitsize=32,
        optimal_T=optimal_T, use_fro_BE=use_fro_BE,
    )


def test_central_encoder_swaps():
    us = _mk(use_fro_BE=False)
    fro = _mk(use_fro_BE=True)
    assert isinstance(us.C_inner, SVDBlockEncodingInterferometer)
    assert isinstance(fro.C_inner, BlockDiagonalClassicalMatrixBlockEncoding)
    # The central Frobenius BE is block-indexed by N_k over the padded N_IP register.
    assert fro.C_inner.n_blocks == fro.N_k
    assert fro.C_inner.n_rows == fro.n_rows_inner


def test_outer_reflections_unchanged():
    us = _mk(use_fro_BE=False)
    fro = _mk(use_fro_BE=True)
    # Only the central piece differs; the X reflections and momentum ops are identical.
    assert us.B_up == fro.B_up
    assert us.B_down == fro.B_down
    assert us.mod_neg == fro.mod_neg
    assert us.mod_sub == fro.mod_sub
    cg = fro.build_call_graph(None)
    assert cg[fro.B_up] == 2
    assert cg[fro.B_down] == 2
    assert cg[fro.C_inner] == 1


def test_controlled_overhead_negligible():
    fro = _mk(use_fro_BE=True)
    cfro = fro.controlled()
    t_unc, t_ctrl = _toffoli(fro), _toffoli(cfro)
    assert t_ctrl >= t_unc
    assert (t_ctrl - t_unc) < 0.01 * t_unc
    # Controlling only the central Frobenius BE adds the control line plus a small
    # constant of bookkeeping ancilla -- still negligible vs the thousands of qubits.
    assert _qubits(cfro) <= _qubits(fro) + 4


def test_counts_finite_across_N_k():
    for N_k in (1, 8, 27, 216):
        for opt in (False, True):
            b = _mk(use_fro_BE=True, N_k=N_k, optimal_T=opt)
            assert _toffoli(b) > 0
            assert _qubits(b) > 0


def test_optimal_T_lowers_toffoli():
    # Blocking the Frobenius QROAM should cut Toffolis well below the un-blocked default.
    default = _mk(use_fro_BE=True, N_k=216, optimal_T=False)
    opt = _mk(use_fro_BE=True, N_k=216, optimal_T=True)
    assert _toffoli(opt) < _toffoli(default) / 10


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All exchange use_fro_BE tests passed.")
