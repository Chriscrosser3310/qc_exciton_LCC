"""Tests for the ``hermitian_central`` option on ``ExchangeCoulombBlockEncoding``.

When ``hermitian_central=True`` the central W^q tensor is wrapped in
:class:`DirectHermitianBlockEncoding` (W = (H (x) I) S (H (x) I), S = [[0,U],[U^dag,0]]),
so the central *unitary* becomes Hermitian (W = W^dag) while block-encoding the SAME
Hermitian Coulomb kernel A'_Q.  These tests verify:

* the central encoder is wrapped (and only the central piece changes);
* the encoded matrix / subnormalization (alpha) and the system register are unchanged;
* exactly one extra ancilla qubit is added (the Hermitian flag), for every central variant;
* the central Toffoli cost roughly doubles (two controlled calls to the base encoding);
* the controlled variant and all (SVD / reflection / Frobenius) central choices compose.
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

from qualtran.resource_counting import get_cost_value, QubitCount

from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    DirectHermitianBlockEncoding,
)
from integrations.qualtran.exchange_Coulomb_block_encoding import (
    ExchangeCoulombBlockEncoding,
)
from integrations.qualtran.utils import get_Toffoli_counts


def _toffoli(bloq) -> int:
    # Project convention: Toffoli/CCZ count incl. rotation synthesis (utils.get_Toffoli_counts).
    return int(get_Toffoli_counts(bloq))


def _mk(hermitian_central, N_k=8, optimal_T=True, **extra):
    return ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=64, N_k=N_k, phase_bitsize=32,
        optimal_T=optimal_T, hermitian_central=hermitian_central, **extra,
    )


def test_central_is_hermitian_wrapped():
    base = _mk(hermitian_central=False)
    herm = _mk(hermitian_central=True)
    assert not isinstance(base.C_inner, DirectHermitianBlockEncoding)
    assert isinstance(herm.C_inner, DirectHermitianBlockEncoding)
    # the wrapped inner is exactly the base central encoding (same matrix)
    assert herm.C_inner.inner == base.C_inner


def test_same_matrix_alpha_and_system_unchanged():
    base = _mk(hermitian_central=False)
    herm = _mk(hermitian_central=True)
    assert herm.alpha == base.alpha            # SAME encoded matrix -> SAME subnormalization
    assert herm.system_bitsize == base.system_bitsize


def test_exactly_one_extra_ancilla_all_variants():
    for extra in ({}, {"central_via_reflection": True}, {"use_fro_BE": True}):
        base = _mk(hermitian_central=False, **extra)
        herm = _mk(hermitian_central=True, **extra)
        assert herm.ancilla_bitsize == base.ancilla_bitsize + 1
        assert isinstance(herm.C_inner, DirectHermitianBlockEncoding)


def test_central_cost_roughly_doubles():
    base = _mk(hermitian_central=False)
    herm = _mk(hermitian_central=True)
    cb = _toffoli(base.C_inner)
    ch = _toffoli(herm.C_inner)
    assert 1.9 * cb <= ch <= 2.6 * cb          # 2x base + cheap controlled/Clifford overhead


def test_outer_pieces_unchanged():
    base = _mk(hermitian_central=False)
    herm = _mk(hermitian_central=True)
    assert herm.B_up == base.B_up and herm.B_down == base.B_down
    assert herm.mod_neg == base.mod_neg and herm.mod_sub == base.mod_sub


def test_controlled_and_counts_finite():
    for N_k in (1, 8, 27):
        herm = _mk(hermitian_central=True, N_k=N_k)
        assert _toffoli(herm) > 0
        assert int(get_cost_value(herm, QubitCount())) > 0
        cbe = herm.controlled()
        assert "ctrl" in [r.name for r in cbe.signature]
        assert _toffoli(cbe) > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All exchange hermitian_central tests passed.")
