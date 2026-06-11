"""Tests for the ``outer_synthesis`` option (column-by-column vs reflection) for the
rectangular X / B tensors in the direct and exchange Coulomb block encodings.

* :class:`ColumnIsometryRectangularBlockEncoding` is a drop-in for
  :class:`ReflectionRectangularBlockEncoding`: identical (system, ancilla=1, resource,
  alpha=1, epsilon, signature) interface, only the internal synthesis differs.
* ``DirectCoulombBlockEncoding`` / ``ExchangeCoulombBlockEncoding`` accept
  ``outer_synthesis="column"``; this swaps only the B tensors (B_mu/B_lam, B_up/B_down),
  leaving the block-encoding interface (alpha, ancilla_bitsize, system_bitsize,
  signature) and the central kernel unchanged, while changing the Toffoli/qubit cost.
* The default remains ``"reflection"`` (no behavior change), bad values raise, and the
  controlled variant still works with the column option.
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

from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    ColumnIsometryRectangularBlockEncoding,
)
from integrations.qualtran.direct_Coulomb_block_encoding import DirectCoulombBlockEncoding
from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
from integrations.qualtran.rectangular_block_encoding_reflection import (
    ReflectionRectangularBlockEncoding,
)


def _t(bloq) -> int:
    return int(get_cost_value(bloq, QECGatesCost()).total_t_count())


def _q(bloq) -> int:
    return int(get_cost_value(bloq, QubitCount()))


# --------------------------------------------------------------------------
# The rectangular isometry block encoding itself
# --------------------------------------------------------------------------


def test_column_rect_be_interface_parity_with_reflection():
    for (nb, nrows, nrefl, b) in [(1, 16, 8, 16), (8, 32, 26, 16), (64, 64, 32, 32)]:
        col = ColumnIsometryRectangularBlockEncoding(
            n_blocks=nb, n_rows=nrows, phase_bitsize=b, n_reflections=nrefl, optimal_T=True
        )
        refl = ReflectionRectangularBlockEncoding(
            n_blocks=nb, n_rows=nrows, phase_bitsize=b, n_reflections=nrefl, optimal_T=True
        )
        assert col.system_bitsize == refl.system_bitsize
        assert col.ancilla_bitsize == refl.ancilla_bitsize == 1
        assert col.resource_bitsize == refl.resource_bitsize
        assert col.alpha == refl.alpha == 1.0
        assert col.epsilon == refl.epsilon
        assert [r.name for r in col.signature] == [r.name for r in refl.signature]


def test_column_rect_be_costs_finite_and_controlled():
    col = ColumnIsometryRectangularBlockEncoding(
        n_blocks=8, n_rows=32, phase_bitsize=16, n_reflections=26, optimal_T=True
    )
    assert _t(col) > 0
    assert _q(col) >= col.signature.n_qubits()
    # controlled variant decomposes and costs at least the bare version
    assert _t(col.controlled()) >= _t(col)


# --------------------------------------------------------------------------
# Direct / exchange wiring
# --------------------------------------------------------------------------

_PARAMS = dict(N_up=4, N_down=22, N_IP=26, N_k=8, phase_bitsize=16)


def _be(cls, **extra):
    return cls(**_PARAMS, **extra)


def test_default_outer_synthesis_is_reflection():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        be = _be(cls)
        assert be.outer_synthesis == "reflection"
        b_tensor = be.B_up if cls is ExchangeCoulombBlockEncoding else be.B_mu
        assert isinstance(b_tensor, ReflectionRectangularBlockEncoding)


def test_column_option_swaps_only_b_tensors_and_preserves_interface():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        refl = _be(cls, optimal_T=True)
        col = _be(cls, optimal_T=True, outer_synthesis="column")
        # block-encoding interface invariants are identical
        assert col.alpha == refl.alpha
        assert col.ancilla_bitsize == refl.ancilla_bitsize
        assert col.system_bitsize == refl.system_bitsize
        assert col.resource_bitsize == refl.resource_bitsize
        assert [r.name for r in col.signature] == [r.name for r in refl.signature]
        # B tensors swapped to the column synthesizer
        if cls is ExchangeCoulombBlockEncoding:
            assert isinstance(col.B_up, ColumnIsometryRectangularBlockEncoding)
            assert isinstance(col.B_down, ColumnIsometryRectangularBlockEncoding)
            # central kernel untouched by the outer-synthesis choice
            assert col.C_inner == refl.C_inner
        else:
            assert isinstance(col.B_mu, ColumnIsometryRectangularBlockEncoding)
            assert isinstance(col.B_lam, ColumnIsometryRectangularBlockEncoding)
            assert col.C_diag == refl.C_diag


def test_column_option_changes_cost():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        refl = _be(cls, optimal_T=True)
        col = _be(cls, optimal_T=True, outer_synthesis="column")
        t_refl, t_col = _t(refl), _t(col)
        assert t_col > 0 and t_refl > 0
        assert t_col != t_refl, (cls.__name__, t_col, t_refl)


def test_optimal_T_with_column_not_worse():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        plain = _be(cls, outer_synthesis="column")
        opt = _be(cls, outer_synthesis="column", optimal_T=True)
        assert _t(opt) <= _t(plain), cls.__name__


def test_controlled_with_column_option():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        col = _be(cls, optimal_T=True, outer_synthesis="column")
        ctl = col.controlled()
        assert ctl.signature.n_qubits() == col.signature.n_qubits() + 1
        assert _t(ctl) > 0


def test_bad_outer_synthesis_raises():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        with pytest.raises(ValueError):
            _be(cls, outer_synthesis="bogus")


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
