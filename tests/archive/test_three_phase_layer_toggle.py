"""Tests for the ``three_phase_layer_prep`` toggle across the unitary-synthesis proposals.

Each proposal that uses a QROAM state preparation
(``unitary_reflection``, ``block_unitary_reflection``, ``classical_matrix_block_encoding``,
``rectangular_block_encoding_reflection``) gains a ``three_phase_layer_prep`` flag that swaps the
standard amplitude+phase state preparation for the "three diagonal phase layers + Hadamards" ansatz
(arXiv:2409.11748 p.14, :class:`ThreePhaseLayerStatePreparation`).

For each consumer this pins:
  * the flag is OFF by default (and OFF reproduces the standard prep class),
  * ON swaps in ``ThreePhaseLayerStatePreparation`` with the same register contract,
  * the swap actually changes the synthesized cost, and
  * the decomposable level stays self-consistent (``QECGatesCost`` of the composite == call graph),
    including the controlled variants.
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

from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_state_preparation_QROAM import (
    BlockStatePreparationViaQROAMRotations,
)
from integrations.qualtran.block_unitary_reflection_QROAM import BlockUnitaryReflectionQROAM
from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
    ClassicalMatrixBlockEncoding,
)
from integrations.qualtran.rectangular_block_encoding_reflection import (
    RectangularBlockEncodingReflection,
    ReflectionRectangularBlockEncoding,
)
from integrations.qualtran.state_prep_QROAM import StatePreparationViaQROAMRotations
from integrations.qualtran.three_phase_layer_state_prep_QROAM import (
    ThreePhaseLayerStatePreparation,
)
from integrations.qualtran.unitary_reflection_QROAM import UnitaryReflectionQROAM


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost())


def _self_consistent(bloq) -> bool:
    """QECGatesCost via the call graph equals that of the one-level decomposition."""
    return _cost(bloq) == _cost(bloq.decompose_bloq())


# ---------------------------------------------------------------------------
# unitary_reflection_QROAM.UnitaryReflectionQROAM
# ---------------------------------------------------------------------------


def test_unitary_reflection_toggle():
    off = UnitaryReflectionQROAM.from_shape(16, 8, n_reflections=4)
    on = UnitaryReflectionQROAM.from_shape(16, 8, n_reflections=4, three_phase_layer_prep=True)

    # default OFF, and OFF reproduces the standard prep class.
    assert off.three_phase_layer_prep is False
    assert isinstance(off.reflection(0).prepare_w.state_prep, StatePreparationViaQROAMRotations)
    assert isinstance(on.reflection(0).prepare_w.state_prep, ThreePhaseLayerStatePreparation)

    # the swap changes the synthesized cost.
    assert _cost(off) != _cost(on)

    # decomposable (data-free) levels stay self-consistent for both settings.
    for top in (off, on):
        assert _self_consistent(top.reflection(0))  # Householder reflection
        assert _self_consistent(top.reflection(0).prepare_w)  # state-prep wrapper

    # the adjoint (un-preparation) keeps the toggle.
    assert on.reflection(0).prepare_w.adjoint().three_phase_layer_prep is True


# ---------------------------------------------------------------------------
# block_unitary_reflection_QROAM.BlockUnitaryReflectionQROAM
# ---------------------------------------------------------------------------


def test_block_unitary_reflection_toggle():
    for opt in (False, True):
        off = BlockUnitaryReflectionQROAM.from_shape(4, 16, 8, n_reflections=4, optimal_T=opt)
        on = BlockUnitaryReflectionQROAM.from_shape(
            4, 16, 8, n_reflections=4, optimal_T=opt, three_phase_layer_prep=True
        )
        assert isinstance(
            off.reflection(0).prepare_w.state_prep, BlockStatePreparationViaQROAMRotations
        )
        assert isinstance(
            on.reflection(0).prepare_w.state_prep, ThreePhaseLayerStatePreparation
        )
        assert _cost(off) != _cost(on), opt
        for top in (off, on):
            assert _self_consistent(top.reflection(0).prepare_w), opt
        # the controlled reflection (get_ctrl_system) stays consistent with the toggle on.
        assert _self_consistent(on.reflection(0).controlled()), opt


def test_block_unitary_reflection_n_blocks_one_drops_block_register():
    on = BlockUnitaryReflectionQROAM.from_shape(1, 16, 8, n_reflections=2, three_phase_layer_prep=True)
    sp = on.reflection(0).prepare_w.state_prep
    assert [r.name for r in sp.signature] == ["prepare_control", "target_state", "phase_gradient"]
    assert _self_consistent(on.reflection(0).prepare_w)


# ---------------------------------------------------------------------------
# classical_matrix_block_encoding_QROAM
# ---------------------------------------------------------------------------


def test_classical_matrix_toggle():
    for opt in (False, True):
        off = ClassicalMatrixBlockEncoding.from_bitsize(16, 8, optimal_T=opt)
        on = ClassicalMatrixBlockEncoding.from_bitsize(16, 8, optimal_T=opt, three_phase_layer_prep=True)
        # U_R block prep and U_L^dag plain prep both swap to the ansatz.
        assert isinstance(off._prep_psi_fwd(), BlockStatePreparationViaQROAMRotations)
        assert isinstance(off._prep_phi_adj(), StatePreparationViaQROAMRotations)
        assert isinstance(on._prep_psi_fwd(), ThreePhaseLayerStatePreparation)
        assert isinstance(on._prep_phi_adj(), ThreePhaseLayerStatePreparation)
        assert _cost(off) != _cost(on), opt
        for be in (off, on):
            assert _self_consistent(be), opt
            assert _self_consistent(be.controlled()), opt


def test_block_diagonal_classical_matrix_toggle():
    for opt in (False, True):
        off = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(4, 16, 8, optimal_T=opt)
        on = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(
            4, 16, 8, optimal_T=opt, three_phase_layer_prep=True
        )
        # flag, U_R, U_L^dag all route through the ansatz.
        for prep in (on._prep_flag(), on._prep_psi_fwd(), on._prep_phi_adj()):
            assert isinstance(prep, ThreePhaseLayerStatePreparation)
        assert isinstance(off._prep_flag(), BlockStatePreparationViaQROAMRotations)
        assert _cost(off) != _cost(on), opt
        for be in (off, on):
            assert _self_consistent(be), opt
            assert _self_consistent(be.controlled()), opt


# ---------------------------------------------------------------------------
# rectangular_block_encoding_reflection
# ---------------------------------------------------------------------------


def test_rectangular_block_encoding_toggle():
    for opt in (False, True):
        off = RectangularBlockEncodingReflection(
            n_blocks=2, m_rows=5, n_cols=11, phase_bitsize=8, optimal_T=opt
        )
        on = RectangularBlockEncodingReflection(
            n_blocks=2, m_rows=5, n_cols=11, phase_bitsize=8, optimal_T=opt, three_phase_layer_prep=True
        )
        # the toggle threads through into the inner BlockUnitaryReflectionQROAM.
        assert off.reflection_synthesis.three_phase_layer_prep is False
        assert on.reflection_synthesis.three_phase_layer_prep is True
        assert isinstance(
            on.reflection_synthesis.reflection(0).prepare_w.state_prep,
            ThreePhaseLayerStatePreparation,
        )
        for be in (off, on):
            assert _self_consistent(be), opt


def test_reflection_rectangular_block_encoding_toggle():
    off = ReflectionRectangularBlockEncoding(n_blocks=4, n_rows=16, phase_bitsize=8, n_reflections=4)
    on = ReflectionRectangularBlockEncoding(
        n_blocks=4, n_rows=16, phase_bitsize=8, n_reflections=4, three_phase_layer_prep=True
    )
    assert isinstance(
        on.synth.reflection(0).prepare_w.state_prep, ThreePhaseLayerStatePreparation
    )
    assert _cost(off) != _cost(on)
    for be in (off, on):
        assert _self_consistent(be)
        assert _self_consistent(be.controlled())


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
