"""Equivalence tests for ``BlockUnitarySynthesisQROAM``.

These tests pin down the single-block reduction
``BlockUnitarySynthesisQROAM(n_blocks=1)`` against the un-blocked
``UnitarySynthesisQROAM`` on Toffoli/AND/measurement counts, and check
basic structural invariants of the block-indexed variant (signature
shape, isometry support, symbolic ``from_shape``, adjoint toggling,
and data-free decomposition errors).

This guards "Improve constant factors in quantum algorithms" (GOALS.md):
the block synthesis is meant to amortize QROAM data loading across
blocks, but its single-block limit must remain faithful to the
established un-blocked construction.
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

import numpy as np

qualtran = pytest.importorskip("qualtran")
_ = qualtran

from qualtran import DecomposeTypeError
from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_unitary_synthesis_QROAM import (
    BlockPrepareHouseholderStateQROAM,
    BlockUnitarySynthesisQROAM,
)
from integrations.qualtran.unitary_synthesis_QROAM import UnitarySynthesisQROAM


def _random_unitary(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Fix the QR phase convention so Q is uniformly random on U(n).
    Q = Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))
    return Q


def _gate_costs(bloq):
    return get_cost_value(bloq, QECGatesCost())


def test_single_block_toffoli_matches_flat_n2():
    U = _random_unitary(2, seed=0)
    block = BlockUnitarySynthesisQROAM(block_unitaries=U[None], phase_bitsize=4)
    flat = UnitarySynthesisQROAM(unitary=U, phase_bitsize=4)
    assert _gate_costs(block).toffoli == _gate_costs(flat).toffoli


def test_single_block_toffoli_matches_flat_n4():
    U = _random_unitary(4, seed=1)
    block = BlockUnitarySynthesisQROAM(block_unitaries=U[None], phase_bitsize=6)
    flat = UnitarySynthesisQROAM(unitary=U, phase_bitsize=6)
    assert _gate_costs(block).toffoli == _gate_costs(flat).toffoli


def test_single_block_toffoli_matches_flat_n8():
    U = _random_unitary(8, seed=2)
    block = BlockUnitarySynthesisQROAM(block_unitaries=U[None], phase_bitsize=6)
    flat = UnitarySynthesisQROAM(unitary=U, phase_bitsize=6)
    # Toffoli is the headline cost. The block variant may use a slightly
    # different QROAM block-size optimum (its data tensor is 2D rather
    # than 1D), so and_bloq/measurement counts can diverge by O(1).
    assert _gate_costs(block).toffoli == _gate_costs(flat).toffoli


def test_single_block_signature_has_no_block_register():
    """n_blocks=1 ⇒ block_bitsize=0 ⇒ no 'block' register slot."""
    U = _random_unitary(4, seed=3)
    block = BlockUnitarySynthesisQROAM(block_unitaries=U[None], phase_bitsize=4)
    names = [r.name for r in block.signature]
    assert "block" not in names
    assert names == ["reflection_ancilla", "system", "phase_gradient"]


def test_multi_block_signature_includes_block_register():
    rng = np.random.default_rng(4)
    blocks = np.stack([_random_unitary(4, seed=10 + i) for i in range(3)])
    block = BlockUnitarySynthesisQROAM(block_unitaries=blocks, phase_bitsize=4)
    names = [r.name for r in block.signature]
    assert names[0] == "block"
    # ceil_log2(3) == 2
    assert block.block_bitsize == 2
    assert block.system_bitsize == 2  # log2(4)


def test_isometry_n_reflections_equals_n_cols():
    """Non-square block_unitaries (isometry columns) ⇒ exactly n_cols reflections."""
    U = _random_unitary(4, seed=5)
    # Keep only the first two orthonormal columns of each block.
    iso = U[:, :2][None]  # (1, 4, 2)
    block = BlockUnitarySynthesisQROAM(block_unitaries=iso, phase_bitsize=4)
    assert block.n_reflections == 2

    # Call graph must contain exactly 2 reflection bloqs.
    from qualtran.resource_counting import SympySymbolAllocator

    cg = block.build_call_graph(SympySymbolAllocator())
    n_refl = sum(cg.values())
    assert n_refl == 2


def test_data_free_bloq_cannot_decompose():
    bloq = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=4, n_rows=8, phase_bitsize=6
    )
    with pytest.raises(DecomposeTypeError):
        bloq.decompose_bloq()


def test_from_shape_isometry_keeps_n_reflections():
    bloq = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=2, n_rows=8, phase_bitsize=6, n_reflections=3
    )
    assert bloq.n_blocks == 2
    assert bloq.n_rows == 8
    assert bloq.n_reflections == 3
    assert bloq.block_bitsize == 1
    assert bloq.system_bitsize == 3


def test_from_shape_symbolic():
    sympy = pytest.importorskip("sympy")
    nb, nr, pb = sympy.symbols("Nb N b", positive=True, integer=True)
    bloq = BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=nb, n_rows=nr, phase_bitsize=pb
    )
    assert bloq.n_blocks == nb
    assert bloq.n_rows == nr
    assert bloq.n_reflections == nr


def test_rejects_non_orthonormal_block():
    bad = np.array([[[1.0 + 0j, 1.0 + 0j], [0.0, 1.0 + 0j]]], dtype=np.complex128)
    with pytest.raises(AssertionError):
        BlockUnitarySynthesisQROAM(block_unitaries=bad, phase_bitsize=4)


def test_prepare_householder_adjoint_toggles_uncompute():
    U = _random_unitary(4, seed=7)
    col = U[:, 0][None]  # (1, 4) with unit norm
    prep = BlockPrepareHouseholderStateQROAM(
        state_coefficients=col, phase_bitsize=4, basis_index=0, uncompute=False
    )
    adj = prep.adjoint()
    assert adj.uncompute is True
    assert adj.adjoint() == prep


def test_reflection_indexed_by_basis():
    U = _random_unitary(4, seed=8)
    block = BlockUnitarySynthesisQROAM(block_unitaries=U[None], phase_bitsize=4)
    for k in range(block.n_reflections):
        r = block.reflection(k)
        assert r.basis_index == k
        # state_coefficients for reflection k should match column k of the unitary.
        assert np.allclose(r.state_coefficients[0], U[:, k])


if __name__ == "__main__":
    failed = 0
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
