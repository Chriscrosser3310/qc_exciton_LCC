"""Qubit-count regression tests for ``BlockUnitaryReflectionQROAM``.

Cycle 9 of the autonomous loop discovered that Qualtran's ``QubitCount``
raised ``KeyError: 'block'`` on ``BlockPrepareHouseholderStateQROAM``
whenever ``n_blocks == 1`` (block_bitsize=0, so the signature omits the
``block`` register but ``build_composite_bloq`` popped it
unconditionally). This file pins:

  1. ``QubitCount`` succeeds on every grid point that
     ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` tabulates (the same 49-point
     ``(n_blocks, N) ∈ {1,2,4,8,16,32,64} × {4,...,256}`` grid the
     Toffoli analytic estimator covers).
  2. ``QubitCount`` for ``n_blocks=1`` agrees with the un-blocked
     ``UnitaryReflectionQROAM`` (the same single-block-equivalence
     contract the Toffoli side pins in
     ``test_block_unitary_reflection_equivalence``).
  3. The transient QROAMClean workspace
     ``T = QubitCount - signature.n_qubits()`` is monotone
     non-decreasing in both ``n_blocks`` and ``n_rows`` along the
     doubling sequence — a sanity invariant that future QROAMClean
     optimizer changes must preserve.

These tests guard "Improve constant factors in quantum algorithms" and
"Any potential improvements on the final Toffoli complexity/qubit
counts/scaling" (GOALS.md) on the qubit side.
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

import numpy as np

from qualtran.resource_counting import QubitCount, get_cost_value

from integrations.qualtran.block_unitary_reflection_QROAM import BlockUnitaryReflectionQROAM
from integrations.qualtran.model_resource_counts import (
    SYNTHESIS_PER_REFLECTION_INTERCEPT,
    block_unitary_synthesis_signature_qubits,
)
from integrations.qualtran.unitary_reflection_QROAM import UnitaryReflectionQROAM


def _qubit_count(bloq) -> int:
    return int(get_cost_value(bloq, QubitCount()))


def test_qubit_count_succeeds_for_single_block():
    """Regression: QubitCount must succeed when n_blocks=1.

    Before the cycle-10 fix, ``BlockPrepareHouseholderStateQROAM.build_composite_bloq``
    unconditionally popped the ``block`` soquet, which is absent when
    ``block_bitsize == 0`` (i.e., n_blocks=1). This made QubitCount
    raise ``KeyError: 'block'`` for the entire n_blocks=1 row of the
    49-point analytic grid.
    """
    bloq = BlockUnitaryReflectionQROAM.from_shape(
        n_blocks=1, n_rows=4, phase_bitsize=4, n_reflections=1
    )
    qc = _qubit_count(bloq)
    # signature width is 1 (refl_ancilla) + 2 (system) + 4 (phase_grad) = 7
    assert qc >= bloq.signature.n_qubits() == 7


def test_qubit_count_works_over_full_intercept_grid():
    """QubitCount must succeed across the same grid the Toffoli estimator covers."""
    for (n_blocks, n_rows) in SYNTHESIS_PER_REFLECTION_INTERCEPT.keys():
        bloq = BlockUnitaryReflectionQROAM.from_shape(
            n_blocks=n_blocks, n_rows=n_rows, phase_bitsize=4, n_reflections=1
        )
        qc = _qubit_count(bloq)
        sig = bloq.signature.n_qubits()
        assert qc >= sig, f"({n_blocks},{n_rows}): qubit count {qc} below signature {sig}"
        # And the closed-form signature lower bound matches the bloq's signature width.
        assert sig == block_unitary_synthesis_signature_qubits(n_blocks, n_rows, 4)


def test_qubit_count_single_block_matches_unblocked():
    """n_blocks=1 ⇒ QubitCount equals the un-blocked ``UnitaryReflectionQROAM``.

    This is the qubit-side analog of the Toffoli single-block equivalence
    pinned in ``test_block_unitary_reflection_equivalence``. Both should
    reduce to the Sec. 4 un-blocked construction of arXiv:1812.00954.
    """
    rng = np.random.default_rng(0)
    for n in (2, 4, 8):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))
        block = BlockUnitaryReflectionQROAM(block_unitaries=Q[None], phase_bitsize=4)
        flat = UnitaryReflectionQROAM(unitary=Q, phase_bitsize=4)
        # The headline qubit count must agree; the per-bloq decomposition
        # paths are different but the Sec. 4 contract is identical.
        assert _qubit_count(block) == _qubit_count(flat), (
            f"n={n}: block={_qubit_count(block)} flat={_qubit_count(flat)}"
        )


def test_workspace_monotone_in_n_blocks():
    """Workspace = QubitCount - signature must be monotone non-decreasing in n_blocks.

    At fixed ``(n_rows, phase_bitsize)``, doubling ``n_blocks`` quadruples
    or keeps the QROAMClean table length, so the workspace overhead must
    not shrink. (It can stay flat where the optimizer's block-size step
    doesn't advance.)
    """
    n_rows = 16
    phase_bitsize = 4
    workspaces = []
    for n_blocks in (1, 2, 4, 8, 16, 32, 64):
        bloq = BlockUnitaryReflectionQROAM.from_shape(
            n_blocks=n_blocks, n_rows=n_rows, phase_bitsize=phase_bitsize, n_reflections=1
        )
        ws = _qubit_count(bloq) - bloq.signature.n_qubits()
        workspaces.append((n_blocks, ws))
    for (nb_prev, ws_prev), (nb_next, ws_next) in zip(workspaces, workspaces[1:]):
        assert ws_next >= ws_prev, (
            f"workspace dropped from {ws_prev} at n_blocks={nb_prev} to "
            f"{ws_next} at n_blocks={nb_next}"
        )


def test_workspace_monotone_in_n_rows():
    """Workspace = QubitCount - signature must be monotone non-decreasing in n_rows.

    At fixed ``(n_blocks, phase_bitsize)``, doubling ``n_rows`` doubles the
    QROAMClean table length and so cannot shrink the workspace overhead.
    """
    n_blocks = 4
    phase_bitsize = 4
    workspaces = []
    for n_rows in (4, 8, 16, 32, 64, 128, 256):
        bloq = BlockUnitaryReflectionQROAM.from_shape(
            n_blocks=n_blocks, n_rows=n_rows, phase_bitsize=phase_bitsize, n_reflections=1
        )
        ws = _qubit_count(bloq) - bloq.signature.n_qubits()
        workspaces.append((n_rows, ws))
    for (nr_prev, ws_prev), (nr_next, ws_next) in zip(workspaces, workspaces[1:]):
        assert ws_next >= ws_prev, (
            f"workspace dropped from {ws_prev} at n_rows={nr_prev} to "
            f"{ws_next} at n_rows={nr_next}"
        )


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
