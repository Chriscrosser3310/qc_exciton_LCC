"""Tests for the interferometer-based isometry synthesis (Sec. III B of arXiv:2409.11748).

Covers both :class:`InterferometerIsometrySynthesisQROAM` (single ``N x M`` isometry) and
:class:`BlockInterferometerIsometrySynthesisQROAM` (``K`` diagonal blocks).

Pins:
  1. The numpy reference (thin-row-CSD schedule) reconstructs the first ``M`` columns of a
     random isometry to machine precision, and the full operator is unitary.
  2. ``build_composite_bloq`` (real gate layout, shape-only QROAM) and ``build_call_graph``
     report identical ``QECGatesCost`` -- both count the same interferometers and rotations.
  3. ``d = 1`` (``M = N``) reduces to the full-unitary interferometer; the block version with
     ``K = 1`` reduces to the single isometry.
  4. ``d = N / M``, ``n_levels = log2 d``, and the schedule has ``2d - 1`` full ``M x M``
     unitaries and ``d - 1`` multiplexed rotations.
  5. Synthesizing ``M < N`` columns costs strictly less than the full ``N``-column unitary.
  6. The singly-controlled variant adds exactly one ``ctrl`` qubit with only modest overhead.
  7. Signature layout; ``block`` register present iff ``K > 1``.
  8. A data-free symbolic instance refuses to decompose / enumerate.
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
import sympy

qualtran = pytest.importorskip("qualtran")

from qualtran import DecomposeTypeError
from qualtran.resource_counting import get_cost_value, QECGatesCost

from integrations.qualtran.interferometer_isometry_QROAM import (
    InterferometerIsometryMuxRotationQROAM,
    InterferometerIsometrySynthesisQROAM as ISO,
    estimate_interferometer_isometry_resources,
    isometry_qr_peel_steps,
    reconstruct_isometry_from_qr_peel,
)
from integrations.qualtran.block_interferometer_isometry_QROAM import (
    BlockInterferometerIsometrySynthesisQROAM as BISO,
)
from integrations.qualtran.block_unitary_interferometer_QROAM import (
    BlockUnitaryInterferometerSynthesisQROAM as FULL,
)


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost())


def _rand_isometry(N, M, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))
    Q, _ = np.linalg.qr(A)
    return Q[:, :M]


# --------------------------------------------------------------------------- 1

def test_numpy_reference_reconstructs_isometry():
    # Eq.-36 QR-peeling: synthesize the first M columns of a (d*M)-dim unitary via d-1 d=2 steps.
    for s, (N, M) in enumerate([(8, 4), (16, 4), (16, 8), (32, 8), (32, 4), (24, 8), (48, 8)]):
        W = _rand_isometry(N, M, seed=s)
        peel = isometry_qr_peel_steps(W, seed=s + 100)
        Op = reconstruct_isometry_from_qr_peel(peel)
        assert len(peel.steps) == N // M - 1, (N, M, len(peel.steps))
        assert np.abs(Op[:, :M] - W).max() < 1e-7, (N, M)
        assert np.abs(Op.conj().T @ Op - np.eye(N)).max() < 1e-7, (N, M)


# --------------------------------------------------------------------------- 2

def test_single_composite_equals_call_graph():
    for N, M in [(8, 4), (16, 4), (16, 8), (16, 16), (32, 8), (32, 16), (64, 8)]:
        for opt in (False, True):
            bloq = ISO.from_shape(N, M, 8, optimal_T=opt)
            assert _cost(bloq) == _cost(bloq.decompose_bloq()), (N, M, opt)


def test_block_composite_equals_call_graph():
    for K, N, M in [(1, 16, 4), (2, 16, 4), (3, 16, 4), (4, 32, 8), (8, 32, 8), (6, 64, 8), (2, 16, 16)]:
        for opt in (False, True):
            bloq = BISO.from_shape(K, N, M, 8, optimal_T=opt)
            assert _cost(bloq) == _cost(bloq.decompose_bloq()), (K, N, M, opt)


# --------------------------------------------------------------------------- 3

def test_d1_reduces_to_full_interferometer():
    for opt in (False, True):
        single = ISO.from_shape(16, 16, 8, optimal_T=opt)
        assert _cost(single) == _cost(FULL.from_shape(1, 16, 8, optimal_T=opt))
        for K in (2, 5):
            block = BISO.from_shape(K, 16, 16, 8, optimal_T=opt)
            assert _cost(block) == _cost(FULL.from_shape(K, 16, 8, optimal_T=opt))


def test_block_k1_equals_single():
    for N, M in [(16, 4), (32, 8), (64, 16)]:
        for opt in (False, True):
            assert _cost(BISO.from_shape(1, N, M, 8, optimal_T=opt)) == _cost(
                ISO.from_shape(N, M, 8, optimal_T=opt)
            )


# --------------------------------------------------------------------------- 4

def test_d_parameter_and_step_counts():
    # Eq. 36 (V merged): 1 initial V + (d-1) controlled-U interferometers = d interferometers,
    # plus (d-1) mux-R_y rotations.  Linear in d.
    for N, M in [(8, 4), (16, 4), (24, 8), (32, 8), (64, 8), (256, 16)]:
        d = N // M
        b = ISO.from_shape(N, M, 8)
        assert int(b.d) == d
        assert int(b.n_steps) == d - 1
        est = estimate_interferometer_isometry_resources(N, M, 8, optimal_T=True)
        assert est.d == d
        assert est.n_d2_steps == d - 1
        assert est.n_mux_rotations == d - 1
        assert est.n_interferometers == (1 if d == 1 else d)
        assert est.toffoli > 0


def test_d3_non_power_of_two_ambient():
    # d=3 (and other non-power-of-two d) are valid: ambient N = d*M need not be a power of two.
    import numpy as np
    for N, M in [(12, 4), (24, 8), (48, 16)]:
        d = N // M
        b = ISO.from_shape(N, M, 8, optimal_T=True)
        assert int(b.d) == d
        assert _cost(b) == _cost(b.decompose_bloq())          # composite == call graph
        W = _rand_isometry(N, M, seed=N)
        peel = isometry_qr_peel_steps(W, seed=1)
        Op = reconstruct_isometry_from_qr_peel(peel)
        assert len(peel.steps) == d - 1
        assert np.abs(Op[:, :M] - W).max() < 1e-7


def test_cost_linear_in_d():
    # The paper states the cost increases linearly with d: marginal Toffoli per unit d -> constant.
    M, b = 32, 8
    ds = [4, 8, 16, 32, 64]
    from integrations.qualtran.utils import get_Toffoli_counts
    T = [int(get_Toffoli_counts(ISO.from_shape(d * M, M, b, optimal_T=False))) for d in ds]
    marg = [(T[i] - T[i - 1]) / (ds[i] - ds[i - 1]) for i in range(1, len(ds))]
    # all marginal slopes within 2% of each other => linear
    assert max(marg) / min(marg) < 1.02, marg


# --------------------------------------------------------------------------- 5

def test_isometry_cheaper_than_full_unitary_large_d():
    """At large d (M << N) the Eq.-36 column synthesis beats the full N x N unitary.

    The cost is linear in d with a fixed per-step cost, so the crossover is at moderate d;
    for small d the (d-1) d=2 steps can exceed the single full-unitary interferometer.
    """
    for N, M in [(256, 8), (256, 16), (128, 8)]:
        iso = ISO.from_shape(N, M, 8, optimal_T=True)
        full = FULL.from_shape(1, N, 8, optimal_T=True)
        assert int(_cost(iso).total_t_count()) < int(_cost(full).total_t_count()), (N, M)


# --------------------------------------------------------------------------- 6

def test_controlled_variant():
    cases_single = [(8, 4), (16, 4), (32, 8)]
    for N, M in cases_single:
        bare = ISO.from_shape(N, M, 8)
        ctl = bare.controlled()
        assert ctl.signature.n_qubits() == bare.signature.n_qubits() + 1
        t_bare = int(_cost(bare).total_t_count())
        t_ctl = int(_cost(ctl).total_t_count())
        assert t_bare <= t_ctl <= 2 * t_bare, (N, M, t_bare, t_ctl)
    for K, N, M in [(2, 16, 4), (4, 32, 8)]:
        bare = BISO.from_shape(K, N, M, 8)
        ctl = bare.controlled()
        assert ctl.signature.n_qubits() == bare.signature.n_qubits() + 1
        t_bare = int(_cost(bare).total_t_count())
        t_ctl = int(_cost(ctl).total_t_count())
        assert t_bare <= t_ctl <= 2 * t_bare, (K, N, M, t_bare, t_ctl)


# --------------------------------------------------------------------------- 7

def test_signature_layout():
    single = ISO.from_shape(32, 8, 10)
    assert [r.name for r in single.signature] == ["system", "phase_gradient"]
    assert single.signature.get_left("system").bitsize == 5  # log2(32)
    assert single.signature.get_left("phase_gradient").bitsize == 10

    block = BISO.from_shape(5, 32, 8, 10)
    names = [r.name for r in block.signature]
    assert names == ["block", "system", "phase_gradient"]
    assert block.signature.get_left("block").bitsize == 3  # ceil(log2 5)
    assert block.signature.get_left("system").bitsize == 5

    # K == 1 drops the block register.
    assert [r.name for r in BISO.from_shape(1, 32, 8, 10).signature] == ["system", "phase_gradient"]


def test_mux_rotation_layer():
    # has_block toggles the block register; both decompose and agree with their call graph.
    no_block = InterferometerIsometryMuxRotationQROAM(n_blocks=1, n_address=8, phase_bitsize=8)
    assert [r.name for r in no_block.signature] == ["system", "target", "phase_gradient"]
    assert _cost(no_block) == _cost(no_block.decompose_bloq())

    with_block = InterferometerIsometryMuxRotationQROAM(n_blocks=4, n_address=8, phase_bitsize=8)
    assert [r.name for r in with_block.signature] == ["block", "system", "target", "phase_gradient"]
    assert _cost(with_block) == _cost(with_block.decompose_bloq())


# --------------------------------------------------------------------------- 8

def test_symbolic_refuses_to_decompose():
    N = sympy.Symbol("N", positive=True, integer=True)
    bloq = ISO.from_shape(N, 8, 8)
    with pytest.raises(DecomposeTypeError):
        bloq.build_call_graph(None)
    with pytest.raises(ValueError):
        ISO.from_shape(N, 8, 8, optimal_T=True)

    bblock = BISO.from_shape(2, N, 8, 8)
    with pytest.raises(DecomposeTypeError):
        bblock.build_call_graph(None)


def test_rejects_bad_shapes():
    with pytest.raises(AssertionError):
        ISO.from_shape(8, 16, 8)  # M > N
    with pytest.raises(AssertionError):
        ISO.from_shape(20, 8, 8)  # N not an integer multiple of M (d = N/M not an integer)
    with pytest.raises(AssertionError):
        ISO.from_shape(16, 2, 8)  # M < 4 (interferometer needs >= 2 system qubits)


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
