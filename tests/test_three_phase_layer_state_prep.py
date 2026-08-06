"""Tests for the three-diagonal-phase-layer state-preparation ansatz.

Pins both the *ansatz arithmetic* and the *resource model* of
``integrations.qualtran.three_phase_layer_state_prep_QROAM``:

Ansatz (numpy reference)
  * :func:`walsh_hadamard` is the normalized ``H^{otimes n}`` transform.
  * :func:`reconstruct_three_phase_state` applies ``D3 H^n D2 H^n D1 |+^n>``: norm-preserving,
    reduces to ``|+^n>`` at zero angles, and matches an independent ``np.diag`` matrix product.

Resource model (qualtran bloqs)
  * ``build_composite_bloq`` (real layout, shape-only QROAM) and ``build_call_graph`` report identical
    ``QECGatesCost`` for the top bloq and for the coherent ``DiagonalPhaseQROAM`` building block.
  * Signature ``target_state, phase_gradient``; counts: ``3n`` Hadamards, three diagonal layers.
  * ``measure_reset`` never increases cost (and keeps the final layer coherent); ``optimal_T`` never
    increases cost; symbolic parameters refuse to decompose.
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

from qualtran import DecomposeTypeError
from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.three_phase_layer_state_prep_QROAM import (
    DiagonalPhaseQROAM,
    ThreePhaseLayerResourceEstimate,
    ThreePhaseLayerStatePreparation,
    estimate_three_phase_layer_resources,
    plus_state,
    reconstruct_three_phase_state,
    walsh_hadamard,
)

T = ThreePhaseLayerStatePreparation


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost())


# ---------------------------------------------------------------------------
# Ansatz arithmetic (numpy reference)
# ---------------------------------------------------------------------------


def test_walsh_hadamard_is_unitary_and_involutive():
    for n in range(1, 6):
        N = 1 << n
        Hn = walsh_hadamard(n)
        assert Hn.shape == (N, N)
        assert np.allclose(Hn @ Hn.conj().T, np.eye(N))
        assert np.allclose(Hn @ Hn, np.eye(N))  # H^2 = I
        assert np.allclose(Hn @ plus_state(n), np.eye(N)[0])  # H^n|+^n> = |0^n>


def test_zero_angles_reconstruct_plus_state():
    """With all phases zero the ansatz collapses to D3 H H D2 H H D1 |+> = |+^n>."""
    for n in range(1, 6):
        N = 1 << n
        z = np.zeros(N)
        assert np.allclose(reconstruct_three_phase_state(z, z, z), plus_state(n))


def test_reconstruct_preserves_norm():
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        N = 1 << n
        t1, t2, t3 = (rng.uniform(0, 2 * np.pi, N) for _ in range(3))
        psi = reconstruct_three_phase_state(t1, t2, t3)
        assert np.isclose(np.linalg.norm(psi), 1.0)


def test_reconstruct_matches_independent_matrix_product():
    """The elementwise reference equals an independent D3 @ H^n @ D2 @ H^n @ D1 @ |+> product."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in range(1, 6):
        N = 1 << n
        thetas = [rng.uniform(0, 2 * np.pi, N) for _ in range(3)]
        Hn = walsh_hadamard(n)
        Ds = [np.diag(np.exp(1j * t)) for t in thetas]
        M = Ds[2] @ Hn @ Ds[1] @ Hn @ Ds[0]
        ref = M @ plus_state(n)
        got = reconstruct_three_phase_state(*thetas)
        worst = max(worst, float(np.linalg.norm(got - ref)))
    assert worst < 1e-12, worst


def test_phase_tables_are_full_length_N():
    """Each layer carries N independent angles (no half-identity assumption)."""
    n = 4
    N = 1 << n
    rng = np.random.default_rng(2)
    thetas = [rng.uniform(0, 2 * np.pi, N) for _ in range(3)]
    # Perturbing any single entry of any layer changes the output -> all N entries are live.
    base = reconstruct_three_phase_state(*thetas)
    for layer in range(3):
        for x in (0, N // 2, N - 1):
            bumped = [t.copy() for t in thetas]
            bumped[layer][x] += 0.7
            assert not np.allclose(reconstruct_three_phase_state(*bumped), base)


# ---------------------------------------------------------------------------
# Resource model: bloq structure / cost
# ---------------------------------------------------------------------------


def test_composite_equals_call_graph():
    cases = [
        (8, 6, False),
        (8, 6, True),
        (16, 8, False),
        (16, 8, True),
        (4, 4, False),
        (32, 10, True),
    ]
    for nrows, b, mr in cases:
        bloq = T.from_bitsize(nrows, b, measure_reset=mr)
        cg = _cost(bloq)
        comp = _cost(bloq.decompose_bloq())
        assert cg == comp, (nrows, b, mr, cg, comp)
    # An explicit (non-default) QROAM block-size split must stay self-consistent too.
    bloq = T.from_bitsize(32, 10, log_block_sizes=2, adjoint_log_block_sizes=2)
    assert _cost(bloq) == _cost(bloq.decompose_bloq())


def test_diagonal_layer_composite_equals_call_graph():
    """The coherent (default) building block is internally consistent."""
    for nrows, b in [(8, 6), (16, 8), (32, 10)]:
        d = DiagonalPhaseQROAM(n_rows=nrows, phase_bitsize=b)
        assert _cost(d) == _cost(d.decompose_bloq()), (nrows, b)


def test_signature_layout():
    bloq = T.from_bitsize(16, 10)
    assert [r.name for r in bloq.signature] == ["target_state", "phase_gradient"]
    assert bloq.signature.n_qubits() == bloq.system_bitsize + 10
    d = DiagonalPhaseQROAM(n_rows=16, phase_bitsize=10)
    assert [r.name for r in d.signature] == ["system", "phase_gradient"]


def test_hadamard_and_layer_counts():
    """build_call_graph has exactly 3n Hadamards and three DiagonalPhaseQROAM layers."""
    from qualtran.bloqs.basic_gates import Hadamard

    for n in range(1, 6):
        N = 1 << n
        bloq = T.from_bitsize(N, 8)
        cg = bloq.build_call_graph(None)
        assert cg[Hadamard()] == 3 * n
        n_layers = sum(c for b, c in cg.items() if isinstance(b, DiagonalPhaseQROAM))
        assert n_layers == 3


def test_measure_reset_keeps_final_layer_coherent():
    """measure_reset erases D1, D2 by measurement but D3 stays coherent (still loads adjoint)."""
    bloq = T.from_bitsize(16, 8, measure_reset=True)
    layers = {bloq.diagonal_layer(is_final=f) for f in (False, True)}
    assert len(layers) == 2  # the two non-final and the final layer differ
    assert bloq.diagonal_layer(is_final=False).measure_reset is True
    assert bloq.diagonal_layer(is_final=True).measure_reset is False


def test_measure_reset_never_increases_toffoli():
    for nrows, b in [(8, 6), (16, 8), (32, 10)]:
        coherent = int(_cost(T.from_bitsize(nrows, b, measure_reset=False)).toffoli)
        reset = int(_cost(T.from_bitsize(nrows, b, measure_reset=True)).toffoli)
        assert reset <= coherent, (nrows, b, coherent, reset)
    # Dropping two coherent adjoints must strictly help for a non-trivial table.
    assert int(_cost(T.from_bitsize(32, 10, measure_reset=True)).toffoli) < int(
        _cost(T.from_bitsize(32, 10, measure_reset=False)).toffoli
    )


def test_default_block_size_is_T_optimal():
    """The default (None) QROAM split is never worse than any fixed log_block_sizes choice.

    QROAM auto-optimizes the load and adjoint splits independently, so for these diagonal tables
    the default beats every single shared block size (and certainly the Lambda=1 baseline).
    """
    for nrows, b in [(8, 6), (64, 8), (256, 8)]:
        n = int(nrows).bit_length() - 1
        default_t = int(_cost(T.from_bitsize(nrows, b)).total_t_count())
        for lbs in range(n + 1):
            fixed_t = int(
                _cost(
                    T.from_bitsize(nrows, b, log_block_sizes=lbs, adjoint_log_block_sizes=lbs)
                ).total_t_count()
            )
            assert default_t <= fixed_t, (nrows, b, lbs, default_t, fixed_t)
    # Strictly better than the un-batched Lambda=1 baseline for a non-trivial table.
    lambda1_t = int(
        _cost(T.from_bitsize(256, 8, log_block_sizes=0, adjoint_log_block_sizes=0)).total_t_count()
    )
    assert int(_cost(T.from_bitsize(256, 8)).total_t_count()) < lambda1_t


def test_symbolic_refuses_to_decompose():
    import sympy

    N = sympy.Symbol("N", positive=True, integer=True)
    bloq = T.from_bitsize(N, 8)
    with pytest.raises(DecomposeTypeError):
        bloq.build_composite_bloq(None)


def test_non_power_of_two_rejected():
    with pytest.raises(AssertionError):
        T.from_bitsize(12, 8)
    with pytest.raises(AssertionError):
        DiagonalPhaseQROAM(n_rows=12, phase_bitsize=8)


# ---------------------------------------------------------------------------
# Resource model: analytic estimate
# ---------------------------------------------------------------------------


def test_estimate_structural_counts():
    for n in range(1, 7):
        N = 1 << n
        e = estimate_three_phase_layer_resources(N, 8)
        assert isinstance(e, ThreePhaseLayerResourceEstimate)
        assert e.n_phase_layers == 3
        assert e.n_hadamards == 3 * n
        assert e.n_phase_additions == 3
        assert e.n_loaded_phase_words == 3 * N
        assert e.register_qubits == n + 8


def test_estimate_method_matches_function_and_bloq():
    bloq = T.from_bitsize(16, 12, measure_reset=True)
    e_method = bloq.resource_estimate()
    e_func = estimate_three_phase_layer_resources(16, 12, measure_reset=True)
    assert e_method == e_func
    assert e_method.toffoli == int(_cost(bloq).toffoli)
    assert e_method.t_count == int(_cost(bloq).total_t_count())


# ---------------------------------------------------------------------------
# Extended drop-in interface: block address, control, uncompute
# ---------------------------------------------------------------------------


def test_block_register_and_cost():
    """n_blocks>1 prepends a block register; cost grows with n_blocks; cg==composite."""
    assert [r.name for r in T.from_bitsize(16, 8, n_blocks=4).signature] == [
        "block",
        "target_state",
        "phase_gradient",
    ]
    assert int(_cost(T.from_bitsize(64, 10, n_blocks=8)).toffoli) > int(
        _cost(T.from_bitsize(64, 10)).toffoli
    )
    for nb in (1, 2, 8):
        b = T.from_bitsize(16, 8, n_blocks=nb)
        assert _cost(b) == _cost(b.decompose_bloq()), nb


def test_control_register_and_hadamard_rotations():
    """control_bitsize=1 prepends prepare_control; controlled-H contributes rotations."""
    assert [r.name for r in T.from_bitsize(16, 8, control_bitsize=1).signature] == [
        "prepare_control",
        "target_state",
        "phase_gradient",
    ]
    n = 6  # N = 64
    # 3n controlled-Hadamards, 2 rotations each; the uncontrolled bloq has none.
    assert int(_cost(T.from_bitsize(64, 10, control_bitsize=1)).rotation) == 3 * n * 2
    assert int(_cost(T.from_bitsize(64, 10)).rotation) == 0
    for ctrl in (0, 1):
        b = T.from_bitsize(16, 8, control_bitsize=ctrl)
        assert _cost(b) == _cost(b.decompose_bloq()), ctrl


def test_full_block_and_control_signature():
    assert [r.name for r in T.from_bitsize(16, 8, n_blocks=4, control_bitsize=1).signature] == [
        "prepare_control",
        "block",
        "target_state",
        "phase_gradient",
    ]


def test_control_bitsize_validation():
    with pytest.raises(AssertionError):
        T.from_bitsize(16, 8, control_bitsize=2)
    with pytest.raises(AssertionError):
        DiagonalPhaseQROAM(n_rows=16, phase_bitsize=8, control_bitsize=2)


def test_uncompute_matches_forward_cost():
    """Data-free: the adjoint (uncompute) preparation costs exactly the same as the forward."""
    for nb, ctrl in [(1, 0), (4, 1)]:
        fwd = _cost(T.from_bitsize(16, 8, n_blocks=nb, control_bitsize=ctrl, uncompute=False))
        adj = _cost(T.from_bitsize(16, 8, n_blocks=nb, control_bitsize=ctrl, uncompute=True))
        assert fwd == adj, (nb, ctrl)


def test_extended_config_matrix_composite_equals_call_graph():
    import itertools

    for nb, ctrl, unc, mr in itertools.product([1, 4], [0, 1], [False, True], [False, True]):
        b = T.from_bitsize(16, 8, n_blocks=nb, control_bitsize=ctrl, uncompute=unc, measure_reset=mr)
        assert _cost(b) == _cost(b.decompose_bloq()), (nb, ctrl, unc, mr)


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
