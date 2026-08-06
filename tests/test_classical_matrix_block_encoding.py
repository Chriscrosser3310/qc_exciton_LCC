"""Tests for ``ClassicalMatrixBlockEncoding`` (Clader et al., arXiv:2206.03505).

These verify the minimal-T-count Frobenius block-encoding ``U_A = U_R^dag U_L``:

* the mathematical identity ``<0,j| U_A |0,k> = A_jk / ||A||_F`` (incl. zero-padding of
  non-power-of-two matrices), checked at the matrix level on the coefficient data;
* the standard ``BlockEncoding`` interface (registers, alpha = ||A||_F, signal state);
* call-graph structure (exactly U_L + swap + U_R^dag) and a single-qubit controlled
  variant whose Toffoli/T overhead over the uncontrolled bloq is negligible;
* data-free resource estimates and the data-free decomposition guard.
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

from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount

from integrations.qualtran.block_state_preparation_QROAM import (
    BlockStatePreparationViaQROAMRotations,
)
from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    _block_coeffs_from_matrices,
    _coeffs_from_matrix,
    BlockDiagonalClassicalMatrixBlockEncoding,
    ClassicalMatrixBlockEncoding,
    DirectHermitianBlockEncoding,
    HermitianOffDiagonalBlockEncoding,
)
from integrations.qualtran.state_prep_QROAM import StatePreparationViaQROAMRotations


def _gate_costs(bloq):
    return get_cost_value(bloq, QECGatesCost())


# --------------------------- Mathematical correctness ---------------------------


def _reconstruct(A: np.ndarray) -> tuple:
    """Return ``(U_A_topleft * alpha, A_padded)`` for the ``U_A = U_L^dag U_R`` construction.

    ``from_matrix`` builds the coefficient tables from ``A^T`` so that
    ``<0,j| U_L^dag U_R |0,l> = phi_l * psi_{l,j} = A_{j,l} / ||A||_F`` (real case),
    i.e. the encoded operator is ``A`` itself.
    """
    psi, phi, alpha, N = _coeffs_from_matrix(np.asarray(A).T)
    phi = np.asarray(phi)
    M = (phi[None, :] * psi.T).real  # M[j, l] = phi[l] * psi[l, j]
    Apad = np.zeros((N, N))
    Apad[: A.shape[0], : A.shape[1]] = A
    return M * alpha, Apad


def test_identity_square_real():
    rng = np.random.default_rng(0)
    for n in (2, 4, 8, 16):
        A = rng.standard_normal((n, n))
        recon, Apad = _reconstruct(A)
        assert np.allclose(recon, Apad, atol=1e-12)


def test_identity_nonsquare_padding():
    rng = np.random.default_rng(1)
    for shape in [(3, 5), (5, 3), (6, 6), (1, 4), (7, 2)]:
        A = rng.standard_normal(shape)
        recon, Apad = _reconstruct(A)
        # padded to a power of two >= max(rows, cols)
        N = recon.shape[0]
        assert N == (1 << (max(shape) - 1).bit_length())
        assert np.allclose(recon, Apad, atol=1e-12)


def test_alpha_is_frobenius_norm():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 4))
    be = ClassicalMatrixBlockEncoding.from_matrix(A, phase_bitsize=12)
    assert np.isclose(be.alpha, np.linalg.norm(A))


def test_zero_matrix_rejected():
    with pytest.raises(ValueError):
        ClassicalMatrixBlockEncoding.from_matrix(np.zeros((4, 4)), phase_bitsize=8)


# ------------------------------ BlockEncoding interface -------------------------


def test_signature_and_bitsizes():
    be = ClassicalMatrixBlockEncoding.from_bitsize(n_rows=256, phase_bitsize=20)
    names = [r.name for r in be.signature]
    assert names == ["system", "ancilla", "resource"]
    assert be.system_bitsize == 8  # log2(256)
    assert be.ancilla_bitsize == 8
    assert be.resource_bitsize == 20


def test_from_bitsize_rejects_non_pow2():
    with pytest.raises(ValueError):
        ClassicalMatrixBlockEncoding.from_bitsize(n_rows=100, phase_bitsize=10)


# ------------------------------- Call-graph structure ---------------------------


def test_call_graph_is_two_preps_and_a_swap():
    # U_A = U_L^dag U_R: the block prep U_R runs FORWARD (uncompute=False) with
    # measurement-only QROAM reset; the |phi> prep U_L^dag is the coherent adjoint.
    be = ClassicalMatrixBlockEncoding.from_bitsize(n_rows=64, phase_bitsize=16)
    cg = be.build_call_graph(None)
    kinds = {type(k).__name__ for k in cg}
    assert StatePreparationViaQROAMRotations.__name__ in kinds  # U_L^dag
    assert BlockStatePreparationViaQROAMRotations.__name__ in kinds  # U_R
    for k in cg:
        if isinstance(k, BlockStatePreparationViaQROAMRotations):
            assert k.uncompute is False
            assert k.measure_reset is True
        if isinstance(k, StatePreparationViaQROAMRotations):
            assert k.uncompute is True


def test_data_free_resource_counts():
    # Data-free (shape-concrete) bloqs decompose into the call-graph wiring, so both
    # qubit count and T-count are available without concrete matrix data.
    be = ClassicalMatrixBlockEncoding.from_bitsize(n_rows=16, phase_bitsize=10)
    cb = be.decompose_bloq()
    assert len(cb.bloq_instances) == 3
    assert _gate_costs(be).total_t_count() > 0
    assert get_cost_value(be, QubitCount()) > be.signature.n_qubits()


def test_optimal_T_trades_qubits_for_T():
    # T-optimal must spend fewer T gates but more qubits than the qubit-optimal regime.
    bt = ClassicalMatrixBlockEncoding.from_bitsize(256, 32, optimal_T=True)
    bq = ClassicalMatrixBlockEncoding.from_bitsize(256, 32, optimal_T=False)
    t_topt = _gate_costs(bt).total_t_count()
    t_qopt = _gate_costs(bq).total_t_count()
    q_topt = get_cost_value(bt, QubitCount())
    q_qopt = get_cost_value(bq, QubitCount())
    assert t_topt < t_qopt
    assert q_topt > q_qopt


def test_optimal_T_row_prep_is_toffoli_minimal():
    """The T-optimal forward U_R must batch its dominant N x N phase QROAM: its Toffoli
    count must be <= every uniform select-swap (log_k_block, log_k_row) choice (all with the
    same forward measurement-reset).  Guards against regressing to the per-layer heuristic,
    which left the phase table un-batched (O(N^2) Toffoli)."""
    from integrations.qualtran.block_state_preparation_QROAM import (
        BlockStatePreparationViaQROAMRotations as BSP,
    )

    N, b, n = 256, 32, 8
    opt = ClassicalMatrixBlockEncoding.from_bitsize(N, b, optimal_T=True)._prep_psi_fwd()
    opt_T = _gate_costs(opt).total_t_count()
    for lbb in range(0, n + 1):
        for lbr in range(0, n + 1):
            t = (lbb, lbr)
            try:
                cand = BSP.from_bitsize(
                    N, N, b, uncompute=False,
                    amp_log_block_sizes=t, amp_adjoint_log_block_sizes=t,
                    phase_log_block_sizes=t, phase_adjoint_log_block_sizes=t,
                    per_layer_optimal=False, measure_reset=True,
                )
                cand_T = _gate_costs(cand).total_t_count()
            except Exception:  # noqa: BLE001
                continue
            assert opt_T <= cand_T + 1e-9, f"optimal_T not minimal: {opt_T} > {cand_T} at {t}"


def test_data_bearing_decomposes():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((4, 4))
    be = ClassicalMatrixBlockEncoding.from_matrix(A, phase_bitsize=10)
    cb = be.decompose_bloq()
    assert len(cb.bloq_instances) == 3  # U_L, swap, U_R^dag


# --------------------------------- Controlled variant ---------------------------


def test_controlled_signature_adds_ctrl():
    be = ClassicalMatrixBlockEncoding.from_bitsize(n_rows=64, phase_bitsize=16)
    cbe = be.controlled()
    names = [r.name for r in cbe.signature]
    assert names == ["ctrl", "system", "ancilla", "resource"]


def test_controlled_overhead_is_small():
    be = ClassicalMatrixBlockEncoding.from_bitsize(n_rows=256, phase_bitsize=20)
    cbe = be.controlled()
    base = _gate_costs(be)
    ctrl = _gate_costs(cbe)
    base_tof = base.toffoli + base.and_bloq
    ctrl_tof = ctrl.toffoli + ctrl.and_bloq
    # The control only thickens the rotation-bearing sub-bloqs; overhead must be tiny.
    assert ctrl_tof >= base_tof
    assert ctrl_tof <= base_tof * 1.01 + 1000


def test_controlled_data_bearing_decomposes():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((8, 8))
    cbe = ClassicalMatrixBlockEncoding.from_matrix(A, phase_bitsize=10).controlled()
    cb = cbe.decompose_bloq()
    assert len(cb.bloq_instances) == 3


# --------------------------------- Resource sanity ------------------------------


def test_resource_counts_grow_with_N():
    costs = []
    for n_rows in (64, 128, 256):
        be = ClassicalMatrixBlockEncoding.from_bitsize(n_rows=n_rows, phase_bitsize=20)
        costs.append(_gate_costs(be).total_t_count())
    assert costs[0] < costs[1] < costs[2]


# =============================================================================
# Block-diagonal version:  A_block = sum_k |k><k| (x) A_k / F_max
# =============================================================================


def _reconstruct_block(mats):
    """Return (U*F_max per block, [A_k padded], F_max) for the ``U_A = U_L^dag U_R`` form.

    ``from_matrices`` builds the tables from ``A_k^T``, so per block k
    ``<0,(k,j)| U |0,(k,l)> = (F_k/F_max) phi_{k,l} psi_{k,l,j} = (A_k)_{j,l}/F_max``."""
    flag, phi, psi, Fmax, N, Kp = _block_coeffs_from_matrices([np.asarray(m).T for m in mats])
    recon, padded = [], []
    for k in range(Kp):
        ck = flag[k, 0].real
        # M[j, l] = c_k * phi[k, l] * psi[k*N + l, j]
        M = ck * (phi[k][None, :].real * psi[k * N:(k + 1) * N].real.T)
        recon.append(M * Fmax)
        ap = np.zeros((N, N))
        if k < len(mats):
            ap[: mats[k].shape[0], : mats[k].shape[1]] = mats[k]
        padded.append(ap)
    return recon, padded, Fmax, Kp


def test_block_identity_equal_norm_blocks():
    rng = np.random.default_rng(10)
    mats = [rng.standard_normal((4, 4)) for _ in range(3)]
    recon, padded, _, Kp = _reconstruct_block(mats)
    for k in range(Kp):
        assert np.allclose(recon[k], padded[k], atol=1e-12)


def test_block_identity_mixed_dims_and_norms():
    rng = np.random.default_rng(11)
    mats = [rng.standard_normal(s) * scale
            for s, scale in [((3, 5), 1.0), ((2, 2), 5.0), ((5, 5), 0.1), ((4, 1), 2.0)]]
    recon, padded, _, Kp = _reconstruct_block(mats)
    for k in range(Kp):
        assert np.allclose(recon[k], padded[k], atol=1e-12)


def test_block_padded_blocks_encode_zero():
    # K=3 -> K_pad=4: the appended block must encode the zero matrix.
    rng = np.random.default_rng(12)
    mats = [rng.standard_normal((4, 4)) for _ in range(3)]
    recon, _, _, Kp = _reconstruct_block(mats)
    assert Kp == 4
    assert np.allclose(recon[3], 0.0, atol=1e-12)


def test_block_alpha_is_max_frobenius():
    rng = np.random.default_rng(13)
    mats = [rng.standard_normal((4, 4)), 3.0 * rng.standard_normal((4, 4))]
    be = BlockDiagonalClassicalMatrixBlockEncoding.from_matrices(mats, phase_bitsize=12)
    fmax = max(np.linalg.norm(m) for m in mats)
    assert np.isclose(be.alpha, fmax)


def test_block_signature_and_bitsizes():
    be = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 256, 20)
    assert [r.name for r in be.signature] == ["system", "ancilla", "resource"]
    assert be.block_bitsize == 3           # log2(8)
    assert be.matrix_bitsize == 8          # log2(256)
    assert be.system_bitsize == 11         # k + l
    assert be.ancilla_bitsize == 9         # prep + flag
    assert be.resource_bitsize == 20


def test_block_single_block_reduces_to_dense():
    # K=1 block-diagonal encodes A_0 / ||A_0||_F, matching the single-matrix construction.
    rng = np.random.default_rng(14)
    A = rng.standard_normal((4, 4))
    recon, padded, fmax, _ = _reconstruct_block([A])
    assert np.isclose(fmax, np.linalg.norm(A))
    assert np.allclose(recon[0], padded[0], atol=1e-12)


def test_block_call_graph_has_flag_two_preps_and_swap():
    # The logical sub-bloqs are exactly: flag rotation + U_L + swap + U_R^dag.
    be = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 64, 16)
    cg = be.build_call_graph(None)
    assert sum(cg.values()) == 4
    n_block_preps = sum(
        v for k, v in cg.items()
        if isinstance(k, BlockStatePreparationViaQROAMRotations)
    )
    assert n_block_preps == 3  # flag + U_L + U_R^dag
    # U_R^dag is the only adjoint prep.
    assert sum(
        v for k, v in cg.items()
        if isinstance(k, BlockStatePreparationViaQROAMRotations) and k.uncompute
    ) == 1


def test_block_data_bearing_decomposes():
    rng = np.random.default_rng(15)
    mats = [rng.standard_normal((4, 4)) for _ in range(3)]
    be = BlockDiagonalClassicalMatrixBlockEncoding.from_matrices(mats, phase_bitsize=10)
    be.decompose_bloq()  # must not raise (includes Split/Join wiring bloqs)
    be.controlled().decompose_bloq()  # must not raise


def test_block_data_free_resource_counts():
    be = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 64, 16)
    be.decompose_bloq()  # data-free (shape-concrete) wiring decomposes
    assert _gate_costs(be).total_t_count() > 0
    assert get_cost_value(be, QubitCount()) > be.signature.n_qubits()


def test_block_optimal_T_trades_qubits_for_T():
    bt = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 256, 32, optimal_T=True)
    bq = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 256, 32, optimal_T=False)
    assert _gate_costs(bt).total_t_count() < _gate_costs(bq).total_t_count()
    assert get_cost_value(bt, QubitCount()) > get_cost_value(bq, QubitCount())


def test_block_controlled_overhead_is_small():
    be = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 256, 32, optimal_T=True)
    cbe = be.controlled()
    assert [r.name for r in cbe.signature] == ["ctrl", "system", "ancilla", "resource"]
    base = _gate_costs(be).toffoli + _gate_costs(be).and_bloq
    ctrl = _gate_costs(cbe).toffoli + _gate_costs(cbe).and_bloq
    assert base <= ctrl <= base * 1.01 + 1000


# =============================================================================
# Off-diagonal Hermitian block-encoding via S = [[0, U], [U^dag, 0]]  (App. A 1)
# =============================================================================


def test_hermitian_dilation_S_algebra():
    """Pin the construction: for ANY block-encoding U of A, the off-diagonal dilation
    S = [[0, U], [U^dag, 0]] block-encodes [[0, A], [A^dag, 0]] at the same alpha
    (projecting only U's ancilla).  Verified on an explicit (cosine-sine) dilation U."""
    from scipy.linalg import sqrtm

    rng = np.random.default_rng(1)
    for n in (2, 3):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        alpha = 3.0 * np.linalg.norm(A, 2)
        B = A / alpha
        T1 = sqrtm(np.eye(n) - B @ B.conj().T)
        T2 = sqrtm(np.eye(n) - B.conj().T @ B)
        U = np.block([[B, T1], [T2, -B.conj().T]])         # unitary, top-left = A/alpha
        assert np.allclose(U.conj().T @ U, np.eye(2 * n))
        blkU = U[:n, :n]
        blkUd = U.conj().T[:n, :n]
        M_S = np.block([[np.zeros((n, n)), blkU], [blkUd, np.zeros((n, n))]])
        Abar = np.block([[np.zeros((n, n)), A / alpha], [(A / alpha).conj().T, np.zeros((n, n))]])
        assert np.allclose(M_S, Abar)                      # S block-encodes [[0,A],[A^dag,0]]
        assert np.allclose(Abar, Abar.conj().T)            # Hermitian


def test_hermitian_signature_and_alpha():
    inner = ClassicalMatrixBlockEncoding.from_bitsize(256, 32, optimal_T=True)
    be = HermitianOffDiagonalBlockEncoding.from_inner(inner)
    assert [r.name for r in be.signature] == ["system", "ancilla", "resource"]
    assert be.system_bitsize == inner.system_bitsize + 1   # + off-diagonal index qubit
    assert be.ancilla_bitsize == inner.ancilla_bitsize     # e lives in the system, NOT ancilla
    assert be.resource_bitsize == inner.resource_bitsize
    assert be.alpha == inner.alpha                          # alpha = ||A||_F (not 2||A||_F)


def test_hermitian_from_matrix_alpha_is_frobenius():
    rng = np.random.default_rng(22)
    A = rng.standard_normal((8, 4))
    be = HermitianOffDiagonalBlockEncoding.from_matrix(A, phase_bitsize=12)
    assert np.isclose(be.alpha, np.linalg.norm(A))


def test_hermitian_cost_is_two_inner_calls():
    inner = ClassicalMatrixBlockEncoding.from_bitsize(256, 32, optimal_T=True)
    be = HermitianOffDiagonalBlockEncoding.from_inner(inner)
    t_inner = _gate_costs(inner).total_t_count()
    t_be = _gate_costs(be).total_t_count()
    # S = C[U] + C[U^dag] + X  ->  ~2x inner, never below 2x-inner
    assert 1.95 * t_inner <= t_be <= 2.1 * t_inner + 100


def test_hermitian_cheaper_than_doubled_space():
    # The S-construction (reuse inner twice, ~2x) must beat a fresh block-encoding over the
    # doubled 2N-dim space (the P^dag SWAP P route), which costs ~that of a 2N x 2N matrix.
    N, b = 256, 32
    s_cost = _gate_costs(
        HermitianOffDiagonalBlockEncoding.from_bitsize(N, N, b, optimal_T=True)
    ).total_t_count()
    doubled = _gate_costs(
        ClassicalMatrixBlockEncoding.from_bitsize(2 * N, b, optimal_T=True)
    ).total_t_count()
    assert s_cost < doubled


def test_hermitian_wraps_any_inner():
    for inner in (
        ClassicalMatrixBlockEncoding.from_bitsize(64, 16, optimal_T=True),
        BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 64, 16, optimal_T=True),
    ):
        be = HermitianOffDiagonalBlockEncoding.from_inner(inner)
        assert be.system_bitsize == inner.system_bitsize + 1
        assert be.ancilla_bitsize == inner.ancilla_bitsize
        assert _gate_costs(be).total_t_count() >= 2 * _gate_costs(inner).total_t_count() * 0.95


def test_hermitian_controlled_works():
    be = HermitianOffDiagonalBlockEncoding.from_bitsize(64, 64, 16, optimal_T=True)
    cbe = be.controlled()
    assert "ctrl" in [r.name for r in cbe.signature]
    assert _gate_costs(cbe).total_t_count() > 0


# =============================================================================
# Direct Hermitian-unitary block-encoding:  W = (H (x) I) S (H (x) I)
# =============================================================================


def test_direct_hermitian_signature_and_alpha():
    inner = ClassicalMatrixBlockEncoding.from_bitsize(256, 32, optimal_T=True)
    w = DirectHermitianBlockEncoding(inner)
    assert [r.name for r in w.signature] == ["system", "ancilla", "resource"]
    assert w.system_bitsize == inner.system_bitsize
    assert w.ancilla_bitsize == inner.ancilla_bitsize + 1   # one extra Hermitian-flag qubit
    assert w.resource_bitsize == inner.resource_bitsize
    assert w.alpha == inner.alpha                            # subnormalization preserved


def test_direct_hermitian_cost_is_two_inner_calls():
    inner = ClassicalMatrixBlockEncoding.from_bitsize(256, 32, optimal_T=True)
    w = DirectHermitianBlockEncoding(inner)
    t_inner = _gate_costs(inner).total_t_count()
    t_w = _gate_costs(w).total_t_count()
    # W = 2 controlled inner calls (C[U], C[U^dag]) + Cliffords -> ~2x, never below 2x-inner
    assert 1.95 * t_inner <= t_w <= 2.1 * t_inner + 100
    # one extra qubit beyond the (cheap) controlled inner
    assert get_cost_value(w, QubitCount()) <= get_cost_value(inner.controlled(), QubitCount()) + 1


def test_direct_hermitian_wraps_any_block_encoding():
    # Works as a generic wrapper over the other constructions in this module.
    for inner in (
        ClassicalMatrixBlockEncoding.from_bitsize(64, 16, optimal_T=True),
        BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(8, 64, 16, optimal_T=True),
        HermitianOffDiagonalBlockEncoding.from_bitsize(64, 64, 16, optimal_T=True),
    ):
        w = DirectHermitianBlockEncoding(inner)
        assert w.ancilla_bitsize == inner.ancilla_bitsize + 1
        assert _gate_costs(w).total_t_count() >= _gate_costs(inner).total_t_count()


if __name__ == "__main__":
    failed = 0
    tests = [
        (name, fn)
        for name, fn in dict(globals()).items()
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
