"""Tests for the recursive-CSD (block-)unitary synthesizer with Eq.-24 / Eq.-40 phase lookups.

Pins both the *algorithm* and the *resource model* of
``integrations.qualtran.recursive_csd_synthesis_QROAM``:

Algorithm (numpy reference)
  * :func:`recursive_csd_rotations` decomposes an arbitrary ``U(2^n)`` into a flat sequence of
    uniformly-controlled one-qubit gates that reconstruct ``U`` to machine precision.
  * Berry Eq. 24 (:func:`csd_phase_words`) reconstructs every one-qubit gate of the decomposition.
  * The structural schedule matches the closed-form multiplexed-gate counts.

Resource model (qualtran bloqs)
  * ``build_composite_bloq`` (real layout, shape-only QROAM) and ``build_call_graph`` report identical
    ``QECGatesCost``.
  * Signature ``block, system, phase_gradient``; block register is a clean per-block QROAM address.
  * Controlled variant adds exactly one qubit with modest overhead.
  * ``optimal_T`` never increases cost; the analytic estimate's lookup/reconstruction split tracks the
    internal block size ``M`` while the total stays invariant.
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

from collections import Counter

import numpy as np

qualtran = pytest.importorskip("qualtran")

from qualtran import DecomposeTypeError
from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_isometry_column_synthesis_QROAM import eq24_reconstruct
from integrations.qualtran.recursive_csd_synthesis_QROAM import (
    CSDMuxPhaseLayerQROAM,
    MultiControlledKQubitUnitaryQROAM,
    PaperRecursiveCSDUnitarySynthesis,
    RecursiveCSDSynthesisQROAM,
    constructed_paper_unitary_tcount,
    csd_mux_schedule,
    csd_phase_words,
    estimate_paper_unitary_resources,
    estimate_recursive_csd_resources,
    num_csd_muxes,
    optimal_constructed_paper_unitary,
    paper_block_tcount,
    recursive_csd_rotations,
    reconstruct_from_ops,
    total_csd_one_qubit_gates,
)

R = RecursiveCSDSynthesisQROAM


def _haar_unitary(n: int, rng) -> np.ndarray:
    N = 1 << n
    X = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    Q, _ = np.linalg.qr(X)
    return Q @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, N)))


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost())


# ---------------------------------------------------------------------------
# Algorithm: recursive CSD + Eq. 24
# ---------------------------------------------------------------------------


def test_recursive_csd_reconstructs_unitary():
    """The flattened uniformly-controlled one-qubit rotations rebuild U(2^n) to machine precision."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in range(1, 6):
        for real in (False, True):
            for _ in range(3):
                if real:
                    X = rng.standard_normal((1 << n, 1 << n))
                    U, _ = np.linalg.qr(X)
                    U = U.astype(complex)
                else:
                    U = _haar_unitary(n, rng)
                ops = recursive_csd_rotations(U)
                worst = max(worst, float(np.linalg.norm(reconstruct_from_ops(ops, n) - U)))
    assert worst < 1e-9, worst


def test_eq24_words_reconstruct_every_one_qubit_gate():
    """Berry Eq. 24 (4 phase words) reproduces every 2x2 gate of the decomposition."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in range(2, 5):
        U = _haar_unitary(n, rng)
        for op in recursive_csd_rotations(U):
            words = csd_phase_words(op, phase_words=4)
            for j, gate in enumerate(op.gates):
                phi, theta, phi0, phi1 = words[j]
                worst = max(worst, float(np.linalg.norm(eq24_reconstruct(phi0, phi1, theta, phi) - gate)))
    assert worst < 1e-9, worst


def test_mux_schedule_matches_closed_form():
    """Structural schedule tally == num_csd_muxes(n, c); total ops == 2*4^(n-1) - 1."""
    for n in range(1, 7):
        tally = Counter(len(controls) for _, controls in csd_mux_schedule(n))
        for c in range(n):
            assert tally.get(c, 0) == num_csd_muxes(n, c), (n, c)
        assert sum(tally.values()) == 2 * 4 ** (n - 1) - 1, n


def test_total_one_qubit_gate_count():
    """Total one-qubit unitaries == N^2 - 3N/2 (generic-unitary parameter scaling)."""
    for n in range(1, 7):
        N = 1 << n
        assert total_csd_one_qubit_gates(n) == N * N - 3 * N // 2, n


def test_block_register_is_clean_address():
    """The (target, controls) schedule depends only on n, so the block label a is a pure QROAM
    address shared across every block U_a of a block-diagonal unitary."""
    rng = np.random.default_rng(2)
    n = 3
    schedules = []
    for _ in range(4):  # several independent U_a blocks
        U = _haar_unitary(n, rng)
        ops = recursive_csd_rotations(U)
        schedules.append([(op.target, op.controls) for op in ops])
    assert all(s == schedules[0] for s in schedules)
    assert schedules[0] == csd_mux_schedule(n)


# ---------------------------------------------------------------------------
# Resource model: bloq structure / cost
# ---------------------------------------------------------------------------


def test_composite_equals_call_graph():
    cases = [
        (1, 2, 6, None, 4, False),
        (1, 3, 6, 1, 4, False),
        (4, 3, 6, 2, 4, True),
        (1, 4, 8, 2, 4, False),
        (2, 3, 6, 0, 2, True),
        (8, 3, 8, 1, 4, True),
    ]
    for nb, n, b, m, pw, opt in cases:
        bloq = R.from_shape(nb, 1 << n, b, block_log_size=m, phase_words=pw, optimal_T=opt)
        cg = _cost(bloq)
        comp = _cost(bloq.decompose_bloq())
        assert cg == comp, (nb, n, b, m, pw, opt, cg, comp)


def test_signature_layout():
    bloq = R.from_shape(8, 16, 10, block_log_size=2)
    assert [r.name for r in bloq.signature] == ["block", "system", "phase_gradient"]
    assert bloq.signature.n_qubits() == bloq.block_bitsize + bloq.system_bitsize + 10


def test_controlled_variant():
    for nb, n, b in [(1, 4, 8), (8, 3, 8)]:
        bare = R.from_shape(nb, 1 << n, b)
        ctl = bare.controlled()
        assert ctl.signature.n_qubits() == bare.signature.n_qubits() + 1
        t_bare = int(_cost(bare).total_t_count())
        t_ctl = int(_cost(ctl).total_t_count())
        assert t_bare <= t_ctl <= 2 * t_bare, (nb, n, t_bare, t_ctl)


def test_phase_layer_leaf_has_no_lookup():
    """An ordinary (n_blocks=1) zero-control leaf is a fixed gate: no QROAM, only the Eq.-40 adds."""
    leaf = CSDMuxPhaseLayerQROAM(n_blocks=1, n_controls=0, phase_bitsize=8, phase_words=4)
    assert not leaf.has_qroam
    # With a block label the same leaf becomes addressable -> it does load from QROAM.
    block_leaf = CSDMuxPhaseLayerQROAM(n_blocks=8, n_controls=0, phase_bitsize=8, phase_words=4)
    assert block_leaf.has_qroam


def test_block_costs_more_than_ordinary():
    """Appending an N_k block address enlarges every QROAM table, so cost grows with n_blocks."""
    t1 = int(_cost(R.from_shape(1, 16, 10, optimal_T=True)).total_t_count())
    t8 = int(_cost(R.from_shape(8, 16, 10, optimal_T=True)).total_t_count())
    assert t8 > t1


def test_optimal_T_never_worse():
    for nb, n, b in [(1, 4, 10), (8, 5, 16), (64, 6, 8)]:
        d = int(_cost(R.from_shape(nb, 1 << n, b, optimal_T=False)).total_t_count())
        o = int(_cost(R.from_shape(nb, 1 << n, b, optimal_T=True)).total_t_count())
        assert o <= d, (nb, n, b, d, o)
    # A large table must show a strict reduction (Lambda > 1 helps).
    d = int(_cost(R.from_shape(64, 64, 8, optimal_T=False)).total_t_count())
    o = int(_cost(R.from_shape(64, 64, 8, optimal_T=True)).total_t_count())
    assert o < d, (d, o)


def test_symbolic_refuses_to_decompose():
    import sympy

    N = sympy.Symbol("N", positive=True, integer=True)
    bloq = R.from_shape(1, N, 8)
    with pytest.raises(DecomposeTypeError):
        bloq.build_call_graph(None)
    with pytest.raises(ValueError):
        R.from_shape(1, N, 8, optimal_T=True)


# ---------------------------------------------------------------------------
# Resource model: analytic estimate
# ---------------------------------------------------------------------------


def test_estimate_total_invariant_to_block_size():
    """M only moves cost between the lookup and reconstruction buckets; the total is invariant."""
    n = 4
    totals = set()
    for m in range(0, n + 1):
        e = estimate_recursive_csd_resources(4, 1 << n, 10, block_log_size=m, optimal_T=True)
        assert e.total_toffoli == e.lookup_toffoli + e.reconstruction_toffoli
        totals.add(e.total_toffoli)
    assert len(totals) == 1, totals


def test_estimate_lookup_grows_as_block_shrinks():
    """Smaller M (smaller m) promotes more levels to stable lookups -> larger lookup share."""
    n = 4
    lookups = [
        estimate_recursive_csd_resources(4, 1 << n, 10, block_log_size=m, optimal_T=True).lookup_toffoli
        for m in range(0, n + 1)
    ]
    assert lookups == sorted(lookups, reverse=True), lookups
    assert lookups[0] > 0 and lookups[-1] == 0


def test_estimate_word_hadamard_counts():
    """#loaded words = phase_words * n_blocks * (#one-qubit gates); #H = 2 * #muxes."""
    for n in range(2, 6):
        N = 1 << n
        e = estimate_recursive_csd_resources(1, N, 8, phase_words=4)
        assert e.n_loaded_phase_words == 4 * (N * N - 3 * N // 2)
        n_muxes = 2 * 4 ** (n - 1) - 1
        assert e.n_hadamards == 2 * n_muxes
        assert e.n_phase_additions == 4 * n_muxes


def test_estimate_block_scaling_of_words():
    """The block label multiplies the loaded-word count by n_blocks."""
    e1 = estimate_recursive_csd_resources(1, 16, 8)
    e8 = estimate_recursive_csd_resources(8, 16, 8)
    assert e8.n_loaded_phase_words == 8 * e1.n_loaded_phase_words


def test_resource_estimate_method_matches_function():
    bloq = R.from_shape(8, 16, 12, block_log_size=2, optimal_T=True)
    e_method = bloq.resource_estimate()
    e_func = estimate_recursive_csd_resources(8, 16, 12, block_log_size=2, optimal_T=True)
    assert e_method == e_func


# ---------------------------------------------------------------------------
# Paper-faithful (grouped) estimate: the N^{4/3} route (arXiv:2509.25702)
# ---------------------------------------------------------------------------


def test_paper_estimate_block_structure():
    """2^{n-k} grouped blocks; total T = n_blocks * per-block (Thm 4.3)."""
    for n in (4, 8, 12):
        e = estimate_paper_unitary_resources(n, 32)
        L = n + 32
        assert e.n_blocks == 2 ** (n - e.k)
        assert abs(e.t_per_block - paper_block_tcount(n, e.k, L)) < 1e-6 * e.t_per_block
        assert abs(e.t_count - e.n_blocks * e.t_per_block) < 1e-6 * e.t_count


def test_paper_estimate_optimal_k_minimizes():
    """The chosen k is the integer minimizer of the total T-count over [1, n-1]."""
    for n in (6, 12, 20, 30):
        e = estimate_paper_unitary_resources(n, 32)
        best = min(
            range(1, n),
            key=lambda k: estimate_paper_unitary_resources(n, 32, k=k).t_count,
        )
        assert e.k == best, (n, e.k, best)


def test_paper_estimate_scaling_approaches_four_thirds():
    """The local exponent of the optimal T-count climbs toward 4/3 (the paper's bound)."""
    ns = list(range(20, 41))
    Ts = [estimate_paper_unitary_resources(n, 32).t_count for n in ns]
    logN = np.array(ns, float) * np.log(2.0)
    local = np.diff(np.log(np.array(Ts, float))) / np.diff(logN)
    # In this window the local exponent sits in a tight band around 4/3, well below the
    # previous-best 3/2 and far below the naive 2.
    assert 1.30 <= float(np.mean(local)) <= 1.38, float(np.mean(local))
    assert max(local) < 1.45


def test_paper_optimal_k_tracks_n_over_three():
    """Asymptotically the optimal block size grows like n/3 (within an integer-rounding band)."""
    for n in (24, 30, 36, 40):
        e = estimate_paper_unitary_resources(n, 32)
        assert abs(e.k - n / 3) <= 3, (n, e.k)


def test_paper_route_far_cheaper_than_naive_quadratic():
    """At large N the grouped route is orders of magnitude below an N^2 reference."""
    n = 10
    e = estimate_paper_unitary_resources(n, 32)
    e0 = estimate_paper_unitary_resources(2, 32)
    quadratic_ref = e0.t_count * (e.N / e0.N) ** 2  # anchored N^2 growth from N=4
    assert e.t_count < 0.2 * quadratic_ref, (e.t_count, quadratic_ref)


# ---------------------------------------------------------------------------
# CONSTRUCTED circuit: the scaling must come from real gate counting, not a formula
# ---------------------------------------------------------------------------


def test_constructed_signature():
    bloq = PaperRecursiveCSDUnitarySynthesis(1 << 6, 32, 2)
    assert [r.name for r in bloq.signature] == ["system", "phase_gradient"]
    assert bloq.n_block_ops == 1 << (6 - 2)


def test_constructed_composite_equals_call_graph():
    """The top-level constructed circuit's composite and call graph cost identically."""
    for n, k in [(4, 1), (6, 2), (8, 1), (8, 3), (10, 2)]:
        bloq = PaperRecursiveCSDUnitarySynthesis(1 << n, 32, k, optimal_T=True)
        cg = _cost(bloq)
        comp = _cost(bloq.decompose_bloq())
        assert cg == comp, (n, k, cg, comp)


def test_constructed_block_cost_matches_theorem_4_3():
    """The constructed block's real Toffoli count tracks 2^{(n+k)/2}sqrt(L)+4^k L (within O(1))."""
    L = 42
    for n, k in [(8, 1), (8, 2), (10, 1), (10, 3)]:
        blk = MultiControlledKQubitUnitaryQROAM(
            n_controls=n - k, k=k, phase_bitsize=L, optimal_T=True
        )
        gc = _cost(blk)
        real = int(gc.toffoli + gc.and_bloq)
        formula = paper_block_tcount(n, k, L)
        assert 0.5 <= real / formula <= 3.0, (n, k, real, formula)


def test_constructed_scaling_is_subquadratic_toward_four_thirds():
    """The constructed optimal T-count fits an exponent well below 2 and trending to 4/3."""
    Ns, Ts = [], []
    for n in range(2, 11):
        k, T = optimal_constructed_paper_unitary(n, 32)
        Ns.append(1 << n)
        Ts.append(T)
    alpha = float(np.polyfit(np.log(Ns), np.log(Ts), 1)[0])
    assert 1.1 < alpha < 1.45, alpha  # sub-quadratic; approaches 4/3 from below in this range
    # tail (N >= 32) sits closer to 4/3 than the full-range fit
    tail = float(np.polyfit(np.log(Ns[3:]), np.log(Ts[3:]), 1)[0])
    assert tail > alpha


def test_constructed_optimal_k_grows_with_n():
    """The optimal block size k* increases with n (tracking the asymptotic n/3)."""
    ks = [optimal_constructed_paper_unitary(n, 32)[0] for n in (6, 10, 14)]
    assert ks[0] <= ks[1] <= ks[2] and ks[-1] > ks[0], ks


def test_block_diagonal_construction():
    """Block-diagonal U_a synthesis: block label appended to every QROAM; composite == call graph,
    cost grows with N_k (but < N_k * single, since reconstruction does not grow), exponent unchanged."""
    # composite == call graph with the block register present
    for n, k, nb in [(4, 1, 8), (6, 2, 216), (8, 1, 216)]:
        bloq = PaperRecursiveCSDUnitarySynthesis(1 << n, 32, k, n_blocks=nb, optimal_T=True)
        assert [r.name for r in bloq.signature] == ["block", "system", "phase_gradient"]
        assert _cost(bloq) == _cost(bloq.decompose_bloq()), (n, k, nb)

    # 216 blocks costs more than a single unitary, but far less than 216x (reconstruction shared,
    # only the lookup grows ~sqrt(N_k)); and the optimal k shifts up for the block case.
    Ns, e1 = [], []
    for n in range(4, 11):
        _, t1 = optimal_constructed_paper_unitary(n, 32, n_blocks=1)
        k2, t216 = optimal_constructed_paper_unitary(n, 32, n_blocks=216)
        assert t1 < t216 < 216 * t1, (n, t1, t216)
        Ns.append(1 << n)
        e1.append(t216)
    # block-diagonal N-scaling exponent is sub-quadratic too (sqrt(N_k) is only a prefactor)
    alpha = float(np.polyfit(np.log(Ns), np.log(e1), 1)[0])
    assert 1.1 < alpha < 1.5, alpha


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
