r"""Column-by-column isometry synthesis (Iten et al.) with Berry et al. Eq.-24 QROAM layers.

This synthesizes an ``m -> n`` isometry (a ``2^n x K`` matrix with orthonormal columns,
``K <= N = 2^n``) using the **column-by-column** decomposition of

    Iten, Colbeck, Kukuljan, Home, Christandl,
    "Quantum circuits for isometries", arXiv:1501.06911 (Thm 2 / Lemma 12 / App. A 3, Fig. 4),

where each merged multiplexed single-qubit gate is realized via the efficient QROAM construction of

    Berry, Tong, Khattar, White, Kim, Boixo, Lin, Lee, Chan, Babbush, Rubin,
    "Rapid initial state preparation for the quantum simulation of strongly correlated molecules",
    arXiv:2409.11748 (Sec. III A, Eq. 24).

Skeleton (Iten Lemma 12, Fig. 4).  Write ``V = G^dagger I_{2^n x K}`` with ``G = G_{K-1} ... G_0``.
The column operation ``G_k`` maps column ``k`` to ``|k>`` while fixing columns ``0..k-1``:

    G_k = prod_{s=0}^{n-1} (Delta_s) . C^u_{n-1-s}(U^u_s) . C_{n-1}(U_s).

* ``C^u_{n-1-s}(U^u_s)`` is a uniformly-controlled single-qubit gate (UCG): target = system qubit
  ``q[n-1-s]``, controls = the ``n-1-s`` more-significant qubits, the ``s`` least-significant qubits free.
  It disentangles one qubit (Fig. 1 cascade).  Iten sets ``C^u_0 = I`` (the most-significant qubit is
  either already disentangled or handled by the multi-controlled gate ``C_{n-1}(U_{n-1})``), so the
  active UCG layers have ``c = n-1-s in {1, ..., n-1}``.
* ``Delta_s`` is a diagonal taken "up to a diagonal"; it is **merged into the UCG** (one multiplexed
  gate per step) and then pushed forward, so all basis-state phases collapse into a single final
  ``2^n``-entry diagonal phase layer.
* ``C_{n-1}(U_s)`` is a subleading multi-controlled single-qubit gate, used only when
  ``k_s = 0 and (k mod 2^{s+1}) != 0``; the total count is ``Q(m,n) = 2^m (n - m/2 - 1) - n + m + 1``
  (Cor. 1) for ``K = 2^m``, and ``sum_k Q_k(n)`` in general.

Per-gate realization (Berry Eq. 24).  Each single-qubit unitary is written as
``diag(e^{i phi0}, e^{i phi1}) H diag(e^{i theta}, 1) H diag(e^{i phi}, 1)``: control-independent
Hadamards on the target plus three per-control diagonals loaded from QROAM and applied through a phase
gradient.  Two phase tables are loaded from a single QROAM, QROAM erasure is done by X-basis
measurement (sign-fixups absorbed into the next layer, 0 Toffoli), and the QROAM block size ``Lambda``
is chosen for the Toffoli optimum (``optimal_T``).  This is exactly the layer already shipped as
``BlockInterferometerPhaseLayerQROAM`` (``R(alpha) H R(beta) H``), reused here with the controls being
the upper ``c`` qubits and the target ``q[n-1-c-... ]`` rather than always the least-significant qubit.

A read-only ``block`` register makes this block-diagonal (``sum_j |j><j| (x) V_j``), threaded into every
QROAM lookup exactly like the sibling ``BlockUnitary*QROAM`` bloqs; ``n_blocks = 1`` is the plain
single-isometry case.

The class is a *resource model*: ``build_composite_bloq`` lays out the real gate structure with
shape-only (data-free) QROAM, and ``build_call_graph`` gives the aggregate Toffoli/qubit cost.  The
module-level numpy helpers (:func:`eq24_angles`, :func:`column_by_column_disentangler`) provide a
standalone reference proving the algorithm reproduces the isometry (``G V = I_{2^n x K}``).
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .real_rotation_layers_QROAM import (
        RealMultiControlledRotationQROAM,
        RealPhaseLayerQROAM,
        RealSignLayerQROAM,
    )
    from .block_unitary_interferometer_QROAM import (
        BlockInterferometerFinalPhasesQROAM,
        BlockInterferometerPhaseLayerQROAM,
        _ControlledFinalPhasesQROAM,
        _ControlledPhaseLayerQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script/direct execution
    from real_rotation_layers_QROAM import (
        RealMultiControlledRotationQROAM,
        RealPhaseLayerQROAM,
        RealSignLayerQROAM,
    )
    from block_unitary_interferometer_QROAM import (
        BlockInterferometerFinalPhasesQROAM,
        BlockInterferometerPhaseLayerQROAM,
        _ControlledFinalPhasesQROAM,
        _ControlledPhaseLayerQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from state_prep_QROAM import _to_tuple_or_none

try:
    from .range_safe_qroam import emit_range_safety
except ImportError:  # pragma: no cover
    from range_safe_qroam import emit_range_safety

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


# ============================================================================
# numpy reference: Berry Eq. 24 and the Iten column-by-column disentangler
# ============================================================================

_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def eq24_reconstruct(phi0: float, phi1: float, theta: float, phi: float) -> NDArray[np.complex128]:
    r"""Rebuild the 2x2 unitary from Berry Eq. 24 angles.

    ``U = diag(e^{i phi0}, e^{i phi1}) H diag(e^{i theta}, 1) H diag(e^{i phi}, 1)``.
    """
    Dl = np.diag([np.exp(1j * phi0), np.exp(1j * phi1)])
    P = np.diag([np.exp(1j * theta), 1.0])
    Dr = np.diag([np.exp(1j * phi), 1.0])
    return Dl @ _H @ P @ _H @ Dr


def eq24_angles(U: NDArray[np.complex128]) -> Tuple[float, float, float, float]:
    r"""Return ``(phi0, phi1, theta, phi)`` decomposing the 2x2 unitary ``U`` per Berry Eq. 24.

    ``theta in [0, pi]`` is fixed by the entry magnitudes, ``phi`` by the cross-ratio, and
    ``phi0/phi1`` by phase-aligning the rows; a two-branch search over ``theta`` selects the
    consistent branch.  The result is self-verified to machine precision.
    """
    U = np.asarray(U, dtype=complex)
    theta_mag = 2.0 * np.arctan2(abs(U[0, 1]), abs(U[0, 0]))  # in [0, pi]

    def _solve(theta: float) -> Tuple[float, float, float, float, float]:
        eit = np.exp(1j * theta)
        num = U[0, 0] * U[1, 0]
        den = U[0, 1] * U[1, 1]
        if abs(den) > 1e-12 and abs(num) > 1e-12:
            phi = 0.5 * float(np.angle(num / den))
        else:
            phi = 0.0
        B = _H @ np.diag([eit, 1.0]) @ _H @ np.diag([np.exp(1j * phi), 1.0])

        def _row_phase(r: int) -> float:
            dot = np.vdot(B[r, :], U[r, :])  # <B_row | U_row>
            return 0.0 if abs(dot) < 1e-12 else float(np.angle(dot))

        phi0, phi1 = _row_phase(0), _row_phase(1)
        err = float(np.linalg.norm(eq24_reconstruct(phi0, phi1, theta, phi) - U))
        return (err, phi0, phi1, theta, phi)

    err, phi0, phi1, theta, phi = min(
        _solve(theta_mag), _solve(2.0 * np.pi - theta_mag), key=lambda t: t[0]
    )
    if err > 1e-7:  # pragma: no cover - defensive
        raise ValueError(f"eq24_angles failed to converge (err={err:.2e}) for U=\n{U}")
    return phi0, phi1, theta, phi


def lemma2_gate(c0: complex, c1: complex, target_bit: int) -> NDArray[np.complex128]:
    r"""Iten Lemma 2: an SU(2) ``U`` with ``U (c0, c1)^T = r |target_bit>``, ``r = |(c0,c1)| >= 0``."""
    v = np.array([c0, c1], dtype=complex)
    r = float(np.linalg.norm(v))
    if r < 1e-300:
        return np.eye(2, dtype=complex)
    p0, p1 = v / r
    if target_bit == 0:
        # U0 = |0><psi| + |1><phi|, |phi> = -conj(p1)|0> + conj(p0)|1>  =>  det = 1
        return np.array([[p0.conjugate(), p1.conjugate()], [-p1, p0]])
    return np.array([[-p1, p0], [p0.conjugate(), p1.conjugate()]])


def mcg_needed(k: int, s: int) -> bool:
    r"""Whether column op ``G_k`` uses a multi-controlled gate at step ``s`` (Iten Lemma 11/12).

    True iff ``k_s == 0`` and ``b^k_{s+1} = (k mod 2^{s+1}) != 0``.
    """
    ks = (k >> s) & 1
    b_s1 = k & ((1 << (s + 1)) - 1)
    return ks == 0 and b_s1 != 0


def column_qmn(k: int, n: int) -> int:
    """Number of multi-controlled gates ``Q_k(n)`` used by column op ``G_k`` (Iten Lemma 13)."""
    return sum(1 for s in range(n) if mcg_needed(k, s))


def tree_level_sizes(d: int) -> List[int]:
    r"""Live node count per level of the P-07 rotation tree over ``d`` leaves.

    Level ``s`` holds ``max(0, min(2^s, d - 2^s))`` nodes and the levels sum to ``d - 1``.
    The top level is truncated by ``d - 2^s``: for ``d = 130`` the sizes are
    ``[1, 2, 4, 8, 16, 32, 64, 2]``, not ``[1, 2, 4, ..., 128]``.  That truncation is the
    whole difference between charging ``~d`` and charging ``~2^ceil(log2 d)``.
    """
    out: List[int] = []
    s = 0
    while (1 << s) < int(d):
        out.append(max(0, min(1 << s, int(d) - (1 << s))))
        s += 1
    return out


def capped_adjoint_log_block_sizes(data_shape, word_bits: int):
    r"""Toffoli-optimal ``log_block_sizes`` for a measurement-based uncomputation, capped.

    The uncomputation converts the low address bits to a one-hot unary selector, applies
    the phase by a reduced lookup on the high bits, then erases the selector by measurement
    and Clifford feed-forward.  The selector is one bit per batch entry, so ``Lambda'``
    costs ``Lambda'`` qubits -- cheap, but not free.

    The bound is the WORD WIDTH, not ``2b`` universally.  At ``Lambda = 1`` the forward
    lookup holds a ``word_bits``-wide output register; the uncomputation measures it in the
    X basis, releasing those qubits, and only then builds the selector -- so the selector
    can host at most ``word_bits`` of them, and ``Lambda' <= word_bits`` keeps the peak
    width at what the forward lookup already needed.  For the two-angle phase layers
    ``word_bits = 2b``; for a single-angle table (the diagonal, an MCG angle load) it is
    ``b``, and passing ``2b`` there would claim twice the workspace that exists.

    ⚠ This is an assumption about qubit reuse, and the sources are not aligned on it.
    ``details.tex`` derives its ancilla formulas assuming ``Lambda' <= Lambda``, which at
    ``Lambda = 1`` forces ``Lambda' = 1``; the analytic model's Q-opt mode instead caps at
    a flat ``2b``.  The spread is large -- for a ``(216, 130)`` table at ``b = 20`` the
    erasure is 28,080 at ``Lambda' = 1``, 910 at ``Lambda' <= 2b``, and 347 unbounded --
    so which convention is right is worth settling rather than inheriting.

    ``Lambda' = 1`` is never right: it pays ``L`` Toffolis where ``min(2 sqrt(L), L/(2b))``
    is available inside the workspace already allocated.  For a ``(216, 130)`` table at
    ``b = 20`` that is 28,080 against 910.
    """
    shape = tuple(int(d) for d in data_shape)
    cap = max(1, int(word_bits))
    L = 1
    for d in shape:
        L *= d
    caps = [max(0, int(d).bit_length() - 1) for d in shape]
    best, bestcost = tuple(0 for _ in shape), None
    def rec(i, acc, lam):
        nonlocal best, bestcost
        if i == len(shape):
            c = -(-L // lam) + lam
            if bestcost is None or c < bestcost:
                best, bestcost = tuple(acc), c
            return
        for k in range(caps[i] + 1):
            if lam * (1 << k) > cap:
                break
            rec(i + 1, acc + [k], lam * (1 << k))
    rec(0, [], 1)
    return best


def sublayer_sizes(n_rows: int) -> List[int]:
    r"""Angle-table size of each real-rotation sublayer, phase-first scheme (author, 2026-08-23).

    Sublayer ``s = 1 .. n`` eliminates along bit ``s-1``: it is **multiplex-controlled** on the
    leading ``n-s`` qubits and **multi-controlled** on the trailing ``s-1`` qubits, held at
    ``f = c mod 2^(s-1)``.  Its table therefore has ``ceil(n_rows / 2^s)`` entries, addressed
    contiguously from 0 -- ``f`` never enters the address, only the multi-control.

    Entering sublayer ``s`` the live entries of column ``c`` are exactly those
    ``= f (mod 2^(s-1))``, about ``N/2^(s-1)`` of them; the sublayer pairs them up and kills
    one of each pair, leaving those ``= c (mod 2^s)``.  So the table is half the live set.

    Sum over sublayers is bracketed by ``N-1 <= sum <= N-1+n``, hitting ``N-1`` exactly for a
    power of two.  Verified: 135 for N=130, 209 for N=208, 255 for N=256.  Independent of the
    column, since the address criterion is ``base < N`` and never mentions ``c``.
    """
    N = int(n_rows)
    n = max(1, (N - 1).bit_length())
    return [-(-N // (1 << s)) for s in range(1, n + 1)]


def num_mcgs(n_rows: int, n_cols: int) -> int:
    """Total multi-controlled gates over all ``n_cols`` column operations (Iten Cor. 1 when K=2^m)."""
    n = int(round(np.log2(n_rows)))
    return sum(column_qmn(k, n) for k in range(n_cols))


def _embed_ucg(n: int, s: int, ctrl_gates: Dict[int, NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Build the ``2^n x 2^n`` UCG: target = bit ``s``; control value = bits above ``s``; the same
    ``U_l`` is applied for every free pattern (bits below ``s``)."""
    N = 1 << n
    place = 1 << s
    G = np.eye(N, dtype=complex)
    for l, Ul in ctrl_gates.items():
        for f in range(place):
            i0 = l * (2 * place) + f
            i1 = i0 + place
            G[i0, i0], G[i0, i1] = Ul[0, 0], Ul[0, 1]
            G[i1, i0], G[i1, i1] = Ul[1, 0], Ul[1, 1]
    return G


def _embed_two_level(n: int, i0: int, i1: int, U2: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """A 2-level (multi-controlled single-qubit) gate acting as ``U2`` on basis pair ``(i0, i1)``."""
    G = np.eye(1 << n, dtype=complex)
    G[i0, i0], G[i0, i1] = U2[0, 0], U2[0, 1]
    G[i1, i0], G[i1, i1] = U2[1, 0], U2[1, 1]
    return G


def _column_op(M: NDArray[np.complex128], k: int, n: int) -> Tuple[NDArray[np.complex128], int]:
    """Return ``(G_k, n_mcg)`` disentangling column ``k`` of ``M`` to ``e_k`` (Iten Lemma 10/11/12)."""
    N = 1 << n
    Gk = np.eye(N, dtype=complex)
    cur = M.copy()
    n_mcg = 0
    for s in range(n):
        ks = (k >> s) & 1
        a_s1 = k >> (s + 1)
        b_s1 = k & ((1 << (s + 1)) - 1)
        place = 1 << s
        f = k & (place - 1)  # already-fixed lower bits
        if ks == 0 and b_s1 != 0:  # Lemma 11 multi-controlled fix-up
            i0 = a_s1 * (2 * place) + f
            i1 = i0 + place
            Gm = _embed_two_level(n, i0, i1, lemma2_gate(cur[i0, k], cur[i1, k], 0))
            cur, Gk = Gm @ cur, Gm @ Gk
            n_mcg += 1
        n_ctrl = n - 1 - s
        lo = a_s1 + 1 if b_s1 != 0 else a_s1  # U_l = I below this to preserve earlier columns
        ctrl_gates: Dict[int, NDArray[np.complex128]] = {}
        for l in range(1 << n_ctrl):
            if l < lo:
                continue
            i0 = l * (2 * place) + f
            i1 = i0 + place
            ctrl_gates[l] = lemma2_gate(cur[i0, k], cur[i1, k], ks)
        Gu = _embed_ucg(n, s, ctrl_gates)
        cur, Gk = Gu @ cur, Gu @ Gk
    return Gk, n_mcg


def column_by_column_disentangler(
    V: NDArray[np.complex128],
) -> Tuple[NDArray[np.complex128], int]:
    """Return ``(G, total_mcg)`` with ``G V = I_{2^n x K}`` (Iten column-by-column).

    ``G`` is a ``2^n x 2^n`` unitary mapping column ``k`` of ``V`` to ``|k>`` (up to a phase that the
    final diagonal layer fixes), for ``k = 0 .. K-1``.  The synthesized isometry is ``V = G^dagger
    I_{2^n x K}``.
    """
    V = np.asarray(V, dtype=complex)
    N, K = V.shape
    n = int(round(np.log2(N)))
    assert (1 << n) == N, "V must have 2^n rows"
    G = np.eye(N, dtype=complex)
    M = V.copy()
    total_mcg = 0
    for k in range(K):
        Gk, n_mcg = _column_op(M, k, n)
        M, G = Gk @ M, Gk @ G
        total_mcg += n_mcg
    return G, total_mcg


# ============================================================================
# Resource-model bloqs
# ============================================================================


def _phase_layer_log_block_sizes(
    n_blocks: int, n_rows_layer: int, phase_bitsize: int, optimal_T: bool,
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]],
) -> Optional[Tuple[SymbolicInt, ...]]:
    """Per-layer QROAM ``log_block_sizes`` for a merged multiplexed gate with ``n_rows_layer = 2^(c+1)``.

    With ``optimal_T`` the Toffoli-optimal split for this layer's table is used (the interferometer
    optimum ``Lambda* ~ 0.5 sqrt(n_blocks n_rows_layer / b)`` already matches the two-angle table's
    ``sqrt(M / 2b)``); otherwise the caller-supplied ``log_block_sizes`` (default ``Lambda = 1``).
    """
    if optimal_T:
        return optimal_interferometer_log_block_sizes(n_blocks, n_rows_layer, phase_bitsize)
    return log_block_sizes


@attrs.frozen
class MultiControlledSU2QROAM(Bloq):
    r"""Toffoli model for the Iten multi-controlled single-qubit gate ``C_{n_controls}(U)``.

    This is the subleading ``C_{n-1}(U_s)`` correction in the column-by-column scheme.  It is modeled
    as an AND-ladder over the ``n_controls`` controls (``n_controls - 1`` ``And`` gates, measurement-based
    uncompute at 0 Toffoli) followed by a singly-controlled SU(2) on the target, realized through the
    Eq.-24 decomposition: control-independent Hadamards plus three controlled phase-gradient additions
    (the two ``diag(e^{i theta},1)/diag(e^{i phi},1)`` factors and the ``diag(e^{i phi0},e^{i phi1})``
    relative phase).
    """

    n_controls: SymbolicInt
    phase_bitsize: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature.build(
            controls=self.n_controls, target=1, phase_gradient=self.phase_bitsize
        )

    @property
    def _add(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if not is_symbolic(self.n_controls):
            n_and = max(0, int(self.n_controls) - 1)
            if n_and:
                ret[And()] += n_and
                ret[And().adjoint()] += n_and
        # singly-controlled SU(2) via Eq. 24: 3 controlled phase additions + 2 Hadamards.
        ret[self._add.controlled()] += 3
        ret[Hadamard()] += 2
        return ret

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> 'Tuple[Bloq, AddControlledT]':
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledMultiControlledSU2QROAM(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledMultiControlledSU2QROAM(Bloq):
    """Singly-controlled :class:`MultiControlledSU2QROAM` (one extra control on the SU(2) rotations)."""

    inner: MultiControlledSU2QROAM

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if not is_symbolic(self.inner.n_controls):
            n_and = max(0, int(self.inner.n_controls) - 1)
            if n_and:
                ret[And()] += n_and
                ret[And().adjoint()] += n_and
        ret[self.inner._add.controlled().controlled()] += 3
        ret[Hadamard()] += 2
        return ret


@attrs.frozen
class BlockIsometryColumnSynthesisQROAM(GateWithRegisters):
    r"""Block-diagonal ``m -> n`` isometry synthesis, column-by-column (Iten) with Eq.-24 QROAM layers.

    Synthesizes ``sum_j |j><j| (x) V_j`` where each ``V_j`` is a ``2^n x n_cols`` isometry (the first
    ``n_cols`` columns of a unitary).  ``n_blocks = 1`` is a single isometry.  See the module docstring
    for the algorithm.

    Args:
        n_blocks: number of block-diagonal blocks (read-only address into every QROAM lookup).
        n_rows: ``N = 2^n`` rows (must be a power of two).
        n_cols: ``K`` columns to synthesize, ``1 <= K <= N`` (defaults to ``N`` for a full unitary).
        phase_bitsize: rotation-angle table bitsize ``b``.
        log_block_sizes: QROAM tradeoff for the per-step phase layers when ``optimal_T`` is False.
        final_log_block_sizes / final_adjoint_log_block_sizes: QROAM tradeoff for the final phase layer.
        optimal_T: pick per-layer Toffoli-optimal ``Lambda`` (else qubit-minimal ``Lambda = 1``).
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    n_cols: SymbolicInt
    phase_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    optimal_T: bool = False
    # The synthesized matrix is real (THC factors, real-symmetric Fock eigenvectors).
    # Selects the rotation-only layers of ``real_rotation_layers_QROAM``: one angle table
    # and one phase-gradient addition per layer instead of two/three, and a 1-bit sign
    # layer instead of a b-bit final phase layer.  Default False = general complex.
    real_data: bool = False
    # How the block-dependent multi-controlled-gate angles are supplied:
    #   "per-gate"   -- one n_blocks-entry QROAM per fix-up, b-bit word.  The plain
    #                   reading of the construction; each gate looks up its own angle.
    #   "per-column" (default) -- one lookup per column with a (q_k * b)-bit word
    #                   covering all of that column's fix-ups at once ("load-all").
    #                   Correct because the slice offsets ARE compile-time constants:
    #                   ``mcg_needed(k, s)`` is a pure function of two classical loop
    #                   indices, so the s-th fix-up in column k always reads slice s and
    #                   no runtime routing is needed.  Verified for THC (n_cols 4, 22)
    #                   and DF (n_cols 88): step counts match ``column_qmn`` and sum to
    #                   ``num_mcgs``.
    mcg_angle_load: str = "per-gate"
    # Whether the angle-register erasure is absorbed into the following layer (X-basis
    # measurement with the sign fix-up folded into the next layer's classical data, the
    # convention the phase layers already use) or charged explicitly.
    absorb_mcg_erasure: bool = True
    # Whether every column charges the FULL 2^(c+1) prefix range at each layer, or only
    # the live nodes of its own P-07 rotation tree.
    #
    # Column ``j`` has support on ``d = n_rows - j`` entries, so its tree level ``s`` holds
    # ``max(0, min(2^s, d - 2^s))`` nodes -- the top level truncated by ``d - 2^s``.  The
    # padded form assumes every level is full, which charges ``~2^n`` per column instead of
    # ``~d``.  Total rotations are then ``M(N-1) - M(M-1)/2`` (P-07, executed: random
    # complex isometries reduced to the identity embedding with residual <= 9e-16 at
    # non-power-of-two N), against ``M(2^n - 1)`` padded.  At n_rows=130, n_cols=22 that
    # is 2607 against 5610, a 2.15x overcharge; at 208 it is 1.30x and at 256 it is 1.04x.
    #
    # WARNING: ``build_composite_bloq`` and ``_column_op`` below still implement the
    # PADDED elimination order, which spends ~2x the angles of the P-07 construction.  So
    # with the default the cost model and the in-module decomposition disagree, and the
    # decomposition is the one that is wrong.  Reconciling them means replacing
    # ``_column_op`` with the P-07 order, which needs its own numerical verification.
    pad_layer_tables: bool = False
    # Whether this instance is the INVERSE isometry V^dagger rather than the forward V.
    #
    # Every phase layer erases its angle register by X-basis measurement, which leaves an
    # address-dependent +-1 phase.  Whether that phase is free depends entirely on the
    # direction, and the reason is the staircase:
    #
    #   FORWARD -- the multiplex grows.  Qubit 1 is rotated unmultiplexed, qubit 2 is
    #   multiplexed by qubit 1, qubit 3 by qubits 1-2, ..., and the last layer is a phase
    #   layer multiplexed by all n qubits.  So the phase left by layer t is diagonal on a
    #   SUBSET of the qubits that layer t+1 multiplexes on.  A diagonal on control qubits
    #   commutes through a gate controlled by those qubits, so it commutes trivially, and
    #   every phase can be pushed to the final all-qubit phase layer and absorbed into its
    #   classical data.  Cost of erasure: zero.
    #
    #   INVERSE -- the multiplex shrinks.  Now the phase left by layer t is diagonal on
    #   qubits that layer t+1 ROTATES, not qubits it controls on.  A diagonal does not
    #   commute through a rotation on the same qubit, so it cannot be deferred: each layer
    #   must pay its own explicit measurement-based uncomputation.
    #
    # So the inverse charges one QROAMCleanAdjoint per phase layer, over that layer's own
    # (n_blocks, n_s) table.  It is measurement-based, hence independent of the word width,
    # which is why the penalty is a b-free term.  Same argument kills the
    # ``absorb_mcg_erasure`` premise, so the inverse charges those explicitly too.
    #
    # Effect (n_blocks=216, b=20): at Q-optimal the inverse is ~2x the forward, since
    # compute and uncompute are equal halves at Lambda=1.  At T-optimal the leading
    # ``sqrt(b)`` term is unchanged and a b-free term appears, ~5 M sqrt(N N_k).
    inverse: bool = False
    # Which construction the cost model charges.
    #
    # 'phase-first' (author's scheme, 2026-08-23; DEFAULT) -- per column: one diagonal phase
    # layer over N entries making the column real, then n real R_y sublayers, sublayer s
    # multiplex-controlled on the leading n-s qubits and multi-controlled on the trailing
    # s-1 qubits at f = c mod 2^(s-1).  Tables are ceil(N/2^s), word b (ONE real angle).
    # NO multi-controlled fix-up gates are needed: whenever a pair contains an index below c
    # the entry to be zeroed is already zero, since column c is orthogonal to e_0..e_{c-1},
    # so the Givens angle is 0 and the gate is the identity.
    #
    # Amplitude does transiently leave the physical range when N is not a power of two -- a
    # sublayer's surviving side can sit above N -- but the destination is always the *keep*
    # side, hence inside the next sublayer's congruence class, so a later sublayer always has
    # an address for it.  That is why ceil(N/2^s) suffices and no range gymnastics are needed.
    # Verified: 39/39 random complex isometries reduced to the identity embedding at residual
    # ~1e-16, n_phys in {21,26,30,37,45,50,60,128,130,208}, no fix-ups.
    #
    # 'iten' -- the older column-by-column path: general two-level gates, 2b-wide two-angle
    # tables, and num_mcgs() Lemma-11 fix-ups.  Kept for comparison; ~2x dearer.
    scheme: str = 'phase-first'

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows):
            # The column-by-column scheme walks the ceil(log2 n_rows)-qubit register; it
            # does not need n_rows itself to be a power of two.  Requiring it forced
            # callers to pad (208 -> 256), which the analytic model does not do.
            assert self.n_rows >= 2
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        if not is_symbolic(self.n_rows, self.n_cols):
            assert 1 <= self.n_cols <= self.n_rows
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if self.optimal_T and is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise ValueError("optimal_T=True requires concrete n_blocks, n_rows, phase_bitsize")

    @classmethod
    def from_shape(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_cols: Optional[SymbolicInt] = None,
        log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        final_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        final_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        optimal_T: bool = False,
        real_data: bool = False,
        mcg_angle_load: str = "per-gate",
        absorb_mcg_erasure: bool = True,
        pad_layer_tables: bool = False,
        inverse: bool = False,
        scheme: str = 'phase-first',
    ) -> 'BlockIsometryColumnSynthesisQROAM':
        return cls(
            n_blocks=n_blocks,
            n_rows=n_rows,
            n_cols=n_rows if n_cols is None else n_cols,
            phase_bitsize=phase_bitsize,
            log_block_sizes=log_block_sizes,
            final_log_block_sizes=final_log_block_sizes,
            final_adjoint_log_block_sizes=final_adjoint_log_block_sizes,
            optimal_T=optimal_T,
            real_data=real_data,
            mcg_angle_load=mcg_angle_load,
            absorb_mcg_erasure=absorb_mcg_erasure,
            pad_layer_tables=pad_layer_tables,
            inverse=inverse,
            scheme=scheme,
        )

    @classmethod
    def from_isometry(
        cls, V: NDArray[np.complex128], phase_bitsize: SymbolicInt, **kwargs
    ) -> 'BlockIsometryColumnSynthesisQROAM':
        """Convenience constructor inferring ``(n_rows, n_cols)`` from a single dense isometry."""
        V = np.asarray(V)
        n_rows, n_cols = V.shape
        return cls.from_shape(1, n_rows, phase_bitsize, n_cols=n_cols, **kwargs)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize, system=self.system_bitsize, phase_gradient=self.phase_bitsize
        )

    def _control_counts(self) -> List[int]:
        """The per-column active UCG control-counts ``c`` (Iten sets ``C^u_0 = I`` for ``n >= 2``)."""
        n = int(self.system_bitsize)
        if n <= 1:
            return [0]  # single-qubit state prep is the whole operation
        return list(range(1, n))

    def phase_layer(self, c: int):
        """The merged multiplexed gate with ``c`` controls (table shape ``(n_blocks, 2^c)``).

        With ``real_data`` this is the rotation-only :class:`RealPhaseLayerQROAM`: one
        ``b``-bit angle table and one phase-gradient addition instead of two.
        """
        return self.phase_layer_of_width(1 << (c + 1))

    def phase_layer_of_width(self, n_rows_layer: int):
        """The same merged multiplexed gate, given the table width directly."""
        n_rows_layer = int(n_rows_layer)
        lbs = _phase_layer_log_block_sizes(
            int(self.n_blocks), n_rows_layer, int(self.phase_bitsize), self.optimal_T,
            self.log_block_sizes,
        )
        if self.real_data:
            return RealPhaseLayerQROAM(
                n_blocks=self.n_blocks, n_rows=n_rows_layer,
                phase_bitsize=self.phase_bitsize, log_block_sizes=lbs,
            )
        return BlockInterferometerPhaseLayerQROAM(
            n_blocks=self.n_blocks, n_rows=n_rows_layer, phase_bitsize=self.phase_bitsize,
            log_block_sizes=lbs,
        )

    @property
    def final_phase_layer(self):
        """The single merged final diagonal layer over all ``n`` qubits (table ``(n_blocks, N)``).

        With ``real_data`` the residual diagonal is a sign, so this becomes the 1-bit
        :class:`RealSignLayerQROAM` applied by a Clifford ``Z``.
        """
        if self.optimal_T:
            lbs = optimal_interferometer_log_block_sizes(
                int(self.n_blocks), int(self.n_rows), int(self.phase_bitsize)
            )
            fwd = adj = lbs
        else:
            fwd = self.final_log_block_sizes
            adj = self.final_adjoint_log_block_sizes
            if tuple(adj or ()) in ((), (0, 0)):
                # Lambda'=1 was the old default and is strictly dominated: the measurement
                # -based uncomputation can batch inside the workspace the forward lookup
                # already holds.  Default to the capped optimum instead.
                # The FINAL layer is a different shape from the tree layers: one phase
                # per ROW, target_bitsizes=(b,), so its word is b and not 2b.
                adj = capped_adjoint_log_block_sizes(
                    (int(self.n_blocks), int(self.n_rows)), int(self.phase_bitsize))
        if self.real_data:
            return RealSignLayerQROAM(
                n_blocks=self.n_blocks, n_rows=self.n_rows,
                log_block_sizes=fwd, adjoint_log_block_sizes=adj,
            )
        return BlockInterferometerFinalPhasesQROAM(
            n_blocks=self.n_blocks, n_rows=self.n_rows, phase_bitsize=self.phase_bitsize,
            log_block_sizes=fwd, adjoint_log_block_sizes=adj,
        )

    @property
    def mcg(self):
        r"""The subleading gate is a **real rotation**, not a general SU(2).

        Iten's fix-up ``C_{n-1}(U_s)`` only has to annihilate one of two amplitudes, i.e.
        it only needs ``c1/c0`` real -- so only the *relative* phase of the pair matters,
        not either absolute phase.  That relative phase can be folded into the preceding
        uniformly-controlled layer, which already loads a per-control phase pair, so it
        is free; the residual global phase falls into the single final diagonal layer
        that the scheme already has.

        Crucially the phase always lands on a control gate the scheme is *allowed* to
        touch: the two affected basis states sit at control values ``l0 = a_{s+1}`` and
        ``l1 = a_{s+1}+1`` of the previous layer, and ``l1`` is always ``>= lo``, so the
        identity-pinned gates that protect already-disentangled columns are never
        disturbed.  Verified numerically over random COMPLEX isometries: 440/440 fix-ups
        absorbed, none blocked, and ``| |G V| - I |`` stays at 9e-16
        (``tests/test_bse_walk_operator.py::test_mcg_is_a_real_rotation``).

        This holds for complex data; it is not a real-data special case.  Cost: one
        controlled phase-gradient addition instead of three.
        """
        return RealMultiControlledRotationQROAM(
            n_controls=self.system_bitsize - 1, phase_bitsize=self.phase_bitsize
        )

    def _angle_load(self, n_angles: int, adjoint: bool = False):
        """QROAM supplying block-dependent multi-controlled-gate angles.

        The isometry is **block diagonal** over the block register (here the momentum
        ``k``), so each block has its OWN fix-up angles.  They are classical data and must
        be looked up from an ``n_blocks``-entry table -- this was missing entirely:
        :class:`MultiControlledSU2QROAM` and its rotation-only sibling emit no QROAM at all
        and carry no block register, i.e. they silently assume a compile-time-constant
        angle.  That is only correct for ``n_blocks == 1``.

        ``n_angles`` is 1 for the ``per-gate`` policy and ``column_qmn(k, n)`` for
        ``per-column``.  ``Lambda`` is chosen by measured optimum in both cases -- it is
        NOT the same choice: a wide (``>= 4`` angle) word wants ``Lambda = 1`` while a
        single ``b``-bit word wants ``Lambda = 2``, and hardcoding either is wrong for the
        other (216 entries, b=32: 214 vs 138).
        """
        try:
            from .qroam_block_sizes import optimal_log_block_sizes_measured as _opt
        except ImportError:
            from qroam_block_sizes import optimal_log_block_sizes_measured as _opt
        word = int(n_angles) * int(self.phase_bitsize)
        lbs = _opt((int(self.n_blocks),), (word,), adjoint=adjoint)
        if adjoint:
            return QROAMCleanAdjoint.build_from_bitsize(
                (self.n_blocks,), target_bitsizes=(word,), log_block_sizes=lbs)
        return QROAMClean.build_from_bitsize(
            (self.n_blocks,), target_bitsizes=(word,), log_block_sizes=lbs)

    def _real_sweep_layer(self, table_rows: int):
        """One real-R_y sublayer: ``(n_blocks, table_rows)`` of single ``b``-bit angles."""
        b = int(self.phase_bitsize)
        # One b-bit angle per address, so the interferometer helper (sized for two-angle
        # tables) is the wrong optimum here; ask the counter directly.
        if self.optimal_T:
            try:
                from .qroam_block_sizes import optimal_log_block_sizes_measured as _opt
            except ImportError:
                from qroam_block_sizes import optimal_log_block_sizes_measured as _opt
            lbs = _opt((int(self.n_blocks), max(1, int(table_rows))), (b,))
        else:
            lbs = self.log_block_sizes
        return RealPhaseLayerQROAM(
            n_blocks=self.n_blocks, n_rows=2 * int(table_rows),
            phase_bitsize=self.phase_bitsize, log_block_sizes=lbs)

    def _column_diagonal(self):
        """The per-column diagonal phase layer: ``(n_blocks, N)`` single ``b``-bit angles."""
        b = int(self.phase_bitsize)
        if self.optimal_T:
            try:
                from .qroam_block_sizes import optimal_log_block_sizes_measured as _opt
            except ImportError:
                from qroam_block_sizes import optimal_log_block_sizes_measured as _opt
            shape = (int(self.n_blocks), int(self.n_rows))
            lbs = _opt(shape, (b,))
            adj = _opt(shape, (b,), adjoint=True)
        else:
            lbs = self.final_log_block_sizes
            adj = capped_adjoint_log_block_sizes(
                (int(self.n_blocks), int(self.n_rows)), b)
        # b-BIT phase per entry: this layer removes an arbitrary complex phase so that
        # every later sublayer can be a bare real R_y.  RealSignLayerQROAM is the 1-bit
        # sign table and is NOT this -- it is what the forward's final layer needs, where
        # only deferred +-1 erasure signs have to be absorbed.
        return BlockInterferometerFinalPhasesQROAM(
            n_blocks=self.n_blocks, n_rows=self.n_rows, phase_bitsize=self.phase_bitsize,
            log_block_sizes=lbs, adjoint_log_block_sizes=adj)

    def _sweep_erasure(self, table_rows: int):
        """Immediate measurement-based erasure of one sublayer (INVERSE direction only).

        Word is ``b`` -- a single real angle per address -- so the batching cap is ``b``,
        not ``2b``: the selector can borrow only the ``b`` qubits the forward lookup held.
        """
        b = int(self.phase_bitsize)
        shape = (int(self.n_blocks), max(1, int(table_rows)))
        # The cap is the workspace the FORWARD lookup already holds, Lambda_fwd * b.  At
        # Q-optimal Lambda_fwd = 1 so that is b.  At T-optimal Lambda_fwd ~ sqrt(L/b), so the
        # workspace is ~sqrt(Lb) and the erasure can reach its own unconstrained optimum;
        # hardcoding b there would over-charge it several-fold.
        fwd = self._real_sweep_layer(table_rows)
        lam_fwd = 1
        for k in (fwd.log_block_sizes or ()):
            lam_fwd *= 1 << int(k)
        return QROAMCleanAdjoint.build_from_bitsize(
            shape, target_bitsizes=(b,),
            log_block_sizes=capped_adjoint_log_block_sizes(shape, lam_fwd * b))

    def _phase_layer_erasure(self, table_rows: int):
        """Measurement-based erasure of one phase layer's angle register.

        Only the INVERSE needs this -- see ``inverse``.  The table is the layer's own
        ``(n_blocks, table_rows)`` holding two ``b``-bit angles; the adjoint is
        measurement-based so its cost does not depend on the word width.
        """
        try:
            from .qroam_block_sizes import optimal_log_block_sizes_measured as _opt
        except ImportError:
            from qroam_block_sizes import optimal_log_block_sizes_measured as _opt
        b = int(self.phase_bitsize)
        shape = (int(self.n_blocks), int(table_rows))
        # The erasure tradeoff parameter is optimized in BOTH operating points, including
        # Q-optimal -- but it is CAPPED, and the cap is the point.
        #
        # A measurement-based uncomputation converts the low address bits to a one-hot
        # unary selector, applies the phase by a reduced lookup on the high bits, then
        # erases the selector by measurement and Clifford feed-forward.  The selector is
        # one bit per batch entry, so Lambda' costs Lambda' qubits -- cheap per qubit, but
        # not free.  The workspace it can borrow is the word being erased, which here is
        # the two b-bit angles, so Lambda' <= 2b keeps the uncomputation from ever raising
        # the peak width above what the forward lookup already needed.  This is the same
        # convention the analytic model uses (its Q-opt mode caps primed lambdas at 2b).
        #
        # NOTE Qualtran's QubitCount reports QROAMCleanAdjoint as flat in Lambda' (44
        # qubits from Lambda'=1 to 512), i.e. it does not appear to charge the selector.
        # We do not rely on that: an uncapped Lambda' would pick ~sqrt(L), which at
        # L = 28080 is 335 against 32 and would quietly claim ~10x the workspace.
        #
        # Consequence: the erasure costs ~L/(2b) per layer rather than ~2 sqrt(L), so the
        # inverse's overhead over the forward is ~1/(2b) -- b-dependent, and ~2.5% at
        # b = 20 -- instead of the b-free ~4.83/sqrt(N_k N).
        lbs = capped_adjoint_log_block_sizes(shape, 2 * b)   # two b-bit angles per word
        return QROAMCleanAdjoint.build_from_bitsize(
            shape, target_bitsizes=(b, b), log_block_sizes=lbs)

    def _emit_mcg_angle_loads(self, ret: 'Counter[Bloq]', K: int, n: int) -> None:
        """Emit the angle lookups for every multi-controlled fix-up.

        Only needed when the isometry is actually multiplexed -- a single isometry has
        compile-time-constant angles and needs no lookup at all.

        **Erasure.**  With ``absorb_mcg_erasure`` the angle register is erased by X-basis
        measurement and the sign fix-up is folded into the next layer's classical angle
        data, at zero Toffoli -- the convention the phase layers already use (only the
        *final* phase layer, which has no successor, pays an explicit adjoint).  Every
        fix-up here is followed by a uniformly-controlled layer within the same column, so
        a successor always exists.  Set False to charge the erasure explicitly instead;
        it costs ~26 Toffoli per lookup at ``n_blocks = 216`` and is independent of the
        word width (measurement-based, so no ``b``).
        """
        if is_symbolic(self.n_blocks) or int(self.n_blocks) <= 1:
            return
        if self.mcg_angle_load == "per-column":
            widths = [column_qmn(k, n) for k in range(K)]
            widths = [q for q in widths if q]
        elif self.mcg_angle_load == "per-gate":
            widths = [1] * num_mcgs(int(self.n_rows), K)
        else:
            raise ValueError(f"unknown mcg_angle_load {self.mcg_angle_load!r}")
        for q in widths:
            ret[self._angle_load(q)] += 1
            emit_range_safety(ret, (int(self.n_blocks),))   # P-13
            if not self.absorb_mcg_erasure or self.inverse:
                # The inverse has no successor layer to fold the sign into either.
                ret[self._angle_load(q, adjoint=True)] += 1

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if is_symbolic(self.n_rows, self.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self}")
        K = int(self.n_cols)
        n = int(self.system_bitsize)
        if self.scheme == 'phase-first':
            # Author's scheme.  Per column: one diagonal phase layer over N, then n real
            # R_y sublayers of ceil(N/2^s).  No fix-up gates.  Forward defers every erasure
            # into one final phase layer; the inverse pays each one immediately, because its
            # staircase decreases and the measurement sign lands on a qubit a later
            # sublayer rotates rather than on one it controls.
            # WHAT EACH SUB-BLOQ ALREADY CHARGES -- read before adding anything here.
            #   BlockInterferometerFinalPhasesQROAM (the diagonal): its QROAM, P-13 range
            #     safety, one AddIntoPhaseGrad, AND its own measurement-based adjoint.  So
            #     the one-erasure-per-column is already paid; do not add one.
            #   RealPhaseLayerQROAM (a sublayer): its QROAM, P-13 range safety, one
            #     SignedCtrlAddIntoPhaseGrad (b-2), two Hadamards.  Range safety is already
            #     paid; do not add one.
            #   QROAMCleanAdjoint: itself only.
            # Only the trailing-qubit AND ladders and the inverse's sublayer erasures are
            # genuinely ours to emit.
            #
            # NOTE RealPhaseLayerQROAM is a COST PROXY, not a structural match: it is named
            # for an adjacent-pair beamsplitter layer over the whole register with no
            # multi-control, whereas our sublayer pairs are strided by 2^s and are
            # multi-controlled on the trailing s-1 qubits.  Table shape, word width and
            # rotation count coincide, so the Toffoli count is right; the routing is not
            # what the class name says.
            sizes = sublayer_sizes(int(self.n_rows))
            for _ in range(K):
                ret[self._column_diagonal()] += 1
                # Every diagonal owes exactly ONE erasure per column, in both
                # directions -- and BlockInterferometerFinalPhasesQROAM already charges it
                # internally (its build_call_graph emits qroam_adj_bloq_for_cost).  So
                # nothing extra is added here; an explicit erasure on top was double
                # charging, worth ~12% of the T-optimal forward.
                #
                # Sublayer signs are the only ones that differ by direction.  A sublayer
                # sign is a function of that layer's multiplex address.  Forward the address
                # GROWS, so the sign is a function of a subset of the next layer's address
                # and is absorbed into its classical angle data for free (a +-1 on the
                # target flips the angle sign, R_y(t) diag(-1,1) = diag(-1,1) R_y(-t); a
                # +-1 on the controls commutes).  Inverse the address SHRINKS, so the sign
                # depends on more bits than the next layer's address has and cannot be
                # absorbed -- hence one explicit erasure per sublayer below.
                for s_idx, size in enumerate(sizes, start=1):
                    ret[self._real_sweep_layer(size)] += 1
                    if s_idx >= 3:                 # AND ladder for the trailing controls
                        ret[And()] += s_idx - 2     # (f is classical: X gates are free)
                    if self.inverse:
                        ret[self._sweep_erasure(size)] += 1
            return ret

        if self.pad_layer_tables:
            for c in self._control_counts():
                ret[self.phase_layer(c)] += K
        else:
            # Column j has support on d = n_rows - j entries; charge its own tree.
            for j in range(K):
                for size in tree_level_sizes(int(self.n_rows) - j):
                    if size <= 0:
                        continue
                    w = 2 * size                    # merged two-angle table, always even
                    ret[self.phase_layer_of_width(w)] += 1
                    if w & (w - 1):                 # P-13: non-power-of-two lookup range
                        emit_range_safety(ret, (int(self.n_blocks), w))
                    if self.inverse:
                        # Decreasing staircase: the measurement phase lands on a qubit a
                        # later layer rotates, so it cannot be deferred.  Pay it here.
                        ret[self._phase_layer_erasure(size)] += 1
        n_mcg = num_mcgs(int(self.n_rows), K)
        if n_mcg:
            ret[self.mcg] += n_mcg
            self._emit_mcg_angle_loads(ret, K, n)
        ret[self.final_phase_layer] += 1
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_blocks, self.n_rows, self.n_cols, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")
        n = int(self.system_bitsize)
        K = int(self.n_cols)
        has_block = not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1
        block = soqs.get('block')
        system = soqs['system']
        phase_grad = soqs['phase_gradient']

        if not self.pad_layer_tables:
            # The cost model charges the P-07 per-column rotation trees, but the
            # elimination coded below is Iten Lemma 12, whose control register spans the
            # full high-bit range at every step and therefore spends ~2^n angles per
            # column rather than ~d.  Emitting it here would contradict build_call_graph.
            # Refuse rather than return a circuit whose cost is not the reported one.
            raise DecomposeTypeError(
                "build_composite_bloq implements the Iten Lemma-12 elimination, which "
                "costs ~2^n angles per column; the cost model charges the cheaper P-07 "
                "per-column trees.  Use pad_layer_tables=True to decompose the "
                "Lemma-12 circuit (and get its higher, self-consistent cost), or "
                "implement the P-07 elimination here."
            )

        active_c = set(self._control_counts())
        for k in range(K):
            for s in range(n):
                c = n - 1 - s
                if mcg_needed(k, s):
                    block, system, phase_grad = self._add_mcg(
                        bb, n, s, has_block, block, system, phase_grad
                    )
                if c in active_c:
                    block, system, phase_grad = self._add_phase_layer(
                        bb, n, c, has_block, block, system, phase_grad
                    )
        # Single merged final diagonal phase layer over all n qubits.
        fsoqs: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
        if has_block:
            fsoqs['block'] = block
        out = bb.add_d(self.final_phase_layer, **fsoqs)
        system, phase_grad = out['system'], out['phase_gradient']
        if has_block:
            block = out['block']

        result: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
        if has_block:
            result['block'] = block
        return result

    def _add_phase_layer(self, bb, n, c, has_block, block, system, phase_grad):
        """Apply the merged multiplexed gate at control-count ``c`` on the top ``c+1`` system qubits.

        Target = ``q[n-1-c]`` (the least-significant of the active sub-register), controls = the upper
        ``c`` qubits, free = the lower ``n-1-c`` qubits.
        """
        qubits = bb.split(system)
        sub = bb.join(qubits[: c + 1], dtype=QUInt(c + 1))  # controls (upper c) + target (LSB)
        layer = self.phase_layer(c)
        lsoqs: Dict[str, SoquetT] = {'system': sub, 'phase_gradient': phase_grad}
        if has_block:
            lsoqs['block'] = block
        out = bb.add_d(layer, **lsoqs)
        phase_grad = out['phase_gradient']
        if has_block:
            block = out['block']
        qubits[: c + 1] = bb.split(out['system'])
        system = bb.join(qubits, dtype=QUInt(n))
        return block, system, phase_grad

    def _add_mcg(self, bb, n, s, has_block, block, system, phase_grad):
        """Apply the subleading multi-controlled gate ``C_{n-1}(U_s)``: target ``q[n-1-s]``, the other
        ``n-1`` system qubits as controls.  ``block`` is unaffected (it is not part of this gate)."""
        qubits = bb.split(system)
        target_idx = n - 1 - s
        ctrl_idx = [i for i in range(n) if i != target_idx]
        controls = bb.join(np.array([qubits[i] for i in ctrl_idx]), dtype=QUInt(n - 1))
        mcg = MultiControlledSU2QROAM(n_controls=n - 1, phase_bitsize=self.phase_bitsize)
        out = bb.add_d(mcg, controls=controls, target=qubits[target_idx], phase_gradient=phase_grad)
        phase_grad = out['phase_gradient']
        ctrl_qubits = bb.split(out['controls'])
        for j, i in enumerate(ctrl_idx):
            qubits[i] = ctrl_qubits[j]
        qubits[target_idx] = out['target']
        system = bb.join(qubits, dtype=QUInt(n))
        return block, system, phase_grad

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> 'Tuple[Bloq, AddControlledT]':
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBlockIsometryColumnSynthesisQROAM(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledBlockIsometryColumnSynthesisQROAM(GateWithRegisters):
    """Singly-controlled :class:`BlockIsometryColumnSynthesisQROAM`.

    The external control reaches only the phase-gradient additions inside every layer (and one extra
    control on each multi-controlled gate); QROAM loads, Hadamards, and measurement-based erasures stay
    uncontrolled because they self-cancel (or are identity) when the control is 0.  The overhead is
    therefore a single extra control bit per ``AddIntoPhaseGrad`` -- ~0 Toffoli relative to the bare
    construction.
    """

    inner: BlockIsometryColumnSynthesisQROAM

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if is_symbolic(self.inner.n_rows, self.inner.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self.inner}")
        K = int(self.inner.n_cols)
        n = int(self.inner.system_bitsize)
        for c in self.inner._control_counts():
            ret[_ControlledPhaseLayerQROAM(self.inner.phase_layer(c))] += K
        n_mcg = num_mcgs(int(self.inner.n_rows), K)
        if n_mcg:
            # Dispatch through the mcg's own controlled form so this tracks whatever
            # ``mcg`` actually is -- it is a rotation-only gate (ONE addition), and
            # hardcoding the SU(2) wrapper here charged three.
            ret[self.inner.mcg.controlled()] += n_mcg
            # The block-dependent angle lookups are part of the operator, not of the
            # control, so the controlled form pays for them too.
            self.inner._emit_mcg_angle_loads(ret, K, n)
        ret[_ControlledFinalPhasesQROAM(self.inner.final_phase_layer)] += 1
        return ret


# ============================================================================
# Rectangular-isometry BlockEncoding backed by column-by-column synthesis.
#
# Drop-in alternative to ``ReflectionRectangularBlockEncoding``
# (rectangular_block_encoding_reflection.py): identical (1, 1, epsilon)
# BlockEncoding interface (system = block + matrix, ancilla = 1 signal, resource =
# phase gradient, alpha = 1), but the synthesized unitary's first ``n_reflections``
# columns are produced by :class:`BlockIsometryColumnSynthesisQROAM` instead of by
# Householder reflections.  The single ancilla is the BlockEncoding signal qubit
# (left in |0>); the column synthesizer itself needs no reflection ancilla, so this
# keeps the same register footprint as the reflection variant for an apples-to-apples
# synthesis-cost comparison inside the direct/exchange Coulomb encodings.
# ============================================================================


@attrs.frozen
class ColumnIsometryRectangularBlockEncoding(BlockEncoding):
    r"""$(1, 1, \epsilon)$ column-by-column isometry block encoding.

    Encodes $\sum_k |k\rangle\langle k| \otimes A_k$ where each $A_k$ is an
    ``n_rows x n_reflections`` isometry sitting in the top-left block of an
    ``n_rows``-dimensional unitary, synthesized column-by-column via
    :class:`BlockIsometryColumnSynthesisQROAM` (Iten 1501.06911 + Berry Eq. 24).

    The interface mirrors :class:`ReflectionRectangularBlockEncoding` exactly so the
    two are interchangeable inside the direct/exchange Coulomb constructions; only the
    internal synthesis (and hence the Toffoli/qubit cost) differs.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_reflections: Optional[SymbolicInt] = None
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    optimal_T: bool = False
    # The synthesized matrix is real (THC factors, real-symmetric Fock eigenvectors).
    # Selects the rotation-only layers of ``real_rotation_layers_QROAM``: one angle table
    # and one phase-gradient addition per layer instead of two/three, and a 1-bit sign
    # layer instead of a b-bit final phase layer.  Default False = general complex.
    real_data: bool = False
    # How the block-dependent multi-controlled-gate angles are supplied:
    #   "per-gate"   -- one n_blocks-entry QROAM per fix-up, b-bit word.  The plain
    #                   reading of the construction; each gate looks up its own angle.
    #   "per-column" -- one lookup per column with a (q_k * b)-bit word covering all of
    #                   that column's fix-ups at once ("load-all").  Cheaper, but relies
    #                   on the slice offsets being compile-time constants.
    mcg_angle_load: str = "per-gate"
    # Whether the angle-register erasure is absorbed into the following layer (X-basis
    # measurement with the sign fix-up folded into the next layer's classical data, the
    # convention the phase layers already use) or charged explicitly.
    absorb_mcg_erasure: bool = True
    # See ``BlockIsometryColumnSynthesisQROAM.pad_layer_tables``.
    pad_layer_tables: bool = False
    # See ``BlockIsometryColumnSynthesisQROAM.inverse``.
    inverse: bool = False
    # See ``BlockIsometryColumnSynthesisQROAM.scheme``.
    scheme: str = 'phase-first'

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def matrix_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def n_reflections_effective(self) -> SymbolicInt:
        return self.n_reflections if self.n_reflections is not None else self.n_rows

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.block_bitsize + self.matrix_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return 1

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return 1.0

    @property
    def epsilon(self) -> SymbolicFloat:
        return 2.0 ** (-self.phase_bitsize)

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return PrepareIdentity.from_bitsizes((self.ancilla_bitsize,))

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QBit()),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    @property
    def synth(self) -> BlockIsometryColumnSynthesisQROAM:
        return BlockIsometryColumnSynthesisQROAM.from_shape(
            self.n_blocks,
            self.n_rows,
            self.phase_bitsize,
            n_cols=self.n_reflections_effective,
            log_block_sizes=self.log_block_sizes,
            final_log_block_sizes=self.final_log_block_sizes,
            final_adjoint_log_block_sizes=self.final_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
            real_data=self.real_data,
            mcg_angle_load=self.mcg_angle_load,
            absorb_mcg_erasure=self.absorb_mcg_erasure,
            pad_layer_tables=self.pad_layer_tables,
            inverse=self.inverse,
            scheme=self.scheme,
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return Counter({self.synth: 1})

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Split ``system`` into ``block`` + ``matrix`` and call the column synthesizer.

        The ``ancilla`` (block-encoding signal qubit) is a spectator here -- the
        column synthesizer acts entirely on the matrix register -- and passes through
        untouched, matching the reflection variant's 1-ancilla footprint.
        """
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise DecomposeTypeError("build_composite_bloq requires concrete parameters")
        has_block = int(self.n_blocks) > 1
        system, ancilla, resource = soqs['system'], soqs['ancilla'], soqs['resource']

        if has_block:
            sys_arr = bb.split(system)
            bb_size = int(self.block_bitsize)
            block = bb.join(sys_arr[:bb_size], dtype=QUInt(self.block_bitsize))
            matrix = bb.join(sys_arr[bb_size:], dtype=QUInt(self.matrix_bitsize))
        else:
            matrix = system

        in_soqs: Dict[str, SoquetT] = {'system': matrix, 'phase_gradient': resource}
        if has_block:
            in_soqs['block'] = block
        out = bb.add_d(self.synth, **in_soqs)
        matrix, resource = out['system'], out['phase_gradient']
        if has_block:
            block = out['block']

        if has_block:
            system = bb.join(
                np.concatenate([bb.split(block), bb.split(matrix)]), dtype=QAny(self.system_bitsize)
            )
        else:
            system = matrix
        return {'system': system, 'ancilla': ancilla, 'resource': resource}

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> 'Tuple[Bloq, AddControlledT]':
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledColumnIsometryRectangularBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledColumnIsometryRectangularBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`ColumnIsometryRectangularBlockEncoding`.

    Promotes only the column synthesizer to its controlled variant (the external
    control reaches the phase-gradient additions inside each layer); the spectator
    ancilla and register surgery are unchanged.
    """

    inner: "ColumnIsometryRectangularBlockEncoding"

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.inner.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.inner.ancilla_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.inner.resource_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.inner.alpha

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.inner.epsilon

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return self.inner.signal_state

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('ctrl', QBit()),
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QBit()),
            Register('resource', QAny(self.inner.resource_bitsize)),
        ])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return Counter({self.inner.synth.controlled(): 1})
