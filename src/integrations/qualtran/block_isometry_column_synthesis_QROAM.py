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
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockInterferometerFinalPhasesQROAM,
        BlockInterferometerPhaseLayerQROAM,
        _ControlledFinalPhasesQROAM,
        _ControlledPhaseLayerQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script/direct execution
    from block_unitary_interferometer_QROAM import (
        BlockInterferometerFinalPhasesQROAM,
        BlockInterferometerPhaseLayerQROAM,
        _ControlledFinalPhasesQROAM,
        _ControlledPhaseLayerQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from state_prep_QROAM import _to_tuple_or_none

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

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows):
            assert self.n_rows == 2 ** bit_length(self.n_rows - 1), "n_rows must be a power of two"
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

    def phase_layer(self, c: int) -> BlockInterferometerPhaseLayerQROAM:
        """The merged multiplexed gate with ``c`` controls (table shape ``(n_blocks, 2^c)``)."""
        n_rows_layer = 1 << (c + 1)
        lbs = _phase_layer_log_block_sizes(
            int(self.n_blocks), n_rows_layer, int(self.phase_bitsize), self.optimal_T,
            self.log_block_sizes,
        )
        return BlockInterferometerPhaseLayerQROAM(
            n_blocks=self.n_blocks, n_rows=n_rows_layer, phase_bitsize=self.phase_bitsize,
            log_block_sizes=lbs,
        )

    @property
    def final_phase_layer(self) -> BlockInterferometerFinalPhasesQROAM:
        """The single merged final diagonal phase layer over all ``n`` qubits (table ``(n_blocks, N)``)."""
        if self.optimal_T:
            lbs = optimal_interferometer_log_block_sizes(
                int(self.n_blocks), int(self.n_rows), int(self.phase_bitsize)
            )
            return BlockInterferometerFinalPhasesQROAM(
                n_blocks=self.n_blocks, n_rows=self.n_rows, phase_bitsize=self.phase_bitsize,
                log_block_sizes=lbs, adjoint_log_block_sizes=lbs,
            )
        return BlockInterferometerFinalPhasesQROAM(
            n_blocks=self.n_blocks, n_rows=self.n_rows, phase_bitsize=self.phase_bitsize,
            log_block_sizes=self.final_log_block_sizes,
            adjoint_log_block_sizes=self.final_adjoint_log_block_sizes,
        )

    @property
    def mcg(self) -> MultiControlledSU2QROAM:
        return MultiControlledSU2QROAM(
            n_controls=self.system_bitsize - 1, phase_bitsize=self.phase_bitsize
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if is_symbolic(self.n_rows, self.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self}")
        K = int(self.n_cols)
        for c in self._control_counts():
            ret[self.phase_layer(c)] += K
        n_mcg = num_mcgs(int(self.n_rows), K)
        if n_mcg:
            ret[self.mcg] += n_mcg
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
        for c in self.inner._control_counts():
            ret[_ControlledPhaseLayerQROAM(self.inner.phase_layer(c))] += K
        n_mcg = num_mcgs(int(self.inner.n_rows), K)
        if n_mcg:
            ret[_ControlledMultiControlledSU2QROAM(self.inner.mcg)] += n_mcg
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
