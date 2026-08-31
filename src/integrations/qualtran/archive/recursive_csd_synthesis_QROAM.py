r"""Recursive (block-)unitary synthesis via cosine-sine decomposition + phase-data lookups.

This is a *prototype resource model* for the recursive-CSD synthesis scheme of

    Xinyu Tan, "Unitary synthesis with fewer T gates", arXiv:2509.25702,

in which the usual Clifford+T compilation of each one-qubit gate is replaced by

  * the one-qubit phase / Hadamard decomposition of
        Berry, Tong, Khattar, White, Kim, Boixo, Lin, Lee, Chan, Babbush, Rubin,
        arXiv:2409.11748, **Eq. (24)**:
            ``U = diag(e^{i phi0}, e^{i phi1}) . H . diag(e^{i theta}, 1) . H . diag(e^{i phi}, 1)`` ,
    i.e. two control-independent Hadamards on the target plus (up to) four per-address phase
    *words*, and

  * the phase-lookup construction of
        Gidney & Fowler / Sanders, Low, Scherer, Berry, ... arXiv:1812.00954, **Eq. (40)**:
        load a ``b``-bit phase value from a QRO(A)M, add it into a phase-gradient register, then
        uncompute (here: erase) the lookup.

Algorithm (ordinary unitary).  An ``N x N = 2^n x 2^n`` unitary is recursively decomposed by the
cosine-sine decomposition (``scipy.linalg.cossin``).  One CSD step on the most-significant qubit of
a ``k``-qubit sub-block ``U`` gives

    U = (W_u (+) W_u) . Rz_u . (V_u (+) V_u) . CS . (W_v (+) W_v) . Rz_v . (V_v (+) V_v),

where ``CS`` is a uniformly-controlled ``Ry`` on the MSB (controlled by the lower ``k-1`` qubits),
``Rz_u`` / ``Rz_v`` are uniformly-controlled ``Rz`` rotations obtained by *demultiplexing* the
block-diagonal CSD factors, and ``W_*, V_*`` are ``(k-1)``-qubit unitaries recursed into the lower
qubits.  Unrolling the recursion turns ``U`` into a flat, ordered sequence of **uniformly-controlled
one-qubit rotations** (multiplexed gates).  A multiplexed gate whose target is qubit ``t`` is
controlled by the ``c = n-1-t`` less-significant qubits, hence carries ``2^c`` one-qubit unitaries.

Each one-qubit unitary is realized by Eq. (24): the ``phase_words`` (default **4** -- the two
leftmost-diagonal phases ``phi0, phi1`` plus the two diagonal phases ``theta, phi``) are stored as
``b``-bit fixed-point integers and applied through the phase gradient (Eq. (40)), interleaved with
the two Hadamards.  For a multiplexed gate with ``c`` controls the ``2^c`` phase-word tuples are
**batched into a single QRO(A)M** addressed by the control register, loaded, applied, and erased
(measurement-based reset, 0 Toffoli, sign-fixup absorbed downstream) -- exactly the
``BlockInterferometerPhaseLayerQROAM`` pattern, generalized to ``phase_words`` words.

Internal block size ``M = 2^m`` (``block_log_size = m``).  The recursion descends to single-qubit
gates, but ``m`` partitions the multiplexed-gate *levels* into

  * **stable / lookup** levels ``c >= m``: the target lies in the outer ``n-m`` qubits, the phase
    data depends only on the *stable* outer controls (and, in the block case, the block label), so it
    is batched into the precomputed-style lookups, and
  * **inner / reconstruction** levels ``c < m``: these are the uniformly-controlled one-qubit
    rotations *inside* the current ``M x M`` block, whose phase data depends on the still-evolving
    target bits and is therefore reconstructed locally -- loaded and uncomputed immediately around
    the rotation -- rather than precomputed once for the whole circuit.

The two buckets use the same per-gate QROAM phase-layer cost; ``m`` only moves cost between the
reported *lookup* and *reconstruction* totals (and sets which loads must persist, affecting ancilla).

Block-diagonal synthesis.  For ``U = sum_a |a><a| (x) U_a`` set ``n_blocks = N_k``: a read-only
``block`` register of ``ceil(log2 N_k)`` qubits is appended to **every** QROAM address, exactly like
the sibling ``BlockUnitary*QROAM`` / ``BlockIsometry*QROAM`` bloqs.  The phase data is then batched
over ``(block label a, stable outer controls, CSD-rotation index, phase-word index)``.  ``n_blocks =
1`` recovers the ordinary single-unitary case.

Two cost regimes (IMPORTANT).  arXiv:2509.25702 shows that recursive CSD factors any ``U(2^n)`` into
exactly ``2^n - 1`` multi-controlled *single-qubit* unitaries (Thm 3.2/3.5), and then **groups**
consecutive runs of ``2^k - 1`` of them that share the same ``k``-qubit target into a single
``(n-k)``-controlled ``k``-qubit unitary (there are ``2^{n-k}`` such blocks), each synthesized with a
generalized Gosset-Kothari-Wu diagonal-synthesis (polynomial factoring) at T-count
``O(2^{(n+k)/2} sqrt(L) + 4^k L)`` (Thm 4.3, ``L = n + log(1/eps)``).  Summed over the ``2^{n-k}``
blocks and minimized at ``k ~ n/3`` this gives the paper's headline **T-count ``O(2^{4n/3} L^{2/3})``**
(Thm 1.1) -- i.e. ``~ N^{4/3}`` up to logs.

  * :func:`estimate_paper_unitary_resources` implements that *grouped* cost (Thm 4.3 + 1.1, optimal
    ``k``); this is the model that reproduces the ``N^{4/3}`` scaling.
  * The Qualtran bloqs below (:class:`CSDMuxPhaseLayerQROAM` / :class:`RecursiveCSDSynthesisQROAM`)
    and :func:`estimate_recursive_csd_resources` lay out the **explicit, un-grouped** circuit: every
    multiplexed gate is realized individually as a QROAM phase layer.  That is the *naive* strategy the
    paper improves on in Sec. 3.1 (decomposing each ``k``-qubit block into ``2^k - 1`` single-qubit
    multiplexed gates), so its cost scales as ``~ N^2`` -- it is a faithful, decomposable circuit, NOT
    the ``N^{4/3}`` route.  Use it for an explicit gate layout / equivalence checks; use
    :func:`estimate_paper_unitary_resources` for the optimal-T-count scaling.

Resource-model status.  As with the sibling modules this is a *resource model*: ``build_composite_bloq``
lays out the real gate structure with shape-only (data-free) QROAM, ``build_call_graph`` gives the
aggregate cost via closed-form multiplexed-gate counts, and the module-level numpy helpers
(:func:`recursive_csd_rotations`, :func:`reconstruct_from_ops`) provide a standalone reference proving
the decomposition reproduces ``U`` to machine precision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, log2, sqrt
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
    QBit,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        _measure_x_reset,
        _positive_power_of_two,
        _qroam_log_block_sizes,
        split_interferometer_log_block_sizes,
    )
    from .block_isometry_column_synthesis_QROAM import eq24_angles
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script/direct execution
    from block_unitary_interferometer_QROAM import (
        _measure_x_reset,
        _positive_power_of_two,
        _qroam_log_block_sizes,
        split_interferometer_log_block_sizes,
    )
    from block_isometry_column_synthesis_QROAM import eq24_angles
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


# ============================================================================
# numpy reference: recursive CSD -> uniformly-controlled one-qubit rotations
# ============================================================================

try:
    from scipy.linalg import cossin as _cossin
except ImportError:  # pragma: no cover - scipy always present in this env
    _cossin = None

_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


@attrs.frozen
class MuxOp:
    """One uniformly-controlled one-qubit gate produced by the recursive CSD.

    ``target`` is the (big-endian) index of the qubit acted on; ``controls`` are the less-significant
    qubit indices selecting which ``2x2`` unitary is applied.  ``gates`` maps a control value (the
    integer read off ``controls``, MSB-first) to the ``2x2`` unitary applied on ``target``.
    """

    target: int
    controls: Tuple[int, ...]
    gates: Tuple[NDArray[np.complex128], ...]  # indexed by control value 0 .. 2^c - 1

    @property
    def n_controls(self) -> int:
        return len(self.controls)


def _demultiplex(
    u1: NDArray[np.complex128], u2: NDArray[np.complex128]
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    r"""Demultiplex ``blkdiag(u1, u2) = (W (+) W) . diag(D, D^dagger) . (V (+) V)``.

    Returns ``(W, V, phi)`` with ``u1 = W D V``, ``u2 = W D^dagger V`` and ``D = diag(e^{i phi})``.
    Uses the standard eigendecomposition ``u1 u2^dagger = W D^2 W^dagger`` (Shende-Bullock-Markov).
    """
    M = u1 @ u2.conj().T  # unitary = W D^2 W^dagger
    evals, W = np.linalg.eig(M)
    d = np.sqrt(evals.astype(complex))  # principal branch of D = sqrt(D^2)
    V = (np.diag(d).conj().T) @ W.conj().T @ u1
    return W, V, np.angle(d)


def _ry(angle: float) -> NDArray[np.complex128]:
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz_diag(phi: float) -> NDArray[np.complex128]:
    """The demultiplexed ``diag(e^{i phi}, e^{-i phi})`` one-qubit factor."""
    return np.diag([np.exp(1j * phi), np.exp(-1j * phi)])


def recursive_csd_rotations(U: NDArray[np.complex128], qubits: Optional[List[int]] = None) -> List[MuxOp]:
    """Decompose an ``N x N`` unitary into an ordered list of :class:`MuxOp` (circuit order).

    ``qubits`` are the big-endian indices the unitary acts on (``qubits[0]`` is the MSB); it defaults
    to ``range(n)``.  The returned list is in *circuit order* (first-applied first); composing the
    embedded gates left-to-right reproduces ``U`` (see :func:`reconstruct_from_ops`).
    """
    if _cossin is None:  # pragma: no cover
        raise RuntimeError("scipy.linalg.cossin is required for the recursive-CSD reference")
    U = np.asarray(U, dtype=complex)
    N = U.shape[0]
    n = int(round(log2(N)))
    assert (1 << n) == N == U.shape[1], "U must be a 2^n x 2^n matrix"
    if qubits is None:
        qubits = list(range(n))
    return _rec(U, qubits)


def _rec(Uk: NDArray[np.complex128], qubits: List[int]) -> List[MuxOp]:
    k = len(qubits)
    if k == 1:
        return [MuxOp(target=qubits[0], controls=(), gates=(Uk.copy(),))]
    half = 1 << (k - 1)
    u, cs, vdh = _cossin(Uk, half, half)
    u1, u2 = u[:half, :half], u[half:, half:]
    v1, v2 = vdh[:half, :half], vdh[half:, half:]
    cos_d = np.diag(cs[:half, :half]).real
    sin_d = np.diag(cs[half:, :half]).real
    theta = 2.0 * np.arctan2(sin_d, cos_d)  # CS = multiplexed Ry(theta_j)
    W_u, V_u, phi_u = _demultiplex(u1, u2)
    W_v, V_v, phi_v = _demultiplex(v1, v2)
    t = qubits[0]
    lower = qubits[1:]
    cdim = 1 << (k - 1)
    ry_gates = tuple(_ry(theta[j]) for j in range(cdim))
    rz_u_gates = tuple(_rz_diag(phi_u[j]) for j in range(cdim))
    rz_v_gates = tuple(_rz_diag(phi_v[j]) for j in range(cdim))
    ctrls = tuple(lower)
    ops: List[MuxOp] = []
    # circuit order (first-applied first), reading the factorization right-to-left:
    ops += _rec(V_v, lower)
    ops.append(MuxOp(target=t, controls=ctrls, gates=rz_v_gates))
    ops += _rec(W_v, lower)
    ops.append(MuxOp(target=t, controls=ctrls, gates=ry_gates))
    ops += _rec(V_u, lower)
    ops.append(MuxOp(target=t, controls=ctrls, gates=rz_u_gates))
    ops += _rec(W_u, lower)
    return ops


def _embed_mux(op: MuxOp, n: int) -> NDArray[np.complex128]:
    """Embed a :class:`MuxOp` as a ``2^n x 2^n`` matrix (big-endian: index 0 = MSB)."""
    N = 1 << n
    G = np.eye(N, dtype=complex)
    tb = n - 1 - op.target
    cbits = [n - 1 - c for c in op.controls]
    seen = np.zeros(N, dtype=bool)
    for x in range(N):
        if seen[x] or ((x >> tb) & 1):
            continue
        x1 = x | (1 << tb)
        cval = 0
        for cb in cbits:
            cval = (cval << 1) | ((x >> cb) & 1)
        Uc = op.gates[cval]
        G[x, x], G[x, x1] = Uc[0, 0], Uc[0, 1]
        G[x1, x], G[x1, x1] = Uc[1, 0], Uc[1, 1]
        seen[x] = seen[x1] = True
    return G


def reconstruct_from_ops(ops: Iterable[MuxOp], n: int) -> NDArray[np.complex128]:
    """Compose the embedded ops (circuit order) into the full ``2^n x 2^n`` unitary."""
    M = np.eye(1 << n, dtype=complex)
    for op in ops:
        M = _embed_mux(op, n) @ M
    return M


def csd_phase_words(op: MuxOp, phase_words: int = 4) -> NDArray[np.float64]:
    """Eq.-24 phase words for every control value of a :class:`MuxOp`.

    Returns an array of shape ``(2^c, phase_words)``; for ``phase_words == 4`` the columns are
    ``(phi, theta, phi0, phi1)`` from :func:`eq24_angles` (Berry Eq. 24).  ``phase_words < 4`` keeps
    the leading ``(phi, theta, ...)`` entries; ``phase_words > 4`` zero-pads.
    """
    rows = []
    for Uc in op.gates:
        phi0, phi1, theta, phi = eq24_angles(Uc)
        full = [phi, theta, phi0, phi1]
        if phase_words <= 4:
            rows.append(full[:phase_words])
        else:
            rows.append(full + [0.0] * (phase_words - 4))
    return np.asarray(rows, dtype=float)


def csd_mux_schedule(n: int) -> List[Tuple[int, Tuple[int, ...]]]:
    """Structural schedule (no data) of ``(target, controls)`` for every multiplexed gate.

    Mirrors :func:`recursive_csd_rotations` branching, so the list length and per-control-count tally
    match the data-bearing decomposition exactly.  Used to lay out :meth:`build_composite_bloq`.
    """
    def rec(qubits: List[int]) -> List[Tuple[int, Tuple[int, ...]]]:
        k = len(qubits)
        if k == 1:
            return [(qubits[0], ())]
        t, lower = qubits[0], qubits[1:]
        c = tuple(lower)
        out: List[Tuple[int, Tuple[int, ...]]] = []
        out += rec(lower)
        out.append((t, c))
        out += rec(lower)
        out.append((t, c))
        out += rec(lower)
        out.append((t, c))
        out += rec(lower)
        return out

    return rec(list(range(n)))


def num_csd_muxes(n: int, control_count: int) -> int:
    """Closed form: number of multiplexed gates with exactly ``control_count`` controls.

    Targeting qubit ``t = n-1-c`` there are ``4^t`` sub-blocks, each contributing 3 multiplexed gates
    (two ``Rz`` + one ``Ry``) for ``c >= 1`` and ``4^{n-1}`` single-qubit leaves for ``c == 0``.
    """
    if n <= 0:
        return 0
    c = control_count
    if c < 0 or c > n - 1:
        return 0
    if c == 0:
        return 4 ** (n - 1)
    return 3 * 4 ** (n - 1 - c)


def total_csd_one_qubit_gates(n: int) -> int:
    """Total one-qubit unitaries (= Eq.-24 instances) over all multiplexed gates: ``N^2 - 3N/2``."""
    return sum(num_csd_muxes(n, c) * (1 << c) for c in range(n))


# ============================================================================
# QROAM phase-layer building block (Eq. 24 + Eq. 40), generalized to phase_words
# ============================================================================


def _optimal_layer_log_block_sizes(
    n_blocks: int, control_count: int, phase_bitsize: int, phase_words: int
) -> Optional[Tuple[int, ...]]:
    """Toffoli-optimal QROAM split for a ``(n_blocks, 2^c)`` phase-word table.

    The forward QROAM load of ``phase_words`` words of ``b`` bits costs ``~ entries/Lambda +
    phase_words*b*Lambda``, optimal at ``Lambda* ~ sqrt(entries / (phase_words*b))``.  For ``c == 0``
    (a ``<= n_blocks``-entry table) the layer is negligible and QROAM's own default is used.
    """
    if control_count == 0:
        return None
    entries = max(1, n_blocks) * (1 << control_count)
    lam = max(1.0, sqrt(entries / max(1, phase_words * phase_bitsize)))
    lam = 2 ** max(0, round(log2(lam)))
    return split_interferometer_log_block_sizes(lam, max(1, n_blocks), 1 << (control_count + 1))


@attrs.frozen
class CSDMuxPhaseLayerQROAM(GateWithRegisters):
    r"""One uniformly-controlled one-qubit gate via Berry Eq. 24 + Eq.-40 phase lookups.

    Loads ``phase_words`` ``b``-bit phase words from a QROAM addressed by ``(block, controls)``
    (table shape ``(n_blocks, 2^{n_controls})``), applies them to the ``target`` qubit through the
    phase gradient interleaved with the two Eq.-24 Hadamards, then erases the load by X-measurement
    (0 Toffoli; the sign-fixup is absorbed into the next layer / the enclosing block encoding).

    ``n_blocks == 1`` and ``n_controls == 0`` is a single fixed one-qubit gate (no address): the
    phase words are classical constants applied as ``phase_words`` constant phase-gradient additions
    with no QROAM.
    """

    n_blocks: SymbolicInt
    n_controls: SymbolicInt
    phase_bitsize: SymbolicInt
    phase_words: int = 4
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        if not is_symbolic(self.n_controls):
            assert self.n_controls >= 0
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        assert self.phase_words >= 2, "Eq. 24 needs at least two phase words (theta, phi)"

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def has_ctrl(self) -> bool:
        return not is_symbolic(self.n_controls) and int(self.n_controls) > 0

    @property
    def has_qroam(self) -> bool:
        return self.has_block or self.has_ctrl

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            controls=self.n_controls,
            target=1,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        sel: List[SymbolicInt] = []
        if self.has_block:
            sel.append(self.block_bitsize)
        if self.has_ctrl:
            sel.append(self.n_controls)
        return tuple(sel)

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        shape: List[SymbolicInt] = []
        if self.has_block:
            shape.append(self.n_blocks)
        if self.has_ctrl:
            shape.append(1 << int(self.n_controls))
        return tuple(shape)

    @property
    def qroam_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _qroam_log_block_sizes(self.log_block_sizes, self.qroam_data_shape)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize,) * self.phase_words,
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=self.qroam_log_block_sizes,
        )

    @property
    def _ctrl_add(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()

    def _apply_eq24(self, bb, target, words, phase_grad):
        """Apply ``phase_words`` controlled phase-gradient additions interleaved with two Hadamards.

        The exact branch (target == 0 vs 1) of each diagonal phase does not change the resource
        count; the data-free layout uses the ``target``-controlled add for every word and places the
        two Hadamards around the middle word, matching the Eq.-24 ``Dl H P H Dr`` structure.
        """
        ctrl_add = self._ctrl_add
        h_after = max(0, len(words) // 2 - 1)  # second Hadamard lands mid-schedule
        target = bb.add(Hadamard(), q=target)
        for i, w in enumerate(words):
            target, w, phase_grad = bb.add(ctrl_add, ctrl=target, x=w, phase_grad=phase_grad)
            words[i] = w
            if i == h_after:
                target = bb.add(Hadamard(), q=target)
        return target, words, phase_grad

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_blocks, self.n_controls, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")
        target = soqs['target']
        phase_grad = soqs['phase_gradient']

        if not self.has_qroam:
            # Fixed single-qubit gate: phase words are classical constants (no lookup).
            words = [bb.allocate(self.phase_bitsize) for _ in range(self.phase_words)]
            target, words, phase_grad = self._apply_eq24(bb, target, words, phase_grad)
            for w in words:
                bb.free(w)
            return {'target': target, 'phase_gradient': phase_grad}

        qroam = self.qroam_bloq_for_cost
        sel_names = [r.name for r in qroam.selection_registers]
        sel_in: Dict[str, SoquetT] = {}
        idx = 0
        if self.has_block:
            sel_in[sel_names[idx]] = soqs['block']
            idx += 1
        if self.has_ctrl:
            sel_in[sel_names[idx]] = soqs['controls']
        qroam_out = bb.add_d(qroam, **sel_in)

        words = [qroam_out[f'target{i}_'] for i in range(self.phase_words)]
        target, words, phase_grad = self._apply_eq24(bb, target, words, phase_grad)

        # Measurement-reset erase the loaded data + junk (0 Toffoli).
        for i, treg in enumerate(qroam.target_registers):
            _measure_x_reset(bb, words[i])
            junk_name = 'junk_' + treg.name
            if junk_name in qroam_out:
                _measure_x_reset(bb, qroam_out[junk_name])

        out: Dict[str, SoquetT] = {'target': target, 'phase_gradient': phase_grad}
        idx = 0
        if self.has_block:
            out['block'] = qroam_out[sel_names[idx]]
            idx += 1
        if self.has_ctrl:
            out['controls'] = qroam_out[sel_names[idx]]
        return out

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if self.has_qroam:
            ret[self.qroam_bloq_for_cost] += 1
        ret[self._ctrl_add] += self.phase_words
        ret[Hadamard()] += 2
        return ret

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> 'Tuple[Bloq, AddControlledT]':
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledCSDMuxPhaseLayerQROAM(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledCSDMuxPhaseLayerQROAM(GateWithRegisters):
    """Singly-controlled :class:`CSDMuxPhaseLayerQROAM`.

    Only the ``phase_words`` phase-gradient additions become doubly-controlled; the QROAM load,
    Hadamards, and measurement-reset stay uncontrolled (they self-cancel / are identity when the
    external control is 0).
    """

    inner: CSDMuxPhaseLayerQROAM

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if self.inner.has_qroam:
            ret[self.inner.qroam_bloq_for_cost] += 1
        ret[self.inner._ctrl_add.controlled()] += self.inner.phase_words
        ret[Hadamard()] += 2
        return ret


# ============================================================================
# Top-level recursive-CSD synthesizer (block-aware; n_blocks = 1 is ordinary)
# ============================================================================


@attrs.frozen
class RecursiveCSDSynthesisQROAM(GateWithRegisters):
    r"""Block-diagonal dense unitary synthesis by recursive CSD with Eq.-24 / Eq.-40 phase lookups.

    Synthesizes ``sum_a |a><a| (x) U_a`` where each ``U_a`` is an ``N x N`` unitary (``N = 2^n``).
    ``n_blocks = 1`` is a single ``N x N`` unitary.  See the module docstring for the algorithm.

    Args:
        n_blocks: number of block-diagonal blocks ``N_k`` (read-only address into every QROAM).
        n_rows: ``N = 2^n`` (must be a power of two).
        phase_bitsize: phase-word bitsize ``b`` (fixed-point precision of every stored angle).
        block_log_size: internal block exponent ``m`` (``M = 2^m``).  Multiplexed-gate levels with
            ``c >= m`` are *stable* lookups; levels with ``c < m`` are *inner* reconstructions.
            Defaults to ``n`` (the whole register is one ``M``-block -> everything reconstructed
            locally, nothing precomputed for the full circuit).
        phase_words: Eq.-24 phase words per one-qubit gate (default 4).
        optimal_T: pick the per-layer Toffoli-optimal QROAM ``Lambda`` (else QROAM's default split).
        log_block_sizes: explicit per-layer QROAM split when ``optimal_T`` is False.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    block_log_size: Optional[SymbolicInt] = None
    phase_words: int = 4
    optimal_T: bool = False
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows):
            assert _positive_power_of_two(self.n_rows), "n_rows must be a power of two"
            assert self.n_rows >= 2
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        assert self.phase_words >= 2
        if self.block_log_size is not None and not is_symbolic(self.block_log_size, self.n_rows):
            assert 0 <= int(self.block_log_size) <= int(self.system_bitsize)
        if self.optimal_T and is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise ValueError("optimal_T=True requires concrete n_blocks, n_rows, phase_bitsize")

    @classmethod
    def from_shape(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        block_log_size: Optional[SymbolicInt] = None,
        phase_words: int = 4,
        optimal_T: bool = False,
        log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    ) -> 'RecursiveCSDSynthesisQROAM':
        return cls(
            n_blocks=n_blocks,
            n_rows=n_rows,
            phase_bitsize=phase_bitsize,
            block_log_size=block_log_size,
            phase_words=phase_words,
            optimal_T=optimal_T,
            log_block_sizes=log_block_sizes,
        )

    @classmethod
    def from_unitary(
        cls, U: NDArray[np.complex128], phase_bitsize: SymbolicInt, **kwargs
    ) -> 'RecursiveCSDSynthesisQROAM':
        """Convenience constructor inferring ``n_rows`` from a single dense unitary (``n_blocks=1``)."""
        U = np.asarray(U)
        assert U.ndim == 2 and U.shape[0] == U.shape[1]
        return cls.from_shape(1, U.shape[0], phase_bitsize, **kwargs)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def m(self) -> int:
        """Internal block exponent ``m`` (defaults to the full register width ``n``)."""
        if self.block_log_size is None:
            return int(self.system_bitsize)
        return int(self.block_log_size)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize, system=self.system_bitsize, phase_gradient=self.phase_bitsize
        )

    def _layer_log_block_sizes(self, c: int) -> Optional[Tuple[SymbolicInt, ...]]:
        if self.optimal_T:
            return _optimal_layer_log_block_sizes(
                int(self.n_blocks), c, int(self.phase_bitsize), self.phase_words
            )
        return self.log_block_sizes

    def phase_layer(self, c: int) -> CSDMuxPhaseLayerQROAM:
        """The multiplexed-gate phase layer with ``c`` controls (table shape ``(n_blocks, 2^c)``)."""
        return CSDMuxPhaseLayerQROAM(
            n_blocks=self.n_blocks,
            n_controls=c,
            phase_bitsize=self.phase_bitsize,
            phase_words=self.phase_words,
            log_block_sizes=self._layer_log_block_sizes(c),
        )

    def is_lookup_level(self, c: int) -> bool:
        """True for *stable / lookup* levels ``c >= m``; False for *inner / reconstruction* ``c < m``."""
        return c >= self.m

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.n_rows):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self}")
        n = int(self.system_bitsize)
        ret: 'Counter[Bloq]' = Counter()
        for c in range(n):
            count = num_csd_muxes(n, c)
            if count:
                ret[self.phase_layer(c)] += count
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")
        n = int(self.system_bitsize)
        has_block = self.has_block
        block = soqs.get('block')
        system = soqs['system']
        phase_grad = soqs['phase_gradient']

        for target_q, controls in csd_mux_schedule(n):
            c = len(controls)
            layer = self.phase_layer(c)
            qubits = bb.split(system)
            lsoqs: Dict[str, SoquetT] = {'target': qubits[target_q], 'phase_gradient': phase_grad}
            if has_block:
                lsoqs['block'] = block
            if c > 0:
                ctrl_reg = bb.join(np.array([qubits[i] for i in controls]), dtype=QUInt(c))
                lsoqs['controls'] = ctrl_reg
            out = bb.add_d(layer, **lsoqs)
            phase_grad = out['phase_gradient']
            if has_block:
                block = out['block']
            qubits[target_q] = out['target']
            if c > 0:
                ctrl_qubits = bb.split(out['controls'])
                for j, i in enumerate(controls):
                    qubits[i] = ctrl_qubits[j]
            system = bb.join(qubits, dtype=QUInt(n))

        result: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
        if has_block:
            result['block'] = block
        return result

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    def resource_estimate(self) -> 'RecursiveCSDResourceEstimate':
        """Symbolic-friendly aggregate resource estimate (see :class:`RecursiveCSDResourceEstimate`)."""
        return estimate_recursive_csd_resources(
            int(self.n_blocks),
            int(self.n_rows),
            int(self.phase_bitsize),
            block_log_size=self.m,
            phase_words=self.phase_words,
            optimal_T=self.optimal_T,
        )

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> 'Tuple[Bloq, AddControlledT]':
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledRecursiveCSDSynthesisQROAM(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledRecursiveCSDSynthesisQROAM(GateWithRegisters):
    """Singly-controlled :class:`RecursiveCSDSynthesisQROAM` (control reaches only the phase adds)."""

    inner: RecursiveCSDSynthesisQROAM

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.inner.n_rows):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self.inner}")
        n = int(self.inner.system_bitsize)
        ret: 'Counter[Bloq]' = Counter()
        for c in range(n):
            count = num_csd_muxes(n, c)
            if count:
                ret[_ControlledCSDMuxPhaseLayerQROAM(self.inner.phase_layer(c))] += count
        return ret


# ============================================================================
# Symbolic resource estimate
# ============================================================================


@dataclass(frozen=True)
class RecursiveCSDResourceEstimate:
    """Aggregate resource estimate for one :class:`RecursiveCSDSynthesisQROAM` configuration.

    All Toffoli figures are exact under the per-layer QROAM cost model; the lookup/reconstruction
    split is set by the internal block size ``M = 2^m`` (stable outer levels ``c >= m`` vs inner
    levels ``c < m``).
    """

    n_blocks: int
    n_rows: int
    phase_bitsize: int
    block_log_size: int
    phase_words: int
    lookup_toffoli: int          # sum over stable levels c >= m
    reconstruction_toffoli: int  # sum over inner levels c < m
    n_loaded_phase_words: int    # phase_words * n_blocks * (#one-qubit gates)
    n_hadamards: int             # 2 per multiplexed gate
    n_phase_additions: int       # phase_words per multiplexed gate (Eq. 40 adds)
    ancilla: int                 # peak ancilla beyond block+system+phase-gradient registers
    qubits: int                  # total qubit footprint
    total_toffoli: int           # lookup + reconstruction


def _qroam_layer_toffoli(
    n_blocks: int, control_count: int, phase_bitsize: int, phase_words: int, optimal_T: bool
) -> int:
    """Toffoli of one ``CSDMuxPhaseLayerQROAM`` (forward QROAM load + ``phase_words`` controlled adds).

    Erasure is measurement-based (0 Toffoli).  A controlled ``AddIntoPhaseGrad(b, b)`` is charged
    ``b - 1`` Toffoli.  The QROAM forward cost uses the standard ``ceil(entries/Lambda) +
    phase_words*b*(Lambda - 1)`` model with the per-layer ``Lambda`` (optimal or 1).
    """
    add_t = phase_words * max(0, phase_bitsize - 1)
    if n_blocks <= 1 and control_count == 0:
        return add_t  # fixed gate, no lookup
    entries = max(1, n_blocks) * (1 << control_count)
    if optimal_T and control_count > 0:
        lam = max(1.0, sqrt(entries / max(1, phase_words * phase_bitsize)))
        lam = 2 ** max(0, round(log2(lam)))
    else:
        lam = 1
    load_t = ceil(entries / lam) + phase_words * phase_bitsize * (lam - 1)
    return int(load_t + add_t)


def _qroam_layer_ancilla(
    n_blocks: int, control_count: int, phase_bitsize: int, phase_words: int, optimal_T: bool
) -> int:
    """Peak clean ancilla of one phase layer: ``phase_words * b * Lambda`` loaded data qubits."""
    if n_blocks <= 1 and control_count == 0:
        return phase_words * phase_bitsize  # transient constant register
    entries = max(1, n_blocks) * (1 << control_count)
    if optimal_T and control_count > 0:
        lam = max(1.0, sqrt(entries / max(1, phase_words * phase_bitsize)))
        lam = 2 ** max(0, round(log2(lam)))
    else:
        lam = 1
    return int(phase_words * phase_bitsize * lam)


def estimate_recursive_csd_resources(
    n_blocks: int,
    n_rows: int,
    phase_bitsize: int,
    *,
    block_log_size: Optional[int] = None,
    phase_words: int = 4,
    optimal_T: bool = False,
) -> RecursiveCSDResourceEstimate:
    """Closed-form cost of the *explicit, un-grouped* circuit (the Sec. 3.1 naive route, ``~ N^2``).

    Each multiplexed gate is charged a QROAM phase layer individually.  For the paper's optimal
    ``~ N^{4/3}`` T-count use :func:`estimate_paper_unitary_resources` instead (see module docstring).
    """
    assert _positive_power_of_two(n_rows)
    assert n_blocks >= 1
    assert phase_bitsize > 1
    assert phase_words >= 2
    n = int(log2(n_rows))
    m = n if block_log_size is None else int(block_log_size)

    lookup_t = 0
    recon_t = 0
    n_muxes_total = 0
    n_gates_total = 0
    peak_layer_anc = 0
    for c in range(n):
        count = num_csd_muxes(n, c)
        if not count:
            continue
        n_muxes_total += count
        n_gates_total += count * (1 << c)
        layer_t = _qroam_layer_toffoli(n_blocks, c, phase_bitsize, phase_words, optimal_T)
        total_c = count * layer_t
        if c >= m:
            lookup_t += total_c
        else:
            recon_t += total_c
        peak_layer_anc = max(
            peak_layer_anc,
            _qroam_layer_ancilla(n_blocks, c, phase_bitsize, phase_words, optimal_T),
        )

    n_loaded = phase_words * n_blocks * n_gates_total
    n_hadamards = 2 * n_muxes_total
    n_phase_adds = phase_words * n_muxes_total
    base_qubits = bit_length(n_blocks - 1) + n + phase_bitsize
    qubits = base_qubits + peak_layer_anc

    return RecursiveCSDResourceEstimate(
        n_blocks=n_blocks,
        n_rows=n_rows,
        phase_bitsize=phase_bitsize,
        block_log_size=m,
        phase_words=phase_words,
        lookup_toffoli=int(lookup_t),
        reconstruction_toffoli=int(recon_t),
        n_loaded_phase_words=int(n_loaded),
        n_hadamards=int(n_hadamards),
        n_phase_additions=int(n_phase_adds),
        ancilla=int(peak_layer_anc),
        qubits=int(qubits),
        total_toffoli=int(lookup_t + recon_t),
    )


# ============================================================================
# Paper-faithful (grouped) cost model -- the N^{4/3} route of arXiv:2509.25702
# ============================================================================


@dataclass(frozen=True)
class PaperUnitaryEstimate:
    """Grouped recursive-CSD T-count estimate (Tan, arXiv:2509.25702, Thm 4.3 + Thm 1.1).

    Leading-order (O-notation constants set to 1) cost of synthesizing an arbitrary ``n``-qubit
    unitary to error ``eps`` by grouping the ``2^n - 1`` multi-controlled single-qubit unitaries into
    ``2^{n-k}`` multi-controlled ``k``-qubit unitaries, each synthesized at T-count
    ``2^{(n+k)/2} sqrt(L) + 4^k L`` with ``L = n + log2(1/eps)``.
    """

    n: int
    N: int
    phase_bitsize: int
    k: int                 # internal block size exponent (target-register width); optimal ~ n/3
    n_blocks: int          # 2^{n-k} grouped multi-controlled k-qubit unitaries
    t_per_block: float     # Thm 4.3 per-block T-count
    t_count: float         # total T-count = n_blocks * t_per_block
    ancilla: float         # peak ancillae ~ 2^{(n+k)/2} sqrt(L) + L


def _optimal_kblock_log_block_sizes(
    n_controls: int, num_words: int, phase_bitsize: int, n_blocks: int = 1
) -> Optional[Tuple[int, ...]]:
    """Select-swap ``lambda* ~ sqrt(N_k * 2^{n_controls} / (num_words * L))`` for the block QROAM.

    The lookup is addressed by ``(block label a, n_controls stable controls)`` -> table
    ``(n_blocks, 2^{n_controls})``.  With a block register the optimum is split across the two
    selection dimensions; without one it is a single split over the controls.
    """
    D = n_blocks * (1 << n_controls)
    lam = max(1.0, sqrt(D / max(1, num_words * phase_bitsize)))
    if n_blocks > 1:
        # split lambda across (block, control) dims for the (n_blocks, 2^{n_controls}) table.
        return split_interferometer_log_block_sizes(
            2 ** max(0, round(log2(lam))), n_blocks, 1 << (n_controls + 1)
        )
    if n_controls == 0:
        return None
    lbs = min(n_controls, max(0, round(log2(lam))))
    return (lbs,)


@attrs.frozen
class MultiControlledKQubitUnitaryQROAM(GateWithRegisters):
    r"""A single ``(n-k)``-controlled ``k``-qubit unitary -- the grouped block of arXiv:2509.25702.

    This is the actual circuit (not a formula) for the paper's building block (Thm 4.3): a QROAM
    addressed by the ``n_controls = n-k`` stable control qubits loads the ``num_words`` (default
    ``4^k``, the parameter count of a ``k``-qubit unitary) ``b``-bit phase words describing the
    target-block unitary; the words are applied to the ``k`` target qubits through the phase gradient
    (Eq. 40) interleaved with the Hadamards of the generalized GKW diagonal synthesis, then erased by
    X-measurement (0 Toffoli).  Counted by ``QECGatesCost`` the cost is the select-swap-amortized
    lookup ``~ sqrt(N_k) 2^{(n+k)/2} sqrt(b)`` plus the reconstruction ``~ 4^k b`` -- i.e. Thm 4.3
    emerges from real gate counting, not an analytic assertion.  For a block-diagonal unitary
    ``sum_a |a><a| (x) U_a`` the ``N_k = n_blocks`` block label ``a`` is appended to the lookup address
    (table ``(n_blocks, 2^{n_controls})``), so only the lookup term grows -- by ``sqrt(N_k)`` under the
    select-swap optimum.  ``n_blocks = 1`` is the single-unitary case.
    """

    n_controls: SymbolicInt
    k: SymbolicInt
    phase_bitsize: SymbolicInt
    n_blocks: SymbolicInt = 1
    num_words: Optional[int] = None  # defaults to 4^k
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    optimal_T: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_controls):
            assert self.n_controls >= 1, "use the k=n single-block case via n_controls>=1"
        if not is_symbolic(self.k):
            assert self.k >= 1
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1

    @property
    def words(self) -> int:
        return int(self.num_words) if self.num_words is not None else 4 ** int(self.k)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            controls=self.n_controls,
            target=self.k,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def _effective_lbs(self) -> Optional[Tuple[SymbolicInt, ...]]:
        if self.optimal_T:
            return _optimal_kblock_log_block_sizes(
                int(self.n_controls), self.words, int(self.phase_bitsize), int(self.n_blocks)
            )
        return self.log_block_sizes

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        D = 1 << int(self.n_controls)
        return (int(self.n_blocks), D) if self.has_block else (D,)

    @property
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        return (self.block_bitsize, self.n_controls) if self.has_block else (self.n_controls,)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize,) * self.words,
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=_qroam_log_block_sizes(self._effective_lbs, self.qroam_data_shape),
        )

    @property
    def _ctrl_add(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()

    @property
    def _n_hadamards(self) -> int:
        # the k-qubit unitary is 2^k - 1 multi-controlled single-qubit gates, each two Hadamards
        return 2 * (2 ** int(self.k) - 1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        ret[self._ctrl_add] += self.words
        ret[Hadamard()] += self._n_hadamards
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_controls, self.k, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")
        controls = soqs['controls']
        target = soqs['target']
        phase_grad = soqs['phase_gradient']

        qroam = self.qroam_bloq_for_cost
        sel_names = [r.name for r in qroam.selection_registers]
        sel_in: Dict[str, SoquetT] = {}
        if self.has_block:
            sel_in[sel_names[0]] = soqs['block']
            sel_in[sel_names[1]] = controls
        else:
            sel_in[sel_names[0]] = controls
        qroam_out = bb.add_d(qroam, **sel_in)
        controls = qroam_out[sel_names[1]] if self.has_block else qroam_out[sel_names[0]]
        words = [qroam_out[f'target{i}_'] for i in range(self.words)]

        tq = bb.split(target)
        ctrl_add = self._ctrl_add
        n_h = self._n_hadamards
        # Interleave the Eq.-40 additions with the GKW Hadamards on the k target qubits.
        for i, w in enumerate(words):
            tq_i = i % int(self.k)
            tq[tq_i], w, phase_grad = bb.add(ctrl_add, ctrl=tq[tq_i], x=w, phase_grad=phase_grad)
            words[i] = w
        for h in range(n_h):
            qi = h % int(self.k)
            tq[qi] = bb.add(Hadamard(), q=tq[qi])
        target = bb.join(tq, dtype=QUInt(int(self.k)))

        for i, treg in enumerate(qroam.target_registers):
            _measure_x_reset(bb, words[i])
            junk_name = 'junk_' + treg.name
            if junk_name in qroam_out:
                _measure_x_reset(bb, qroam_out[junk_name])

        out: Dict[str, SoquetT] = {'controls': controls, 'target': target, 'phase_gradient': phase_grad}
        if self.has_block:
            out['block'] = qroam_out[sel_names[0]]
        return out


@attrs.frozen
class PaperRecursiveCSDUnitarySynthesis(GateWithRegisters):
    r"""Arbitrary ``N x N`` unitary synthesis via the grouped recursive CSD of arXiv:2509.25702.

    Constructs the circuit as a sequence of ``2^{n-k}`` :class:`MultiControlledKQubitUnitaryQROAM`
    blocks (the paper's grouping of the ``2^n-1`` multi-controlled single-qubit unitaries into
    multi-controlled ``k``-qubit unitaries).  The total ``QECGatesCost`` therefore comes from real
    gate counting; minimizing over ``k`` reproduces the paper's ``~ N^{4/3}`` (up to logs) T-count.

    The blocks act on the same ``k``-qubit target (the low ``k`` qubits) controlled by the other
    ``n-k`` qubits; in this data-free resource model every block is identical by shape, so the count
    is ``2^{n-k}`` (the data, which differs per block, is not materialized -- the standard convention
    of the sibling ``*QROAM`` resource models).
    """

    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    block_log_size: SymbolicInt  # k (target-block width)
    n_blocks: SymbolicInt = 1    # N_k diagonal blocks (block-diagonal unitary); 1 = single unitary
    optimal_T: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows):
            assert _positive_power_of_two(self.n_rows) and self.n_rows >= 2
        if not is_symbolic(self.n_rows, self.block_log_size):
            assert 1 <= int(self.block_log_size) <= int(self.system_bitsize) - 1, "need 1 <= k <= n-1"
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1

    @property
    def system_bitsize(self) -> int:
        return int(bit_length(self.n_rows - 1))

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def k(self) -> int:
        return int(self.block_log_size)

    @property
    def n_controls(self) -> int:
        return self.system_bitsize - self.k

    @property
    def n_block_ops(self) -> int:
        return 1 << self.n_controls  # 2^{n-k}

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize, system=self.system_bitsize, phase_gradient=self.phase_bitsize
        )

    @property
    def block(self) -> MultiControlledKQubitUnitaryQROAM:
        return MultiControlledKQubitUnitaryQROAM(
            n_controls=self.n_controls, k=self.k, phase_bitsize=self.phase_bitsize,
            n_blocks=self.n_blocks, optimal_T=self.optimal_T,
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.n_rows):
            raise DecomposeTypeError(f"cannot enumerate blocks for symbolic {self}")
        return Counter({self.block: self.n_block_ops})

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_rows, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")
        n, k = self.system_bitsize, self.k
        system = soqs['system']
        phase_grad = soqs['phase_gradient']
        block_reg = soqs.get('block')
        block = self.block
        for _ in range(self.n_block_ops):
            q = bb.split(system)
            controls = bb.join(q[: n - k], dtype=QUInt(n - k))
            target = bb.join(q[n - k:], dtype=QUInt(k))
            in_soqs: Dict[str, SoquetT] = {
                'controls': controls, 'target': target, 'phase_gradient': phase_grad
            }
            if self.has_block:
                in_soqs['block'] = block_reg
            out = bb.add_d(block, **in_soqs)
            phase_grad = out['phase_gradient']
            if self.has_block:
                block_reg = out['block']
            system = bb.join(
                np.concatenate([bb.split(out['controls']), bb.split(out['target'])]), dtype=QUInt(n)
            )
        result: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
        if self.has_block:
            result['block'] = block_reg
        return result


def constructed_paper_unitary_tcount(n: int, phase_bitsize: int, k: int, n_blocks: int = 1) -> int:
    """Real ``QECGatesCost`` T-count of the constructed :class:`PaperRecursiveCSDUnitarySynthesis`.

    ``n_blocks = N_k`` synthesizes the block-diagonal unitary ``sum_a |a><a| (x) U_a`` (the block label
    is appended to every lookup); ``n_blocks = 1`` is a single ``2^n x 2^n`` unitary.
    """
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.resource_counting.generalizers import generalize_cswap_approx

    bloq = PaperRecursiveCSDUnitarySynthesis(1 << n, phase_bitsize, k, n_blocks=n_blocks, optimal_T=True)
    cost = get_cost_value(bloq, QECGatesCost(), generalizer=generalize_cswap_approx)
    return int(cost.total_t_count())


def optimal_constructed_paper_unitary(
    n: int, phase_bitsize: int, *, n_blocks: int = 1, k_max: int = 6
) -> Tuple[int, int]:
    """Return ``(k*, T-count)`` minimizing the constructed circuit's T-count over ``k``.

    The search is capped at ``k_max`` because each block materializes ``4^k`` QROAM target registers
    (infeasible for large ``k``); this never misses the optimum, since the ``4^k b`` reconstruction
    term makes large ``k`` strictly worse here (the asymptotic optimum ``k ~ n/3`` is small for the
    constructible range).  ``n_blocks = N_k`` is the block-diagonal case.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    ks = range(1, min(n, k_max + 1))
    best_k = min(ks, key=lambda k: constructed_paper_unitary_tcount(n, phase_bitsize, k, n_blocks))
    return best_k, constructed_paper_unitary_tcount(n, phase_bitsize, best_k, n_blocks)


def paper_block_tcount(n: int, k: int, L: float) -> float:
    """Thm 4.3 (eps-version): T-count of one ``(n-k)``-controlled ``k``-qubit unitary.

    ``2^{(n+k)/2} sqrt(L) + 4^k L`` -- the GKW polynomial-factoring synthesis: a ``sqrt``-amortized
    lookup term over the ``n-k`` stable controls plus a ``4^k`` reconstruction term for the ``k``-qubit
    target block.  ``L = n + log2(1/eps)`` is the precision (Hadamard+T word length per one-qubit gate).
    """
    return 2.0 ** ((n + k) / 2.0) * sqrt(L) + 4.0 ** k * L


def estimate_paper_unitary_resources(
    n: int, phase_bitsize: int, *, k: Optional[int] = None, eps: Optional[float] = None
) -> PaperUnitaryEstimate:
    """Optimal-T-count estimate for ``U(2^n)`` synthesis via the grouped recursive-CSD scheme.

    ``L = n + log2(1/eps)``; by default ``eps = 2^{-phase_bitsize}`` so ``L = n + phase_bitsize``.  With
    ``k`` unset, minimizes the total T-count over integer ``k in [1, n-1]`` (the optimum approaches the
    asymptotic ``k ~ n/3``; for ``n`` comparable to ``L`` it sits lower because the ``4^k L`` term bites).
    Returns the leading-order Thm 1.1 estimate -- ``O(2^{4n/3} L^{2/3})`` at the optimal ``k``.
    """
    assert n >= 1 and phase_bitsize > 0
    L = (n + phase_bitsize) if eps is None else (n + log2(1.0 / eps))

    def total(kk: int) -> float:
        return 2.0 ** (n - kk) * paper_block_tcount(n, kk, L)

    if n == 1:
        kbest = 1
    elif k is not None:
        assert 1 <= k <= n - 1
        kbest = int(k)
    else:
        kbest = min(range(1, n), key=total)
    n_blocks = 2 ** (n - kbest)
    t_per = paper_block_tcount(n, kbest, L)
    ancilla = 2.0 ** ((n + kbest) / 2.0) * sqrt(L) + L
    return PaperUnitaryEstimate(
        n=n,
        N=1 << n,
        phase_bitsize=phase_bitsize,
        k=kbest,
        n_blocks=int(n_blocks),
        t_per_block=float(t_per),
        t_count=float(n_blocks * t_per),
        ancilla=float(ancilla),
    )
