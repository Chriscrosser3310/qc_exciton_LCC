r"""Interferometer synthesis of the first ``M`` columns of a unitary (isometry).

This implements Sec. III B (``Synthesis of columns of a unitary``) of

    Berry, Tong, Khattar, White, Kim, Boixo, Lin, Lee, Chan, Babbush, Rubin,
    "Rapid initial state preparation for the quantum simulation of strongly
    correlated molecules", arXiv:2409.11748 (PRX Quantum 6, 020327).

Where the *full* unitary interferometer of Sec. III A
(:class:`BlockUnitaryInterferometerSynthesisQROAM`) synthesizes every column of an
``N x N`` unitary, Sec. III B synthesizes only the first ``M`` columns -- an
``N x M`` isometry -- at a cost that scales with ``M`` rather than ``N``.

Parameter ``d``.  With ``N`` rows and ``M`` synthesized columns the paper's reduction
parameter is

    ``d = N / M``    (required to be a positive integer),

so ``M = N / d``.  ``M`` (= the bond dimension ``chi``) and ``N`` are powers of two.

Construction (Eq. 36 and surrounding text).  The ``d = 2`` step synthesizes ``M`` columns
("half the columns") of a ``2M x 2M`` unitary as, splitting off the top qubit ``q_0``,

    ``A = U_1 cos(theta) V,    B = U_2 sin(theta) V`` ,                       (Eq. 29)

i.e. (1) a full ``M x M`` interferometer ``V`` on the bottom ``m`` qubits; (2) a
multiplexed ``R_y(2 theta_j)`` on ``q_0`` controlled by those ``m`` qubits (Eq. 32);
(3) the ``q_0``-controlled pair ``U_1`` / ``U_2`` -- a 2-block ``M x M`` interferometer
(Eq. 33).  For ``d > 2`` the paper *iterates this ``d = 2`` step ``d - 1`` times*: a thin
QR of the bottom ``(d-1)M`` rows (Eq. 36) leaves the first ``M`` columns of a
``(d-1)M x (d-1)M`` unitary -- the same problem with ``d`` reduced by one -- so the cost
is "multiplied by ``d - 1``" and is **linear in ``d``**.

This module realizes that staircase directly.  Crucially, the initial ``M x M`` unitary
``V`` at each level **merges with the previous level's controlled-``U``** (Sec. III B: it
"can be combined with prior operations and does not add to the cost"), so the circuit is

  * one initial ``V_0`` -- a full ``M x M`` interferometer on the bottom ``m`` qubits; then
  * ``d - 1`` levels, each a multiplexed ``R_y`` (Eq. 32) followed by ONE 2-block ``M x M``
    interferometer (the merged ``U_i (I (x) V_{i+1})``), on the bottom ``m + 1``-qubit window;
  * an ``AddK`` cyclic shift of the system register by ``M`` between levels advances the
    active adjacent block-pair window through the ``(d*M)``-dim register (Eq. 34).

So the whole construction uses just ``d`` full ``M x M`` interferometers (``1`` initial ``V_0``
+ ``d - 1`` controlled ``U``'s) and ``d - 1`` rotations -- NOT ``2(d-1)`` interferometers --
matching Eq. 33's ``N_un/2`` phasing layers per ``d = 2`` step.  ``d`` need not be a power of
two (e.g. ``d = 3`` -> ``N = 3M``; the register is padded to ``2^ceil(log2 N)``).  ``d = 1``
(``M = N``) is exactly the Sec. III A full ``M x M`` interferometer.

Resource-model status (as for the sibling ``*_QROAM`` modules):
:meth:`build_composite_bloq` lays out the real gate structure with shape-only (data-free)
QROAM, :meth:`build_call_graph` walks the actual sub-bloq graph, and the module-level
numpy helpers (:func:`isometry_qr_peel_steps`, :func:`reconstruct_isometry_from_qr_peel`)
are a standalone reference proving the QR-peeling staircase reproduces the first ``M``
columns of a target isometry (with exactly ``d - 1`` ``d = 2`` steps) to machine precision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2, sqrt
from typing import Dict, Optional, Tuple, TYPE_CHECKING

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
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        _ControlledBlockUnitaryInterferometerSynthesisQROAM,
        _positive_power_of_two,
        _qroam_log_block_sizes,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script/direct execution
    from block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        _ControlledBlockUnitaryInterferometerSynthesisQROAM,
        _positive_power_of_two,
        _qroam_log_block_sizes,
    )
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


# ============================================================================
# numpy reference: thin-row-CSD recursion -> flat interferometer-isometry schedule
# ============================================================================


@dataclass(frozen=True)
class IsometryQRPeelSteps:
    """The Eq.-36 QR-peeling staircase of a ``(d*M) x M`` isometry (numpy reference).

    ``steps`` is a list of ``(block_i, U2)`` in circuit order (first applied first): ``U2`` is a
    ``2M x 2M`` unitary whose first ``M`` columns are the ``d=2`` isometry synthesized at that
    iteration, acting on the adjacent block pair ``{block_i, block_i+1}`` of the ``(d*M)``-dim
    register (block ``j`` = indices ``j*M .. (j+1)*M-1``).  There are exactly ``d-1`` steps.
    """

    d: int
    m: int  # log2 M
    steps: Tuple[Tuple[int, NDArray[np.complex128]], ...]


def _complete_to_unitary(iso: NDArray[np.complex128], rng: np.random.Generator) -> NDArray[np.complex128]:
    """Complete an ``R x C`` isometry (``C <= R``) to an ``R x R`` unitary whose first ``C`` cols are ``iso``."""
    R, C = iso.shape
    extra = rng.standard_normal((R, R - C)) + 1j * rng.standard_normal((R, R - C))
    Q, _ = np.linalg.qr(np.concatenate([iso, extra], axis=1))
    Q[:, :C] = iso
    return Q


def isometry_qr_peel_steps(W: NDArray[np.complex128], *, seed: int = 0) -> IsometryQRPeelSteps:
    r"""Decompose a ``(d*M) x M`` isometry into the Eq.-36 ``d-1`` ``d=2`` staircase steps.

    Implements the recursion of arXiv:2409.11748 Sec. III B: split the rows of the
    ``(D*M) x M`` isometry into the top block ``A`` (``M x M``) and the rest ``Brest``
    (``(D-1)M x M``); a thin QR ``Brest = Q R_{11}`` gives the ``d=2`` isometry
    ``[A; R_{11}]`` (``2M x M``, synthesized on the top block pair) and reduces the problem
    to the first ``M`` columns of the ``(D-1)M``-dim unitary ``Q`` (recursed one block down).
    """
    W = np.asarray(W, dtype=complex)
    N, M = W.shape
    m = int(round(log2(M)))
    assert (1 << m) == M and N % M == 0, "M must be a power of two dividing N"
    d = N // M
    rng = np.random.default_rng(seed)
    steps: List[Tuple[int, NDArray[np.complex128]]] = []

    def rec(Wk: NDArray[np.complex128], base_block: int) -> None:
        Dk = Wk.shape[0] // M
        if Dk == 2:  # base: synthesize the whole 2M x M isometry directly (the explicit d=2 step)
            steps.append((base_block, _complete_to_unitary(Wk, rng)))
            return
        A = Wk[:M, :]
        Q, R11 = np.linalg.qr(Wk[M:, :])          # Q: (Dk-1)M x M, R11: M x M
        d2_iso = np.vstack([A, R11])               # 2M x M  (the d=2 isometry on the top block pair)
        steps.append((base_block, _complete_to_unitary(d2_iso, rng)))
        rec(Q, base_block + 1)

    if d == 1:  # full M x M unitary -- no QR peeling
        return IsometryQRPeelSteps(d=1, m=m, steps=())
    rec(W, 0)
    return IsometryQRPeelSteps(d=d, m=m, steps=tuple(steps))


def reconstruct_isometry_from_qr_peel(peel: IsometryQRPeelSteps) -> NDArray[np.complex128]:
    """Build the full ``(d*M) x (d*M)`` operator from the staircase; its first ``M`` columns are ``W``."""
    M = 1 << peel.m
    N = peel.d * M
    Op = np.eye(N, dtype=complex)
    for block, U2 in peel.steps:  # circuit order: first applied first
        lo = block * M
        G = np.eye(N, dtype=complex)
        G[lo:lo + 2 * M, lo:lo + 2 * M] = U2
        Op = G @ Op
    return Op


# ============================================================================
# Multiplexed R_y layer (Eq. 29 cosine-sine rotation) backed by QROAM
# ============================================================================


@attrs.frozen
class InterferometerIsometryMuxRotationQROAM(GateWithRegisters):
    r"""Multiplexed ``R_y`` on one target qubit, angles loaded from QROAM.

    Applies ``R_y(2 theta_{x, j})`` to ``target`` for address ``(block x, system j)``, i.e.
    the cosine-sine rotation ``|j>|0> -> |j>(cos theta |0> + sin theta |1>)`` of Eq. 29.
    The angle is loaded by ``QROAMClean`` from a ``(n_blocks, n_address)`` table, applied
    through the phase gradient interleaved with two Hadamards (the standard
    Hadamard-sandwich realization of an ``R_y`` on ``target``), and uncomputed with
    ``QROAMCleanAdjoint``.  ``n_blocks == 1`` drops the ``block`` register.
    """

    n_blocks: SymbolicInt
    n_address: SymbolicInt
    phase_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        if not is_symbolic(self.n_address):
            assert _positive_power_of_two(self.n_address) and self.n_address >= 2
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def address_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_address - 1)

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            system=self.address_bitsize,
            target=1,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        if self.has_block:
            return (self.n_blocks, self.n_address)
        return (self.n_address,)

    @property
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        if self.has_block:
            return (self.block_bitsize, self.address_bitsize)
        return (self.address_bitsize,)

    def _capped(self, raw: Optional[Tuple[SymbolicInt, ...]]) -> Optional[Tuple[SymbolicInt, ...]]:
        return _qroam_log_block_sizes(raw, self.qroam_data_shape)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize,),
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=self._capped(self.log_block_sizes),
        )

    @property
    def qroam_adj_bloq_for_cost(self) -> QROAMCleanAdjoint:
        qroam = self.qroam_bloq_for_cost
        return QROAMCleanAdjoint.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            target_shapes=(qroam.block_sizes,),
            log_block_sizes=self._capped(self.adjoint_log_block_sizes),
        )

    @property
    def ctrl_phase_grad_add(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Circuit: QROAM(block, j) -> theta; H target; ctrl-theta -> grad; H target; QROAM^dag."""
        if is_symbolic(self.n_blocks, self.n_address, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")
        has_block = self.has_block
        target = soqs['target']
        phase_grad = soqs['phase_gradient']
        system = soqs['system']

        qroam = self.qroam_bloq_for_cost
        sel_names = [r.name for r in qroam.selection_registers]
        if has_block:
            q_out = bb.add_d(qroam, **{sel_names[0]: soqs['block'], sel_names[1]: system})
            block = q_out[sel_names[0]]
            system = q_out[sel_names[1]]
        else:
            q_out = bb.add_d(qroam, **{sel_names[0]: system})
            system = q_out[sel_names[0]]
        theta = q_out['target0_']

        # R_y(2 theta) on target via Hadamard sandwich + controlled phase-gradient add.
        target = bb.add(Hadamard(), q=target)
        target, theta, phase_grad = bb.add(
            self.ctrl_phase_grad_add, ctrl=target, x=theta, phase_grad=phase_grad
        )
        target = bb.add(Hadamard(), q=target)

        # Clean uncompute of the loaded angle (reshape forward target+junk into the
        # adjoint's expected target shape = the forward QROAM block_sizes).
        qroam_adj = self.qroam_adj_bloq_for_cost
        adj_sel_names = [r.name for r in qroam_adj.selection_registers]
        junk = np.asarray(q_out['junk_target0_']) if 'junk_target0_' in q_out else np.array([])
        adj_target = next(iter(qroam_adj.target_registers))
        adj_soqs: Dict[str, SoquetT] = {
            adj_target.name: np.array([theta, *junk]).reshape(qroam_adj.target_shapes[0])
        }
        if has_block:
            adj_soqs[adj_sel_names[0]] = block
            adj_soqs[adj_sel_names[1]] = system
        else:
            adj_soqs[adj_sel_names[0]] = system
        adj_out = bb.add_d(qroam_adj, **adj_soqs)

        out: Dict[str, SoquetT] = {
            'system': adj_out[adj_sel_names[1]] if has_block else adj_out[adj_sel_names[0]],
            'target': target,
            'phase_gradient': phase_grad,
        }
        if has_block:
            out['block'] = adj_out[adj_sel_names[0]]
        return out

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        ret[self.ctrl_phase_grad_add] += 1
        ret[Hadamard()] += 2
        ret[self.qroam_adj_bloq_for_cost] += 1
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledMuxRotationQROAM(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledMuxRotationQROAM(GateWithRegisters):
    """Singly-controlled :class:`InterferometerIsometryMuxRotationQROAM`.

    Only the single phase-gradient add becomes doubly-controlled (a cc-``R_y`` on the
    target); the QROAM forward/uncompute pair and the two Hadamards stay uncontrolled
    because they self-cancel / are identity when the external control is 0.
    """

    inner: "InterferometerIsometryMuxRotationQROAM"

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        b = self.inner.phase_bitsize
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner.qroam_bloq_for_cost] += 1
        ret[AddIntoPhaseGrad(b, b).controlled().controlled()] += 1
        ret[Hadamard()] += 2
        ret[self.inner.qroam_adj_bloq_for_cost] += 1
        return ret


# ============================================================================
# Single isometry synthesis (no block register; n_blocks implicitly 1)
# ============================================================================


@attrs.frozen
class InterferometerIsometrySynthesisQROAM(GateWithRegisters):
    r"""Synthesize the first ``M`` columns of an ``N x N`` unitary (Sec. III B isometry).

    ``n_rows = N`` and ``n_cols = M`` are powers of two with ``M <= N`` and ``d = N / M``
    a positive integer (a power of two here).  The synthesized operator acts on
    ``system`` (``n = log2 N`` qubits); initialized with the top ``t = log2 d`` qubits in
    ``|0>`` it maps ``|k>`` (``0 <= k < M`` on the bottom ``m = log2 M`` qubits) to column
    ``k`` of the target isometry.  See the module docstring for the layered construction.

    ``d = 1`` (``M = N``) is the full Sec. III A interferometer.
    """

    n_rows: SymbolicInt
    n_cols: SymbolicInt
    phase_bitsize: SymbolicInt
    n_layers: Optional[SymbolicInt] = None
    optimal_T: bool = False
    interferometer_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    interferometer_final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    interferometer_final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    rotation_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    rotation_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_cols):
            # Every V / U is an M x M interferometer; Sec. III A's interferometer needs at least
            # two system qubits, so M = n_cols must be a power of two >= 4.
            assert _positive_power_of_two(self.n_cols) and self.n_cols >= 4
        if not is_symbolic(self.n_rows, self.n_cols):
            # d = N / M must be a positive integer (the paper's reduction parameter); N itself
            # need NOT be a power of two (e.g. d=3 -> N=3M), the register is padded to 2^ceil.
            assert self.n_rows >= self.n_cols and self.n_rows % self.n_cols == 0, \
                "n_rows must be a positive-integer multiple d of n_cols (M)"
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if self.optimal_T and is_symbolic(self.n_rows, self.n_cols, self.phase_bitsize):
            raise ValueError("optimal_T=True requires concrete n_rows, n_cols, phase_bitsize")

    @classmethod
    def from_shape(
        cls,
        n_rows: SymbolicInt,
        n_cols: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_layers: Optional[SymbolicInt] = None,
        optimal_T: bool = False,
        **kwargs,
    ) -> "InterferometerIsometrySynthesisQROAM":
        """Data-free isometry synthesis bloq for resource estimates (``d = n_rows / n_cols``)."""
        return cls(
            n_rows=n_rows,
            n_cols=n_cols,
            phase_bitsize=phase_bitsize,
            n_layers=n_layers,
            optimal_T=optimal_T,
            **kwargs,
        )

    @classmethod
    def from_isometry(
        cls, W: NDArray[np.complex128], phase_bitsize: SymbolicInt, **kwargs
    ) -> "InterferometerIsometrySynthesisQROAM":
        """Infer ``(n_rows, n_cols)`` from a dense ``N x M`` isometry."""
        W = np.asarray(W)
        assert W.ndim == 2 and W.shape[1] <= W.shape[0]
        return cls.from_shape(W.shape[0], W.shape[1], phase_bitsize, **kwargs)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def col_bitsize(self) -> SymbolicInt:
        """``m = log2 M`` -- the number of bottom qubits the interferometers act on."""
        return bit_length(self.n_cols - 1)

    @property
    def d(self) -> SymbolicInt:
        """The reduction parameter ``d = N / M``."""
        return self.n_rows // self.n_cols

    @property
    def n_steps(self) -> SymbolicInt:
        """Number of ``d=2`` staircase steps = ``d - 1`` (Eq. 36); ``d=1`` is one full unitary."""
        return self.d - 1

    @property
    def signature(self) -> Signature:
        return Signature.build(system=self.system_bitsize, phase_gradient=self.phase_bitsize)

    # -- sub-bloq factories --

    def full_interferometer(self, n_blocks_eff: int) -> BlockUnitaryInterferometerSynthesisQROAM:
        """A full ``M x M`` interferometer multiplexed over ``n_blocks_eff`` blocks."""
        return BlockUnitaryInterferometerSynthesisQROAM(
            n_blocks=n_blocks_eff,
            n_rows=self.n_cols,
            phase_bitsize=self.phase_bitsize,
            n_layers=self.n_layers,
            log_block_sizes=self.interferometer_log_block_sizes,
            final_log_block_sizes=self.interferometer_final_log_block_sizes,
            final_adjoint_log_block_sizes=self.interferometer_final_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    def mux_rotation(self, n_blocks_eff: int) -> InterferometerIsometryMuxRotationQROAM:
        """The multiplexed ``R_y`` layer (Eq. 29/32 cosine-sine rotation) addressed by ``(n_blocks_eff, M)``.

        ``optimal_T`` defers the angle-table split to QROAM's own (validated) T-optimal
        heuristic (``log_block_sizes=None``); the rotation tables are tiny relative to the
        interferometers, so this matches the three-phase-layer T-optimal convention used
        elsewhere and sidesteps fully-consuming a small block dimension.
        """
        if self.optimal_T:
            lbs = adj = None
        else:
            lbs, adj = self.rotation_log_block_sizes, self.rotation_adjoint_log_block_sizes
        return InterferometerIsometryMuxRotationQROAM(
            n_blocks=n_blocks_eff,
            n_address=self.n_cols,
            phase_bitsize=self.phase_bitsize,
            log_block_sizes=lbs,
            adjoint_log_block_sizes=adj,
        )

    def _shift(self, k: int) -> AddK:
        """Cyclic shift of the ``system`` register by ``k`` (advances the active 2M window)."""
        return AddK(dtype=QUInt(self.system_bitsize), k=k % (1 << int(self.system_bitsize)))

    def _apply_level(
        self, bb: BloqBuilder, system: SoquetT, phase_grad: SoquetT
    ) -> Tuple[SoquetT, SoquetT]:
        r"""One column-synthesis level on the bottom window: mux-``R_y`` (Eq.32) then 2-block ``U``.

        The controlled-``U`` here is the *merged* ``U_i (I (x) V_{i+1})`` -- the next level's ``V``
        is absorbed into this interferometer at no extra cost (Sec. III B), so a level adds only
        one 2-block ``M x M`` interferometer plus one rotation.  Acts on the bottom ``m + 1``
        qubits: ``q`` (the rotated block bit, position ``n-m-1``) + the bottom ``m`` qubits.
        """
        m, n = int(self.col_bitsize), int(self.system_bitsize)
        qs = bb.split(system)
        q = qs[n - m - 1]
        bottom = bb.join(qs[n - m:], dtype=QUInt(m))

        r_out = bb.add_d(self.mux_rotation(1), system=bottom, target=q, phase_gradient=phase_grad)
        q, bottom, phase_grad = r_out['target'], r_out['system'], r_out['phase_gradient']

        q_reg = bb.join(np.array([q], dtype=object), dtype=QUInt(1))
        u_out = bb.add_d(self.full_interferometer(2), block=q_reg, system=bottom, phase_gradient=phase_grad)
        q = bb.split(u_out['block'])[0]
        bottom, phase_grad = u_out['system'], u_out['phase_gradient']

        qs[n - m - 1] = q
        qs[n - m:] = bb.split(bottom)
        return bb.join(qs, dtype=QUInt(n)), phase_grad

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_rows, self.n_cols, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")
        phase_grad = soqs['phase_gradient']
        system = soqs['system']
        d, M, m, n = int(self.d), int(self.n_cols), int(self.col_bitsize), int(self.system_bitsize)

        # Initial V_0 : a full M x M interferometer on the bottom m qubits (Eq. 29).
        qs = bb.split(system)
        v_out = bb.add_d(self.full_interferometer(1),
                         system=bb.join(qs[n - m:], dtype=QUInt(m)), phase_gradient=phase_grad)
        qs[n - m:] = bb.split(v_out['system'])
        phase_grad = v_out['phase_gradient']
        system = bb.join(qs, dtype=QUInt(n))
        if d == 1:  # full M x M unitary -- just V_0
            return {'system': system, 'phase_gradient': phase_grad}

        # (d-1) column-synthesis levels (Eq. 36), each advancing the window by a -M shift.
        for i in range(d - 1):
            system, phase_grad = self._apply_level(bb, system, phase_grad)
            if i < d - 2:
                system = bb.add(self._shift(-M), x=system)
        if d > 2:
            system = bb.add(self._shift((d - 2) * M), x=system)  # restore the net shift
        return {'system': system, 'phase_gradient': phase_grad}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        if is_symbolic(self.n_rows, self.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self}")
        d, M = int(self.d), int(self.n_cols)
        ret: "Counter[Bloq]" = Counter()
        ret[self.full_interferometer(1)] += 1           # initial V_0
        if d == 1:
            return ret
        # Each of the d-1 levels: a 2-block U (with the next level's V merged in, free) + an R_y.
        ret[self.full_interferometer(2)] += d - 1
        ret[self.mux_rotation(1)] += d - 1
        if d > 2:
            ret[self._shift(-M)] += d - 2               # advance window between levels
            ret[self._shift((d - 2) * M)] += 1          # restore net shift
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledInterferometerIsometrySynthesisQROAM(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledInterferometerIsometrySynthesisQROAM(GateWithRegisters):
    """Singly-controlled :class:`InterferometerIsometrySynthesisQROAM`.

    The control propagates only into the phase-gradient adds of each sub-bloq (the cheap
    ``_Controlled*`` variants); QROAM loads/uncomputes, Hadamards, and the ``AddK`` window
    shifts stay uncontrolled (the shifts net to identity, so they cancel when ctrl=0).
    """

    inner: "InterferometerIsometrySynthesisQROAM"

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        if is_symbolic(self.inner.n_rows, self.inner.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self.inner}")
        inner = self.inner
        d, M = int(inner.d), int(inner.n_cols)
        CI = _ControlledBlockUnitaryInterferometerSynthesisQROAM
        ret: "Counter[Bloq]" = Counter()
        ret[CI(inner.full_interferometer(1))] += 1      # initial V_0
        if d == 1:
            return ret
        ret[CI(inner.full_interferometer(2))] += d - 1
        ret[_ControlledMuxRotationQROAM(inner.mux_rotation(1))] += d - 1
        if d > 2:
            ret[inner._shift(-M)] += d - 2
            ret[inner._shift((d - 2) * M)] += 1
        return ret


# ============================================================================
# Symbolic resource estimate
# ============================================================================


@dataclass(frozen=True)
class InterferometerIsometryResourceEstimate:
    """Aggregate Toffoli / qubit estimate for one isometry-synthesis configuration (Eq. 36)."""

    n_rows: int
    n_cols: int
    d: int
    phase_bitsize: int
    n_d2_steps: int              # d - 1 fixed d=2 steps (Eq. 36); d=1 is one full unitary
    n_mux_rotations: int         # one R_y per d=2 step = d - 1
    n_interferometers: int       # 2 per d=2 step (V + 2-block U) = 2(d-1); 1 when d=1
    toffoli: int
    qubits: int


def estimate_interferometer_isometry_resources(
    n_rows: int,
    n_cols: int,
    phase_bitsize: int,
    *,
    optimal_T: bool = True,
) -> InterferometerIsometryResourceEstimate:
    """Resource estimate for the Eq.-36 isometry synthesis (``d - 1`` fixed ``d=2`` steps).

    The total Toffoli is obtained by walking the real sub-bloq graph; it is linear in ``d``
    (``cost multiplied by d-1``, arXiv:2409.11748).  ``d=1`` is a single full ``M x M``
    interferometer.  The qubit figure adds the largest single-step QROAM workspace.
    """
    assert _positive_power_of_two(n_cols)
    assert n_cols <= n_rows and n_rows % n_cols == 0 and phase_bitsize > 1
    from math import log2 as _log2

    bloq = InterferometerIsometrySynthesisQROAM.from_shape(
        n_rows, n_cols, phase_bitsize, optimal_T=optimal_T
    )
    d = int(bloq.d)
    n_steps = max(0, d - 1)

    from qualtran.resource_counting import get_cost_value, QECGatesCost
    from qualtran.resource_counting.generalizers import ignore_split_join

    gates = get_cost_value(bloq, QECGatesCost(), generalizer=ignore_split_join)
    # Each AND (uncomputed via measurement) counts as one Toffoli-equivalent.
    toffoli = int(gates.toffoli + gates.and_bloq)

    n = int(bloq.system_bitsize)
    base_qubits = n + phase_bitsize
    # Largest single-step QROAM workspace: the 2-block U interferometer (~ 2*b*lambda).
    n_blocks_eff = 2 if d >= 2 else 1
    if optimal_T:
        lam = max(1, 2 ** max(0, round(_log2(max(1.0, sqrt((n_blocks_eff * n_cols) / phase_bitsize))))))
    else:
        lam = 1
    qubits = base_qubits + 2 * phase_bitsize * lam

    return InterferometerIsometryResourceEstimate(
        n_rows=n_rows,
        n_cols=n_cols,
        d=d,
        phase_bitsize=phase_bitsize,
        n_d2_steps=n_steps,
        n_mux_rotations=n_steps,
        # 1 initial V_0 + (d-1) merged controlled-U interferometers (V_1.. absorbed, free).
        n_interferometers=(1 if d == 1 else d),
        toffoli=toffoli,
        qubits=int(qubits),
    )
