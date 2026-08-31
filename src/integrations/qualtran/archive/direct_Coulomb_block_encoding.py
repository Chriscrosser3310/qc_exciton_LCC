r"""Data-free block encoding of the periodic THC two-electron integral tensor,
*direct (Coulomb) orientation*.

This is the orientation-flipped sibling of
:class:`~.exchange_Coulomb_block_encoding.ExchangeCoulombBlockEncoding`.  Both encode
the tensor-hypercontraction (THC) factorization of arXiv:2601.16379 Eq. (9):

    (mu_{k_mu} nu_{k_nu} | lam_{k_lam} sig_{k_sig})
        = sum_{IJ}  X_I^{mu, k_mu}  X_I^{nu, -k_nu}  X_J^{lam, k_lam}  X_J^{sig, -k_sig}  W_{IJ}^{q}

with momentum transfer ``q``.  The interpolating vectors ``X`` (shape N_AO x N_IP per
k-point) are the "other tensors"; the Coulomb kernel ``W`` (shape N_IP x N_IP per q) is
the "central block".

**Orientation.**  In the *exchange* construction the two reflection block encodings load
the bra-pair (mu, nu) and the central block encoding acts on the resulting two
interpolation registers.  In the *direct* construction here the orientation is flipped:
the **inputs** are ``(mu, k_mu)`` and ``(lambda, k_lambda)`` while the **outputs** are the
other two indices ``(nu, k_nu)`` and ``(sig, k_sig)``.  Concretely (the recipe this module
implements):

  1.  Add an ancilla momentum register ``|Q>`` initialized to the uniform superposition
      over all ``N_k`` momentum values (``PrepareUniformSuperposition(N_k)``).  ``Q`` is the
      shared momentum transfer; the whole construction is an LCU over ``Q``.
  2.  Apply the two rectangular block encodings of ``X`` (same as exchange):
        B_mu  = BE of sum_{k} |k><k| (x) X^{mu, k}      [isometry N_up  x N_IP]
        B_lam = BE of sum_{k} |k><k| (x) X^{lam, k}     [isometry N_down x N_IP]
      producing interpolation indices ``I`` (from mu) and ``J`` (from lambda).  Each X
      channel acts on an ``N_k x N_IP`` system register (momentum ``k`` + an N_IP-sized
      matrix register), and the full construction acts on two such registers (double).
      When ``restrict_input`` is set, a ``LessThanConstant`` comparator first flags
      ``mu < N_up`` (= N_o, occupied) and ``lambda < N_down`` (= N_v, virtual) into a
      per-channel ancilla, restraining the encoded operator's input to the physical
      orbital subspace (computed here and uncomputed at the end).
  3.  Modular-add ``Q`` into each input momentum, overwriting it with the output momentum:
        |k_mu>     -->  |k_mu + Q mod N_k>      (= k_nu)
        |k_lambda> -->  |k_lambda + Q mod N_k>  (= k_sig).
      These two additions are the part of the encoded operator that moves the momentum
      indices from inputs to outputs; they are *not* undone.
  4.  Apply the diagonal Coulomb kernel
        C = sum_{Q, I, J}  W_{IJ}^{Q}  |Q, I, J><Q, I, J|,
      block-indexed by ``Q`` with ``I``, ``J`` the outputs of the ``X`` matrices on mu and
      lambda.  This is realized by :class:`DiagonalCoulombKernelBlockEncoding`: a QROAMClean
      loads the angle ``theta = arccos(W_{IJ}^{Q})``, one (Hadamard-sandwiched) controlled
      ``AddIntoPhaseGrad`` followed by a ``Z`` executes the reflection ``Z R_y`` on a
      block-encoding ancilla, and a QROAMCleanAdjoint uncomputes the angle register.  Its QROAM tradeoff parameters are
      tunable (and ``optimal_T`` chooses the closed-form Toffoli optimum).
  5.  Apply the two ``X^dagger`` (``B_mu^dagger``, ``B_lam^dagger``) on the corresponding
      registers, same as exchange.
  6.  Unprepare ``|Q>`` (``PrepareUniformSuperposition(N_k).adjoint()``), projecting it onto
      ``|0>``.  This is the LCU postselection that supplies the ``1/N_k`` central
      subnormalization.

This module is *data-free*: it only describes the structural circuit and emits the right
sub-bloqs in ``build_call_graph``.  All Toffoli and qubit counts come from Qualtran's
resource counter walking that call graph.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from math import log2
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.arithmetic import Add, LessThanConstant  # ModAdd has Qualtran bugs; Add as cost proxy
from qualtran.bloqs.basic_gates import Hadamard, ZGate
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from .block_unitary_interferometer_QROAM import _data_max_log_block_sizes
    from .block_unitary_reflection_QROAM import (
        optimal_reflection_adjoint_log_block_sizes,
        optimal_reflection_log_block_sizes,
    )
    from .rectangular_block_encoding_reflection import (
        ReflectionRectangularBlockEncoding,
    )
    from .qroam_block_sizes import optimal_log_block_sizes_measured
    from .state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none
except ImportError:
    from block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from block_unitary_interferometer_QROAM import _data_max_log_block_sizes
    from block_unitary_reflection_QROAM import (
        optimal_reflection_adjoint_log_block_sizes,
        optimal_reflection_log_block_sizes,
    )
    from rectangular_block_encoding_reflection import (
        ReflectionRectangularBlockEncoding,
    )
    from qroam_block_sizes import optimal_log_block_sizes_measured
    from state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none

try:
    from .range_safe_qroam import emit_range_safety
except ImportError:  # pragma: no cover
    from range_safe_qroam import emit_range_safety

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _next_power_of_two(n: int) -> int:
    return 1 << max(1, (int(n) - 1).bit_length())


def _distribute(total: int, caps: Tuple[int, ...]) -> Tuple[int, ...]:
    """Greedily fill ``total`` log-block-size across dims, largest (last) dim first."""
    out = [0] * len(caps)
    rem = int(total)
    for i in range(len(caps) - 1, -1, -1):
        take = min(rem, int(caps[i]))
        out[i] = take
        rem -= take
    return tuple(out)


def optimal_diag_log_block_sizes(
    data_shape: Tuple[int, ...], phase_bitsize: int, *, adjoint: bool
) -> Tuple[int, ...]:
    r"""Closed-form Toffoli-optimal QROAM ``log_block_sizes`` for the diagonal kernel.

    Forward QROAM cost ``T(L) = ceil(M/L) + b*(L-1)`` is minimized at
    ``L* ~ sqrt(M/b)``; the adjoint cost ``T(L) = ceil(M/L) + (L-1)`` at ``L* ~ sqrt(M)``.
    ``M`` is the total table size ``prod(data_shape)``.  The resulting total
    ``log2(L*)`` is distributed across dimensions (largest first), each capped by
    ``floor(log2(dim))``.
    """
    # Exact brute force over the (small) grid.  The previous closed form floored
    # ``log2`` of each dimension, so a non-power-of-two table was costed as if it were
    # smaller -- which selected too small a Lambda and made LARGER tables come out
    # CHEAPER (N_IP=224 -> 9046 but N_IP=256 -> 5686).  See qroam_block_sizes.
    return optimal_log_block_sizes_measured(
        tuple(int(d) for d in data_shape), (int(phase_bitsize),), adjoint=adjoint
    )


@attrs.frozen
class DiagonalCoulombKernelBlockEncoding(BlockEncoding):
    r"""$(1, \cdot, \epsilon)$ block encoding of the diagonal Coulomb kernel.

    Encodes the diagonal operator

    .. math:: C = \sum_{Q, I, J} W_{IJ}^{Q}\, |Q, I, J\rangle\langle Q, I, J|,

    with ``|W_{IJ}^Q| <= 1``, via the QROAM -> rotation -> QROAM^dagger pattern:

      1. ``QROAMClean`` loads ``theta_{Q,I,J} = arccos(W_{IJ}^Q)`` into a phase register,
         selected by ``(Q, I, J)``.
      2. One ``AddIntoPhaseGrad`` (Hadamard-sandwiched, controlled on a block-encoding
         ancilla) realizes ``R_y(2 theta)`` on that ancilla, followed by a ``Z`` gate so
         the rotation becomes a reflection (``Z R_y(2 theta)``).
      3. ``QROAMCleanAdjoint`` uncomputes the angle register.

    The QROAM block sizes (forward and adjoint) are tunable tradeoff parameters; with
    ``optimal_T=True`` they are set to the closed-form Toffoli optimum.

    Attributes:
        N_k: number of momentum-transfer blocks ``Q`` (range ``[0, N_k)``).
        N_IP: number of interpolation points; ``I`` and ``J`` range over ``[0, N_IP)``.
        phase_bitsize: bitsize ``b`` of the angle / phase-gradient registers.
        diag_log_block_sizes: ``log_block_sizes`` for the forward angle QROAM.
        diag_adjoint_log_block_sizes: ``log_block_sizes`` for the QROAM uncomputation.
    """

    N_k: SymbolicInt
    N_IP: SymbolicInt
    phase_bitsize: SymbolicInt = 32
    diag_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0, 0), converter=_to_tuple_or_none
    )
    diag_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0, 0), converter=_to_tuple_or_none
    )
    optimal_T: bool = False
    # The encoded diagonal entries are REAL.  W(r-r') is a real function, so the
    # momentum-space kernel obeys zeta_{-Q} = conj(zeta_Q) and its transform
    # zeta~_R = N_k^-1 sum_Q e^{iQR} zeta_Q is real.  Verified on the ISDF data with the
    # correct 3-D mesh arithmetic: ||W[-Q] - conj(W[Q])||/max|W| = 1.7e-4 (2x2x2),
    # 3.9e-8 (3x3x3), 1.7e-4 (4x4x4), and max|Im(zeta~)|/max|Re(zeta~)| = 3.7e-5, 8.3e-9,
    # 8.3e-6 -- i.e. real to the ISDF fitting error.
    #
    # TRAP: a 1-D DFT over the LINEAR momentum index gives Im/Re ~ 0.25-0.77 and looks
    # emphatically complex.  It is wrong -- the -Q partner is per-axis on the 3-D mesh
    # and linear-index negation does not find it.  An earlier pass here drew exactly that
    # false conclusion and doubled this bloq's word and rotation count.
    #
    # Real => one rotation, a b-bit word, AND Z R_y stays an involution, so the
    # ov-direct / oo / vv templates keep their free self-inverseness.
    complex_data: bool = False

    def __attrs_post_init__(self):
        if self.optimal_T:
            if is_symbolic(self.N_k, self.N_IP, self.phase_bitsize):
                raise ValueError("optimal_T=True requires concrete N_k, N_IP, phase_bitsize")
            shape = self.diag_data_shape
            # Size Lambda against the EFFECTIVE word: 2b for a complex entry.
            word = 2 * int(self.phase_bitsize) if self.complex_data else int(self.phase_bitsize)
            object.__setattr__(
                self,
                'diag_log_block_sizes',
                optimal_diag_log_block_sizes(shape, word, adjoint=False),
            )
            object.__setattr__(
                self,
                'diag_adjoint_log_block_sizes',
                optimal_diag_log_block_sizes(shape, word, adjoint=True),
            )

    # ----------------------------- shape helpers -----------------------------

    @property
    def k_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_k - 1)

    @property
    def ip_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_IP - 1)

    @property
    def diag_data_shape(self) -> Tuple[SymbolicInt, ...]:
        # (Q, I, J) when there is more than one block; drop the Q axis otherwise.
        if is_symbolic(self.N_k) or self.N_k > 1:
            return (self.N_k, self.N_IP, self.N_IP)
        return (self.N_IP, self.N_IP)

    # ------------------------- BlockEncoding interface -------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        # Q (if present) + I + J.
        q = self.k_bitsize if (is_symbolic(self.N_k) or self.N_k > 1) else 0
        return q + 2 * self.ip_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return 1

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        # |W_{IJ}^Q| <= 1, so the diagonal block encoding has subnormalization 1.
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

    # --------------------------- Sub-bloq factories ----------------------------

    def _capped_lbs(self, raw: Optional[Tuple[SymbolicInt, ...]]):
        return _cap_log_block_sizes(raw, _data_max_log_block_sizes(self.diag_data_shape))

    @property
    def _diag_target_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        """One word: ``2b`` bits for a complex entry (magnitude angle + phase), ``b`` for real.

        Carried as a single wide target rather than two ``b``-bit registers -- identical
        for costing, and the data shape here is 3-D ``(N_k, N_IP, N_IP)`` so Qualtran's
        one-array-per-target convention does not admit two.
        """
        if self.complex_data:
            return (2 * self.phase_bitsize,)
        return (self.phase_bitsize,)

    @property
    def diag_qroam(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.diag_data_shape,
            target_bitsizes=self._diag_target_bitsizes,
            log_block_sizes=self._capped_lbs(self.diag_log_block_sizes),
        )

    @property
    def diag_qroam_adjoint(self) -> QROAMCleanAdjoint:
        adj_lbs = self._capped_lbs(self.diag_adjoint_log_block_sizes)
        fwd_lbs = self._capped_lbs(self.diag_log_block_sizes)
        kwargs = dict(target_bitsizes=self._diag_target_bitsizes, log_block_sizes=adj_lbs)
        if fwd_lbs is not None:
            kwargs['target_shapes'] = (tuple(1 << b for b in fwd_lbs),)
        return QROAMCleanAdjoint.build_from_bitsize(self.diag_data_shape, **kwargs)

    @property
    def ctrl_phase_grad_add(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()

    # ----------------------------- Resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.diag_qroam] += 1                    # load theta (and phi if complex)
        emit_range_safety(ret, self.diag_data_shape)  # P-13: out-of-range -> |0>
        # Magnitude rotation, plus a second rotation carrying the phase when the entry
        # is complex.  One rotation is only correct for a real diagonal.
        ret[self.ctrl_phase_grad_add] += 2 if self.complex_data else 1
        ret[ZGate()] += 1                            # Z after R_y
        ret[self.diag_qroam_adjoint] += 1            # uncompute the angle register
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Single-qubit control: only the rotation gains a control; QROAM pair cancels."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledDiagonalCoulombKernelBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledDiagonalCoulombKernelBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`DiagonalCoulombKernelBlockEncoding`.

    The external control reaches only the singular-value-style ``R_y`` (promoting the
    controlled phase-gradient add to a doubly-controlled one).  The QROAM
    forward/uncompute pair is left uncontrolled because it cancels when ``ctrl = 0``.
    """

    inner: "DiagonalCoulombKernelBlockEncoding"

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
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        b = self.inner.phase_bitsize
        ret: "Counter[Bloq]" = Counter()
        # Controlling a QROAM-driven rotation costs +1 Toffoli, not a doubled rotation:
        # the control is absorbed into the unary-iteration tree (measured: +1 exactly
        # for every table size and shape).  With the load controlled off, the angle
        # register stays |0>, so R_y(0) = I and the Hadamards cancel -- the rotations
        # themselves need no control at all.
        ret[self.inner.diag_qroam] += 1
        ret[And()] += 1                                   # the control, on the load
        emit_range_safety(ret, self.inner.diag_data_shape)
        ret[AddIntoPhaseGrad(b, b).controlled()] += 1      # unchanged by the control
        ret[ZGate().controlled()] += 1                     # Clifford, free
        ret[self.inner.diag_qroam_adjoint] += 1
        return ret


@attrs.frozen
class DirectCoulombBlockEncoding(BlockEncoding):
    """Data-free direct (Coulomb) THC block encoding (orientation-flipped exchange).

    Inputs ``(mu, k_mu)``, ``(lambda, k_lambda)``; outputs ``(nu, k_nu)``,
    ``(sig, k_sig)`` with ``k_nu = k_mu + Q`` and ``k_sig = k_lambda + Q`` for a shared
    momentum transfer ``Q`` summed over (LCU).  See the module docstring for the full
    six-step recipe.

    Construction policy (mirrors :class:`ExchangeCoulombBlockEncoding`):
      * The rectangular ``X`` matrices (``X^{mu}``: N_up x N_IP, ``X^{lambda}``:
        N_down x N_IP) are block encoded by :class:`ReflectionRectangularBlockEncoding`.
      * The central Coulomb kernel is the **diagonal** operator
        :class:`DiagonalCoulombKernelBlockEncoding` (QROAM -> Z R_y -> QROAM^dagger),
        block-indexed by ``Q`` with ``I``, ``J`` the outputs of the ``X`` matrices.

    Attributes:
        N_up: row count of ``X^{mu}``.
        N_down: row count of ``X^{lambda}``.
        N_IP: shared column count of ``X^{mu}``, ``X^{lambda}`` (= range of ``I``, ``J``).
        N_k: number of momentum values; ``k_mu``, ``k_lambda``, ``Q`` each range ``[0, N_k)``.
        phase_bitsize: bitsize of phase / angle registers used inside each sub-bloq.
        outer_amp_* / outer_phase_*: ``log_block_sizes`` for the QROAMs inside each
            rectangular reflection block encoding (B_mu and B_lam).
        diag_log_block_sizes / diag_adjoint_log_block_sizes: ``log_block_sizes`` for the
            central diagonal kernel QROAM forward / uncompute.
        optimal_T: when True, set every QROAM tradeoff parameter to its closed-form
            Toffoli optimum.
    """

    N_up: int
    N_down: int
    N_IP: int
    N_k: int
    phase_bitsize: int = 32
    outer_amp_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    outer_amp_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    outer_phase_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    outer_phase_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    diag_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0, 0)
    diag_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0, 0)
    optimal_T: bool = False
    # Synthesis method for the rectangular X matrices (B_mu, B_lam): "reflection"
    # (LKS Householder, default) or "column" (column-by-column isometry synthesis,
    # Iten 1501.06911 + Berry Eq. 24).  The central diagonal kernel is unaffected.
    outer_synthesis: str = "reflection"
    # Restrict each orbital input to its physical range with a comparator.  Each X channel
    # acts on an N_k x N_IP system register (block k + matrix sized to ceil(log2 N_IP)),
    # but the input orbital index is only valid over the first N_up (= N_o, occupied) /
    # N_down (= N_v, virtual) entries.  When True a ``LessThanConstant`` flags ``x < N_o``
    # (mu) / ``x < N_v`` (lambda) into a 1-qubit ancilla per channel, restraining the
    # encoded operator to that input subspace (computed + uncomputed; O(log N_IP) Toffoli).
    restrict_input: bool = True

    def __attrs_post_init__(self):
        if self.outer_synthesis not in ("reflection", "column"):
            raise ValueError(
                f"outer_synthesis must be 'reflection' or 'column', got {self.outer_synthesis!r}"
            )
        if self.optimal_T:
            n_rows_outer = max(self.n_rows_up, self.n_rows_down)
            opt_outer_fwd = optimal_reflection_log_block_sizes(
                int(self.N_k), int(n_rows_outer), int(self.phase_bitsize)
            )
            opt_outer_adj = optimal_reflection_adjoint_log_block_sizes(
                int(self.N_k), int(n_rows_outer)
            )
            object.__setattr__(self, 'outer_amp_log_block_sizes', opt_outer_fwd)
            object.__setattr__(self, 'outer_amp_adjoint_log_block_sizes', opt_outer_adj)
            object.__setattr__(self, 'outer_phase_log_block_sizes', opt_outer_fwd)
            object.__setattr__(self, 'outer_phase_adjoint_log_block_sizes', opt_outer_adj)
            shape = self.C_diag.diag_data_shape
            object.__setattr__(
                self, 'diag_log_block_sizes',
                optimal_diag_log_block_sizes(shape, int(self.phase_bitsize), adjoint=False),
            )
            object.__setattr__(
                self, 'diag_adjoint_log_block_sizes',
                optimal_diag_log_block_sizes(shape, int(self.phase_bitsize), adjoint=True),
            )

    # ------------------------- shape helpers -------------------------

    @cached_property
    def n_rows_up(self) -> int:
        return _next_power_of_two(max(self.N_up, self.N_IP))

    @cached_property
    def n_rows_down(self) -> int:
        return _next_power_of_two(max(self.N_down, self.N_IP))

    @cached_property
    def k_bitsize(self) -> int:
        return bit_length(self.N_k - 1)

    # --------------------- BlockEncoding interface ---------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        # k_mu + sys_mu + k_lambda + sys_lambda + Q (ancilla momentum register).
        return (
            self.k_bitsize + bit_length(self.n_rows_up - 1)
            + self.k_bitsize + bit_length(self.n_rows_down - 1)
            + self.k_bitsize
        )

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        # Two outer BEs' ancillas (matrix-row aux + flag) + central diagonal BE ancilla,
        # plus (when restrict_input) one input-range comparator flag per orbital channel.
        n_up = bit_length(self.n_rows_up - 1)
        n_down = bit_length(self.n_rows_down - 1)
        cmp_flags = 2 if self.restrict_input else 0
        return (n_up + 1) + (n_down + 1) + 1 + cmp_flags

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        # Single phase-gradient workspace shared across all sub-bloqs.
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        # B_mu, B_lam each subnormalize by n_rows (reflection BE), applied as B and
        # B^dagger; the Q-LCU contributes a factor N_k; the diagonal kernel has alpha = 1.
        return (
            float(self.n_rows_up) * float(self.n_rows_down)
            * float(self.N_k)
            * float(self.n_rows_up) * float(self.n_rows_down)
        )

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
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    # --------------------- Sub-bloq factories ---------------------

    def _outer_be(self, n_rows: int, n_reflections: int) -> BlockEncoding:
        """Rectangular X-matrix isometry block encoding, per ``outer_synthesis``.

        Both choices share the same (system = block + matrix, ancilla = 1, resource =
        phase-gradient, alpha = 1) interface so they are drop-in interchangeable; only
        the internal synthesis (Householder reflections vs column-by-column) differs.
        """
        if self.outer_synthesis == "column":
            return ColumnIsometryRectangularBlockEncoding(
                n_blocks=self.N_k,
                n_rows=n_rows,
                phase_bitsize=self.phase_bitsize,
                n_reflections=n_reflections,
                optimal_T=self.optimal_T,
            )
        return ReflectionRectangularBlockEncoding(
            n_blocks=self.N_k,
            n_rows=n_rows,
            phase_bitsize=self.phase_bitsize,
            n_reflections=n_reflections,
            amp_log_block_sizes=self.outer_amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.outer_amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.outer_phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.outer_phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    @property
    def B_mu(self) -> BlockEncoding:
        # Rectangular X^{mu} (N_up x N_IP): isometry on the n_rows_up-dim register,
        # synthesizing only min(N_up, N_IP) columns (short direction).
        return self._outer_be(self.n_rows_up, min(self.N_up, self.N_IP))

    @property
    def B_lam(self) -> BlockEncoding:
        # Rectangular X^{lambda} (N_down x N_IP): min(N_down, N_IP) synthesized vectors.
        return self._outer_be(self.n_rows_down, min(self.N_down, self.N_IP))

    @property
    def C_diag(self) -> DiagonalCoulombKernelBlockEncoding:
        # Central diagonal Coulomb kernel sum_{Q,I,J} W_{IJ}^Q |Q,I,J><Q,I,J|.
        return DiagonalCoulombKernelBlockEncoding(
            N_k=self.N_k,
            N_IP=self.N_IP,
            phase_bitsize=self.phase_bitsize,
            diag_log_block_sizes=self.diag_log_block_sizes,
            diag_adjoint_log_block_sizes=self.diag_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    @property
    def uniform_prep(self) -> Bloq:
        # Prepares (and, as adjoint, postselects) the |Q> momentum ancilla.
        return PrepareUniformSuperposition(n=self.N_k)

    @property
    def mod_add(self) -> Bloq:
        # |k> -> |k + Q mod N_k>.  A true modular add is Add + comparison + conditional
        # subtract, all O(n_k) Toffoli -- negligible next to the QROAM costs.  Modeled by
        # Add on QUInt(n_k) (ModAdd has Qualtran bugs).
        return Add(a_dtype=QUInt(self.k_bitsize))

    @property
    def input_comparator_mu(self) -> Bloq:
        # Restrain the mu input (occupied): flag x < N_up (= N_o) on the N_IP-sized matrix
        # register of the up channel.  Negligible O(log N_IP) Toffoli.
        return LessThanConstant(bitsize=bit_length(self.n_rows_up - 1), less_than_val=self.N_up)

    @property
    def input_comparator_lam(self) -> Bloq:
        # Restrain the lambda input (virtual): flag x < N_down (= N_v) on the down channel.
        return LessThanConstant(bitsize=bit_length(self.n_rows_down - 1), less_than_val=self.N_down)

    # --------------------- Resource counts ---------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        # Restrain each orbital input to its physical N_o / N_v range (compute + uncompute).
        if self.restrict_input:
            ret[self.input_comparator_mu] += 2
            ret[self.input_comparator_lam] += 2
        # Outer X block encodings: forward (step 2) and adjoint (step 5).
        ret[self.B_mu] += 2
        ret[self.B_lam] += 2
        if self.N_k > 1:
            # |Q> LCU: prepare (step 1) + unprepare/postselect (step 6).
            ret[self.uniform_prep] += 2
            # Two momentum modular additions k_mu += Q and k_lambda += Q (step 3),
            # each applied once (overwriting input momentum with output momentum).
            ret[self.mod_add] += 2
        # Central diagonal Coulomb kernel W_{IJ}^Q (step 4).
        ret[self.C_diag] += 1
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Cheap single-qubit control.

        ``.controlled()`` returns :class:`_ControlledDirectCoulombBlockEncoding`, which
        controls the central diagonal kernel ``C`` and the two momentum modular additions
        (so that ``ctrl = 0`` leaves the momenta unshifted and ``C = I``, making the outer
        ``B``/``B^dagger`` reflection pair and ``|Q>`` LCU self-cancel).  The expensive
        reflection block encodings stay uncontrolled.
        """
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledDirectCoulombBlockEncoding(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledDirectCoulombBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`DirectCoulombBlockEncoding`.

    The external control reaches the central diagonal kernel ``C`` and the two momentum
    modular additions.  When ``ctrl = 0`` the additions are skipped and ``C = I``, so the
    outer ``B``/``B^dagger`` reflection pair and the ``|Q>`` prepare/unprepare cancel; the
    expensive reflections therefore stay uncontrolled.
    """

    inner: DirectCoulombBlockEncoding

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
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if self.inner.restrict_input:
            ret[self.inner.input_comparator_mu] += 2
            ret[self.inner.input_comparator_lam] += 2
        ret[self.inner.B_mu] += 2
        ret[self.inner.B_lam] += 2
        if self.inner.N_k > 1:
            ret[self.inner.uniform_prep] += 2
            # Momentum additions gain the external control (cheap) so that ctrl = 0
            # leaves the momenta unshifted and the outer reflections cancel.
            ret[self.inner.mod_add.controlled()] += 2
        # Central diagonal kernel is the main piece promoted to a controlled BE.
        ret[self.inner.C_diag.controlled()] += 1
        return ret

