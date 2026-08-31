r"""Diagonal Coulomb-kernel block encoding.

.. note::
   **Split out of ``direct_Coulomb_block_encoding.py`` on 2026-08-13, content unchanged.**
   That module also held ``DirectCoulombBlockEncoding`` -- the v1 direct template, which
   the Sec.-2 construction does not use (``DirectTemplate`` in ``bse_block_encoding.py``
   composes the isometries and this diagonal itself).  Keeping them together forced a dead
   class, and two reflection modules it alone imported, to stay live.  The dead half is now
   in ``archive/direct_Coulomb_block_encoding.py``, intact.

   Everything below is verbatim from that module; only the import block is pruned.

Original module docstring follows.
Data-free block encoding of the periodic THC two-electron integral tensor,
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
    from .block_unitary_interferometer_QROAM import _data_max_log_block_sizes
    from .qroam_block_sizes import optimal_log_block_sizes_measured
    from .state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none
except ImportError:
    from block_unitary_interferometer_QROAM import _data_max_log_block_sizes
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


def capped_adjoint_log_block_sizes(
    data_shape: Tuple[int, ...], word_bits: int
) -> Tuple[int, ...]:
    r"""Toffoli-optimal adjoint ``log_block_sizes`` subject to ``prod(2^k) <= word_bits``.

    A measurement-based uncomputation converts the low address bits to a one-hot unary
    selector, applies the phase by a reduced lookup on the high bits, then erases the
    selector by measurement and Clifford feed-forward.  The selector is one bit per batch
    entry, so ``Lambda'`` costs ``Lambda'`` qubits -- cheap, but not free.  The workspace it
    can borrow is the word the forward lookup already holds, which at ``Lambda = 1`` is
    ``word_bits``.

    Leaving ``Lambda' = 1`` therefore pays the FULL table length where ``~L/word_bits`` is
    available inside qubits already allocated: at ``(N_k, N_IP, N_IP) = (216, 208, 208)`` and
    ``b = 32`` that is 9,345,024 against 292,064, i.e. the adjoint doubling the whole
    primitive instead of adding 3%.  This is the same defect that was found and fixed twice in
    the isometry path.
    """
    shape = tuple(int(d) for d in data_shape)
    total = 1
    for d in shape:
        total *= d
    caps = [max(0, int(d).bit_length() - 1) for d in shape]
    best: Tuple[int, ...] = tuple(0 for _ in shape)
    best_cost = None

    def walk(i: int, acc: list, lam: int) -> None:
        nonlocal best, best_cost
        if i == len(shape):
            cost = -(-total // lam) + lam
            if best_cost is None or cost < best_cost:
                best, best_cost = tuple(acc), cost
            return
        for k in range(caps[i] + 1):
            if lam * (1 << k) > max(1, int(word_bits)):
                break
            walk(i + 1, acc + [k], lam * (1 << k))

    walk(0, [], 1)
    return best


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
        elif not is_symbolic(self.N_k, self.N_IP, self.phase_bitsize):
            # Q-optimal: the FORWARD lookup stays at Lambda = 1, but the measurement-based
            # adjoint always takes its tradeoff, bounded by the word the forward already
            # holds.  Lambda' = 1 is strictly dominated -- more Toffolis for identical
            # qubits -- so it is never the qubit-minimizing choice.
            if tuple(self.diag_adjoint_log_block_sizes or ()) in ((), (0, 0, 0)):
                word = (2 * int(self.phase_bitsize) if self.complex_data
                        else int(self.phase_bitsize))
                object.__setattr__(
                    self,
                    'diag_adjoint_log_block_sizes',
                    capped_adjoint_log_block_sizes(self.diag_data_shape, word),
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
        try:
            from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        except ImportError:
            from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        return SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)   # App. A (2007.07391): b-2

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
        try:
            from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        except ImportError:
            from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        ret[self.inner.diag_qroam] += 1
        ret[And()] += 1                                   # the control, on the load
        emit_range_safety(ret, self.inner.diag_data_shape)
        ret[SignedCtrlAddIntoPhaseGrad(b)] += 1            # App. A: b-2 (outer ctrl on load only)
        ret[ZGate().controlled()] += 1                     # Clifford, free
        ret[self.inner.diag_qroam_adjoint] += 1
        return ret

