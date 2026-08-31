r"""Controlled state preparation by the **load-all** rotation tree.

Prepares one of a multiplexed family of dense states

.. math::
    S_{\chi_{\mathbf k,\mu}}\,|0\rangle \;\propto\; \sum_a \chi_{a\mathbf k,\mu}\,|a\rangle ,

selected by a read-only address register :math:`(\mathbf k, \mu)`.  This is the primitive
`main.tex` Sec. 2 assigns to the **virtual (electron-side)** THC factors of the :math:`ov`
exchange template: "the THC index labels a family of states, so the more efficient
primitive is controlled state preparation".

Load-all vs. layer-by-layer
---------------------------
A rotation tree over :math:`N = 2^n` amplitudes has :math:`n` layers.  The usual
("layer-by-layer") compilation issues **one QROAM lookup per layer**, each addressed by
the full :math:`(\mathbf k, \mu)` index.  Here that address has
:math:`N_k N_{\mathrm{THC}}` values, so paying it :math:`n` times dominates.

The **load-all** variant issues a *single* QROAM lookup whose output word holds **every**
angle of the tree at once, then walks the tree by swapping the relevant :math:`b`-bit
slice into place:

* one :class:`QROAMClean` over the address, target width :math:`(N-1)b` for real data
  (:math:`(2N-1)b` when a phase per amplitude is also needed);
* :math:`N-1` controlled swaps routing the active angle to the rotation register;
* :math:`n` controlled phase-gradient additions -- the actual rotations;
* one :class:`QROAMCleanAdjoint`, measurement-based.

The wide output word is the price; a single address traversal is what is bought.  For
:math:`N_k N_{\mathrm{THC}} \gg N` -- exactly this problem's regime -- that is the better
trade, which is why it is the default here.

Complex data is the default
---------------------------
The THC factors :math:`\chi_{p\mathbf k,\mu}` are built from Bloch orbitals and are
**complex** at a general :math:`\mathbf k` (measured :math:`\lvert\mathrm{Im}\rvert /
\lvert\mathrm{Re}\rvert \approx 0.7` on the 3x3x3 and 4x4x4 meshes; they are real only on
2x2x2, a time-reversal-special mesh).  So the tree carries a phase per amplitude as well
as an angle: the word is :math:`(2N-1)b`, not :math:`(N-1)b`, and there are :math:`n+1`
rotations, not :math:`n`.  ``real_data=True`` is available for a genuinely real family.

Self-inverse use
----------------
This bloq is *not* itself an involution -- it is a state preparation.  It appears in the
block encodings only in ``S ... S^dagger`` pairs, where the pair is a congruence and the
composite is an involution provided the enclosed central encoding is.  See
:mod:`eigendecomposition_block_encoding`.

Data-free: structure only; counts come from Qualtran's resource counter.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import TwoBitCSwap
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .range_safe_qroam import emit_range_safety
except ImportError:  # pragma: no cover
    from range_safe_qroam import emit_range_safety

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _to_tuple_or_none(x):
    return tuple(x) if x is not None else None


@attrs.frozen
class LoadAllStatePreparationQROAM(Bloq):
    r"""Multiplexed dense state preparation, load-all rotation tree.

    Attributes:
        n_addr: number of distinct states in the family (the QROAM address size, e.g.
            :math:`N_k N_{\mathrm{THC}}`).
        n_rows: amplitude count :math:`N` of each state (a power of two).
        phase_bitsize: angle bitsize :math:`b`.
        real_data: amplitudes are real -- angles only, word width :math:`(N-1)b`.
            When False the word is :math:`(2N-1)b` (angles plus per-amplitude phases).
        uncompute: this instance is the adjoint :math:`S^\dagger`.  Structurally the same
            layers in reverse; the QROAM erasure is measurement-based either way, so the
            Toffoli count matches the forward direction.
        log_block_sizes / adjoint_log_block_sizes: QROAM tradeoff :math:`\Lambda`,
            :math:`\Lambda'`.

    Registers:
        selection: the read-only ``(k, mu)`` address.
        target: the ``n``-qubit register the state is prepared on.
        phase_gradient: ``phase_bitsize``-qubit phase-gradient workspace.
    """

    n_addr: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    real_data: bool = False
    uncompute: bool = False
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    # ------------------------------ shape helpers ------------------------------

    @property
    def selection_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_addr - 1)

    @property
    def target_bitsize(self) -> SymbolicInt:
        """``n = log2 N``: the number of tree layers, and the register the state lives on."""
        return bit_length(self.n_rows - 1)

    @property
    def word_bits(self) -> SymbolicInt:
        r""":math:`(N-1)b` real / :math:`(2N-1)b` complex -- the whole tree in one word."""
        n_values = (self.n_rows - 1) if self.real_data else (2 * self.n_rows - 1)
        return n_values * self.phase_bitsize

    @property
    def n_rotations(self) -> SymbolicInt:
        r"""``n`` amplitude rotations, plus one phase rotation when the data is complex.

        A complex amplitude needs both a magnitude angle and a phase, so the tree's
        ``n`` :math:`R_y` layers are followed by a phase layer -- ``n+1`` rotations in
        total.  Real data needs only the ``n``.
        """
        return self.target_bitsize + (0 if self.real_data else 1)

    @property
    def n_values(self) -> SymbolicInt:
        r"""Values held in the loaded word: :math:`2N-1` complex, :math:`N-1` real.

        A complex amplitude carries a magnitude *and* a phase, so the tree stores
        :math:`N-1` rotation angles plus :math:`N` phases.
        """
        return (self.n_rows - 1) if self.real_data else (2 * self.n_rows - 1)

    @property
    def n_swaps(self) -> SymbolicInt:
        r"""``2(n_values - n - 1)`` controlled swaps of :math:`b`-bit words.

        Routing an angle slice out of the loaded word is **reversible**: the slice has to
        be swapped back before the QROAM adjoint can erase the word, so every routing is
        paid twice -- hence the leading 2.  The ``- n - 1`` drops the layers whose slice
        is already in position.

        For complex data this is :math:`2(2N - n - 2)` swaps, i.e. :math:`2b(2N-n-2)`
        Toffolis, which is the analytic model's routing term exactly.  An earlier version
        here counted the forward pass only and so undercounted by ~1.7x.
        """
        return 2 * (self.n_values - self.target_bitsize - 1)

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('selection', QAny(self.selection_bitsize)),
            Register('target', QAny(self.target_bitsize)),
            Register('phase_gradient', QAny(self.phase_bitsize)),
        ])

    # ------------------------------ sub-bloqs ----------------------------------

    @property
    def qroam(self) -> QROAMClean:
        """The single wide lookup: every tree angle for this ``(k, mu)``."""
        return QROAMClean.build_from_bitsize(
            (self.n_addr,),
            target_bitsizes=(self.word_bits,),
            log_block_sizes=self.log_block_sizes,
        )

    @property
    def qroam_adjoint(self) -> QROAMCleanAdjoint:
        kwargs = dict(
            target_bitsizes=(self.word_bits,), log_block_sizes=self.adjoint_log_block_sizes
        )
        if self.log_block_sizes is not None:
            kwargs['target_shapes'] = (tuple(1 << b for b in self.log_block_sizes),)
        return QROAMCleanAdjoint.build_from_bitsize((self.n_addr,), **kwargs)

    @property
    def routing_swap(self) -> Bloq:
        """One bit of the controlled swap that routes an angle slice into place."""
        return TwoBitCSwap()

    @property
    def rotation(self) -> Bloq:
        try:
            from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        except ImportError:
            from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        return SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)   # App. A (2007.07391): b-2

    # ---------------------------- resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam] += 1
        emit_range_safety(ret, (self.n_addr,))          # P-13
        if not is_symbolic(self.n_rows, self.phase_bitsize):
            # Route the active b-bit angle slice out of the loaded word: N-1 controlled
            # swaps of b-bit words.
            ret[self.routing_swap] += int(self.n_swaps) * int(self.phase_bitsize)
        ret[self.rotation] += self.n_rotations        # n amplitude rotations (+1 phase if complex)
        ret[self.qroam_adjoint] += 1
        return ret

    def adjoint(self) -> 'LoadAllStatePreparationQROAM':
        return attrs.evolve(self, uncompute=not self.uncompute)

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledLoadAllStatePreparationQROAM(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledLoadAllStatePreparationQROAM(Bloq):
    """Singly-controlled :class:`LoadAllStatePreparationQROAM`.

    Only the rotations gain the control; the QROAM load/erase pair is left uncontrolled
    (it cancels at ``ctrl = 0``, and the loaded word is never acted on).
    """

    inner: LoadAllStatePreparationQROAM

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner.qroam] += 1
        emit_range_safety(ret, (self.inner.n_addr,))    # P-13
        if not is_symbolic(self.inner.n_rows, self.inner.phase_bitsize):
            ret[self.inner.routing_swap] += int(self.inner.n_swaps) * int(
                self.inner.phase_bitsize
            )
        ret[And()] += 1                                   # the control, on the load: +1
        ret[self.inner.rotation] += self.inner.n_rotations  # rotations unchanged
        ret[self.inner.qroam_adjoint] += 1
        return ret
