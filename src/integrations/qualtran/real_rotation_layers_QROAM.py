r"""Rotation-only QROAM layers for **real** isometries and unitaries.

The Berry et al. (arXiv:2409.11748, Eq. 24) decomposition writes a general single-qubit
unitary as

.. math::
    U = \mathrm{diag}(e^{i\phi_0}, e^{i\phi_1})\; H\; \mathrm{diag}(e^{i\theta},1)\; H\;
        \mathrm{diag}(e^{i\phi},1),

i.e. **three** angle tables and three phase-gradient additions per gate (two for the
merged multiplexed beamsplitter layer, which fuses one pair of them).

When a rotation is enough
-------------------------
These layers apply where the synthesized matrix is real.  **That is not the general case
here** -- the THC factors and the Bloch-basis Fock blocks are complex at a general
:math:`\mathbf k`, so the complex layers are the default.  What *is* unconditional is the
subleading multi-controlled fix-up, which needs only a relative phase and so is always a
rotation (see :class:`RealMultiControlledRotationQROAM`).  For a real orthogonal
:math:`2\times2` gate,

.. math::
    R = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}
      = H\, \mathrm{diag}(e^{i\theta}, 1)\, H \quad (\text{up to a Clifford } Z),

so the two relative-phase tables collapse and **one** rotation table survives.  Concretely
this module implements the "remove the phase first, then rotate only" strategy:

1. all basis-state phases of a real matrix are :math:`\pm1`, i.e. a **sign**, not an angle;
2. signs are pulled out into one final layer that needs a **1-bit** table, not a
   :math:`b`-bit one, and are applied by Clifford :math:`Z` gates (0 Toffoli beyond the
   lookup);
3. what remains in every intermediate layer is a pure :math:`R_y`, one table, one
   phase-gradient addition.

Effect on cost.  Per multiplexed layer the QROAM output width drops :math:`2b \to b` and
the controlled additions drop :math:`2 \to 1`; per subleading multi-controlled gate the
additions drop :math:`3 \to 1`; the final diagonal drops from a :math:`b`-bit phase table
to a 1-bit sign table.  Roughly a factor two on the dominant layers.

The layer bloqs here are drop-in replacements for the corresponding ones in
:mod:`block_unitary_interferometer_QROAM` / :mod:`block_isometry_column_synthesis_QROAM`,
selected with ``real_data=True`` on the consuming synthesis bloq (default ``False``).
:class:`RealMultiControlledRotationQROAM` is **not** gated on that flag -- it is used
unconditionally, because the fix-up is a rotation even for complex isometries.

Data-free: structure only; counts come from Qualtran's resource counter.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QBit, Register, Signature
from qualtran.bloqs.basic_gates import Hadamard, ZGate

try:
    from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
except ImportError:  # pragma: no cover
    from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
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
class RealPhaseLayerQROAM(Bloq):
    r"""One merged multiplexed **real** beamsplitter layer: ``QROAM -> H -> R_y -> H``.

    The real counterpart of ``BlockInterferometerPhaseLayerQROAM``.  That bloq loads two
    :math:`b`-bit tables :math:`(\alpha, \beta)` and performs two controlled additions; a
    real rotation needs one table and one addition.

    Attributes:
        n_blocks: block-diagonal multiplexer address size.
        n_rows: dimension of the sub-register this layer acts on (``2^(c+1)``).
        phase_bitsize: rotation-angle bitsize ``b``.
        log_block_sizes: QROAM tradeoff.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def pair_bitsize(self) -> SymbolicInt:
        return self.system_bitsize - 1

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.n_rows // 2,)
        return (self.n_blocks, self.n_rows // 2)

    @property
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.pair_bitsize,)
        return (self.block_bitsize, self.pair_bitsize)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        # ONE b-bit angle table (the complex sibling loads two).
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize,),
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=self.log_block_sizes,
        )

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        try:
            from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        except ImportError:
            from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        emit_range_safety(ret, self.qroam_data_shape)   # P-13
        ret[SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)] += 1   # App. A (2007.07391): b-2
        ret[Hadamard()] += 2
        return ret


@attrs.frozen
class RealMultiControlledRotationQROAM(Bloq):
    r"""Subleading multi-controlled **real** rotation ``C_{n_controls}(R_y)``.

    The real counterpart of ``MultiControlledSU2QROAM``: an ``And`` ladder over the
    controls followed by a *single* controlled phase-gradient addition (the complex
    version needs three).
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

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if not is_symbolic(self.n_controls):
            n_and = max(0, int(self.n_controls) - 1)
            if n_and:
                ret[And()] += n_and
                ret[And().adjoint()] += n_and
        # App. A (arXiv:2007.07391): a phase-gradient rotation CONTROLLED on a qubit costs
        # b-2, not the ~2(b-1) that .controlled() charges -- the control drives Cliffords
        # around a single uncontrolled addition.  Same fix the Aug-2026 sweep applied to
        # the interferometer phase layers; this gate was missed.
        ret[SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)] += 1
        ret[Hadamard()] += 2
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledRealMultiControlledRotationQROAM(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledRealMultiControlledRotationQROAM(Bloq):
    """Singly-controlled :class:`RealMultiControlledRotationQROAM`."""

    inner: RealMultiControlledRotationQROAM

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if not is_symbolic(self.inner.n_controls):
            n_and = max(0, int(self.inner.n_controls) - 1)
            if n_and:
                ret[And()] += n_and
                ret[And().adjoint()] += n_and
        # The angle is supplied by the column's angle load, which carries the control;
        # with it off the angle is 0 and this rotation is already the identity.
        ret[self.inner._add.controlled()] += 1
        ret[Hadamard()] += 2
        return ret


@attrs.frozen
class RealSignLayerQROAM(Bloq):
    r"""Final diagonal layer for a real matrix: a **1-bit sign table**, applied by ``Z``.

    The complex sibling (``BlockInterferometerFinalPhasesQROAM``) loads a :math:`b`-bit
    phase per basis state and adds it into the phase gradient.  A real matrix's residual
    diagonal is :math:`\pm1`, so the table is one bit wide and the "rotation" is a Clifford
    :math:`Z` controlled on that bit -- no phase-gradient addition at all.

    This is the "remove the phase first" half of the rotation-only strategy: doing it
    up front is what lets every other layer be a bare :math:`R_y`.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(block=self.block_bitsize, system=self.system_bitsize)

    @property
    def data_shape(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.n_rows,)
        return (self.n_blocks, self.n_rows)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.data_shape, target_bitsizes=(1,), log_block_sizes=self.log_block_sizes
        )

    @property
    def qroam_adj_bloq_for_cost(self) -> QROAMCleanAdjoint:
        kwargs = dict(target_bitsizes=(1,), log_block_sizes=self.adjoint_log_block_sizes)
        if self.log_block_sizes is not None:
            kwargs['target_shapes'] = (tuple(1 << b for b in self.log_block_sizes),)
        return QROAMCleanAdjoint.build_from_bitsize(self.data_shape, **kwargs)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        emit_range_safety(ret, self.data_shape)  # P-13
        ret[ZGate().controlled()] += 1     # Clifford: no phase-gradient addition
        ret[self.qroam_adj_bloq_for_cost] += 1
        return ret
