r"""Rectangular block encoding of $\sum_k |k\rangle\langle k| \otimes A_k$ for
non-square, non-power-of-2 $A_k$ via the state-reflection (Householder) construction.

Given each $A_k$ of size $M \times N$ (with $\|A_k\| \le 1$), the bloq encodes the
zero-padded matrix $A_{k,\text{pad}}$ inside a $D \times D$ unitary, where

    D = smallest power of 2 $\ge M + N$.

Two cases:

* $M \le N$ (wider-or-equal): we synthesize the first $M$ *rows* of a $D \times D$
  unitary using $M$ reflections.  This is the **transposed** state-reflection method:
  the reflection vectors are the rows of (the padded) $A_k$.
* $M > N$ (taller): we synthesize the first $N$ *columns* of a $D \times D$ unitary
  using $N$ reflections.  This is the **standard** state-reflection method, identical
  to ``BlockUnitaryReflectionQROAM`` with ``n_reflections = N``.

In both cases the synthesis call is structurally identical: one
``BlockUnitaryReflectionQROAM(n_blocks=K, n_rows=D, n_reflections=min(M,N), ...)``,
whose reflection vectors are loaded by QROAM with the appropriate state
coefficients.

The non-power-of-2 row / column counts are absorbed into the *data* loaded by QROAM:
choosing $D = 2^{\lceil \log_2(M+N) \rceil}$ leaves the rows or columns $\ge \min(M,N)$
zero-padded in the natural $D$-dim Hilbert space.  No comparator is needed in the
circuit (the same argument as in :mod:`svd_block_encoding_interferometer`).

Implements both ``build_call_graph`` and ``build_composite_bloq`` (the latter delegates
to the underlying synthesis bloq for the actual circuit DAG).
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

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
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .block_unitary_reflection_QROAM import (
        BlockUnitaryReflectionQROAM,
        optimal_reflection_adjoint_log_block_sizes,
        optimal_reflection_log_block_sizes,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:
    from block_unitary_reflection_QROAM import (
        BlockUnitaryReflectionQROAM,
        optimal_reflection_adjoint_log_block_sizes,
        optimal_reflection_log_block_sizes,
    )
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _next_pow2_at_least(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << bit_length(int(n) - 1)


@attrs.frozen
class RectangularBlockEncodingReflection(BlockEncoding):
    r"""Block encoding of $\sum_k |k\rangle\langle k| \otimes A_k$ for rectangular
    $A_k$ of size $M \times N$ via the state-reflection construction.

    Padding:  $D = $ smallest power of $2 \ge M + N$ is the dimension of the
    synthesized unitary's system register.  If $M \le N$ the bloq uses the
    *transposed* reflection method (reflections build the $M$ rows of $A_k$);
    if $M > N$, the *standard* method (reflections build the $N$ columns of $A_k$).

    Attributes:
        n_blocks: number $K$ of blocks $A_k$.
        m_rows: logical row count $M$ (may be non-power-of-2).
        n_cols: logical column count $N$ (may be non-power-of-2).
        phase_bitsize: bitsize $b$ of phase registers.
        amp_log_block_sizes: ``log_block_sizes`` driving the amp-staircase forward QROAMs
            inside each reflection (per-layer cap applied).
        phase_log_block_sizes: ``log_block_sizes`` for the final phase layer forward QROAM.
        phase_adjoint_log_block_sizes: ``log_block_sizes`` for the final phase layer
            adjoint QROAM uncompute.

    Registers (``BlockEncoding`` interface):
        system: ``block_bitsize + matrix_bitsize`` qubits, where
                ``matrix_bitsize = log2 D = ceil(log2(M + N))``.
        ancilla: 1-qubit reflection ancilla (block-encoding ancilla, signal state $|0\rangle$).
        resource: ``phase_bitsize``-qubit phase-gradient workspace.
    """

    n_blocks: SymbolicInt
    m_rows: SymbolicInt
    n_cols: SymbolicInt
    phase_bitsize: SymbolicInt
    amp_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    amp_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    phase_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    phase_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    optimal_T: bool = False

    def __attrs_post_init__(self):
        if self.optimal_T:
            if is_symbolic(self.n_blocks, self.m_rows, self.n_cols, self.phase_bitsize):
                raise ValueError("optimal_T=True requires concrete n_blocks, m_rows, n_cols, phase_bitsize")
            opt_fwd = optimal_reflection_log_block_sizes(
                int(self.n_blocks), int(self.matrix_dim), int(self.phase_bitsize)
            )
            opt_adj = optimal_reflection_adjoint_log_block_sizes(
                int(self.n_blocks), int(self.matrix_dim)
            )
            object.__setattr__(self, 'amp_log_block_sizes', opt_fwd)
            object.__setattr__(self, 'amp_adjoint_log_block_sizes', opt_adj)
            object.__setattr__(self, 'phase_log_block_sizes', opt_fwd)
            object.__setattr__(self, 'phase_adjoint_log_block_sizes', opt_adj)

    # --------------------------- Shape helpers ---------------------------

    @cached_property
    def transposed(self) -> bool:
        """True when ``M <= N`` (we build rows of A); False when ``M > N`` (we build columns)."""
        if is_symbolic(self.m_rows, self.n_cols):
            raise NotImplementedError("transposed flag requires concrete M, N")
        return int(self.m_rows) <= int(self.n_cols)

    @cached_property
    def n_reflections(self) -> SymbolicInt:
        """Number of state reflections needed: ``min(M, N)``."""
        if is_symbolic(self.m_rows, self.n_cols):
            return self.m_rows  # symbolic; caller should specialize
        return min(int(self.m_rows), int(self.n_cols))

    @cached_property
    def matrix_dim(self) -> SymbolicInt:
        """Padded unitary dimension ``D = next_pow2(M + N)``."""
        if is_symbolic(self.m_rows, self.n_cols):
            raise NotImplementedError("matrix_dim requires concrete M, N")
        return _next_pow2_at_least(int(self.m_rows) + int(self.n_cols))

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def matrix_bitsize(self) -> SymbolicInt:
        return bit_length(self.matrix_dim - 1)

    # ------------------------- BlockEncoding interface -------------------------

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
    def signature(self) -> Signature:
        regs = [
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QBit()),
            Register('resource', QAny(self.resource_bitsize)),
        ]
        return Signature(regs)

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return PrepareIdentity.from_bitsizes((1,))

    # --------------------------- Sub-bloq factory ----------------------------

    @cached_property
    def reflection_synthesis(self) -> BlockUnitaryReflectionQROAM:
        """The (data-free) reflection synthesis bloq.

        ``n_blocks = K``, ``n_rows = D``, ``n_reflections = min(M, N)``.  When ``M > N``
        this is the standard column-synthesis (``transpose=False``); when ``M <= N`` it
        builds the ``M`` rows of the padded ``D x D`` unitary via the reversed
        (``transpose=True``) reflection sequence.  Both have identical gate cost.
        """
        return BlockUnitaryReflectionQROAM.from_shape(
            n_blocks=self.n_blocks,
            n_rows=self.matrix_dim,
            phase_bitsize=self.phase_bitsize,
            n_reflections=self.n_reflections,
            amp_log_block_sizes=self.amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
            transpose=self.transposed,
        )

    # ----------------------------- Resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.reflection_synthesis] += 1
        return ret

    # --------------------------- Composite circuit -----------------------------

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Split ``system`` into ``block`` + ``matrix`` and call the synthesis sub-bloq."""
        if is_symbolic(self.n_blocks, self.m_rows, self.n_cols, self.phase_bitsize):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")

        has_block = int(self.n_blocks) > 1
        system = soqs['system']
        ancilla = soqs['ancilla']
        resource = soqs['resource']

        if has_block:
            sys_arr = bb.split(system)
            bb_size = int(self.block_bitsize)
            block = bb.join(sys_arr[:bb_size], dtype=QUInt(self.block_bitsize))
            matrix = bb.join(sys_arr[bb_size:], dtype=QUInt(self.matrix_bitsize))
        else:
            matrix = system

        synth = self.reflection_synthesis
        in_soqs: Dict[str, SoquetT] = {
            'system': matrix,
            'reflection_ancilla': ancilla,
            'phase_gradient': resource,
        }
        if has_block:
            in_soqs['block'] = block
        out = bb.add_d(synth, **in_soqs)
        matrix = out['system']
        ancilla = out['reflection_ancilla']
        resource = out['phase_gradient']
        if has_block:
            block = out['block']

        if has_block:
            system = bb.join(
                np.concatenate([bb.split(block), bb.split(matrix)]),
                dtype=QAny(self.system_bitsize),
            )
        else:
            system = matrix

        return {'system': system, 'ancilla': ancilla, 'resource': resource}


# =============================================================================
# Square-isometry variant: simpler API for ``A_k`` already shaped (n_rows, n_reflections)
# with no extra (M + N) padding.  Kept alongside ``RectangularBlockEncodingReflection``
# because callers that already know the synthesis dimension (e.g. SVD dilation, where
# ``n_rows`` is a power of 2) avoid the padding overhead.
# =============================================================================


@attrs.frozen
class ReflectionRectangularBlockEncoding(BlockEncoding):
    r"""$(1, 1, \epsilon)$ Householder-reflection isometry block encoding.

    Encodes $\sum_k |k\rangle\langle k| \otimes A_k$ where each $A_k$ is an
    ``n_rows x n_reflections`` isometry that sits in the top-left block of an
    ``n_rows``-dimensional unitary, synthesized as ``n_reflections`` block
    Householder reflections via :class:`BlockUnitaryReflectionQROAM`.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_reflections: Optional[SymbolicInt] = None
    amp_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    amp_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    phase_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    phase_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    optimal_T: bool = False
    # When True, synthesize the first ``n_reflections`` *rows* of each U_k (reflection
    # sequence reversed -> builds U^dagger), instead of its columns.  Same cost; lets a
    # rectangular A_k be encoded along whichever side has ``n_reflections`` vectors.
    transpose: bool = False

    def __attrs_post_init__(self):
        if self.optimal_T:
            if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
                raise ValueError("optimal_T=True requires concrete n_blocks, n_rows, phase_bitsize")
            opt_fwd = optimal_reflection_log_block_sizes(
                int(self.n_blocks), int(self.n_rows), int(self.phase_bitsize)
            )
            opt_adj = optimal_reflection_adjoint_log_block_sizes(
                int(self.n_blocks), int(self.n_rows)
            )
            object.__setattr__(self, 'amp_log_block_sizes', opt_fwd)
            object.__setattr__(self, 'amp_adjoint_log_block_sizes', opt_adj)
            object.__setattr__(self, 'phase_log_block_sizes', opt_fwd)
            object.__setattr__(self, 'phase_adjoint_log_block_sizes', opt_adj)

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
    def synth(self) -> BlockUnitaryReflectionQROAM:
        return BlockUnitaryReflectionQROAM.from_shape(
            n_blocks=self.n_blocks,
            n_rows=self.n_rows,
            phase_bitsize=self.phase_bitsize,
            n_reflections=self.n_reflections_effective,
            amp_log_block_sizes=self.amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
            transpose=self.transpose,
        )

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        return Counter({self.synth: 1})

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Split ``system`` into ``block`` + ``matrix`` and call the synthesis sub-bloq."""
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")
        has_block = int(self.n_blocks) > 1
        system, ancilla, resource = soqs['system'], soqs['ancilla'], soqs['resource']

        if has_block:
            sys_arr = bb.split(system)
            bb_size = int(self.block_bitsize)
            block = bb.join(sys_arr[:bb_size], dtype=QUInt(self.block_bitsize))
            matrix = bb.join(sys_arr[bb_size:], dtype=QUInt(self.matrix_bitsize))
        else:
            matrix = system

        in_soqs: Dict[str, SoquetT] = {
            'system': matrix, 'reflection_ancilla': ancilla, 'phase_gradient': resource,
        }
        if has_block:
            in_soqs['block'] = block
        out = bb.add_d(self.synth, **in_soqs)
        matrix, ancilla, resource = out['system'], out['reflection_ancilla'], out['phase_gradient']
        if has_block:
            block = out['block']

        if has_block:
            system = bb.join(
                np.concatenate([bb.split(block), bb.split(matrix)]), dtype=QAny(self.system_bitsize)
            )
        else:
            system = matrix
        return {'system': system, 'ancilla': ancilla, 'resource': resource}

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledReflectionRectangularBlockEncoding(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledReflectionRectangularBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`ReflectionRectangularBlockEncoding`.

    Each Householder reflection is promoted to its controlled variant
    (:class:`_ControlledHouseholderReflection` via ``reflection(i).controlled()``);
    the QROAM prepare/uncompute pair inside each reflection stays uncontrolled because
    it self-cancels on ``ctrl = 0``.  The overhead is one extra control bit on one
    ``MultiControlZ`` per reflection -- tiny next to the QROAM cost.
    """

    inner: "ReflectionRectangularBlockEncoding"

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
            Register('resource', QAny(self.resource_bitsize)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        synth = self.inner.synth
        ret: "Counter[Bloq]" = Counter()
        n_refl = synth.n_reflections
        if is_symbolic(n_refl):
            ret[synth.reflection(0).controlled()] += n_refl
        else:
            for basis_index in range(int(n_refl)):
                ret[synth.reflection(basis_index).controlled()] += 1
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """ctrl-controlled sequence of Householder reflections (only the reflect-about-0
        gets the extra control; the QROAM prepare/uncompute stays uncontrolled)."""
        inner = self.inner
        if is_symbolic(inner.n_blocks, inner.n_rows, inner.phase_bitsize):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")
        ctrl = soqs.pop('ctrl')
        system, ancilla, resource = soqs.pop('system'), soqs.pop('ancilla'), soqs.pop('resource')
        has_block = int(inner.n_blocks) > 1

        if has_block:
            sys_arr = bb.split(system)
            bb_size = int(inner.block_bitsize)
            block = bb.join(sys_arr[:bb_size], dtype=QUInt(inner.block_bitsize))
            matrix = bb.join(sys_arr[bb_size:], dtype=QUInt(inner.matrix_bitsize))
        else:
            matrix = system

        synth = inner.synth
        # Match the uncontrolled order: reversed reflection sequence when transpose=True.
        order = range(int(synth.n_reflections))
        if synth.transpose:
            order = reversed(order)
        for basis_index in order:
            cr = synth.reflection(basis_index).controlled()
            in_soqs: Dict[str, SoquetT] = {
                'ctrl': ctrl, 'system': matrix,
                'reflection_ancilla': ancilla, 'phase_gradient': resource,
            }
            if has_block:
                in_soqs['block'] = block
            out = bb.add_d(cr, **in_soqs)
            ctrl, matrix = out['ctrl'], out['system']
            ancilla, resource = out['reflection_ancilla'], out['phase_gradient']
            if has_block:
                block = out['block']

        if has_block:
            system = bb.join(
                np.concatenate([bb.split(block), bb.split(matrix)]), dtype=QAny(inner.system_bitsize)
            )
        else:
            system = matrix
        return {'ctrl': ctrl, 'system': system, 'ancilla': ancilla, 'resource': resource}
