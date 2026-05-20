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
  to ``BlockUnitarySynthesisQROAM`` with ``n_reflections = N``.

In both cases the synthesis call is structurally identical: one
``BlockUnitarySynthesisQROAM(n_blocks=K, n_rows=D, n_reflections=min(M,N), ...)``,
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
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .block_unitary_synthesis_QROAM import BlockUnitarySynthesisQROAM
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:
    from block_unitary_synthesis_QROAM import BlockUnitarySynthesisQROAM
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
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
        log_block_sizes: ``log_block_sizes`` forwarded to the QROAM inside each reflection.
        adjoint_log_block_sizes: ``log_block_sizes`` for the QROAM uncompute inside each
            reflection.

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
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

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
    def reflection_synthesis(self) -> BlockUnitarySynthesisQROAM:
        """The (data-free) reflection synthesis bloq.

        ``n_blocks = K``, ``n_rows = D``, ``n_reflections = min(M, N)``.  When ``M > N``
        this is the standard column-synthesis; when ``M <= N`` it is the same call but
        interpreted as building the ``M`` rows of a padded ``D x D`` unitary (the
        transposed semantics is absorbed entirely in the QROAM data the caller loads).
        """
        return BlockUnitarySynthesisQROAM.from_shape(
            n_blocks=self.n_blocks,
            n_rows=self.matrix_dim,
            phase_bitsize=self.phase_bitsize,
            n_reflections=self.n_reflections,
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
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
