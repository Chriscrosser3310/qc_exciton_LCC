"""SVD-based block encoding of sum_k |k><k| (x) A_k using the block-unitary interferometer.

For each block A_k (size 2^n x 2^n) with SVD A_k = U_k Sigma_k V_k:

  1. Apply  sum_k |k><k| (x) V_k          via BlockUnitaryInterferometerSynthesisQROAM.
  2. Apply  sum_k |k><k| (x) Sigma_k      on a single block-encoding ancilla,
     loading angles theta_{k,i} = arccos(sigma_{k,i}) by QROAMClean, executing one
     controlled AddIntoPhaseGrad (Hadamard-sandwiched for an Ry on the ancilla),
     and uncomputing via QROAMCleanAdjoint.
  3. Apply  sum_k |k><k| (x) U_k          via BlockUnitaryInterferometerSynthesisQROAM.

Non-power-of-two matrix dimensions are supported by ``n_rows_logical = M``, where
``M <= n_rows = 2^n`` is the true matrix dimension and ``n_rows`` is the smallest power
of two at least M.  When set, two ``LessThanConstant(matrix_bitsize, M)`` comparators
(input + output side, each computed then uncomputed) flag the indices
``i >= M`` and OR the result into the block-encoding ancilla via a CNOT.  Combined
with theta_{k,i} = pi/2 (sigma_{k,i} = 0) for i >= M, this turns the synthesized
block encoding into a block encoding of the N x N zero-padded matrix A_pad whose
top-left M x M block is A and whose other entries are zero.

The class inherits from ``qualtran.bloqs.block_encoding.BlockEncoding`` so it can be
plugged into anything that consumes the Qualtran ``BlockEncoding`` interface (QSVT,
Qubitization-based phase estimation, hamiltonian simulation, ...).

``build_call_graph`` returns exactly the call counts above, so the Qualtran resource
counter walks the *actual* sub-bloq graph (no analytic formulas live in this file).
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING, Union

import attrs

from qualtran import Bloq, QAny, QBit, Register, Signature
from qualtran.bloqs.arithmetic.comparison import LessThanConstant
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        _data_max_log_block_sizes,
    )
    from .state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none
except ImportError:
    from block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        _data_max_log_block_sizes,
    )
    from state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class SVDBlockEncodingInterferometer(BlockEncoding):
    r"""$(1, 1, \epsilon)$ block encoding of $\sum_k |k\rangle\langle k| \otimes A_k$ via SVD.

    Each $A_k$ (size $2^n \times 2^n$, $\lVert A_k \rVert \le 1$) is decomposed as
    $A_k = U_k \Sigma_k V_k$.  The block encoding is

    .. math:: B = (I_a \otimes U)(R_y \otimes I_s)(I_a \otimes V),

    where the singular-value rotation $R_y$ acts on a single block-encoding ancilla
    controlled on the (block, system) index, loading $\theta_{k,i}$ via QROAMClean and
    uncomputing it via QROAMCleanAdjoint.

    This bloq exposes the standard ``qualtran.bloqs.block_encoding.BlockEncoding``
    interface (system/ancilla/resource registers; alpha, epsilon; signal_state) so it
    can drive QSVT, qubitization, or any other consumer of a Qualtran block encoding.

    Attributes:
        n_blocks: number $K$ of blocks $A_k$.
        n_rows: row count of each $A_k$ (= $2^n$).
        phase_bitsize: bitsize $b$ of phase/angle registers.
        n_layers: number of beamsplitter layers per interferometer (defaults to ``n_rows``).
        interferometer_log_block_sizes: ``log_block_sizes`` for U/V phase-layer QROAM.
        interferometer_final_log_block_sizes: ``log_block_sizes`` for U/V final-phase QROAM.
        interferometer_final_adjoint_log_block_sizes: ``log_block_sizes`` for U/V
            final-phase QROAM uncomputation.
        diag_log_block_sizes: ``log_block_sizes`` for the diagonal angle QROAM forward load.
        diag_adjoint_log_block_sizes: ``log_block_sizes`` for the diagonal QROAM uncomputation.
        n_rows_logical: optional logical row count $M \\le 2^n$; when set and strictly
            less than ``n_rows``, the bloq adds comparator-based flagging of indices
            $i \\ge M$ so the resulting block encoding represents the zero-padded matrix
            $A_{\\text{pad}}$ (top-left $M \\times M$ block is $A$, all other entries are zero).

    Registers:
        system: combined (block, matrix) register of size ``block_bitsize + matrix_bitsize``
                that the encoded operator acts on.
        ancilla: 1-qubit block-encoding ancilla (signal state $|0\rangle$).
        resource: ``phase_bitsize``-qubit phase-gradient workspace.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_layers: Optional[SymbolicInt] = None
    n_rows_logical: Optional[SymbolicInt] = None
    interferometer_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    interferometer_final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    interferometer_final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    diag_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    diag_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    # --------------------------- Local shape helpers ---------------------------

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def matrix_bitsize(self) -> SymbolicInt:
        """Number of qubits encoding a single block A_k (= log2 n_rows)."""
        return bit_length(self.n_rows - 1)

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
        # ||A_k|| <= 1 by assumption, so the block encoding has subnormalization 1.
        return 1.0

    @property
    def epsilon(self) -> SymbolicFloat:
        # Dominated by phase / rotation discretization at b bits.
        return 2.0 ** (-self.phase_bitsize)

    @cached_property
    def signature(self) -> Signature:
        regs = [
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QBit()) if self.ancilla_bitsize == 1
            else Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ]
        return Signature(regs)

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return PrepareIdentity.from_bitsizes((self.ancilla_bitsize,))

    # --------------------------- Sub-bloq factories ----------------------------

    @property
    def interferometer(self) -> BlockUnitaryInterferometerSynthesisQROAM:
        """Single-side (U_k or V_k) block-unitary interferometer; used twice in the call graph."""
        return BlockUnitaryInterferometerSynthesisQROAM(
            n_blocks=self.n_blocks,
            n_rows=self.n_rows,
            phase_bitsize=self.phase_bitsize,
            n_layers=self.n_layers,
            log_block_sizes=self.interferometer_log_block_sizes,
            final_log_block_sizes=self.interferometer_final_log_block_sizes,
            final_adjoint_log_block_sizes=self.interferometer_final_adjoint_log_block_sizes,
        )

    @property
    def diag_data_shape(self) -> Tuple[SymbolicInt, ...]:
        if is_symbolic(self.n_blocks) or self.n_blocks > 1:
            return (self.n_blocks, self.n_rows)
        return (self.n_rows,)

    def _capped_diag_lbs(self, raw: Optional[Tuple[SymbolicInt, ...]]):
        return _cap_log_block_sizes(raw, _data_max_log_block_sizes(self.diag_data_shape))

    @property
    def diag_qroam(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.diag_data_shape,
            target_bitsizes=(self.phase_bitsize,),
            log_block_sizes=self._capped_diag_lbs(self.diag_log_block_sizes),
        )

    @property
    def diag_qroam_adjoint(self) -> QROAMCleanAdjoint:
        lbs = self._capped_diag_lbs(self.diag_adjoint_log_block_sizes)
        target_shapes = (tuple(1 << b for b in lbs),) if lbs is not None else None
        kwargs = dict(target_bitsizes=(self.phase_bitsize,), log_block_sizes=lbs)
        if target_shapes is not None:
            kwargs['target_shapes'] = target_shapes
        return QROAMCleanAdjoint.build_from_bitsize(self.diag_data_shape, **kwargs)

    @property
    def ctrl_phase_grad_add(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()

    @property
    def has_logical_padding(self) -> bool:
        """True iff a comparator-based projection out of the i >= n_rows_logical subspace is needed."""
        if self.n_rows_logical is None:
            return False
        if is_symbolic(self.n_rows_logical) or is_symbolic(self.n_rows):
            return True
        return int(self.n_rows_logical) < int(self.n_rows)

    @property
    def comparator(self) -> Optional[Bloq]:
        """LessThanConstant(matrix_bitsize, n_rows_logical) used to flag i >= M on the BE ancilla.

        Comparator output is XORed into a workspace qubit then ORed into the BE ancilla;
        each comparator is followed by its self-inverse uncompute, so the call graph holds
        two LessThanConstant calls per side (input + output), times two sides.
        """
        if not self.has_logical_padding:
            return None
        return LessThanConstant(bitsize=self.matrix_bitsize, less_than_val=self.n_rows_logical)

    # ----------------------------- Resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.interferometer] += 2                # U_k and V_k
        ret[self.diag_qroam] += 1                    # load angles theta_{k,i}
        ret[self.ctrl_phase_grad_add] += 1           # Ry(2 theta) on BE ancilla
        ret[self.diag_qroam_adjoint] += 1            # uncompute angle register
        if self.has_logical_padding:
            # Two-sided projection: comparator (compute + uncompute) on the matrix
            # register before V_k and after U_k.  Each side: 2 LessThanConstant calls
            # (forward and self-inverse uncompute) writing a workspace bit that is
            # CNOTed into the BE ancilla (CNOT is Clifford, zero Toffoli).
            ret[self.comparator] += 4
        return ret
