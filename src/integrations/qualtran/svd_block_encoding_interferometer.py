"""SVD-based block encoding of sum_k |k><k| (x) A_k using the block-unitary interferometer.

For each block A_k (size 2^n x 2^n) with SVD A_k = U_k Sigma_k V_k:

  1. Apply  sum_k |k><k| (x) V_k          via BlockUnitaryInterferometerSynthesisQROAM.
  2. Apply  sum_k |k><k| (x) Sigma_k      on a single block-encoding ancilla,
     loading angles theta_{k,i} = arccos(sigma_{k,i}) by QROAMClean, executing one
     controlled AddIntoPhaseGrad (Hadamard-sandwiched for an Ry on the ancilla),
     and uncomputing via QROAMCleanAdjoint.
  3. Apply  sum_k |k><k| (x) U_k          via BlockUnitaryInterferometerSynthesisQROAM.

Non-power-of-two matrix dimensions are supported via ``n_rows_logical = M`` with
``M <= n_rows = 2^n``.  Padding is implemented entirely by the *data* baked into the
bloq's QROAM tables and interferometer angles: setting ``theta_{k,i} = pi/2``
(equivalently ``sigma_{k,i} = 0``) for ``i >= M`` together with block-diagonal
``U_pad``, ``V_pad`` (top-left ``M x M`` block carries ``U_M``, ``V_M``; bottom-right
``(N-M) x (N-M)`` block is the identity) makes the synthesized block encoding represent
the zero-padded ``A_pad``.  The circuit (and hence Toffoli / qubit cost) is the same
as for ``M = N``: any explicit projection would be redundant once the data is set up
this way.  ``n_rows_logical`` therefore acts as a documentation / consumer-side hint.

The class inherits from ``qualtran.bloqs.block_encoding.BlockEncoding`` so it can be
plugged into anything that consumes the Qualtran ``BlockEncoding`` interface (QSVT,
Qubitization-based phase estimation, hamiltonian simulation, ...).

``build_call_graph`` returns exactly the call counts above, so the Qualtran resource
counter walks the *actual* sub-bloq graph (no analytic formulas live in this file).
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np

from qualtran import Bloq, BloqBuilder, QAny, QBit, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import Hadamard
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
            less than ``n_rows`` the block encoding represents the zero-padded matrix
            $A_{\\text{pad}}$ (top-left $M \\times M$ block is $A$, all other entries are
            zero).  Padding is enforced by the data fed into this bloq's QROAM tables
            and synthesized U/V angles (block-diagonal extension, ``\\theta_{k,i} =
            \\pi/2`` for ``i >= M``).  The circuit itself does not depend on ``M``.

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

    # ----------------------------- Resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.interferometer] += 2                # U_k and V_k
        ret[self.diag_qroam] += 1                    # load angles theta_{k,i}
        ret[self.ctrl_phase_grad_add] += 1           # Ry(2 theta) on BE ancilla
        ret[self.diag_qroam_adjoint] += 1            # uncompute angle register
        return ret

    # --------------------------- Composite circuit -----------------------------

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Wire the actual circuit:  V_k  ->  Sigma_k (QROAM + Ry + QROAM^dag)  ->  U_k.

        The combined ``system`` register is split into ``block`` (upper
        ``block_bitsize`` bits) and ``matrix`` (lower ``matrix_bitsize`` bits) and
        rejoined at the end.

        Non-power-of-2 logical row counts ``M = n_rows_logical < n_rows`` are handled
        purely by the *data* that the caller bakes into this bloq's QROAM tables and
        interferometer angles: setting ``theta_{k,i} = pi/2`` (so ``sigma_{k,i} = 0``)
        for ``i >= M`` and choosing block-diagonal U_pad, V_pad makes
        ``<0_anc| (U_pad)(R_y)(V_pad^dag) |0_anc>`` equal to ``A_pad`` (top-left
        ``M x M`` block is ``A``, all other entries zero).  The circuit itself is
        unchanged by ``n_rows_logical``.
        """
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")

        has_block = int(self.n_blocks) > 1
        system = soqs['system']
        be_anc = soqs['ancilla']
        phase_grad = soqs['resource']

        # Split system into block (MSBs) and matrix (LSBs).
        if has_block:
            sys_arr = bb.split(system)
            bb_size = int(self.block_bitsize)
            block = bb.join(sys_arr[:bb_size], dtype=QUInt(self.block_bitsize))
            matrix = bb.join(sys_arr[bb_size:], dtype=QUInt(self.matrix_bitsize))
        else:
            matrix = system

        # ---- 1) V_k : block-unitary interferometer ----
        intf = self.interferometer
        v_in = {'system': matrix, 'phase_gradient': phase_grad}
        if has_block:
            v_in['block'] = block
        v_out = bb.add_d(intf, **v_in)
        matrix = v_out['system']
        phase_grad = v_out['phase_gradient']
        if has_block:
            block = v_out['block']

        # ---- 2) Diagonal Sigma_k stage ----
        qroam = self.diag_qroam
        qroam_adj = self.diag_qroam_adjoint
        sel_names = [r.name for r in qroam.selection_registers]
        if has_block:
            q_in = {sel_names[0]: block, sel_names[1]: matrix}
        else:
            q_in = {sel_names[0]: matrix}
        q_out = bb.add_d(qroam, **q_in)
        if has_block:
            block = q_out[sel_names[0]]
            matrix = q_out[sel_names[1]]
        else:
            matrix = q_out[sel_names[0]]
        phi = q_out['target0_']

        # Ry(2*theta) on BE ancilla via Hadamard sandwich + controlled phase-grad add.
        be_anc = bb.add(Hadamard(), q=be_anc)
        be_anc, phi, phase_grad = bb.add(
            self.ctrl_phase_grad_add, ctrl=be_anc, x=phi, phase_grad=phase_grad
        )
        be_anc = bb.add(Hadamard(), q=be_anc)

        # QROAM uncompute (measurement-based; 0 Toffoli for the intermediate-style adjoint).
        block_sizes = qroam.block_sizes
        junk_arr = (
            np.asarray(q_out['junk_target0_']) if 'junk_target0_' in q_out else np.array([])
        )
        adj_sel_names = [r.name for r in qroam_adj.selection_registers]
        adj_target = next(iter(qroam_adj.target_registers))
        adj_soqs: Dict[str, SoquetT] = {
            adj_target.name: np.array([phi, *junk_arr]).reshape(block_sizes)
        }
        if has_block:
            adj_soqs[adj_sel_names[0]] = block
            adj_soqs[adj_sel_names[1]] = matrix
        else:
            adj_soqs[adj_sel_names[0]] = matrix
        adj_out = bb.add_d(qroam_adj, **adj_soqs)
        if has_block:
            block = adj_out[adj_sel_names[0]]
            matrix = adj_out[adj_sel_names[1]]
        else:
            matrix = adj_out[adj_sel_names[0]]

        # ---- 3) U_k : block-unitary interferometer ----
        u_in = {'system': matrix, 'phase_gradient': phase_grad}
        if has_block:
            u_in['block'] = block
        u_out = bb.add_d(intf, **u_in)
        matrix = u_out['system']
        phase_grad = u_out['phase_gradient']
        if has_block:
            block = u_out['block']

        # Rejoin system register.
        if has_block:
            system = bb.join(
                np.concatenate([bb.split(block), bb.split(matrix)]),
                dtype=QAny(self.system_bitsize),
            )
        else:
            system = matrix

        return {'system': system, 'ancilla': be_anc, 'resource': phase_grad}
