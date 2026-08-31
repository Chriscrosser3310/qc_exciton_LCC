r"""Eigendecomposition block encoding ``U D U^dag`` of a block-diagonal Hermitian operator.

Encodes

.. math::  A = \sum_k |k\rangle\langle k| \otimes A_k, \qquad A_k = A_k^\dagger,

by diagonalizing each block *classically* as :math:`A_k = U_k D_k U_k^\dagger` and
building the **congruence**

.. math::  B = (I_a \otimes U)\,(Z R_y \otimes I_s)\,(I_a \otimes U^\dagger).

This is the construction `main.tex` Sec. 2 specifies for the Fock template
("Classically, each matrix is diagonalized as :math:`f^{o/v} = U D U^\dagger`") and for
the operator-norm option on the exchange template's central :math:`\zeta^V_{\mathbf Q}`.

Why not the SVD sibling
-----------------------
:class:`~.svd_block_encoding_interferometer.SVDBlockEncodingInterferometer` builds
:math:`U \Sigma V` with **independent** :math:`U, V`.  That is strictly more general, and
for a Hermitian :math:`A_k` it is also strictly worse in one respect that matters here:
it is **not an involution**, because :math:`(U\Sigma V)^2 \neq I` when :math:`V \neq
U^\dagger`.  The congruence form is:

.. math::
    B^2 = U (Z R_y) U^\dagger U (Z R_y) U^\dagger = U (Z R_y)^2 U^\dagger = I,

since :math:`Z R_y(2\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ -\sin\theta &
-\cos\theta\end{pmatrix}` is real symmetric and squares to the identity.  The qubitization
walk :math:`W = (2\Pi - I) U_A` needs exactly this, and it costs **nothing** extra: the
:math:`Z` is Clifford.

One rotation, not two
---------------------
:math:`A_k` Hermitian means the eigenvalues :math:`d_{k,i}` are **real**, so
:math:`\theta_{k,i} = \arccos(d_{k,i}) \in [0, \pi]` and a single :math:`R_y` carries both
magnitude and sign.  A complex diagonal would need a second rotation for the phase; there
is none here.  This is the same "one rotation" accounting the diagonal Coulomb kernel uses.

Subnormalization
----------------
:math:`\alpha = \max_k \lVert A_k \rVert = \max_k \max_i |d_{k,i}|`, the operator norm --
this is the property that makes the eigendecomposition worth its cost relative to the
Frobenius alternative, whose :math:`\alpha` is :math:`\lVert A \rVert_F \ge \lVert A
\rVert`.  Callers pass the physical value as ``alpha_val``; the default ``1.0`` matches
the data-free convention of the sibling modules (blocks assumed pre-normalized).

Data-free: only the circuit *structure* is described; all Toffoli / qubit counts come from
Qualtran's resource counter walking ``build_call_graph``.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import ZGate
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        optimal_interferometer_log_block_sizes,
    )
except ImportError:
    from block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        optimal_interferometer_log_block_sizes,
    )

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
class EigendecompositionBlockEncoding(BlockEncoding):
    r"""Self-inverse block encoding of $\sum_k |k\rangle\langle k| \otimes A_k$ via $U D U^\dagger$.

    Each $A_k$ is an $n_{\mathrm{rows}} \times n_{\mathrm{rows}}$ **Hermitian** contraction.
    The encoding is an involution ($B = B^\dagger$, $B^2 = I$), so it can be dropped
    straight into a qubitization walk without a Hermitian-repair wrapper.

    Attributes:
        n_blocks: number $K$ of blocks $A_k$ (the multiplexer address, e.g. $N_k$ or $N_Q$).
        n_rows: row count of each $A_k$.  Need not be a power of two -- the
            interferometer only requires an even dimension (it pairs Givens blocks).
        phase_bitsize: bitsize $b$ of the phase / angle registers.
        n_layers: beamsplitter layers per interferometer (defaults to ``n_rows``).
        alpha_val: physical subnormalization $\max_k \lVert A_k\rVert$; ``1.0`` in the
            data-free convention (blocks pre-normalized).
        optimal_T: choose Toffoli-optimal QROAM blocking throughout.
        real_data: the eigenvector matrices $U_k$ are real orthogonal.  Halves the
            per-layer rotation load in the interferometer (no relative-phase table).

    Registers:
        system: ``block_bitsize + matrix_bitsize`` qubits.
        ancilla: 1-qubit block-encoding ancilla (signal state $|0\rangle$).
        resource: ``phase_bitsize``-qubit phase-gradient workspace.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_layers: Optional[SymbolicInt] = None
    alpha_val: SymbolicFloat = 1.0
    interferometer_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    interferometer_final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    interferometer_final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    diag_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    diag_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    optimal_T: bool = False
    real_data: bool = False

    def __attrs_post_init__(self):
        if self.optimal_T:
            if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
                raise ValueError("optimal_T=True requires concrete n_blocks, n_rows, phase_bitsize")
            opt = optimal_interferometer_log_block_sizes(
                int(self.n_blocks), int(self.n_rows), int(self.phase_bitsize)
            )
            if int(self.n_blocks) == 1:
                opt = (opt[-1],)
            for field in (
                'interferometer_log_block_sizes',
                'interferometer_final_log_block_sizes',
                'interferometer_final_adjoint_log_block_sizes',
                'diag_log_block_sizes',
                'diag_adjoint_log_block_sizes',
            ):
                object.__setattr__(self, field, opt)

    # --------------------------- Local shape helpers ---------------------------

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def matrix_bitsize(self) -> SymbolicInt:
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
        return self.alpha_val

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

    @property
    def interferometer(self) -> BlockUnitaryInterferometerSynthesisQROAM:
        """The eigenvector synthesis $U$; its adjoint supplies $U^\\dagger$."""
        return BlockUnitaryInterferometerSynthesisQROAM(
            n_blocks=self.n_blocks,
            n_rows=self.n_rows,
            phase_bitsize=self.phase_bitsize,
            n_layers=self.n_layers,
            log_block_sizes=self.interferometer_log_block_sizes,
            final_log_block_sizes=self.interferometer_final_log_block_sizes,
            final_adjoint_log_block_sizes=self.interferometer_final_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    @property
    def diag_data_shape(self) -> Tuple[SymbolicInt, ...]:
        if is_symbolic(self.n_blocks) or self.n_blocks > 1:
            return (self.n_blocks, self.n_rows)
        return (self.n_rows,)

    @property
    def diag_qroam(self) -> QROAMClean:
        """Load $\\theta_{k,i} = \\arccos(d_{k,i})$; one real angle per eigenvalue."""
        return QROAMClean.build_from_bitsize(
            self.diag_data_shape,
            target_bitsizes=(self.phase_bitsize,),
            log_block_sizes=self.diag_log_block_sizes,
        )

    @property
    def diag_qroam_adjoint(self) -> QROAMCleanAdjoint:
        kwargs = dict(
            target_bitsizes=(self.phase_bitsize,),
            log_block_sizes=self.diag_adjoint_log_block_sizes,
        )
        if self.diag_log_block_sizes is not None:
            kwargs['target_shapes'] = (tuple(1 << b for b in self.diag_log_block_sizes),)
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
        r"""$U^\dagger \to (\text{QROAM} \to Z R_y \to \text{QROAM}^\dagger) \to U$."""
        ret: "Counter[Bloq]" = Counter()
        # U and U^dag: the adjoint is the same layer structure run in reverse with
        # conjugated angle tables, so it has the same Toffoli count.  Emitted as two
        # copies of the same bloq (the convention the SVD sibling uses for U and V) --
        # Qualtran's generic Adjoint wrapper cannot cost a QROAMCleanAdjoint.
        ret[self.interferometer] += 2
        ret[self.diag_qroam] += 1                   # load theta_{k,i} = arccos(d_{k,i})
        emit_range_safety(ret, self.diag_data_shape)  # P-13
        ret[self.ctrl_phase_grad_add] += 1          # ONE R_y: the eigenvalues are real
        ret[ZGate()] += 1                           # Z after R_y -> involution on the ancilla
        ret[self.diag_qroam_adjoint] += 1           # uncompute the angle register
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Cheap single control: only the rotation-bearing pieces gain it.

        The $U$ / $U^\\dagger$ pair cancels at ``ctrl = 0``, so the control only has to
        reach the diagonal's ``R_y`` and ``Z``.  That is *cheaper* than the SVD sibling,
        where both interferometers must be controlled.
        """
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledEigendecompositionBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledEigendecompositionBlockEncoding(BlockEncoding):
    r"""Singly-controlled :class:`EigendecompositionBlockEncoding`.

    Because the encoding is a congruence $U (Z R_y) U^\dagger$, the outer pair is
    *unconditionally* applied: at ``ctrl = 0`` the inner reflection becomes the identity
    and $U U^\dagger = I$.  Only the ``R_y`` and ``Z`` are promoted.
    """

    inner: EigendecompositionBlockEncoding

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
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner.interferometer] += 2              # uncontrolled: U U^dag cancels at ctrl = 0
        ret[self.inner.diag_qroam] += 1                  # uncontrolled (cancels)
        emit_range_safety(ret, self.inner.diag_data_shape)  # P-13
        ret[And()] += 1                                   # the control, on the load: +1
        ret[self.inner.ctrl_phase_grad_add] += 1          # rotation unchanged (angle 0 -> I)
        ret[ZGate().controlled()] += 1                    # Clifford, free
        ret[self.inner.diag_qroam_adjoint] += 1
        return ret
