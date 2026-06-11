"""Block-diagonal unitary synthesis via block-indexed QROAM state preparation."""

from collections import Counter
from math import ceil, log2, sqrt
from typing import cast, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZGate
from qualtran.bloqs.mcmt import MultiControlZ
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.symbolics import bit_length, HasLength, is_symbolic, Shaped, shape, SymbolicInt

try:
    from .block_state_preparation_QROAM import BlockStatePreparationViaQROAMRotations
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:
    from block_state_preparation_QROAM import BlockStatePreparationViaQROAMRotations
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _to_block_unitaries_or_shape(
    x: Union[Shaped, Iterable[Iterable[Iterable[complex]]]]
) -> Union[Shaped, NDArray[np.complex128]]:
    if isinstance(x, Shaped):
        return x
    return np.asarray(x, dtype=np.complex128)


@attrs.frozen
class BlockPrepareHouseholderStateQROAM(GateWithRegisters):
    r"""Prepare block-indexed Householder states.

    For each block ``j`` and column ``k``, this prepares

    $$
        |w_{j,k}\rangle = (|1\rangle |k\rangle - |0\rangle |u_{j,k}\rangle)/\sqrt{2}
    $$

    while leaving the block register ``|j>`` unchanged.
    """

    state_coefficients: Union[Shaped, NDArray[np.complex128]] = attrs.field(
        converter=lambda x: x if isinstance(x, Shaped) else np.asarray(x, dtype=np.complex128),
        eq=lambda x: tuple(shape(x)),
    )
    phase_bitsize: SymbolicInt
    basis_index: int
    uncompute: bool = False
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
        assert len(shape(self.state_coefficients)) == 2
        n_blocks, n_rows = shape(self.state_coefficients)
        if not is_symbolic(n_blocks):
            assert n_blocks >= 1
        if not is_symbolic(n_rows):
            assert n_rows == 2**self.system_bitsize
            assert 0 <= self.basis_index < n_rows
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if isinstance(self.state_coefficients, np.ndarray):
            norms = np.linalg.norm(self.state_coefficients, axis=1)
            assert np.allclose(norms, np.ones_like(norms))

    @property
    def n_blocks(self) -> SymbolicInt:
        return shape(self.state_coefficients)[0]

    @property
    def n_rows(self) -> SymbolicInt:
        return shape(self.state_coefficients)[1]

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            reflection_ancilla=1,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def state_prep(self) -> BlockStatePreparationViaQROAMRotations:
        return BlockStatePreparationViaQROAMRotations(
            state_coefficients=self.state_coefficients,
            phase_bitsize=self.phase_bitsize,
            control_bitsize=1,
            uncompute=self.uncompute,
            amp_log_block_sizes=self.amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
            # Tie per-layer T-optimal row blocking to optimal_T.  When optimal_T is False
            # (qubit-optimal request) we must honor the literal log_block_sizes verbatim --
            # otherwise per_layer_optimal=True silently re-blocks each staircase layer to its
            # T-optimum (lambda ~ sqrt(n_blocks*n_rows/b)), so a (0,0) "no blocking" request
            # would still allocate sqrt-scaling QROAM ancilla instead of the lambda=1 minimum.
            per_layer_optimal=self.optimal_T,
        )

    def adjoint(self) -> "BlockPrepareHouseholderStateQROAM":
        return attrs.evolve(self, uncompute=not self.uncompute)

    def _basis_one_positions(self) -> Tuple[int, ...]:
        if is_symbolic(self.system_bitsize):
            return ()
        return tuple(qi for qi in range(int(self.system_bitsize)) if (self.basis_index >> qi) & 1)

    def _apply_basis_cnot_ladder(
        self, bb: BloqBuilder, reflection_ancilla: Soquet, system_qubits: NDArray
    ) -> Tuple[Soquet, NDArray]:
        for qi in self._basis_one_positions():
            reflection_ancilla, system_qubits[qi] = bb.add(
                CNOT(), ctrl=reflection_ancilla, target=system_qubits[qi]
            )
        return reflection_ancilla, system_qubits

    def _apply_controlled_state_prep(
        self,
        bb: BloqBuilder,
        block: Optional[Soquet],
        reflection_ancilla: Soquet,
        system: Soquet,
        phase_gradient: Soquet,
    ) -> Tuple[Optional[Soquet], Soquet, Soquet, Soquet]:
        reflection_ancilla = bb.add(XGate(), q=reflection_ancilla)
        extra_soqs = {"block": block} if block is not None else {}
        out_soqs = bb.add_d(
            self.state_prep,
            **extra_soqs,
            prepare_control=reflection_ancilla,
            target_state=system,
            phase_gradient=phase_gradient,
        )
        block = cast(Soquet, out_soqs["block"]) if block is not None else None
        reflection_ancilla = cast(Soquet, out_soqs["prepare_control"])
        system = cast(Soquet, out_soqs["target_state"])
        phase_gradient = cast(Soquet, out_soqs["phase_gradient"])
        reflection_ancilla = bb.add(XGate(), q=reflection_ancilla)
        return block, reflection_ancilla, system, phase_gradient

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        # ``block`` is omitted from the signature when ``n_blocks == 1`` (block_bitsize=0).
        block = soqs.pop("block", None)
        reflection_ancilla = soqs.pop("reflection_ancilla")
        system = soqs.pop("system")
        phase_gradient = soqs.pop("phase_gradient")

        system_qubits = bb.split(system)
        if self.uncompute:
            block, reflection_ancilla, system, phase_gradient = self._apply_controlled_state_prep(
                bb, block, reflection_ancilla, bb.join(system_qubits), phase_gradient
            )
            system_qubits = bb.split(system)
            reflection_ancilla, system_qubits = self._apply_basis_cnot_ladder(
                bb, reflection_ancilla, system_qubits
            )
            reflection_ancilla = bb.add(ZGate(), q=reflection_ancilla)
            reflection_ancilla = bb.add(Hadamard(), q=reflection_ancilla)
        else:
            reflection_ancilla = bb.add(Hadamard(), q=reflection_ancilla)
            reflection_ancilla = bb.add(ZGate(), q=reflection_ancilla)
            reflection_ancilla, system_qubits = self._apply_basis_cnot_ladder(
                bb, reflection_ancilla, system_qubits
            )
            block, reflection_ancilla, system, phase_gradient = self._apply_controlled_state_prep(
                bb, block, reflection_ancilla, bb.join(system_qubits), phase_gradient
            )
            system_qubits = bb.split(system)

        if block is not None:
            soqs["block"] = block
        soqs["reflection_ancilla"] = reflection_ancilla
        soqs["system"] = bb.join(system_qubits)
        soqs["phase_gradient"] = phase_gradient
        return soqs

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[Hadamard()] += 1
        ret[ZGate()] += 1
        ret[CNOT()] += len(self._basis_one_positions())
        ret[XGate()] += 2
        ret[self.state_prep] += 1
        return ret


@attrs.frozen
class BlockHouseholderReflectionQROAM(GateWithRegisters):
    """Block-diagonal reflection about block-indexed Householder states."""

    state_coefficients: Union[Shaped, NDArray[np.complex128]] = attrs.field(
        converter=lambda x: x if isinstance(x, Shaped) else np.asarray(x, dtype=np.complex128),
        eq=lambda x: tuple(shape(x)),
    )
    phase_bitsize: SymbolicInt
    basis_index: int
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

    @property
    def n_blocks(self) -> SymbolicInt:
        return shape(self.state_coefficients)[0]

    @property
    def n_rows(self) -> SymbolicInt:
        return shape(self.state_coefficients)[1]

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            reflection_ancilla=1,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def prepare_w(self) -> BlockPrepareHouseholderStateQROAM:
        return BlockPrepareHouseholderStateQROAM(
            state_coefficients=self.state_coefficients,
            phase_bitsize=self.phase_bitsize,
            basis_index=self.basis_index,
            amp_log_block_sizes=self.amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    def _reflect_around_zero(
        self, bb: BloqBuilder, reflection_ancilla: Soquet, system_qubits: NDArray
    ) -> Tuple[Soquet, NDArray]:
        reflection_qubits = np.array([reflection_ancilla, *system_qubits], dtype=object)
        if len(reflection_qubits) == 1:
            reflection_qubits[0] = bb.add(XGate(), q=reflection_qubits[0])
            reflection_qubits[0] = bb.add(ZGate(), q=reflection_qubits[0])
            reflection_qubits[0] = bb.add(XGate(), q=reflection_qubits[0])
        else:
            target = reflection_qubits[-1]
            controls = reflection_qubits[:-1]
            target = bb.add(XGate(), q=target)
            controls, target = bb.add(
                MultiControlZ((0,) * len(controls)), controls=controls, target=target
            )
            target = bb.add(XGate(), q=target)
            reflection_qubits[:-1] = controls
            reflection_qubits[-1] = target
        return reflection_qubits[0], reflection_qubits[1:]

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        soqs = bb.add_d(self.prepare_w.adjoint(), **soqs)
        system_qubits = bb.split(soqs.pop("system"))
        reflection_ancilla, system_qubits = self._reflect_around_zero(
            bb, soqs.pop("reflection_ancilla"), system_qubits
        )
        soqs["reflection_ancilla"] = reflection_ancilla
        soqs["system"] = bb.join(system_qubits)
        soqs = bb.add_d(self.prepare_w, **soqs)
        return soqs

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        n_reflection_qubits = 1 + self.system_bitsize
        ret: "Counter[Bloq]" = Counter()
        ret[self.prepare_w] += 1
        ret[self.prepare_w.adjoint()] += 1
        ret[XGate()] += 2
        if is_symbolic(n_reflection_qubits):
            ret[MultiControlZ(HasLength(n_reflection_qubits - 1))] += 1
        elif n_reflection_qubits == 1:
            ret[ZGate()] += 1
        else:
            ret[MultiControlZ((0,) * int(n_reflection_qubits - 1))] += 1
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledHouseholderReflection(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledHouseholderReflection(GateWithRegisters):
    """Singly-controlled :class:`BlockHouseholderReflectionQROAM`.

    Reflection ``R = W (I - 2|0><0|) W^dag``.  Controlling ``R`` is done by adding the
    external control as one extra positive control to the inner reflection-about-zero;
    the QROAM-based ``prepare_w`` / ``prepare_w^dag`` pair stays *uncontrolled* because
    ``W I W^dag = I`` when the control is 0 (and the X gates around the multi-control-Z
    self-cancel).  So the cost is the bare reflection plus a single extra control bit on
    one ``MultiControlZ``.
    """

    inner: "BlockHouseholderReflectionQROAM"

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        n_reflection_qubits = 1 + self.inner.system_bitsize
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner.prepare_w] += 1
        ret[self.inner.prepare_w.adjoint()] += 1
        ret[XGate()] += 2
        # ctrl (cv=1) plus the n_reflection_qubits-1 original zero-controls.
        if is_symbolic(n_reflection_qubits):
            ret[MultiControlZ(HasLength(n_reflection_qubits))] += 1
        else:
            cvs = (1,) + (0,) * int(n_reflection_qubits - 1)
            ret[MultiControlZ(cvs)] += 1
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        ctrl = soqs.pop('ctrl')
        prepare_w = self.inner.prepare_w

        # 1) uncontrolled W^dag
        soqs = bb.add_d(prepare_w.adjoint(), **soqs)

        # 2) ctrl-controlled reflection about |0...0> on [reflection_ancilla, *system].
        #    Reflection = X_target . MCZ(controls=0..0) . X_target; adding `ctrl` as an
        #    extra positive control to the MCZ controls the whole reflection (when ctrl=0
        #    the MCZ is identity and the two X's cancel).
        reflection_ancilla = soqs.pop('reflection_ancilla')
        system_qubits = bb.split(soqs.pop('system'))
        reflection_qubits = np.array([reflection_ancilla, *system_qubits], dtype=object)
        target = reflection_qubits[-1]
        controls = np.array([ctrl, *reflection_qubits[:-1]], dtype=object)
        cvs = (1,) + (0,) * (len(reflection_qubits) - 1)
        target = bb.add(XGate(), q=target)
        controls, target = bb.add(MultiControlZ(cvs), controls=controls, target=target)
        target = bb.add(XGate(), q=target)
        ctrl = controls[0]
        reflection_qubits[:-1] = controls[1:]
        reflection_qubits[-1] = target

        soqs['reflection_ancilla'] = reflection_qubits[0]
        soqs['system'] = bb.join(reflection_qubits[1:])

        # 3) uncontrolled W
        soqs = bb.add_d(prepare_w, **soqs)
        return {'ctrl': ctrl, **soqs}


@attrs.frozen
class BlockUnitaryReflectionQROAM(GateWithRegisters):
    r"""Synthesize block-diagonal unitary data ``sum_j |j><j| tensor U_j``.

    ``block_unitaries`` has shape ``(n_blocks, N, n_reflections)``.  For each block ``j`` the
    first ``n_reflections`` columns of ``U_j`` are synthesized using the Sec. 4 reflection
    construction, while the block register is left unchanged.

    Transpose mode (``transpose=True``):  each Householder reflection ``R_i`` is its own
    inverse, so the synthesized unitary ``U = R_{n-1} ... R_1 R_0`` has adjoint
    ``U^dagger = R_0 R_1 ... R_{n-1}`` -- the *same* reflections applied in reversed
    order.  Running the reflection sequence backwards therefore builds ``U^dagger``,
    i.e. it synthesizes the first ``n_reflections`` *rows* of ``U`` (= columns of
    ``U^dagger``) instead of its columns.  This lets a rectangular ``M x N`` isometry be
    synthesized along the short side: ``n_reflections = min(M, N)`` reflections, choosing
    ``transpose`` to decide whether those reflections build rows or columns.  The gate
    cost is unchanged (the reflection multiset is identical; only the order flips).
    """

    block_unitaries: Union[Shaped, NDArray[np.complex128]] = attrs.field(
        converter=_to_block_unitaries_or_shape, eq=lambda x: tuple(shape(x))
    )
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
    transpose: bool = False

    def __attrs_post_init__(self):
        assert len(shape(self.block_unitaries)) == 3
        n_blocks, n_rows, n_cols = shape(self.block_unitaries)
        if not is_symbolic(n_blocks):
            assert n_blocks >= 1
        if not is_symbolic(n_rows):
            assert n_rows == 2**self.system_bitsize
        if not is_symbolic(n_rows, n_cols):
            assert n_cols <= n_rows
        if isinstance(self.block_unitaries, np.ndarray):
            for block in range(self.block_unitaries.shape[0]):
                gram = self.block_unitaries[block].conj().T @ self.block_unitaries[block]
                assert np.allclose(gram, np.eye(self.block_unitaries.shape[2]), atol=1e-8)
        if self.optimal_T:
            if is_symbolic(n_blocks, n_rows, self.phase_bitsize):
                raise ValueError("optimal_T=True requires concrete n_blocks, n_rows, phase_bitsize")
            opt_fwd = optimal_reflection_log_block_sizes(
                int(n_blocks), int(n_rows), int(self.phase_bitsize)
            )
            opt_adj = optimal_reflection_adjoint_log_block_sizes(
                int(n_blocks), int(n_rows)
            )
            object.__setattr__(self, 'amp_log_block_sizes', opt_fwd)
            object.__setattr__(self, 'amp_adjoint_log_block_sizes', opt_adj)
            object.__setattr__(self, 'phase_log_block_sizes', opt_fwd)
            object.__setattr__(self, 'phase_adjoint_log_block_sizes', opt_adj)

    @classmethod
    def from_shape(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_reflections: Optional[SymbolicInt] = None,
        amp_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        amp_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        phase_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        phase_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        optimal_T: bool = False,
        transpose: bool = False,
    ) -> "BlockUnitaryReflectionQROAM":
        # ``n_reflections`` is the number of synthesized vectors -- columns of U when
        # ``transpose=False``, rows of U (columns of U^dagger) when ``transpose=True``.
        n_cols = n_reflections if n_reflections is not None else n_rows
        return cls(
            block_unitaries=Shaped((n_blocks, n_rows, n_cols)),
            phase_bitsize=phase_bitsize,
            amp_log_block_sizes=amp_log_block_sizes,
            amp_adjoint_log_block_sizes=amp_adjoint_log_block_sizes,
            phase_log_block_sizes=phase_log_block_sizes,
            phase_adjoint_log_block_sizes=phase_adjoint_log_block_sizes,
            optimal_T=optimal_T,
            transpose=transpose,
        )

    @property
    def n_blocks(self) -> SymbolicInt:
        return shape(self.block_unitaries)[0]

    @property
    def n_rows(self) -> SymbolicInt:
        return shape(self.block_unitaries)[1]

    @property
    def n_reflections(self) -> SymbolicInt:
        return shape(self.block_unitaries)[2]

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            reflection_ancilla=1,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    def reflection(self, basis_index: int) -> BlockHouseholderReflectionQROAM:
        state_coefficients: Union[Shaped, NDArray[np.complex128]]
        if isinstance(self.block_unitaries, Shaped):
            state_coefficients = Shaped((self.n_blocks, self.n_rows))
        else:
            state_coefficients = self.block_unitaries[:, :, basis_index]
        return BlockHouseholderReflectionQROAM(
            state_coefficients=state_coefficients,
            phase_bitsize=self.phase_bitsize,
            basis_index=basis_index,
            amp_log_block_sizes=self.amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if isinstance(self.block_unitaries, Shaped):
            raise DecomposeTypeError(f"cannot decompose data-free {self}")
        # U = R_{n-1} ... R_0  -> forward order builds columns of U.
        # U^dagger = R_0 ... R_{n-1} (each R_i self-inverse) -> reversed order builds rows.
        order = range(int(self.n_reflections))
        if self.transpose:
            order = reversed(order)
        for basis_index in order:
            soqs = bb.add_d(self.reflection(basis_index), **soqs)
        return soqs

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if is_symbolic(self.n_reflections):
            ret[self.reflection(0)] += self.n_reflections
            return ret
        for basis_index in range(int(self.n_reflections)):
            ret[self.reflection(basis_index)] += 1
        return ret


def split_reflection_log_block_sizes(
    lam: float,
    n_blocks: int,
    n_rows: int,
) -> Tuple[int, int]:
    """Split a target QROAM block size ``lam`` across ``(block, row)`` dimensions.

    Returns ``(log_k_block, log_k_row)`` for tables of shape ``(n_blocks, n_rows)``.
    The selected powers of two have product as close as possible to ``lam`` while
    minimizing the batched QROAM table size ``ceil(n_blocks/k_block) * ceil(n_rows/k_row)``.
    """
    assert lam >= 1
    assert n_blocks >= 1
    assert n_rows >= 1
    best: Optional[Tuple[float, int, int, int, int]] = None
    for log_k_block in range(bit_length(n_blocks - 1) + 1):
        k_block = 2**log_k_block
        if k_block > n_blocks:
            continue
        for log_k_row in range(bit_length(n_rows - 1) + 1):
            k_row = 2**log_k_row
            if k_row > n_rows:
                continue
            product_gap = abs(k_block * k_row - lam)
            batched_size = ceil(n_blocks / k_block) * ceil(n_rows / k_row)
            balance_gap = abs(log_k_block - log_k_row)
            candidate = (product_gap, batched_size, balance_gap, log_k_block, log_k_row)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[3], best[4]


def optimal_reflection_log_block_sizes(
    n_blocks: int,
    n_rows: int,
    phase_bitsize: int,
) -> Tuple[int, int]:
    """Pick FORWARD ``log_block_sizes`` for ``BlockUnitaryReflectionQROAM``.

    Targets ``lam* ~ sqrt(n_blocks * n_rows / (2 * phase_bitsize))`` — the
    optimum for the largest (final) state-preparation staircase layer. Per-layer
    QROAM data-shape caps automatically reduce the row split on smaller layers,
    so each layer ends up close to its own optimum without varying the
    user-supplied parameter.
    """
    assert n_blocks >= 1
    assert n_rows >= 1
    assert phase_bitsize > 0
    lam = max(1.0, sqrt((n_blocks * n_rows) / (2 * phase_bitsize)))
    lam = 2 ** max(0, round(log2(lam)))
    return split_reflection_log_block_sizes(lam, n_blocks, n_rows)


def optimal_reflection_adjoint_log_block_sizes(
    n_blocks: int,
    n_rows: int,
) -> Tuple[int, int]:
    """Pick ADJOINT ``log_block_sizes`` for ``BlockUnitaryReflectionQROAM``.

    QROAMCleanAdjoint cost ~ ``N/lambda + lambda``, so the optimum is
    ``lam_adj* ~ sqrt(n_blocks * n_rows)`` (no ``phase_bitsize`` factor).
    """
    assert n_blocks >= 1
    assert n_rows >= 1
    lam = max(1.0, sqrt(n_blocks * n_rows))
    lam = 2 ** max(0, round(log2(lam)))
    return split_reflection_log_block_sizes(lam, n_blocks, n_rows)
