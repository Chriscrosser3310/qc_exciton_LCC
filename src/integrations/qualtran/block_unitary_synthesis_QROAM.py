"""Block-diagonal unitary synthesis via block-indexed QROAM state preparation."""

from collections import Counter
from typing import cast, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, DecomposeTypeError, GateWithRegisters, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZGate
from qualtran.bloqs.mcmt import MultiControlZ
from qualtran.symbolics import bit_length, HasLength, is_symbolic, Shaped, shape, SymbolicInt

try:
    from .block_state_preparation_QROAM import BlockStatePreparationViaQROAMRotations
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:
    from block_state_preparation_QROAM import BlockStatePreparationViaQROAMRotations
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
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
        eq=False,
    )
    phase_bitsize: SymbolicInt
    basis_index: int
    uncompute: bool = False
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

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
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
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
        block: Soquet,
        reflection_ancilla: Soquet,
        system: Soquet,
        phase_gradient: Soquet,
    ) -> Tuple[Soquet, Soquet, Soquet, Soquet]:
        reflection_ancilla = bb.add(XGate(), q=reflection_ancilla)
        out_soqs = bb.add_d(
            self.state_prep,
            block=block,
            prepare_control=reflection_ancilla,
            target_state=system,
            phase_gradient=phase_gradient,
        )
        block = cast(Soquet, out_soqs["block"])
        reflection_ancilla = cast(Soquet, out_soqs["prepare_control"])
        system = cast(Soquet, out_soqs["target_state"])
        phase_gradient = cast(Soquet, out_soqs["phase_gradient"])
        reflection_ancilla = bb.add(XGate(), q=reflection_ancilla)
        return block, reflection_ancilla, system, phase_gradient

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        block = soqs.pop("block")
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
        eq=False,
    )
    phase_bitsize: SymbolicInt
    basis_index: int
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

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
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
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


@attrs.frozen
class BlockUnitarySynthesisQROAM(GateWithRegisters):
    r"""Synthesize block-diagonal unitary data ``sum_j |j><j| tensor U_j``.

    ``block_unitaries`` has shape ``(n_blocks, N, n_reflections)``.  For each block ``j`` the
    first ``n_reflections`` columns of ``U_j`` are synthesized using the Sec. 4 reflection
    construction, while the block register is left unchanged.
    """

    block_unitaries: Union[Shaped, NDArray[np.complex128]] = attrs.field(
        converter=_to_block_unitaries_or_shape, eq=False
    )
    phase_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

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

    @classmethod
    def from_shape(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_reflections: Optional[SymbolicInt] = None,
        log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    ) -> "BlockUnitarySynthesisQROAM":
        n_cols = n_reflections if n_reflections is not None else n_rows
        return cls(
            block_unitaries=Shaped((n_blocks, n_rows, n_cols)),
            phase_bitsize=phase_bitsize,
            log_block_sizes=log_block_sizes,
            adjoint_log_block_sizes=adjoint_log_block_sizes,
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
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if isinstance(self.block_unitaries, Shaped):
            raise DecomposeTypeError(f"cannot decompose data-free {self}")
        for basis_index in range(int(self.n_reflections)):
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
