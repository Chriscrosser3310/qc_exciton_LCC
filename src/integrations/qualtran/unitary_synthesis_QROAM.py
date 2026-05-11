"""Block-encoded unitary synthesis via QROAM-backed state-preparation reflections.

This follows Sec. 4 of Low, Kliuchnikov, and Schaeffer,
`Trading T-gates for dirty qubits in state preparation and unitary synthesis`
(arXiv:1812.00954v2).
"""

from collections import Counter
from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, DecomposeTypeError, GateWithRegisters, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZGate
from qualtran.bloqs.mcmt import MultiControlZ
from qualtran.symbolics import bit_length, HasLength, is_symbolic, Shaped, slen, SymbolicInt

try:
    from .state_prep_QROAM import StatePreparationViaQROAMRotations
except ImportError:
    from state_prep_QROAM import StatePreparationViaQROAMRotations

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _to_tuple_or_none(
    x: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]],
) -> Optional[Tuple[SymbolicInt, ...]]:
    if x is None:
        return None
    if isinstance(x, (int, float, sympy.Basic)):
        return (x,)
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    return tuple(x)


def _to_tuple_or_has_length(x: Union[HasLength, Iterable[complex]]) -> Union[HasLength, Tuple[complex, ...]]:
    if isinstance(x, HasLength):
        return x
    return tuple(complex(v) for v in x)


def _to_complex_matrix_or_shape(
    x: Union[Shaped, Iterable[Iterable[complex]]]
) -> Union[Shaped, NDArray[np.complex128]]:
    if isinstance(x, Shaped):
        return x
    return np.asarray(x, dtype=np.complex128)


@attrs.frozen
class PrepareHouseholderStateQROAM(GateWithRegisters):
    r"""Prepare the Householder state used in Sec. 4.

    For the kth column vector `state_coefficients = u_k`, this prepares, up to a global phase,

    $$
        |w_k\rangle = \frac{|1\rangle |k\rangle - |0\rangle |u_k\rangle}{\sqrt{2}}.
    $$

    `phase_gradient` is a catalyst required by `StatePreparationViaQROAMRotations`; it is returned
    unchanged and is not part of the reflected subspace.
    """

    state_coefficients: Union[HasLength, Tuple[complex, ...]] = attrs.field(
        converter=_to_tuple_or_has_length
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
        n_coeff = slen(self.state_coefficients)
        if not is_symbolic(n_coeff):
            assert n_coeff == 2**self.system_bitsize
            assert 0 <= self.basis_index < n_coeff
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if isinstance(self.state_coefficients, tuple):
            assert np.isclose(np.linalg.norm(self.state_coefficients), 1)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(slen(self.state_coefficients) - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            reflection_ancilla=1,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def state_prep(self) -> StatePreparationViaQROAMRotations:
        return StatePreparationViaQROAMRotations(
            state_coefficients=self.state_coefficients,
            phase_bitsize=self.phase_bitsize,
            control_bitsize=1,
            uncompute=self.uncompute,
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
        )

    def adjoint(self) -> 'PrepareHouseholderStateQROAM':
        return attrs.evolve(self, uncompute=not self.uncompute)

    def _basis_one_positions(self) -> Tuple[int, ...]:
        if is_symbolic(self.system_bitsize):
            return ()
        return tuple(
            qi for qi in range(int(self.system_bitsize)) if (self.basis_index >> qi) & 1
        )

    def _apply_basis_cnot_ladder(
        self, bb: BloqBuilder, reflection_ancilla: Soquet, system_qubits: NDArray
    ) -> Tuple[Soquet, NDArray]:
        for qi in self._basis_one_positions():
            reflection_ancilla, system_qubits[qi] = bb.add(
                CNOT(), ctrl=reflection_ancilla, target=system_qubits[qi]
            )
        return reflection_ancilla, system_qubits

    def _apply_controlled_state_prep(
        self, bb: BloqBuilder, reflection_ancilla: Soquet, system: Soquet, phase_gradient: Soquet
    ) -> Tuple[Soquet, Soquet, Soquet]:
        reflection_ancilla = bb.add(XGate(), q=reflection_ancilla)
        reflection_ancilla, system, phase_gradient = bb.add(
            self.state_prep,
            prepare_control=reflection_ancilla,
            target_state=system,
            phase_gradient=phase_gradient,
        )
        reflection_ancilla = bb.add(XGate(), q=reflection_ancilla)
        return reflection_ancilla, system, phase_gradient

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        reflection_ancilla = soqs.pop("reflection_ancilla")
        system = soqs.pop("system")
        phase_gradient = soqs.pop("phase_gradient")

        system_qubits = bb.split(system)
        if self.uncompute:
            reflection_ancilla, system, phase_gradient = self._apply_controlled_state_prep(
                bb, reflection_ancilla, bb.join(system_qubits), phase_gradient
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
            reflection_ancilla, system, phase_gradient = self._apply_controlled_state_prep(
                bb, reflection_ancilla, bb.join(system_qubits), phase_gradient
            )
            system_qubits = bb.split(system)

        soqs["reflection_ancilla"] = reflection_ancilla
        soqs["system"] = bb.join(system_qubits)
        soqs["phase_gradient"] = phase_gradient
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        ret[Hadamard()] += 1
        ret[ZGate()] += 1
        ret[CNOT()] += len(self._basis_one_positions())
        ret[XGate()] += 2
        ret[self.state_prep] += 1
        return ret


@attrs.frozen
class HouseholderReflectionQROAM(GateWithRegisters):
    r"""Reflect about the Sec. 4 Householder state prepared with QROAM rotations."""

    state_coefficients: Union[HasLength, Tuple[complex, ...]] = attrs.field(
        converter=_to_tuple_or_has_length
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
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(slen(self.state_coefficients) - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            reflection_ancilla=1,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def prepare_w(self) -> PrepareHouseholderStateQROAM:
        return PrepareHouseholderStateQROAM(
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n_reflection_qubits = 1 + self.system_bitsize
        ret: 'Counter[Bloq]' = Counter()
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
class UnitarySynthesisQROAM(GateWithRegisters):
    r"""Synthesize the Sec. 4 block-encoding of an isometry/unitary.

    The input `unitary` is interpreted in the conventional matrix layout: each column is one
    target output state $|u_k\rangle$. For an $N \times K$ isometry, this applies the product of
    `K` Householder reflections

    $$
        I - 2 |w_k\rangle\langle w_k|,\qquad
        |w_k\rangle = (|1\rangle|k\rangle - |0\rangle|u_k\rangle)/\sqrt{2}.
    $$

    For a full $N \times N$ unitary `U`, the product implements the paper's

    $$
        W = |0\rangle\langle 1| \otimes U + |1\rangle\langle 0| \otimes U^\dagger.
    $$

    Thus initializing `reflection_ancilla` in $|1\rangle$ maps
    $|1\rangle|\psi\rangle \mapsto |0\rangle U|\psi\rangle$ for the synthesized columns.
    """

    unitary: Union[Shaped, NDArray[np.complex128]] = attrs.field(
        converter=_to_complex_matrix_or_shape, eq=False
    )
    phase_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if isinstance(self.unitary, np.ndarray):
            assert self.unitary.ndim == 2
        if isinstance(self.unitary, Shaped):
            n_rows, n_cols = self.unitary.shape
            if not is_symbolic(n_rows):
                assert n_rows == 2**bit_length(n_rows - 1)
            if not is_symbolic(n_rows, n_cols):
                assert n_cols <= n_rows
            return
        n_rows, n_cols = self.unitary.shape
        assert n_rows == 2**self.system_bitsize
        assert n_cols <= n_rows
        gram = self.unitary.conj().T @ self.unitary
        assert np.allclose(gram, np.eye(n_cols), atol=1e-8)

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.unitary.shape[0] - 1)

    @property
    def n_reflections(self) -> SymbolicInt:
        return self.unitary.shape[1]

    @classmethod
    def from_shape(
        cls,
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, SymbolicInt]],
        phase_bitsize: SymbolicInt,
        *,
        n_reflections: Optional[SymbolicInt] = None,
        log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    ) -> 'UnitarySynthesisQROAM':
        """Build a dense, data-free unitary/isometry synthesis bloq for resource estimates.

        Args:
            data_len_or_shape: Either `N` for a square `N x N` unitary, or `(N, K)` for an
                `N x K` isometry with `K` specified columns.
            phase_bitsize: Bitsize for state-preparation rotation tables.
            n_reflections: Optional `K` when `data_len_or_shape` is given as `N`.
        """
        if isinstance(data_len_or_shape, tuple):
            n_rows, n_cols = data_len_or_shape
            assert n_reflections is None
        else:
            n_rows = data_len_or_shape
            n_cols = n_reflections if n_reflections is not None else n_rows
        return cls(
            unitary=Shaped((n_rows, n_cols)),
            phase_bitsize=phase_bitsize,
            log_block_sizes=log_block_sizes,
            adjoint_log_block_sizes=adjoint_log_block_sizes,
        )

    @property
    def signature(self) -> Signature:
        return Signature.build(
            reflection_ancilla=1,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    def reflection(self, basis_index: int) -> HouseholderReflectionQROAM:
        state_coefficients: Union[HasLength, Tuple[complex, ...]]
        if isinstance(self.unitary, Shaped):
            state_coefficients = HasLength(self.unitary.shape[0])
        else:
            state_coefficients = tuple(self.unitary[:, basis_index])
        return HouseholderReflectionQROAM(
            state_coefficients=state_coefficients,
            phase_bitsize=self.phase_bitsize,
            basis_index=basis_index,
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if isinstance(self.unitary, Shaped):
            raise DecomposeTypeError(f"cannot decompose data-free {self}")
        for basis_index in range(self.n_reflections):
            soqs = bb.add_d(self.reflection(basis_index), **soqs)
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        if is_symbolic(self.n_reflections):
            ret[self.reflection(0)] += self.n_reflections
            return ret
        for basis_index in range(self.n_reflections):
            ret[self.reflection(basis_index)] += 1
        return ret
