"""Block-aware interferometer unitary synthesis with QROAM-loaded phases.

This implements the resource model from
``notes/block_unitary_interferometer_synthesis_self_contained_with_givens.pdf``.
The circuit uses one rectangular nearest-neighbor beamsplitter skeleton for every
block and loads only the block-dependent phase angles from QROAM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, log2, sqrt
from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union

import attrs

from qualtran import Bloq, BloqBuilder, GateWithRegisters, Signature, SoquetT
from qualtran.bloqs.basic_gates import Hadamard, Toffoli
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, Shaped, SymbolicInt

try:
    from .state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none
except ImportError:
    from state_prep_QROAM import _cap_log_block_sizes, _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _positive_power_of_two(n: SymbolicInt) -> bool:
    return isinstance(n, int) and n > 0 and n == 2 ** bit_length(n - 1)


@attrs.frozen
class BlockInterferometerPhaseLayerQROAM(GateWithRegisters):
    r"""One multiplexed beamsplitter layer.

    For target index ``y = 2j + s`` this applies

    ``R(alpha(block, j)) H R(beta(block, j)) H``

    to the active target bit ``s``. The concrete angle data is intentionally
    represented only by shape for resource estimation.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        if not is_symbolic(self.n_rows):
            assert _positive_power_of_two(self.n_rows)
            assert self.n_rows >= 2
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1

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
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.pair_bitsize,)
        return (self.block_bitsize, self.pair_bitsize)

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.n_rows // 2,)
        return (self.n_blocks, self.n_rows // 2)

    @property
    def qroam_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _cap_log_block_sizes(self.log_block_sizes, self.qroam_selection_bitsizes)

    @property
    def qroam_adjoint_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _cap_log_block_sizes(self.adjoint_log_block_sizes, self.qroam_selection_bitsizes)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize, self.phase_bitsize),
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=self.qroam_log_block_sizes,
        )

    @property
    def qroam_adj_bloq_for_cost(self) -> QROAMCleanAdjoint:
        qroam = self.qroam_bloq_for_cost
        return QROAMCleanAdjoint.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            target_shapes=(qroam.block_sizes,) * len(qroam.target_bitsizes),
            log_block_sizes=self.qroam_adjoint_log_block_sizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        raise NotImplementedError("data-free interferometer layers are for resource estimates")

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        ret[AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)] += 2
        ret[Hadamard()] += 2
        ret[self.qroam_adj_bloq_for_cost] += 1
        return ret


@attrs.frozen
class BlockInterferometerFinalPhasesQROAM(GateWithRegisters):
    """Final block-dependent diagonal phase layer addressed by ``(block, target)``."""

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
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
        return Signature.build(
            block=self.block_bitsize,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.system_bitsize,)
        return (self.block_bitsize, self.system_bitsize)

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            return (self.n_rows,)
        return (self.n_blocks, self.n_rows)

    @property
    def qroam_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _cap_log_block_sizes(self.log_block_sizes, self.qroam_selection_bitsizes)

    @property
    def qroam_adjoint_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _cap_log_block_sizes(self.adjoint_log_block_sizes, self.qroam_selection_bitsizes)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize,),
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=self.qroam_log_block_sizes,
        )

    @property
    def qroam_adj_bloq_for_cost(self) -> QROAMCleanAdjoint:
        qroam = self.qroam_bloq_for_cost
        return QROAMCleanAdjoint.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            target_shapes=(qroam.block_sizes,),
            log_block_sizes=self.qroam_adjoint_log_block_sizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        raise NotImplementedError("data-free final phase layer is for resource estimates")

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        ret[AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)] += 1
        ret[self.qroam_adj_bloq_for_cost] += 1
        return ret


@attrs.frozen
class TargetOnlyCyclicShiftCost(Bloq):
    """Toffoli model for target-only cyclic shifts used to realize odd layers."""

    system_bitsize: SymbolicInt

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if is_symbolic(self.system_bitsize):
            return ret
        n = int(self.system_bitsize)
        ret[Toffoli()] += max(0, n - 2) * (2**n - 1)
        return ret


@attrs.frozen
class BlockUnitaryInterferometerSynthesisQROAM(GateWithRegisters):
    r"""Block-diagonal dense unitary synthesis via multiplexed interferometer layers.

    The synthesized unitary has the form ``sum_x |x><x| tensor U_x``. The block
    register is an address for phase QROAM only; beamsplitters and shifts act on
    the target register.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_layers: Optional[SymbolicInt] = None
    include_final_phases: bool = True
    include_shift_cost: bool = True
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    @classmethod
    def from_shape(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_layers: Optional[SymbolicInt] = None,
        log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        final_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        final_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    ) -> "BlockUnitaryInterferometerSynthesisQROAM":
        return cls(
            n_blocks=n_blocks,
            n_rows=n_rows,
            phase_bitsize=phase_bitsize,
            n_layers=n_layers,
            log_block_sizes=log_block_sizes,
            adjoint_log_block_sizes=adjoint_log_block_sizes,
            final_log_block_sizes=final_log_block_sizes,
            final_adjoint_log_block_sizes=final_adjoint_log_block_sizes,
        )

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def layer_count(self) -> SymbolicInt:
        return self.n_rows if self.n_layers is None else self.n_layers

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def phase_layer(self) -> BlockInterferometerPhaseLayerQROAM:
        return BlockInterferometerPhaseLayerQROAM(
            n_blocks=self.n_blocks,
            n_rows=self.n_rows,
            phase_bitsize=self.phase_bitsize,
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
        )

    @property
    def final_phase_layer(self) -> BlockInterferometerFinalPhasesQROAM:
        return BlockInterferometerFinalPhasesQROAM(
            n_blocks=self.n_blocks,
            n_rows=self.n_rows,
            phase_bitsize=self.phase_bitsize,
            log_block_sizes=self.final_log_block_sizes,
            adjoint_log_block_sizes=self.final_adjoint_log_block_sizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if isinstance(self.n_blocks, Shaped):
            raise NotImplementedError
        raise NotImplementedError("data-free interferometer synthesis is for resource estimates")

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.phase_layer] += self.layer_count
        if self.include_shift_cost:
            ret[TargetOnlyCyclicShiftCost(self.system_bitsize)] += 1
        if self.include_final_phases:
            ret[self.final_phase_layer] += 1
        return ret


@dataclass(frozen=True)
class InterferometerResourceEstimate:
    """Analytic resource estimate for one choice of QROAM tradeoff parameters."""

    toffoli: int
    qubits: int
    layer_log_block_size: int
    final_log_block_size: int
    layer_lambda: int
    final_lambda: int


def estimate_interferometer_resources(
    n_blocks: int,
    n_rows: int,
    phase_bitsize: int,
    *,
    layer_log_block_size: int,
    final_log_block_size: int,
    n_layers: Optional[int] = None,
) -> InterferometerResourceEstimate:
    """Return the note's Toffoli/qubit estimate for fixed QROAM parameters."""

    assert n_blocks >= 1
    assert _positive_power_of_two(n_rows)
    assert phase_bitsize > 1
    n = int(log2(n_rows))
    layers = n_rows if n_layers is None else n_layers
    layer_lambda = 2**layer_log_block_size
    final_lambda = 2**final_log_block_size
    layer_entries = n_blocks * n_rows // 2
    final_entries = n_blocks * n_rows

    layer_t = ceil(layer_entries / layer_lambda) + 2 * layer_lambda * phase_bitsize - 5
    final_t = (
        ceil(final_entries / final_lambda)
        + final_lambda * phase_bitsize
        + ceil(final_entries / final_lambda)
        + final_lambda
        - 6
    )
    shift_t = max(0, n - 2) * (n_rows - 1)
    toffoli = layers * layer_t + final_t + shift_t

    base_qubits = bit_length(n_blocks - 1) + n + phase_bitsize
    layer_workspace = 2 * phase_bitsize * max(1, layer_lambda)
    final_workspace = phase_bitsize * max(1, final_lambda)
    qubits = base_qubits + max(layer_workspace, final_workspace)

    return InterferometerResourceEstimate(
        toffoli=int(toffoli),
        qubits=int(qubits),
        layer_log_block_size=layer_log_block_size,
        final_log_block_size=final_log_block_size,
        layer_lambda=layer_lambda,
        final_lambda=final_lambda,
    )


def optimal_interferometer_log_block_sizes(
    n_blocks: int,
    n_rows: int,
    phase_bitsize: int,
) -> Tuple[int, int]:
    """Continuous optimum rounded to nearby powers of two for layer/final QROAM."""

    layer_entries = n_blocks * n_rows // 2
    final_entries = n_blocks * n_rows
    layer_lam = max(1.0, sqrt(layer_entries / (2 * phase_bitsize)))
    final_lam = max(1.0, sqrt(2 * final_entries / (phase_bitsize + 1)))
    return max(0, round(log2(layer_lam))), max(0, round(log2(final_lam)))
