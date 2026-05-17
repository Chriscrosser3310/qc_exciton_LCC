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
import numpy as np

from qualtran import Bloq, BloqBuilder, GateWithRegisters, QUInt, Signature, SoquetT
from qualtran.bloqs.arithmetic.addition import AddK
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


def _data_max_log_block_sizes(data_shape: Tuple[SymbolicInt, ...]) -> Tuple[SymbolicInt, ...]:
    """Return per-dimension floor(log2(dim)) for use as QROAM log_block_sizes caps.

    QROAM requires 2**lbs <= data_shape[i], which is stricter than lbs <= selection_bitsize
    when data_shape[i] is not a power of two (e.g. n_blocks = 26 → block_bitsize = 5
    but floor(log2(26)) = 4).
    """
    result = []
    for d in data_shape:
        if is_symbolic(d):
            result.append(d)
        else:
            result.append(int(log2(max(1, int(d)))))
    return tuple(result)


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
        return _cap_log_block_sizes(self.log_block_sizes, _data_max_log_block_sizes(self.qroam_data_shape))

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        return QROAMClean.build_from_bitsize(
            self.qroam_data_shape,
            target_bitsizes=(self.phase_bitsize, self.phase_bitsize),
            selection_bitsizes=self.qroam_selection_bitsizes,
            log_block_sizes=self.qroam_log_block_sizes,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Circuit: QROAM(block, j) → α,β; H s; ctrl-s β→grad; H s; ctrl-s α→grad; QROAM†.

        The system register encodes y = 2j + s.  s (LSB) is the mode bit acted on by each
        beamsplitter; j (upper bits) is the pair index used to address the angle table.
        """
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")

        has_block = (self.n_blocks > 1)
        phase_grad = soqs['phase_gradient']

        # Split system register (big-endian: index 0 = MSB, index -1 = LSB = s)
        system_qubits = bb.split(soqs['system'])
        s = system_qubits[-1]
        pair = bb.join(system_qubits[:-1], dtype=QUInt(self.pair_bitsize))

        # QROAM load: (block, j) → α (target0_), β (target1_)
        # With one selection the register is named 'selection'; with two: 'selection0', 'selection1'.
        qroam = self.qroam_bloq_for_cost
        sel_names = [r.name for r in qroam.selection_registers]
        if has_block:
            qroam_out = bb.add_d(qroam, **{sel_names[0]: soqs['block'], sel_names[1]: pair})
            block = qroam_out[sel_names[0]]
            pair = qroam_out[sel_names[1]]
        else:
            qroam_out = bb.add_d(qroam, **{sel_names[0]: pair})
            pair = qroam_out[sel_names[0]]
        alpha = qroam_out['target0_']
        beta = qroam_out['target1_']

        # Apply R(α) H R(β) H on s: circuit order is H, ctrl-β, H, ctrl-α
        ctrl_add = AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()
        s = bb.add(Hadamard(), q=s)
        s, beta, phase_grad = bb.add(ctrl_add, ctrl=s, x=beta, phase_grad=phase_grad)
        s = bb.add(Hadamard(), q=s)
        s, alpha, phase_grad = bb.add(ctrl_add, ctrl=s, x=alpha, phase_grad=phase_grad)

        # Measurement-based uncompute (0 Toffoli, per arXiv:2409.11748 eq 84)
        qroam_adj = QROAMCleanAdjoint.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            target_shapes=(qroam.block_sizes,) * len(qroam.target_bitsizes),
        )
        block_sizes = qroam.block_sizes
        targets_map = {'target0_': alpha, 'target1_': beta}
        adj_sel_names = [r.name for r in qroam_adj.selection_registers]
        adj_soqs: Dict[str, SoquetT] = {}
        if has_block:
            adj_soqs[adj_sel_names[0]] = block
            adj_soqs[adj_sel_names[1]] = pair
        else:
            adj_soqs[adj_sel_names[0]] = pair
        for target, adj_target in zip(qroam.target_registers, qroam_adj.target_registers):
            junk_name = 'junk_' + target.name
            junk_arr = np.asarray(qroam_out[junk_name]) if junk_name in qroam_out else np.array([])
            adj_soqs[adj_target.name] = np.array([targets_map[target.name], *junk_arr]).reshape(block_sizes)
        adj_out = bb.add_d(qroam_adj, **adj_soqs)

        # Rejoin system: [pair_MSB ... pair_0, s]
        pair_out = adj_out[adj_sel_names[1]] if has_block else adj_out[adj_sel_names[0]]
        system_out = bb.join(np.concatenate([bb.split(pair_out), [s]]), dtype=QUInt(self.system_bitsize))

        out: Dict[str, SoquetT] = {'system': system_out, 'phase_gradient': phase_grad}
        if has_block:
            out['block'] = adj_out[adj_sel_names[0]]
        return out

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        ret[AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()] += 2
        ret[Hadamard()] += 2
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
        return _cap_log_block_sizes(self.log_block_sizes, _data_max_log_block_sizes(self.qroam_data_shape))

    @property
    def qroam_adjoint_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _cap_log_block_sizes(self.adjoint_log_block_sizes, _data_max_log_block_sizes(self.qroam_data_shape))

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
        """Circuit: QROAM(block, y) → φ; AddIntoPhaseGrad(φ, grad); QROAM†."""
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")

        has_block = (self.n_blocks > 1)
        phase_grad = soqs['phase_gradient']
        system = soqs['system']

        # QROAM load: (block, system) → φ (target0_)
        # With one selection the register is named 'selection'; with two: 'selection0', 'selection1'.
        qroam = self.qroam_bloq_for_cost
        sel_names = [r.name for r in qroam.selection_registers]
        if has_block:
            qroam_out = bb.add_d(qroam, **{sel_names[0]: soqs['block'], sel_names[1]: system})
            block = qroam_out[sel_names[0]]
            system = qroam_out[sel_names[1]]
        else:
            qroam_out = bb.add_d(qroam, **{sel_names[0]: system})
            system = qroam_out[sel_names[0]]
        phi = qroam_out['target0_']

        # Unconditional phase kick: grad += φ
        phi, phase_grad = bb.add(
            AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize), x=phi, phase_grad=phase_grad
        )

        # Adjoint uncompute with dedicated lambda parameter
        qroam_adj = self.qroam_adj_bloq_for_cost
        adj_sel_names = [r.name for r in qroam_adj.selection_registers]
        junk_name = 'junk_target0_'
        junk_arr = np.asarray(qroam_out[junk_name]) if junk_name in qroam_out else np.array([])
        adj_target = next(iter(qroam_adj.target_registers))
        adj_soqs: Dict[str, SoquetT] = {
            adj_target.name: np.array([phi, *junk_arr]).reshape(qroam_adj.target_shapes[0])
        }
        if has_block:
            adj_soqs[adj_sel_names[0]] = block
            adj_soqs[adj_sel_names[1]] = system
        else:
            adj_soqs[adj_sel_names[0]] = system
        adj_out = bb.add_d(qroam_adj, **adj_soqs)

        out: Dict[str, SoquetT] = {
            'system': adj_out[adj_sel_names[1]] if has_block else adj_out[adj_sel_names[0]],
            'phase_gradient': phase_grad,
        }
        if has_block:
            out['block'] = adj_out[adj_sel_names[0]]
        return out

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

    @property
    def signature(self) -> Signature:
        return Signature.build(system=self.system_bitsize)

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
        final_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        final_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    ) -> "BlockUnitaryInterferometerSynthesisQROAM":
        return cls(
            n_blocks=n_blocks,
            n_rows=n_rows,
            phase_bitsize=phase_bitsize,
            n_layers=n_layers,
            log_block_sizes=log_block_sizes,
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
        r"""Full interferometer: N alternating beamsplitter layers plus a final diagonal phase.

        Even layers (ell % 2 == 0) apply beamsplitters directly.  Odd layers shift the
        system register by +1 (cyclic increment mod N) before the beamsplitter layer and
        shift back afterwards, effectively addressing the offset pair-index grouping.
        """
        if is_symbolic(self.n_blocks, self.n_rows, self.phase_bitsize, self.layer_count):
            raise NotImplementedError("build_composite_bloq requires concrete parameters")

        has_block = (self.n_blocks > 1)
        system = soqs['system']
        phase_grad = soqs['phase_gradient']
        block = soqs.get('block')

        inc = AddK(dtype=QUInt(self.system_bitsize), k=1)
        dec = AddK(dtype=QUInt(self.system_bitsize), k=-1)

        for ell in range(int(self.layer_count)):
            if ell % 2 == 1 and self.include_shift_cost:
                system = bb.add(inc, x=system)
            layer_soqs: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
            if has_block:
                layer_soqs['block'] = block
            out = bb.add_d(self.phase_layer, **layer_soqs)
            system = out['system']
            phase_grad = out['phase_gradient']
            if has_block:
                block = out['block']
            if ell % 2 == 1 and self.include_shift_cost:
                system = bb.add(dec, x=system)

        if self.include_final_phases:
            final_soqs: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
            if has_block:
                final_soqs['block'] = block
            out = bb.add_d(self.final_phase_layer, **final_soqs)
            system = out['system']
            phase_grad = out['phase_gradient']
            if has_block:
                block = out['block']

        result: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
        if has_block:
            result['block'] = block
        return result

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.phase_layer] += self.layer_count
        if self.include_shift_cost:
            if is_symbolic(self.layer_count):
                ret[TargetOnlyCyclicShiftCost(self.system_bitsize)] += 1
            else:
                n_odd = int(self.layer_count) // 2
                if n_odd > 0:
                    ret[AddK(dtype=QUInt(self.system_bitsize), k=1)] += n_odd
                    ret[AddK(dtype=QUInt(self.system_bitsize), k=-1)] += n_odd
        if self.include_final_phases:
            ret[self.final_phase_layer] += 1
        return ret


@dataclass(frozen=True)
class InterferometerResourceEstimate:
    """Analytic resource estimate for one choice of QROAM tradeoff parameters."""

    toffoli: int
    qubits: int
    layer_load_log_block_size: int
    final_load_log_block_size: int
    final_adjoint_log_block_size: int
    layer_load_lambda: int
    final_load_lambda: int
    final_adjoint_lambda: int


def estimate_interferometer_resources(
    n_blocks: int,
    n_rows: int,
    phase_bitsize: int,
    *,
    layer_log_block_size: int,
    final_log_block_size: int,
    final_adjoint_log_block_size: Optional[int] = None,
    n_layers: Optional[int] = None,
) -> InterferometerResourceEstimate:
    """Return the note's Toffoli/qubit estimate for fixed QROAM parameters.

    The per-layer QROAM uncompute uses measurement-based erasure (0 Toffoli),
    per arXiv:2409.11748 eq (84).  Only the final phase layer charges for its
    adjoint (via final_adjoint_log_block_size).
    """

    assert n_blocks >= 1
    assert _positive_power_of_two(n_rows)
    assert phase_bitsize > 1
    if final_adjoint_log_block_size is None:
        final_adjoint_log_block_size = final_log_block_size
    n = int(log2(n_rows))
    layers = n_rows if n_layers is None else n_layers
    layer_load_lambda = 2**layer_log_block_size
    final_load_lambda = 2**final_log_block_size
    final_adjoint_lambda = 2**final_adjoint_log_block_size
    layer_entries = n_blocks * n_rows // 2
    final_entries = n_blocks * n_rows

    layer_t = (
        ceil(layer_entries / layer_load_lambda)
        + 2 * layer_load_lambda * phase_bitsize
        - 5
    )
    final_t = (
        ceil(final_entries / final_load_lambda)
        + final_load_lambda * phase_bitsize
        + ceil(final_entries / final_adjoint_lambda)
        + final_adjoint_lambda
        - 6
    )
    shift_t = max(0, n - 2) * (n_rows - 1)
    toffoli = layers * layer_t + final_t + shift_t

    base_qubits = bit_length(n_blocks - 1) + n + phase_bitsize
    qubits = base_qubits + max(
        2 * phase_bitsize * max(1, layer_load_lambda),
        phase_bitsize * max(1, final_load_lambda),
        max(1, final_adjoint_lambda),
    )

    return InterferometerResourceEstimate(
        toffoli=int(toffoli),
        qubits=int(qubits),
        layer_load_log_block_size=layer_log_block_size,
        final_load_log_block_size=final_log_block_size,
        final_adjoint_log_block_size=final_adjoint_log_block_size,
        layer_load_lambda=layer_load_lambda,
        final_load_lambda=final_load_lambda,
        final_adjoint_lambda=final_adjoint_lambda,
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
