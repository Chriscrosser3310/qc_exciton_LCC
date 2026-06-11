"""Block-controlled state preparation via QROAM-backed rotations.

This is a minimal block-indexed variant of ``state_prep_QROAM``.  The extra ``block`` register is
threaded into every QROAM lookup, so rotation tables are queried as

    |k>|x>|0> -> |k>|x>|f_k(x)>.

The block register is otherwise unchanged.
"""

from collections import Counter
from math import log2, sqrt
from typing import cast, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, DecomposeTypeError, GateWithRegisters, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.basic_gates.rotation import Rx
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, Shaped, shape, slen, SymbolicInt

try:
    from .state_prep_QROAM import (
        RotationTree,
        _cap_log_block_sizes,
        _measure_x_reset,
        _to_tuple_or_none,
    )
except ImportError:
    from state_prep_QROAM import (
        RotationTree,
        _cap_log_block_sizes,
        _measure_x_reset,
        _to_tuple_or_none,
    )

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _per_layer_log_block_sizes(
    n_blocks: int, layer_n_rows: int, phase_bitsize: int, log_k_block: int
) -> Tuple[int, int]:
    """Per-staircase-layer (log_k_block, log_k_row) split for the FORWARD QROAM.

    Keeps ``log_k_block`` fixed at the supplied global value and computes
    ``log_k_row`` so that the resulting lambda matches the per-layer target
    ``lambda* ~ sqrt(2 * n_blocks * layer_n_rows / phase_bitsize)``.  The row
    component is clipped to ``[0, log2(layer_n_rows)]`` so it remains a valid
    QROAM batch size.
    """
    assert n_blocks >= 1 and phase_bitsize > 0 and layer_n_rows >= 1
    lam = max(1.0, sqrt(2.0 * n_blocks * layer_n_rows / phase_bitsize))
    log_lam = max(0, round(log2(lam)))
    log_k_row = max(0, log_lam - int(log_k_block))
    sel = bit_length(layer_n_rows - 1) if layer_n_rows > 1 else 0
    log_k_row = min(log_k_row, sel)
    return (int(log_k_block), int(log_k_row))


def _per_layer_adjoint_log_block_sizes(
    n_blocks: int, layer_n_rows: int, log_k_block: int
) -> Tuple[int, int]:
    """Per-staircase-layer (log_k_block, log_k_row) split for the ADJOINT QROAM.

    QROAMCleanAdjoint cost ~ N/lambda + lambda, so the optimum is
    ``lambda_adj* ~ sqrt(n_blocks * layer_n_rows)`` (no ``phase_bitsize`` factor).
    Same convention as the forward helper: ``log_k_block`` fixed, ``log_k_row``
    derived and clipped to the layer's row capacity.
    """
    assert n_blocks >= 1 and layer_n_rows >= 1
    lam = max(1.0, sqrt(n_blocks * layer_n_rows))
    log_lam = max(0, round(log2(lam)))
    log_k_row = max(0, log_lam - int(log_k_block))
    sel = bit_length(layer_n_rows - 1) if layer_n_rows > 1 else 0
    log_k_row = min(log_k_row, sel)
    return (int(log_k_block), int(log_k_row))


def _to_block_array_or_shape(x: Union[Shaped, Iterable[Iterable[complex]]]) -> Union[Shaped, NDArray[np.complex128]]:
    if isinstance(x, Shaped):
        return x
    return np.asarray(x, dtype=np.complex128)


def _coeff_shape_key(x: Union[Shaped, NDArray[np.complex128]]):
    """Hashable equality key for ``state_coefficients``: its (block, coeff) SHAPE.

    Raw amplitudes are excluded from equality (NDArrays are unhashable, and two tables of
    the same shape have identical resource costs), but the shape MUST be included so that
    differently-shaped data-free bloqs -- e.g. an ``(K, 2)`` flag prep vs a ``(K*N, N)``
    row prep -- are not collapsed into one entry by a ``build_call_graph`` ``Counter``.
    """
    return tuple(shape(x))


def _to_int_table_or_shape(x: Union[Shaped, NDArray[np.int_], Iterable[Iterable[int]]]):
    if isinstance(x, Shaped):
        return x
    return tuple(tuple(int(v) for v in row) for row in np.asarray(x, dtype=int))


@attrs.frozen
class BlockPRGAViaPhaseGradientQROAM(Bloq):
    """Controlled rotation array whose QROAM data is selected by ``block`` and ``selection``."""

    block_bitsize: SymbolicInt
    selection_bitsize: SymbolicInt
    phase_bitsize: SymbolicInt
    rom_values: Union[Shaped, Tuple[Tuple[int, ...], ...]] = attrs.field(
        converter=_to_int_table_or_shape
    )
    control_bitsize: SymbolicInt
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    measure_reset: bool = False

    @property
    def qroam_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            log_block_sizes = None if self.log_block_sizes is None else self.log_block_sizes[-1:]
            return _cap_log_block_sizes(log_block_sizes, (self.selection_bitsize,))
        return _cap_log_block_sizes(
            self.log_block_sizes, (self.block_bitsize, self.selection_bitsize)
        )

    @property
    def qroam_adjoint_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            log_block_sizes = (
                None if self.adjoint_log_block_sizes is None else self.adjoint_log_block_sizes[-1:]
            )
            return _cap_log_block_sizes(log_block_sizes, (self.selection_bitsize,))
        return _cap_log_block_sizes(
            self.adjoint_log_block_sizes, (self.block_bitsize, self.selection_bitsize)
        )

    @property
    def n_blocks(self) -> SymbolicInt:
        if isinstance(self.rom_values, Shaped):
            return self.rom_values.shape[0]
        return len(self.rom_values)

    @property
    def n_values(self) -> SymbolicInt:
        if isinstance(self.rom_values, Shaped):
            return self.rom_values.shape[1]
        return len(self.rom_values[0])

    @property
    def qroam_selection_sources(self) -> Tuple[str, ...]:
        sources: List[str] = []
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            if is_symbolic(self.n_values) or self.n_values > 1:
                sources.append("selection")
            return tuple(sources)
        for dim, dim_len in enumerate(self.qroam_bloq.data_shape):
            if is_symbolic(dim_len) or dim_len > 1:
                sources.append("block" if dim == 0 else "selection")
        return tuple(sources)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            control=self.control_bitsize,
            block=self.block_bitsize,
            selection=self.selection_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def qroam_bloq(self) -> QROAMClean:
        if not is_symbolic(self.n_blocks) and self.n_blocks == 1:
            rom_values = (
                Shaped((self.n_values,))
                if isinstance(self.rom_values, Shaped)
                else self.rom_values[0]
            )
            return QROAMClean(
                (rom_values,),
                selection_bitsizes=(self.selection_bitsize,),
                target_bitsizes=(self.phase_bitsize,),
                target_shapes=((),),
                num_controls=self.control_bitsize,
            ).with_log_block_sizes(self.qroam_log_block_sizes)
        return QROAMClean(
            (self.rom_values,),
            selection_bitsizes=(self.block_bitsize, self.selection_bitsize),
            target_bitsizes=(self.phase_bitsize,),
            target_shapes=((),),
            num_controls=self.control_bitsize,
        ).with_log_block_sizes(self.qroam_log_block_sizes)

    @property
    def qroam_bloq_for_cost(self) -> QROAMClean:
        qroam = self.qroam_bloq
        if not qroam.has_data():
            return qroam
        return QROAMClean.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            selection_bitsizes=qroam.selection_bitsizes,
            num_controls=qroam.num_controls,
            log_block_sizes=qroam.log_block_sizes,
        )

    @property
    def qroam_adj_bloq(self) -> QROAMCleanAdjoint:
        qroam = self.qroam_bloq
        if qroam.has_data():
            return QROAMCleanAdjoint.build_from_data(
                *qroam.batched_data_permuted,
                target_bitsizes=qroam.target_bitsizes,
                target_shapes=(qroam.block_sizes,) * len(qroam.batched_data_permuted),
                num_controls=qroam.num_controls,
                log_block_sizes=self.qroam_adjoint_log_block_sizes,
            )
        return QROAMCleanAdjoint.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            target_shapes=(qroam.block_sizes,) * len(qroam.target_bitsizes),
            num_controls=qroam.num_controls,
            log_block_sizes=self.qroam_adjoint_log_block_sizes,
        )

    @property
    def qroam_adj_bloq_for_cost(self) -> QROAMCleanAdjoint:
        qroam = self.qroam_bloq_for_cost
        return QROAMCleanAdjoint.build_from_bitsize(
            qroam.data_shape,
            target_bitsizes=qroam.target_bitsizes,
            target_shapes=(qroam.block_sizes,) * len(qroam.target_bitsizes),
            num_controls=qroam.num_controls,
            log_block_sizes=self.qroam_adjoint_log_block_sizes,
        )

    @property
    def add_into_phase_grad(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)

    def _move_public_selection_to_qroam(self, soqs: Dict[str, SoquetT]) -> Dict[str, SoquetT]:
        for qroam_reg, source_name in zip(
            self.qroam_bloq.selection_registers, self.qroam_selection_sources
        ):
            soqs[qroam_reg.name] = soqs.pop(source_name)
        return soqs

    def _move_qroam_selection_to_public(self, soqs: Dict[str, SoquetT]) -> Dict[str, SoquetT]:
        for qroam_reg, source_name in zip(
            self.qroam_bloq.selection_registers, self.qroam_selection_sources
        ):
            soqs[source_name] = soqs.pop(qroam_reg.name)
        return soqs

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        phase_grad = soqs.pop("phase_gradient")
        soqs = self._move_public_selection_to_qroam(dict(soqs))
        soqs = bb.add_d(self.qroam_bloq, **soqs)
        soqs["target0_"], phase_grad = bb.add(
            self.add_into_phase_grad, x=soqs["target0_"], phase_grad=phase_grad
        )
        if self.measure_reset:
            # Forward-only: erase the loaded angle register (+ junk) by X-basis
            # measurement instead of a coherent QROAMCleanAdjoint -- 0 Toffoli.
            for target in self.qroam_bloq.target_registers:
                _measure_x_reset(bb, soqs.pop(target.name))
                junk_name = "junk_" + target.name
                if junk_name in soqs:
                    _measure_x_reset(bb, soqs.pop(junk_name))
            soqs = self._move_qroam_selection_to_public(dict(soqs))
            soqs["phase_gradient"] = phase_grad
            return soqs
        block_sizes = cast(Tuple[int, ...], self.qroam_bloq.block_sizes)
        for target, adj_target in zip(
            self.qroam_bloq.target_registers, self.qroam_adj_bloq.target_registers
        ):
            junk_name = "junk_" + target.name
            junk_soqs = soqs.pop(junk_name) if junk_name in soqs else np.array([])
            assert isinstance(junk_soqs, np.ndarray)
            soqs[adj_target.name] = np.array([soqs.pop(target.name), *junk_soqs]).reshape(
                block_sizes
            )
        soqs = bb.add_d(self.qroam_adj_bloq, **soqs)
        soqs = self._move_qroam_selection_to_public(dict(soqs))
        soqs["phase_gradient"] = phase_grad
        return soqs

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        if not self.measure_reset:
            ret[self.qroam_adj_bloq_for_cost] += 1  # else 0-Toffoli X-measurement
        ret[self.add_into_phase_grad] += 1
        return ret


@attrs.frozen
class BlockStatePreparationViaQROAMRotations(GateWithRegisters):
    """Prepare one of several dense states selected by an unchanged ``block`` register."""

    state_coefficients: Union[Shaped, NDArray[np.complex128]] = attrs.field(
        converter=_to_block_array_or_shape, eq=_coeff_shape_key
    )
    phase_bitsize: SymbolicInt
    control_bitsize: int = 0
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
    per_layer_optimal: bool = True
    measure_reset: bool = False

    @property
    def _layer_measure_reset(self) -> bool:
        """Per-layer measurement-only uncompute applies only to the FORWARD direction."""
        return self.measure_reset and not self.uncompute

    @classmethod
    def from_bitsize(
        cls,
        n_blocks: SymbolicInt,
        n_coeff: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        control_bitsize: int = 0,
        uncompute: bool = False,
        amp_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        amp_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        phase_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        phase_adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = (0, 0),
        optimal_T: bool = False,
        per_layer_optimal: bool = True,
        measure_reset: bool = False,
    ) -> "BlockStatePreparationViaQROAMRotations":
        if not is_symbolic(n_blocks):
            assert n_blocks >= 1
        if not is_symbolic(n_coeff):
            assert n_coeff == 2 ** bit_length(n_coeff - 1)
        return cls(
            state_coefficients=Shaped((n_blocks, n_coeff)),
            phase_bitsize=phase_bitsize,
            control_bitsize=control_bitsize,
            uncompute=uncompute,
            amp_log_block_sizes=amp_log_block_sizes,
            amp_adjoint_log_block_sizes=amp_adjoint_log_block_sizes,
            phase_log_block_sizes=phase_log_block_sizes,
            phase_adjoint_log_block_sizes=phase_adjoint_log_block_sizes,
            optimal_T=optimal_T,
            per_layer_optimal=per_layer_optimal,
            measure_reset=measure_reset,
        )

    def _per_layer_amp_lbs(self, layer_n_rows: int) -> Tuple[int, int]:
        """Forward log_block_sizes for one staircase amp layer of shape (n_blocks, layer_n_rows).

        Per-layer cap derived from ``amp_log_block_sizes``'s ``log_k_block``: the row
        split is set per-layer via ``lambda_qi ~ sqrt(2 * n_blocks * layer_n_rows / phase_bitsize)``.
        """
        if not self.per_layer_optimal:
            if self.amp_log_block_sizes is None:
                return None  # propagate None -> QROAMClean auto-optimizes this layer
            return self._literal_layer_lbs(self.amp_log_block_sizes, layer_n_rows)
        global_lbs = self.amp_log_block_sizes if self.amp_log_block_sizes is not None else (0, 0)
        log_k_block = int(global_lbs[0]) if len(global_lbs) >= 2 else 0
        return _per_layer_log_block_sizes(
            int(self.n_blocks), int(layer_n_rows), int(self.phase_bitsize), log_k_block
        )

    def _per_layer_amp_adj_lbs(self, layer_n_rows: int) -> Tuple[int, int]:
        """Adjoint log_block_sizes for one staircase amp layer.

        Per-layer cap derived from ``amp_adjoint_log_block_sizes``'s ``log_k_block``: the
        row split is set per-layer via the canonical adjoint optimum
        ``lambda_adj_qi ~ sqrt(n_blocks * layer_n_rows)``.
        """
        if not self.per_layer_optimal:
            if self.amp_adjoint_log_block_sizes is None:
                return None  # propagate None -> QROAMCleanAdjoint auto-optimizes this layer
            return self._literal_layer_lbs(self.amp_adjoint_log_block_sizes, layer_n_rows)
        global_lbs = self.amp_adjoint_log_block_sizes if self.amp_adjoint_log_block_sizes is not None else (0, 0)
        log_k_block = int(global_lbs[0]) if len(global_lbs) >= 2 else 0
        return _per_layer_adjoint_log_block_sizes(
            int(self.n_blocks), int(layer_n_rows), log_k_block
        )

    def _literal_layer_lbs(self, global_lbs: Tuple[SymbolicInt, ...], layer_n_rows: int) -> Tuple[int, int]:
        """Use the supplied ``(log_k_block, log_k_row)`` verbatim (capped to the layer).

        Active only when ``per_layer_optimal`` is False -- this disables the analytic
        ``lambda*`` row-batching override so the caller's literal block sizes are honored.
        ``(0, 0)`` therefore yields an un-batched (``lambda = 1``) qubit-minimal QROAM.
        """
        log_k_block = int(global_lbs[0]) if len(global_lbs) >= 2 else 0
        log_k_row = int(global_lbs[1]) if len(global_lbs) >= 2 else (int(global_lbs[0]) if global_lbs else 0)
        sel = bit_length(layer_n_rows - 1) if layer_n_rows > 1 else 0
        log_k_row = min(max(0, log_k_row), sel)
        return (max(0, log_k_block), log_k_row)

    def __attrs_post_init__(self):
        assert self.control_bitsize >= 0
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        assert len(shape(self.state_coefficients)) == 2
        if not is_symbolic(self.n_coeff):
            assert self.n_coeff == 2**self.state_bitsize
        if isinstance(self.state_coefficients, np.ndarray):
            norms = np.linalg.norm(self.state_coefficients, axis=1)
            assert np.allclose(norms, np.ones_like(norms))

    @property
    def n_blocks(self) -> SymbolicInt:
        return shape(self.state_coefficients)[0]

    @property
    def n_coeff(self) -> SymbolicInt:
        return shape(self.state_coefficients)[1]

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def state_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_coeff - 1)

    @property
    def signature(self) -> Signature:
        return Signature.build(
            prepare_control=self.control_bitsize,
            block=self.block_bitsize,
            target_state=self.state_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def rotation_trees(self) -> List[RotationTree]:
        assert isinstance(self.state_coefficients, np.ndarray) and isinstance(self.phase_bitsize, int)
        return [
            RotationTree(self.state_coefficients[block], self.phase_bitsize, self.uncompute)
            for block in range(int(self.n_blocks))
        ]

    def _amp_layer_lbs(self, qi: int) -> Tuple[Optional[Tuple[SymbolicInt, ...]], Optional[Tuple[SymbolicInt, ...]]]:
        """(forward, adjoint) log_block_sizes for the qi-th amplitude staircase layer."""
        return self._per_layer_amp_lbs(2**qi), self._per_layer_amp_adj_lbs(2**qi)

    @property
    def prga_prepare_amplitude(self) -> List[BlockPRGAViaPhaseGradientQROAM]:
        if isinstance(self.state_coefficients, Shaped) or is_symbolic(self.phase_bitsize):
            if is_symbolic(self.state_bitsize):
                raise DecomposeTypeError(
                    "exact block dense resource estimates require a concrete state bitsize"
                )
            ret = []
            for qi in range(int(self.state_bitsize)):
                fwd_lbs, adj_lbs = self._amp_layer_lbs(qi)
                ret.append(
                    BlockPRGAViaPhaseGradientQROAM(
                        block_bitsize=self.block_bitsize,
                        selection_bitsize=qi,
                        phase_bitsize=self.phase_bitsize,
                        rom_values=Shaped((self.n_blocks, 2**qi)),
                        control_bitsize=self.control_bitsize + 1,
                        log_block_sizes=fwd_lbs,
                        adjoint_log_block_sizes=adj_lbs,
                        measure_reset=self._layer_measure_reset,
                    )
                )
            return ret
        trees = self.rotation_trees
        ret = []
        for qi in range(int(self.state_bitsize)):
            fwd_lbs, adj_lbs = self._amp_layer_lbs(qi)
            ret.append(
                BlockPRGAViaPhaseGradientQROAM(
                    block_bitsize=self.block_bitsize,
                    selection_bitsize=qi,
                    phase_bitsize=self.phase_bitsize,
                    rom_values=np.array([tree.get_rom_vals()[0][qi] for tree in trees], dtype=int),
                    control_bitsize=self.control_bitsize + 1,
                    log_block_sizes=fwd_lbs,
                    adjoint_log_block_sizes=adj_lbs,
                    measure_reset=self._layer_measure_reset,
                )
            )
        return ret

    @property
    def prga_prepare_phases(self) -> BlockPRGAViaPhaseGradientQROAM:
        if isinstance(self.state_coefficients, Shaped) or is_symbolic(self.phase_bitsize):
            data_or_shape: Union[Shaped, Tuple[Tuple[int, ...], ...]] = Shaped(
                (self.n_blocks, self.n_coeff)
            )
        else:
            data_or_shape = np.array([tree.get_rom_vals()[1] for tree in self.rotation_trees], dtype=int)
        return BlockPRGAViaPhaseGradientQROAM(
            block_bitsize=self.block_bitsize,
            selection_bitsize=self.state_bitsize,
            phase_bitsize=self.phase_bitsize,
            rom_values=data_or_shape,
            control_bitsize=self.control_bitsize + 1,
            log_block_sizes=self.phase_log_block_sizes,
            adjoint_log_block_sizes=self.phase_adjoint_log_block_sizes,
            # FINAL layer of a forward prep -> keep the coherent QROAMCleanAdjoint (no
            # measurement shortcut); only the preceding amplitude layers measurement-reset.
            measure_reset=False,
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if isinstance(self.state_coefficients, Shaped) or is_symbolic(self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free {self}")
        if self.uncompute:
            soqs = self._prepare_phases(bb, **soqs)
            soqs = self._prepare_amplitudes(bb, **soqs)
        else:
            soqs = self._prepare_amplitudes(bb, **soqs)
            soqs = self._prepare_phases(bb, **soqs)
        return soqs

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[Rx(angle=-np.pi / 2)] += self.state_bitsize
        ret[Rx(angle=np.pi / 2)] += self.state_bitsize
        ret[XGate()] += 2
        ret[self.prga_prepare_phases] += 1
        for bloq in self.prga_prepare_amplitude:
            ret[bloq] += 1
        return ret

    def _prepare_amplitudes(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        state_qubits = bb.split(cast(Soquet, soqs.pop("target_state")))
        ctrl_rot_bloqs = (
            reversed(list(enumerate(self.prga_prepare_amplitude)))
            if self.uncompute
            else enumerate(self.prga_prepare_amplitude)
        )
        for qi, ctrl_rot_q in ctrl_rot_bloqs:
            state_qubits[qi] = bb.add(Rx(angle=np.pi / 2), q=state_qubits[qi])
            if qi:
                soqs["selection"] = bb.join(state_qubits[:qi])
            if self.control_bitsize > 1:
                soqs["control"] = bb.join(
                    np.array(
                        [*bb.split(cast(Soquet, soqs.pop("prepare_control"))), state_qubits[qi]]
                    )
                )
            elif self.control_bitsize == 1:
                soqs["control"] = bb.join(np.array([soqs.pop("prepare_control"), state_qubits[qi]]))
            else:
                soqs["control"] = state_qubits[qi]
            soqs = bb.add_d(ctrl_rot_q, **soqs)
            separated = bb.split(cast(Soquet, soqs.pop("control")))
            if self.control_bitsize != 0:
                soqs["prepare_control"] = bb.join(separated[:-1])
            state_qubits[qi] = separated[-1]
            if qi:
                state_qubits[:qi] = bb.split(cast(Soquet, soqs.pop("selection")))
            state_qubits[qi] = bb.add(Rx(angle=-np.pi / 2), q=state_qubits[qi])
        soqs["target_state"] = bb.join(state_qubits)
        return soqs

    def _prepare_phases(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        rot_ancilla = bb.allocate(1)
        rot_ancilla = bb.add(XGate(), q=rot_ancilla)
        if self.control_bitsize > 1:
            soqs["control"] = bb.join(
                np.array([*bb.split(cast(Soquet, soqs.pop("prepare_control"))), rot_ancilla])
            )
        elif self.control_bitsize == 1:
            soqs["control"] = bb.join(np.array([soqs.pop("prepare_control"), rot_ancilla]))
        else:
            soqs["control"] = rot_ancilla
        ctrl_rot = self.prga_prepare_phases
        if ctrl_rot.selection_bitsize:
            soqs["selection"] = soqs.pop("target_state")
        soqs = bb.add_d(ctrl_rot, **soqs)
        if ctrl_rot.selection_bitsize:
            soqs["target_state"] = soqs.pop("selection")
        separated = bb.split(cast(Soquet, soqs.pop("control")))
        if self.control_bitsize != 0:
            soqs["prepare_control"] = bb.join(separated[:-1])
        separated[-1] = bb.add(XGate(), q=separated[-1])
        bb.free(separated[-1])
        return soqs
