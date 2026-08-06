r"""State preparation via three full diagonal phase layers interleaved with Hadamards.

This is a *prototype resource model* for the "three diagonal phase layers + Hadamards"
state-preparation ansatz of

    Berry, Tong, Khattar, White, Kim, Boixo, Lin, Lee, Chan, Babbush, Rubin,
    arXiv:2409.11748 (page 14),

which approximates a normalized target state ``|psi>`` of length ``N = 2^n`` by

    |psi> ~= D3 . H^n . D2 . H^n . D1 . |+^n> ,

where ``H^n`` is the normalized Walsh-Hadamard transform ``H^{otimes n}`` and each layer

    D_l = diag(e^{i theta_l[x]}),   x = 0, ..., N-1

is a *generic, full* length-``N`` diagonal unitary (no half-identity / sparsity assumption is
made -- every one of the ``N`` phases ``theta_l[x]`` is an independent free parameter, up to one
irrelevant global phase per layer).  Because ``|+^n> = H^n |0^n>``, preparing ``|psi>`` from
``|0^n>`` is the symmetric three-layer circuit

    |0^n>  --H^n-->  D1  --H^n-->  D2  --H^n-->  D3 ,

i.e. three Hadamard transforms and three full diagonal phase layers.

Diagonal phase layer (``DiagonalPhaseQROAM``).  Each ``D_l`` is realized exactly like the
``BlockInterferometerFinalPhasesQROAM`` diagonal phase of the sibling interferometer module and the
Eq.-40 phase-lookup of arXiv:1812.00954:

  * a QRO(A)M addressed by the full ``n``-qubit system register loads a ``b``-bit fixed-point phase
    word ``theta_l[x]`` into a clean ancilla register,
  * an (uncontrolled) ``AddIntoPhaseGrad`` adds that word into a phase-gradient register, kicking back
    ``e^{2 pi i theta_l[x] / 2^b} = e^{i theta_l[x]}`` onto the basis state ``|x>``, and
  * the load is uncomputed by ``QROAMCleanAdjoint`` (or, as a forward-only optimization for the
    non-final layers, erased by X-basis measurement at 0 Toffoli).

Resource-model status.  Following the integration's data-free convention (``AGENTS.md``) this module
only models the case where the angle tables are *fictitious* -- represented by shape, never by
concrete data -- and does **not** implement angle finding (the inverse problem of fitting
``theta_1, theta_2, theta_3`` to a given ``|psi>``).  ``build_composite_bloq`` lays out the real gate
structure with shape-only QROAM, ``build_call_graph`` gives the aggregate cost, and the module-level
numpy helpers (:func:`walsh_hadamard`, :func:`reconstruct_three_phase_state`) provide a standalone
*forward* reference for the ansatz arithmetic.

References:
    [Rapid initial-state preparation ... three diagonal phase layers] arXiv:2409.11748, page 14.
    [Trading T-gates for dirty qubits in state preparation and unitary synthesis] arXiv:1812.00954.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    DecomposeTypeError,
    GateWithRegisters,
    QUInt,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        _measure_x_reset,
        _positive_power_of_two,
        _qroam_log_block_sizes,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script/direct execution
    from block_unitary_interferometer_QROAM import (
        _measure_x_reset,
        _positive_power_of_two,
        _qroam_log_block_sizes,
    )
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


# ============================================================================
# numpy reference: the forward ansatz  D3 H^n D2 H^n D1 |+^n>
# ============================================================================

_H1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def walsh_hadamard(n: int) -> NDArray[np.complex128]:
    """The normalized Walsh-Hadamard transform ``H^{otimes n}`` as a ``2^n x 2^n`` matrix."""
    H = np.array([[1.0 + 0.0j]])
    for _ in range(int(n)):
        H = np.kron(H, _H1)
    return H


def plus_state(n: int) -> NDArray[np.complex128]:
    """The uniform-superposition state ``|+^n> = H^n |0^n>`` of length ``2^n``."""
    N = 1 << int(n)
    return np.full(N, 1.0 / np.sqrt(N), dtype=complex)


def reconstruct_three_phase_state(
    theta1: Iterable[float], theta2: Iterable[float], theta3: Iterable[float]
) -> NDArray[np.complex128]:
    r"""Apply the ansatz ``D3 H^n D2 H^n D1 |+^n>`` for three full diagonal phase tables.

    Each ``theta_l`` is a length-``N = 2^n`` real angle table giving ``D_l = diag(e^{i theta_l[x]})``.
    Returns the resulting (normalized) state vector.  This is the *forward* map only; it does not
    solve for the angles given a target state.
    """
    thetas = [np.asarray(t, dtype=float).ravel() for t in (theta1, theta2, theta3)]
    N = thetas[0].size
    n = int(round(log2(N)))
    assert (1 << n) == N, "each phase table must have length 2^n"
    assert all(t.size == N for t in thetas), "all three phase tables must have equal length"
    Hn = walsh_hadamard(n)
    state = plus_state(n)
    state = np.exp(1j * thetas[0]) * state  # D1
    state = Hn @ state  # H^n
    state = np.exp(1j * thetas[1]) * state  # D2
    state = Hn @ state  # H^n
    state = np.exp(1j * thetas[2]) * state  # D3
    return state


# ============================================================================
# Diagonal phase layer  D = diag(e^{i theta[x]})  via QROAM + phase gradient
# ============================================================================


@attrs.frozen
class DiagonalPhaseQROAM(GateWithRegisters):
    r"""A full length-``N`` diagonal phase ``D = diag(e^{i theta[x]})`` via a QROAM phase lookup.

    Circuit: ``QROAM(system) -> theta_word``; ``AddIntoPhaseGrad(theta_word, phase_gradient)``;
    erase the load.  Phase kickback imparts ``e^{2 pi i theta_int[x] / 2^b} = e^{i theta[x]}`` onto
    every computational basis state ``|x>`` of the ``system`` register, i.e. a generic diagonal
    unitary with one independent ``b``-bit phase per address.  The angle table is *data-free*: it is
    represented only by its shape ``(N,)`` for resource estimation.

    This mirrors ``BlockInterferometerFinalPhasesQROAM`` (sibling interferometer module) and the
    Eq.-40 phase-lookup of arXiv:1812.00954.

    Block address.  With ``n_blocks > 1`` a read-only ``block`` register of ``ceil(log2 n_blocks)``
    qubits is prepended to the QROAM address (table shape ``(n_blocks, N)``), so the layer realizes a
    block-diagonal family ``sum_a |a><a| (x) diag(e^{i theta_a[x]})`` -- exactly like the sibling
    ``Block*QROAM`` bloqs.

    Control.  With ``control_bitsize == 1`` a ``prepare_control`` qubit gates *only* the
    ``AddIntoPhaseGrad`` (the QROAM load/erase stay uncontrolled because they self-cancel when the
    control is 0), so the diagonal phase is applied iff ``prepare_control = 1``.

    Args:
        n_rows: ``N = 2^n``, the diagonal length (must be a power of two).
        phase_bitsize: ``b``, the fixed-point bitsize of every stored phase word.
        n_blocks: number of block-diagonal blocks (1 = ordinary single diagonal).
        control_bitsize: 0 (uncontrolled) or 1 (a ``prepare_control`` qubit gates the phase add).
        measure_reset: if True, erase the load by X-basis measurement (0 Toffoli) instead of a
            coherent ``QROAMCleanAdjoint``.  Forward-only: the omitted sign-fixup must be absorbed by
            a subsequent operation, so the *final* phase layer of a preparation keeps it False.
        log_block_sizes: QROAM load tradeoff parameter (block sizes ``2**log_block_sizes``); the
            default ``None`` lets QROAM choose its own T-optimal split.
        adjoint_log_block_sizes: optional separate tradeoff for ``QROAMCleanAdjoint`` (``None`` ->
            QROAM's own default).
    """

    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_blocks: SymbolicInt = 1
    control_bitsize: int = 0
    measure_reset: bool = False
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows):
            assert _positive_power_of_two(self.n_rows), "n_rows must be a power of two"
            assert self.n_rows >= 2
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        assert self.control_bitsize in (0, 1), "only uncontrolled or singly-controlled is supported"

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def signature(self) -> Signature:
        return Signature.build(
            prepare_control=self.control_bitsize,
            block=self.block_bitsize,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def qroam_data_shape(self) -> Tuple[SymbolicInt, ...]:
        return (self.n_blocks, self.n_rows) if self.has_block else (self.n_rows,)

    @property
    def qroam_selection_bitsizes(self) -> Tuple[SymbolicInt, ...]:
        if self.has_block:
            return (self.block_bitsize, self.system_bitsize)
        return (self.system_bitsize,)

    @property
    def qroam_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _qroam_log_block_sizes(self.log_block_sizes, self.qroam_data_shape)

    @property
    def qroam_adjoint_log_block_sizes(self) -> Optional[Tuple[SymbolicInt, ...]]:
        return _qroam_log_block_sizes(self.adjoint_log_block_sizes, self.qroam_data_shape)

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

    @property
    def _add_into_phase_grad(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)

    @property
    def _phase_add_bloq(self) -> Bloq:
        """The (possibly ``prepare_control``-controlled) ``AddIntoPhaseGrad`` phase kick."""
        add = self._add_into_phase_grad
        return add.controlled() if self.control_bitsize else add

    def _pack_out(self, system, phase_grad, block, pc) -> Dict[str, SoquetT]:
        out: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
        if self.has_block:
            out['block'] = block
        if self.control_bitsize:
            out['prepare_control'] = pc
        return out

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Circuit: ``QROAM(block?, system) -> theta``; (ctrl-)``AddIntoPhaseGrad``; erase ``theta``."""
        if is_symbolic(self.n_rows, self.phase_bitsize, self.n_blocks):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")

        phase_grad = soqs['phase_gradient']
        system = soqs['system']
        block = soqs.get('block')
        pc = soqs.get('prepare_control')

        # QROAM load: (block?, system) -> theta (target0_)
        qroam = self.qroam_bloq_for_cost
        sel_names = [r.name for r in qroam.selection_registers]
        if self.has_block:
            qroam_out = bb.add_d(qroam, **{sel_names[0]: block, sel_names[1]: system})
            block = qroam_out[sel_names[0]]
            system = qroam_out[sel_names[1]]
        else:
            qroam_out = bb.add_d(qroam, **{sel_names[0]: system})
            system = qroam_out[sel_names[0]]
        theta = qroam_out['target0_']

        # Phase kick: grad += theta  ->  e^{i theta[x]} on |x>  (gated by prepare_control if any).
        if self.control_bitsize:
            pc, theta, phase_grad = bb.add(self._phase_add_bloq, ctrl=pc, x=theta, phase_grad=phase_grad)
        else:
            theta, phase_grad = bb.add(self._phase_add_bloq, x=theta, phase_grad=phase_grad)

        if self.measure_reset:
            # Forward-only: erase the loaded word (and any junk) by X-measurement -- 0 Toffoli.
            # The QROAMCleanAdjoint sign-fixup is omitted and absorbed downstream.
            _measure_x_reset(bb, theta)
            if 'junk_target0_' in qroam_out:
                _measure_x_reset(bb, qroam_out['junk_target0_'])
            return self._pack_out(system, phase_grad, block, pc)

        # Coherent uncompute with a dedicated adjoint lambda.
        qroam_adj = self.qroam_adj_bloq_for_cost
        adj_sel_names = [r.name for r in qroam_adj.selection_registers]
        junk_arr = np.asarray(qroam_out['junk_target0_']) if 'junk_target0_' in qroam_out else np.array([])
        adj_target = next(iter(qroam_adj.target_registers))
        adj_soqs: Dict[str, SoquetT] = {
            adj_target.name: np.array([theta, *junk_arr]).reshape(qroam_adj.target_shapes[0])
        }
        if self.has_block:
            adj_soqs[adj_sel_names[0]] = block
            adj_soqs[adj_sel_names[1]] = system
        else:
            adj_soqs[adj_sel_names[0]] = system
        adj_out = bb.add_d(qroam_adj, **adj_soqs)
        if self.has_block:
            block = adj_out[adj_sel_names[0]]
            system = adj_out[adj_sel_names[1]]
        else:
            system = adj_out[adj_sel_names[0]]
        return self._pack_out(system, phase_grad, block, pc)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        ret[self.qroam_bloq_for_cost] += 1
        ret[self._phase_add_bloq] += 1
        if not self.measure_reset:
            ret[self.qroam_adj_bloq_for_cost] += 1  # else replaced by 0-Toffoli X-measurement
        return ret


# ============================================================================
# Top-level three-phase-layer state preparation
# ============================================================================


@attrs.frozen
class ThreePhaseLayerStatePreparation(GateWithRegisters):
    r"""Prepare ``|psi> ~= D3 H^n D2 H^n D1 |+^n>`` from ``|0^n>`` (arXiv:2409.11748, p. 14).

    The circuit applied to ``target_state`` initialized at ``|0^n>`` is

        H^n  ->  D1  ->  H^n  ->  D2  ->  H^n  ->  D3 ,

    i.e. three normalized Walsh-Hadamard transforms (``n`` Hadamards each) and three *full*
    length-``N`` diagonal phase layers (:class:`DiagonalPhaseQROAM`).  Because ``|+^n> = H^n|0^n>``
    the leading ``H^n`` creates ``|+^n>`` and the result is exactly ``D3 H^n D2 H^n D1 |+^n>``.

    This is a data-free resource model: the three angle tables are fictitious (shape-only QROAM) and
    angle finding is out of scope (see the module docstring).

    Drop-in interface.  The signature and options mirror ``StatePreparationViaQROAMRotations`` /
    ``BlockStatePreparationViaQROAMRotations`` so this ansatz can be toggled in as their replacement:

      * ``n_blocks > 1`` prepends a ``block`` address to every diagonal layer (a block-diagonal
        family ``sum_a |a><a| (x) |psi_a>``);
      * ``control_bitsize == 1`` adds a ``prepare_control`` qubit; the controlled preparation must
        gate the ``H^n`` layers too (controlled Hadamards) because the three transforms do **not**
        pairwise cancel under a single control (unlike the standard prep's self-cancelling Cliffords).
        The controlled-Hadamard cost is negligible next to the QROAM phase loads;
      * ``uncompute`` applies the adjoint ``H^n D1^dagger H^n D2^dagger H^n D3^dagger`` (data-free,
        so identical cost to the forward direction; ``measure_reset`` is forward-only).

    Args:
        n_rows: ``N = 2^n``, the prepared-state length (must be a power of two).
        phase_bitsize: ``b``, fixed-point bitsize of every stored phase word.
        n_blocks: number of block-diagonal blocks (1 = single state).
        control_bitsize: 0 (uncontrolled) or 1 (a ``prepare_control`` qubit).
        uncompute: prepare the adjoint (un-preparation) instead of the forward preparation.
        measure_reset: erase the QROAM load of the two non-final forward layers (``D1``, ``D2``) by
            X-measurement (0 Toffoli, forward-only); ``D3`` keeps the coherent adjoint, and an
            ``uncompute`` preparation keeps every layer coherent.
        log_block_sizes: explicit per-layer QROAM load split; the default ``None`` lets QROAM pick
            its own T-optimal split (which here beats any single fixed value).
        adjoint_log_block_sizes: explicit per-layer ``QROAMCleanAdjoint`` split (``None`` -> default).
    """

    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    n_blocks: SymbolicInt = 1
    control_bitsize: int = 0
    uncompute: bool = False
    measure_reset: bool = False
    log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )
    adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=None, converter=_to_tuple_or_none
    )

    #: Number of diagonal phase layers in the ansatz (and of Hadamard transforms).
    #: A ``ClassVar`` so ``attrs`` treats it as a fixed constant, not an init field.
    N_PHASE_LAYERS: ClassVar[int] = 3

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows):
            assert _positive_power_of_two(self.n_rows), "n_rows must be a power of two"
            assert self.n_rows >= 2
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        assert self.control_bitsize in (0, 1), "only uncontrolled or singly-controlled is supported"

    @classmethod
    def from_bitsize(
        cls,
        n_coeff: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_blocks: SymbolicInt = 1,
        control_bitsize: int = 0,
        uncompute: bool = False,
        measure_reset: bool = False,
        log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
        adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    ) -> 'ThreePhaseLayerStatePreparation':
        """Data-free constructor from the (per-block) dense state-vector length ``n_coeff`` (= ``N``)."""
        if not is_symbolic(n_coeff):
            assert n_coeff == 2 ** bit_length(n_coeff - 1), "n_coeff must be a power of two"
        return cls(
            n_rows=n_coeff,
            phase_bitsize=phase_bitsize,
            n_blocks=n_blocks,
            control_bitsize=control_bitsize,
            uncompute=uncompute,
            measure_reset=measure_reset,
            log_block_sizes=log_block_sizes,
            adjoint_log_block_sizes=adjoint_log_block_sizes,
        )

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def signature(self) -> Signature:
        return Signature.build(
            prepare_control=self.control_bitsize,
            block=self.block_bitsize,
            target_state=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    @property
    def _layer_measure_reset(self) -> bool:
        """Per-layer measurement-only uncompute applies only to the FORWARD direction."""
        return self.measure_reset and not self.uncompute

    def diagonal_layer(self, is_final: bool) -> DiagonalPhaseQROAM:
        """The ``DiagonalPhaseQROAM`` for one ``D_l`` (``is_final`` keeps the coherent adjoint)."""
        return DiagonalPhaseQROAM(
            n_rows=self.n_rows,
            phase_bitsize=self.phase_bitsize,
            n_blocks=self.n_blocks,
            control_bitsize=self.control_bitsize,
            measure_reset=self._layer_measure_reset and not is_final,
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
        )

    def _hadamard_transform(self, bb: BloqBuilder, target_state: SoquetT, pc):
        """Apply ``H^n`` (one Hadamard per qubit), controlled by ``pc`` when ``control_bitsize``."""
        qubits = bb.split(target_state)
        for i in range(len(qubits)):
            if self.control_bitsize:
                pc, qubits[i] = bb.add(Hadamard().controlled(), ctrl=pc, target=qubits[i])
            else:
                qubits[i] = bb.add(Hadamard(), q=qubits[i])
        return bb.join(qubits, dtype=QUInt(int(self.system_bitsize))), pc

    def _apply_layer(self, bb, layer, target_state, phase_grad, block, pc):
        lin: Dict[str, SoquetT] = {'system': target_state, 'phase_gradient': phase_grad}
        if self.has_block:
            lin['block'] = block
        if self.control_bitsize:
            lin['prepare_control'] = pc
        lout = bb.add_d(layer, **lin)
        target_state = lout['system']
        phase_grad = lout['phase_gradient']
        if self.has_block:
            block = lout['block']
        if self.control_bitsize:
            pc = lout['prepare_control']
        return target_state, phase_grad, block, pc

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_rows, self.phase_bitsize, self.n_blocks):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")

        target_state = soqs['target_state']
        phase_grad = soqs['phase_gradient']
        block = soqs.get('block')
        pc = soqs.get('prepare_control')

        last = self.N_PHASE_LAYERS - 1
        for ell in range(self.N_PHASE_LAYERS):
            if not self.uncompute:
                # forward:  ... H^n, D_{ell+1}
                target_state, pc = self._hadamard_transform(bb, target_state, pc)
                layer = self.diagonal_layer(is_final=(ell == last))
                target_state, phase_grad, block, pc = self._apply_layer(
                    bb, layer, target_state, phase_grad, block, pc
                )
            else:
                # adjoint:  ... D^dagger, H^n   (data-free: same bloq, all coherent)
                layer = self.diagonal_layer(is_final=False)
                target_state, phase_grad, block, pc = self._apply_layer(
                    bb, layer, target_state, phase_grad, block, pc
                )
                target_state, pc = self._hadamard_transform(bb, target_state, pc)

        out: Dict[str, SoquetT] = {'target_state': target_state, 'phase_gradient': phase_grad}
        if self.has_block:
            out['block'] = block
        if self.control_bitsize:
            out['prepare_control'] = pc
        return out

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        had = Hadamard().controlled() if self.control_bitsize else Hadamard()
        ret[had] += self.N_PHASE_LAYERS * self.system_bitsize
        last = self.N_PHASE_LAYERS - 1
        if not self.uncompute:
            for ell in range(self.N_PHASE_LAYERS):
                ret[self.diagonal_layer(is_final=(ell == last))] += 1
        else:
            ret[self.diagonal_layer(is_final=False)] += self.N_PHASE_LAYERS
        return ret

    def resource_estimate(self) -> 'ThreePhaseLayerResourceEstimate':
        """Aggregate resource estimate for this configuration (see the dataclass)."""
        return estimate_three_phase_layer_resources(
            int(self.n_rows),
            int(self.phase_bitsize),
            measure_reset=self.measure_reset,
            log_block_sizes=self.log_block_sizes,
            adjoint_log_block_sizes=self.adjoint_log_block_sizes,
        )


# ============================================================================
# Resource estimate
# ============================================================================


@dataclass(frozen=True)
class ThreePhaseLayerResourceEstimate:
    """Aggregate resource estimate for one :class:`ThreePhaseLayerStatePreparation` configuration.

    The Toffoli and T figures are taken from ``QECGatesCost`` of the actual bloq (so they always
    track the real layout); the remaining fields are exact structural counts of the ansatz.
    """

    n_rows: int
    phase_bitsize: int
    n_phase_layers: int
    n_hadamards: int          # N_PHASE_LAYERS * n  (one H^n transform per layer)
    n_phase_additions: int    # one (uncontrolled) AddIntoPhaseGrad per diagonal layer
    n_loaded_phase_words: int # N per layer  ->  N_PHASE_LAYERS * N
    toffoli: int
    t_count: int
    register_qubits: int      # target_state + phase_gradient footprint (QROAM adds transient ancilla)


def estimate_three_phase_layer_resources(
    n_rows: int,
    phase_bitsize: int,
    *,
    measure_reset: bool = False,
    log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
    adjoint_log_block_sizes: Optional[Union[SymbolicInt, Iterable[SymbolicInt]]] = None,
) -> ThreePhaseLayerResourceEstimate:
    """Closed-form structural counts plus ``QECGatesCost`` Toffoli/T for the three-layer ansatz."""
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    assert _positive_power_of_two(n_rows)
    assert phase_bitsize > 1
    bloq = ThreePhaseLayerStatePreparation(
        n_rows=n_rows,
        phase_bitsize=phase_bitsize,
        measure_reset=measure_reset,
        log_block_sizes=log_block_sizes,
        adjoint_log_block_sizes=adjoint_log_block_sizes,
    )
    cost = get_cost_value(bloq, QECGatesCost())
    n = int(bloq.system_bitsize)
    N = int(n_rows)
    L = ThreePhaseLayerStatePreparation.N_PHASE_LAYERS
    return ThreePhaseLayerResourceEstimate(
        n_rows=N,
        phase_bitsize=int(phase_bitsize),
        n_phase_layers=L,
        n_hadamards=L * n,
        n_phase_additions=L,
        n_loaded_phase_words=L * N,
        toffoli=int(cost.toffoli),
        t_count=int(cost.total_t_count()),
        register_qubits=int(bloq.signature.n_qubits()),
    )
