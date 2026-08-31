r"""Block-diagonal interferometer synthesis of isometries (Sec. III B, block version).

This is the ``K``-block generalization of
:class:`interferometer_isometry_QROAM.InterferometerIsometrySynthesisQROAM`: it synthesizes

    ``sum_{a=0}^{K-1} |a><a| (x) W_a`` ,

where each ``W_a`` is the first ``M`` columns of an ``N x N`` unitary (an ``N x M``
isometry) and the ``block`` register ``|a>`` is a read-only QROAM address left unchanged.
It is the Sec. III B (column-synthesis) sibling of the full-unitary block interferometer
:class:`block_unitary_interferometer_QROAM.BlockUnitaryInterferometerSynthesisQROAM`, and
follows the same block-indexing convention as the other ``Block*QROAM`` modules.

The construction is identical to the single-isometry case (see that module's docstring):
the Eq.-36 staircase of ``d - 1`` fixed ``d = 2`` steps (each ``V`` + multiplexed ``R_y``
+ 2-block ``U``) on the bottom ``m + 1``-qubit window, advanced by an ``AddK`` shift
between steps; ``d = 1`` is one full ``M x M`` interferometer per block.  The only change
is that **every** sub-bloq's block address is widened by the ``K`` outer blocks (so each
step's ``V`` / ``R_y`` use ``K`` blocks and its ``U`` uses ``2K`` blocks).  ``n_blocks = 1``
recovers the single isometry exactly.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran.bloqs.arithmetic.addition import AddK
from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QBit,
    QUInt,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        _ControlledBlockUnitaryInterferometerSynthesisQROAM,
        _positive_power_of_two,
    )
    from .interferometer_isometry_QROAM import (
        InterferometerIsometryMuxRotationQROAM,
        _ControlledMuxRotationQROAM,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script/direct execution
    from block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        _ControlledBlockUnitaryInterferometerSynthesisQROAM,
        _positive_power_of_two,
    )
    from interferometer_isometry_QROAM import (
        InterferometerIsometryMuxRotationQROAM,
        _ControlledMuxRotationQROAM,
    )
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class BlockInterferometerIsometrySynthesisQROAM(GateWithRegisters):
    r"""Synthesize ``sum_a |a><a| (x) W_a`` with each ``W_a`` an ``N x M`` isometry.

    ``n_rows = N`` and ``n_cols = M`` are powers of two with ``M <= N`` (both ``>= 4``,
    inherited from the Sec. III A interferometer's two-qubit-system minimum) and
    ``d = N / M`` a positive integer.  Registers are ``block`` (``ceil(log2 K)`` qubits,
    dropped when ``K == 1``), ``system`` (``n = log2 N`` qubits), and ``phase_gradient``.
    Initialized with the top ``t = log2 d`` system qubits in ``|0>``, the bloq maps
    ``|a>|k>`` (``0 <= k < M``) to ``|a>`` (column ``k`` of ``W_a``).

    ``d = 1`` (``M = N``) is the full-unitary block interferometer
    :class:`BlockUnitaryInterferometerSynthesisQROAM`.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    n_cols: SymbolicInt
    phase_bitsize: SymbolicInt
    n_layers: Optional[SymbolicInt] = None
    optimal_T: bool = False
    interferometer_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    interferometer_final_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    interferometer_final_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    rotation_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )
    rotation_adjoint_log_block_sizes: Optional[Tuple[SymbolicInt, ...]] = attrs.field(
        default=(0, 0), converter=_to_tuple_or_none
    )

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_blocks):
            assert self.n_blocks >= 1
        if not is_symbolic(self.n_cols):
            assert _positive_power_of_two(self.n_cols) and self.n_cols >= 4
        if not is_symbolic(self.n_rows, self.n_cols):
            # d = N / M must be a positive integer; N need not be a power of two (e.g. d=3).
            assert self.n_rows >= self.n_cols and self.n_rows % self.n_cols == 0, \
                "n_rows must be a positive-integer multiple d of n_cols (M)"
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if self.optimal_T and is_symbolic(self.n_blocks, self.n_rows, self.n_cols, self.phase_bitsize):
            raise ValueError("optimal_T=True requires concrete n_blocks, n_rows, n_cols, phase_bitsize")

    @classmethod
    def from_shape(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        n_cols: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        n_layers: Optional[SymbolicInt] = None,
        optimal_T: bool = False,
        **kwargs,
    ) -> "BlockInterferometerIsometrySynthesisQROAM":
        """Data-free block-isometry synthesis bloq for resource estimates (``d = n_rows / n_cols``)."""
        return cls(
            n_blocks=n_blocks,
            n_rows=n_rows,
            n_cols=n_cols,
            phase_bitsize=phase_bitsize,
            n_layers=n_layers,
            optimal_T=optimal_T,
            **kwargs,
        )

    @property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_rows - 1)

    @property
    def col_bitsize(self) -> SymbolicInt:
        """``m = log2 M`` -- the bottom qubits the interferometers act on."""
        return bit_length(self.n_cols - 1)

    @property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @property
    def d(self) -> SymbolicInt:
        """The reduction parameter ``d = N / M``."""
        return self.n_rows // self.n_cols

    @property
    def n_steps(self) -> SymbolicInt:
        """Number of ``d=2`` staircase steps = ``d - 1`` (Eq. 36); ``d=1`` is one full unitary."""
        return self.d - 1

    @property
    def has_block(self) -> bool:
        return not is_symbolic(self.n_blocks) and int(self.n_blocks) > 1

    @property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize,
            system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    # -- sub-bloq factories (each step multiplexed by the ``K`` outer blocks) --

    def full_interferometer(self, n_blocks_eff: int) -> BlockUnitaryInterferometerSynthesisQROAM:
        """A full ``M x M`` interferometer multiplexed over ``n_blocks_eff`` blocks."""
        return BlockUnitaryInterferometerSynthesisQROAM(
            n_blocks=n_blocks_eff,
            n_rows=self.n_cols,
            phase_bitsize=self.phase_bitsize,
            n_layers=self.n_layers,
            log_block_sizes=self.interferometer_log_block_sizes,
            final_log_block_sizes=self.interferometer_final_log_block_sizes,
            final_adjoint_log_block_sizes=self.interferometer_final_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    def mux_rotation(self, n_blocks_eff: int) -> InterferometerIsometryMuxRotationQROAM:
        """The multiplexed ``R_y`` layer addressed by ``(n_blocks_eff, M)``."""
        if self.optimal_T:
            lbs = adj = None
        else:
            lbs, adj = self.rotation_log_block_sizes, self.rotation_adjoint_log_block_sizes
        return InterferometerIsometryMuxRotationQROAM(
            n_blocks=n_blocks_eff,
            n_address=self.n_cols,
            phase_bitsize=self.phase_bitsize,
            log_block_sizes=lbs,
            adjoint_log_block_sizes=adj,
        )

    def _shift(self, k: int) -> AddK:
        """Cyclic shift of ``system`` by ``k`` (advance the active 2M window)."""
        return AddK(dtype=QUInt(self.system_bitsize), k=k % (1 << int(self.system_bitsize)))

    # ----------------------------- composite circuit -----------------------------

    def _apply_level(
        self, bb: BloqBuilder, system: SoquetT, phase_grad: SoquetT, gblock: NDArray
    ) -> Tuple[SoquetT, SoquetT, NDArray]:
        r"""One column-synthesis level: mux-``R_y`` (Eq.32) then the merged 2-block ``U``, both
        additionally multiplexed by the ``K`` global blocks (so ``U`` uses ``2K`` blocks).
        The next level's ``V`` is absorbed into this ``U`` at no extra cost (Sec. III B)."""
        m, n, K, nb = (int(self.col_bitsize), int(self.system_bitsize),
                       int(self.n_blocks), int(self.block_bitsize))
        qs = bb.split(system)
        q = qs[n - m - 1]
        bottom = bb.join(qs[n - m:], dtype=QUInt(m))

        r_in: Dict[str, SoquetT] = {'system': bottom, 'target': q, 'phase_gradient': phase_grad}
        if K > 1:
            r_in['block'] = bb.join(gblock, dtype=QUInt(nb))
        r_out = bb.add_d(self.mux_rotation(K), **r_in)
        q, bottom, phase_grad = r_out['target'], r_out['system'], r_out['phase_gradient']
        if K > 1:
            gblock = bb.split(r_out['block'])

        base_block = bb.join(np.concatenate([gblock, [q]]), dtype=QUInt(nb + 1))
        u_out = bb.add_d(self.full_interferometer(2 * K), block=base_block,
                         system=bottom, phase_gradient=phase_grad)
        bottom, phase_grad = u_out['system'], u_out['phase_gradient']
        arr = bb.split(u_out['block'])
        gblock, q = arr[:nb], arr[nb]

        qs[n - m - 1] = q
        qs[n - m:] = bb.split(bottom)
        return bb.join(qs, dtype=QUInt(n)), phase_grad, gblock

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.n_blocks, self.n_rows, self.n_cols, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot decompose data-free symbolic {self}")
        has_block = self.has_block
        nb_bits = int(self.block_bitsize)
        phase_grad = soqs['phase_gradient']
        system = soqs['system']
        gblock = bb.split(soqs['block']) if has_block else np.array([], dtype=object)
        d, M, m, n, K = (int(self.d), int(self.n_cols), int(self.col_bitsize),
                         int(self.system_bitsize), int(self.n_blocks))

        def finish(system, gblock):
            out: Dict[str, SoquetT] = {'system': system, 'phase_gradient': phase_grad}
            if has_block:
                out['block'] = bb.join(gblock, dtype=QUInt(nb_bits))
            return out

        # Initial V_0 : a full M x M interferometer (K blocks) on the bottom m qubits.
        qs = bb.split(system)
        v_in: Dict[str, SoquetT] = {'system': bb.join(qs[n - m:], dtype=QUInt(m)), 'phase_gradient': phase_grad}
        if has_block:
            v_in['block'] = bb.join(gblock, dtype=QUInt(nb_bits))
        v_out = bb.add_d(self.full_interferometer(K), **v_in)
        qs[n - m:] = bb.split(v_out['system'])
        phase_grad = v_out['phase_gradient']
        if has_block:
            gblock = bb.split(v_out['block'])
        system = bb.join(qs, dtype=QUInt(n))
        if d == 1:
            return finish(system, gblock)

        for i in range(d - 1):
            system, phase_grad, gblock = self._apply_level(bb, system, phase_grad, gblock)
            if i < d - 2:
                system = bb.add(self._shift(-M), x=system)
        if d > 2:
            system = bb.add(self._shift((d - 2) * M), x=system)  # restore the net shift
        return finish(system, gblock)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        if is_symbolic(self.n_blocks, self.n_rows, self.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self}")
        d, M, K = int(self.d), int(self.n_cols), int(self.n_blocks)
        ret: "Counter[Bloq]" = Counter()
        ret[self.full_interferometer(K)] += 1              # initial V_0
        if d == 1:
            return ret
        ret[self.full_interferometer(2 * K)] += d - 1      # (d-1) merged 2K-block U's
        ret[self.mux_rotation(K)] += d - 1
        if d > 2:
            ret[self._shift(-M)] += d - 2
            ret[self._shift((d - 2) * M)] += 1
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBlockInterferometerIsometrySynthesisQROAM(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledBlockInterferometerIsometrySynthesisQROAM(GateWithRegisters):
    """Singly-controlled :class:`BlockInterferometerIsometrySynthesisQROAM`.

    The control reaches only the phase-gradient adds inside each sub-bloq (the cheap
    ``_Controlled*`` variants); QROAM loads/uncomputes and the ``AddK`` window shifts stay
    uncontrolled (the shifts net to identity), so the controlled cost ~ the bare cost.
    """

    inner: "BlockInterferometerIsometrySynthesisQROAM"

    @property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        if is_symbolic(self.inner.n_blocks, self.inner.n_rows, self.inner.n_cols):
            raise DecomposeTypeError(f"cannot enumerate layers for symbolic {self.inner}")
        inner = self.inner
        d, M, K = int(inner.d), int(inner.n_cols), int(inner.n_blocks)
        CI = _ControlledBlockUnitaryInterferometerSynthesisQROAM
        ret: "Counter[Bloq]" = Counter()
        ret[CI(inner.full_interferometer(K))] += 1         # initial V_0
        if d == 1:
            return ret
        ret[CI(inner.full_interferometer(2 * K))] += d - 1
        ret[_ControlledMuxRotationQROAM(inner.mux_rotation(K))] += d - 1
        if d > 2:
            ret[inner._shift(-M)] += d - 2
            ret[inner._shift((d - 2) * M)] += 1
        return ret
