r"""P-07 -- column-by-column block-isometry synthesis, fused layout.

Iten's annihilator clears an isometry one column at a time.  For column $c$ it
first removes the phases of the live entries, then combines them pairwise through
a binary tree of real Givens rotations, leaving the column equal to a basis
state; processing columns upward preserves every column already cleared.  The
forward synthesis is that circuit reversed.

The fused layout keeps each block's own power-of-two row capacity $P_r$ and packs
the blocks adjacently in descending $(P_r, M_r)$ order.  Descending powers make
$P_r$ divide each block's offset, so a block's physical low address bits equal its
local low bits and one aligned tree serves every block at once.  At level $q$ the
tree pairs rows $x=h2^{q+1}+\ell_{c,q}$ and $y=x+2^q$; removing the target bit and
packing the $q$ lower bits as high bits turns the required condition into one
contiguous lookup window in the full remaining-bit address register. Its Select
checks the lower bits as part of that lookup, including fixed zero bits.
Because the fused tree is aligned, its pair layout is free --- unlike the indexed
tree, whose layout leaf P-07 leaves open.

Both directions use one fixed rounded reference sequence. After every phase or
Givens lookup, all angle and unused QROAM words are X-measured and recycled.
The classical frame is D=diag(d_j); a reference pair (x,y) loads d_x*d_y*theta.
One final exact sign correction covers the whole binary address register. The
lower-bit condition is retained for column zero as well. Costs are symmetric.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import List, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QBit, Register, Signature
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.basic_gates.x_basis import MeasureX
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

try:
    from .qroam import RangeSafeQROAM
except ImportError:  # pragma: no cover - direct execution
    from qroam import RangeSafeQROAM

try:
    from ..isometry_sign_frame_bloqs import IsometrySignCorrection
except ImportError:
    from isometry_sign_frame_bloqs import IsometrySignCorrection

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def power_of_two_at_least(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


@attrs.frozen
class FusedColumnIsometry(Bloq):
    r"""One direction of the fused P-07 synthesis of ``(+)_r V_r``.

    ``blocks`` lists ``(N_r, M_r)`` --- live rows and columns of each isometry.
    ``row_capacities`` declares the power-of-two row extent $P_r$ of each block;
    left unset it defaults to the smallest power of two admitting $N_r$.  The BSE
    composition supplies (BSE.8)'s common extent instead, so that one adjacent
    P-18 call shares the layout.
    """

    blocks: Tuple[Tuple[int, int], ...] = attrs.field(
        converter=lambda v: tuple((int(a), int(b)) for a, b in v))
    row_capacities: Tuple[int, ...] = attrs.field(
        default=None, converter=lambda v: tuple(int(x) for x in v) if v is not None else None)
    phase_bits: int = 32
    forward: bool = True
    #: Names the data family.  Two factors with identical shapes but different
    #: tables must stay distinct children, because their lookups cannot be shared.
    tag: str = ""
    #: None selects the Toffoli optimum. Explicit values are logarithms of
    #: one flattened swap-bank size, not a separate lower-bit control.
    rotation_log_block_size: int | None = None
    phase_log_block_size: int | None = None
    sign_log_block_size: int | None = None
    external_control: bool = False

    @cached_property
    def layout(self) -> Tuple[Tuple[int, int, int], ...]:
        """``(P_r, N_r, M_r)`` in the declared descending lexicographic order."""
        capacities = self.row_capacities
        if capacities is None:
            capacities = tuple(power_of_two_at_least(rows) for rows, _ in self.blocks)
        if len(capacities) != len(self.blocks):
            raise ValueError("one row capacity is required for each block")
        entries = []
        for (rows, columns), capacity in zip(self.blocks, capacities):
            capacity = int(capacity)
            if capacity < 1 or capacity & (capacity - 1):
                raise ValueError(f"row capacity {capacity} is not a power of two")
            if capacity < rows:
                raise ValueError(f"row capacity {capacity} cannot hold {rows} rows")
            entries.append((capacity, int(rows), int(columns)))
        return tuple(sorted(entries, key=lambda item: (item[0], item[2]), reverse=True))

    @cached_property
    def max_columns(self) -> int:
        """``M_max``."""
        return max(columns for _, _, columns in self.layout)

    @cached_property
    def fused_extent(self) -> int:
        """``L_P = sum_r P_r``."""
        return sum(capacity for capacity, _, _ in self.layout)

    @cached_property
    def register_bits(self) -> int:
        """``n_F``."""
        return max(1, (self.fused_extent - 1).bit_length())

    def _levels(self, column: int) -> List[Tuple[int, int]]:
        """``(q, H_q)`` for every live level of this column, per (P7.5a)-(P7.15)."""
        result = []
        for q in range(self.register_bits):
            active = sum(capacity for capacity, _, _ in self.layout
                         if capacity.bit_length() - 1 > q)
            if active == 0:
                break
            floor_pair = column >> (q + 1)
            beta = (column >> q) & 1
            h_min = floor_pair + beta
            live = any(column < columns and q < capacity.bit_length() - 1
                       and h_min < capacity >> (q + 1)
                       for capacity, _, columns in self.layout)
            if live:
                result.append((q, active >> (q + 1)))
        return result

    @cached_property
    def levels(self) -> Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...]:
        return tuple((column, tuple(self._levels(column)))
                     for column in range(self.max_columns))

    @cached_property
    def phase_lookups(self) -> Tuple[RangeSafeQROAM, RangeSafeQROAM]:
        """Column phase children; the old immediate-uncompute child is diagnostic only."""
        region = ((0, self.fused_extent - 1),)
        tradeoffs = (None if self.phase_log_block_size is None
                     else (1 << self.phase_log_block_size,))
        return (RangeSafeQROAM(region=region, word=self.phase_bits, compute=True,
                              address_widths=(self.register_bits,), tradeoffs=tradeoffs,
                              external_control=self.external_control),
                RangeSafeQROAM(region=region, word=self.phase_bits, compute=False,
                              address_widths=(self.register_bits,),
                              external_control=self.external_control))

    def _givens_lookups(self, column: int, q: int, height: int):
        """One literal P7.16/P7.17 lookup on all bits except the target.

        Pack the q lower system bits above the upper address. A nonmatching
        lower pattern selects zero. The physical address width must retain
        leading zero bits, particularly on column zero and singleton windows.
        """
        high_bits = self.register_bits - q - 1
        if column < 0 or q < 0 or high_bits < 0 or not 1 <= height <= 1 << high_bits:
            raise ValueError("invalid aligned isometry lookup geometry")
        low = (column % (1 << q)) << high_bits
        region = ((low, low + height - 1),)
        tradeoffs = (None if self.rotation_log_block_size is None
                     else (1 << self.rotation_log_block_size,))
        kwargs = dict(address_widths=(self.register_bits - 1,),
                      max_log_block_sizes=(high_bits,), external_control=self.external_control)
        return (RangeSafeQROAM(region=region, word=self.phase_bits, compute=True,
                              tradeoffs=tradeoffs, **kwargs),
                RangeSafeQROAM(region=region, word=self.phase_bits, compute=False,
                              **kwargs))

    @cached_property
    def rotation(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bits, self.phase_bits)

    @cached_property
    def n_levels(self) -> int:
        """``L_iso``: one phase step per column plus every live Givens level."""
        return self.max_columns + sum(len(levels) for _, levels in self.levels)

    @cached_property
    def final_sign_correction(self) -> IsometrySignCorrection:
        return IsometrySignCorrection(
            self.register_bits,
            tradeoff=None if self.sign_log_block_size is None else 1 << self.sign_log_block_size,
            external_control=self.external_control)

    @cached_property
    def toffolis(self) -> int:
        """P7.20/P7.21: all computes and rotations, then one full-address fixup."""
        rotation = self.phase_bits - 2
        compute, _ = self.phase_lookups
        total = self.max_columns * (compute.toffolis + rotation)
        for column, levels in self.levels:
            for q, height in levels:
                load, _ = self._givens_lookups(column, q, height)
                total += load.toffolis + rotation
        return total + self.final_sign_correction.toffolis

    @cached_property
    def clean_ancillas(self) -> int:
        """Recycle each lookup bank; its full address width includes the low bits."""
        b = self.phase_bits
        compute, _ = self.phase_lookups
        bounds = [compute.clean_ancillas + b - 1,
                  self.final_sign_correction.clean_ancillas]
        for column, levels in self.levels:
            for q, height in levels:
                load, _ = self._givens_lookups(column, q, height)
                bounds.append(load.clean_ancillas + b - 1)
        return b + max(bounds)

    def adjoint(self):
        return attrs.evolve(self, forward=not self.forward)

    def reference_from_isometries(self, isometries):
        """Compile concrete fixed gates separately from the data-free cost graph."""
        try:
            from ..isometry_sign_frame import compile_isometry_reference
        except ImportError:
            from isometry_sign_frame import compile_isometry_reference
        if tuple(v.shape for v in isometries) != self.blocks:
            raise ValueError("matrix shapes do not match the resource specification")
        return compile_isometry_reference(isometries, bits=self.phase_bits,
                                           row_capacities=self.row_capacities)

    @cached_property
    def rotation_error_bound(self) -> float:
        """(P7.25) without the gradient-state allocation: ``2 pi L_iso / 2^b``."""
        return 2 * 3.141592653589793 * self.n_levels / float(1 << self.phase_bits)

    @cached_property
    def signature(self) -> Signature:
        registers = [Register("data", QAny(self.register_bits)),
                     Register("phase_gradient", QAny(self.phase_bits))]
        if self.external_control:
            registers.insert(0, Register("ctrl", QBit()))
        return Signature(registers)

    def get_ctrl_system(self, ctrl_spec):
        if self.external_control:
            return super().get_ctrl_system(ctrl_spec)
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=attrs.evolve(self, external_control=True), ctrl_reg_name="ctrl")

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        compute, _ = self.phase_lookups
        loads = [compute] * self.max_columns
        for column, levels in self.levels:
            for q, height in levels:
                load, _ = self._givens_lookups(column, q, height)
                loads.append(load)
        for load in loads:
            count[load] += 1
            count[self.rotation] += 1
            words = 1
            for tradeoff in load.schedule:
                words *= tradeoff
            count[MeasureX()] += self.phase_bits * words
        count[self.final_sign_correction] += 1
        return dict(count)
