r"""P-06/P-15/P-16 -- Berry unitary synthesis, in its fused block layout.

Berry's parent circuit factorizes an even-padded unitary into
:math:`E_r` Givens layers and one final diagonal.  Each layer is block diagonal
with 2-by-2 unitaries drawn from the two alternating pairings, and each 2-by-2
block decomposes into two Hadamards and two programmable phase rotations.  Two
P-05 calls apply those rotations; one range-safe lookup supplies their angles.

P-16 is the fused option: the padded blocks occupy consecutive intervals of one
data register in descending size order, so there is no separate block address.
Layer :math:`t` therefore loads angles over the interval
:math:`[0, A_t-1]`, where :math:`A_t=\sum_{r:E_r>t}E_r` counts only the blocks
still active at that depth, and a global shift moves the window between layers.
The final diagonal loads over the whole live extent :math:`[0, L_E-1]`,
applies one rotation, and unloads.

Berry's two source-specific savings -- one Toffoli per layer for the required
layer control, and three for the combined final sign fixup -- are booked against
the lookups that would otherwise pay them.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, Register, Signature
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

try:
    from .qroam import RangeSafeQROAM
except ImportError:  # pragma: no cover - direct execution
    from qroam import RangeSafeQROAM

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def even_padded_sizes(live_sizes: Sequence[int]) -> Tuple[int, ...]:
    """(P61516.1): even-pad every block, embed a singleton as ``U_r (+) 1``, sort down.

    A block of dimension zero is dropped rather than padded.  (BSE.6) gives a
    missing irrep copy dimension zero, so there is no matrix to synthesize; padding
    it to two would charge a 2-by-2 synthesis for a block that does not exist.
    """
    padded = []
    for size in live_sizes:
        value = int(size)
        if value <= 0:
            continue
        padded.append(2 if value == 1 else value + (value % 2))
    return tuple(sorted(padded, reverse=True))


def shift_toffolis(bits: int) -> int:
    """``C_shift(n) = max(n-2, 0)``, Berry's classical decrement/increment leaf."""
    return max(int(bits) - 2, 0)


@attrs.frozen
class FusedBlockUnitarySynthesis(Bloq):
    r"""P-16: one synthesis of ``(+)_r (U_r (+) I)`` on a single fused register.

    ``live_sizes`` are the unpadded block dimensions :math:`N_r`.  ``packed``
    selects (P61516.10) -- both angles of a layer in one :math:`2b`-bit word --
    against (P61516.10b), which loads them separately to halve the live angle
    register.
    """

    live_sizes: Tuple[int, ...] = attrs.field(converter=lambda v: tuple(int(x) for x in v))
    phase_bits: int = 32
    packed: bool = True

    @cached_property
    def padded_sizes(self) -> Tuple[int, ...]:
        return even_padded_sizes(self.live_sizes)

    @cached_property
    def max_extent(self) -> int:
        """``E_max``, the number of Givens layers."""
        return self.padded_sizes[0]

    @cached_property
    def live_extent(self) -> int:
        """``L_E = sum_r E_r``."""
        return sum(self.padded_sizes)

    @cached_property
    def register_bits(self) -> int:
        """``n_F`` with ``2^{n_F} >= L_E``."""
        return max(1, (self.live_extent - 1).bit_length())

    @cached_property
    def active_extents(self) -> Tuple[int, ...]:
        """``A_t`` for ``0 <= t < E_max``: the active *extent*, in modes."""
        return tuple(sum(size for size in self.padded_sizes if size > depth)
                     for depth in range(self.max_extent))

    @cached_property
    def layer_windows(self) -> Tuple[int, ...]:
        """The number of angle words layer ``t`` loads, which is ``A_t / 2``.

        A Givens layer applies 2-by-2 unitaries to *disjoint pairs* of modes, so an
        ``A_t``-mode layer holds ``A_t / 2`` of them, and (P61516.10) charges two
        rotations per layer rather than two per block.  One lookup multiplexed over
        the pair index therefore serves the whole layer, exactly as P-07's tree
        removes its target bit to give ``H_q = A_q / 2^{q+1}``.
        """
        return tuple(max(1, extent // 2) for extent in self.active_extents)

    @cached_property
    def layer_lookups(self) -> Tuple[Tuple[RangeSafeQROAM, ...], ...]:
        """Per layer, the lookup calls it issues.

        A packed layer issues one ``2b``-bit load; a split layer issues two
        ``b``-bit loads over the same region.  Berry's single layer-control
        saving is booked against the first call of each layer, so a split layer
        is charged that credit once, not twice.
        """
        word = 2 * self.phase_bits if self.packed else self.phase_bits
        layers = []
        for extent in self.layer_windows:
            region = ((0, extent - 1),)
            calls = [RangeSafeQROAM(region=region, word=word, compute=True, adjustment=-1)]
            for _ in range(self.loads_per_layer - 1):
                calls.append(RangeSafeQROAM(region=region, word=word, compute=True))
            layers.append(tuple(calls))
        return tuple(layers)

    @cached_property
    def diagonal_lookups(self) -> Tuple[RangeSafeQROAM, RangeSafeQROAM]:
        region = ((0, self.live_extent - 1),)
        return (RangeSafeQROAM(region=region, word=self.phase_bits, compute=True),
                RangeSafeQROAM(region=region, word=self.phase_bits, compute=False,
                               adjustment=-3))

    @cached_property
    def rotation(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bits, self.phase_bits)

    @cached_property
    def loads_per_layer(self) -> int:
        return 1 if self.packed else 2

    @cached_property
    def toffolis(self) -> int:
        rotation = self.phase_bits - 2
        total = 0
        for calls in self.layer_lookups:
            total += sum(call.toffolis for call in calls) + 2 * rotation
        total += (self.max_extent - 1) * shift_toffolis(self.register_bits)
        compute, uncompute = self.diagonal_lookups
        total += compute.toffolis + rotation + uncompute.toffolis
        return total

    @cached_property
    def clean_ancillas(self) -> int:
        """(P61516.13), the no-overlap bound."""
        b = self.phase_bits
        bounds = [call.clean_ancillas + b - 1
                  for calls in self.layer_lookups for call in calls]
        compute, uncompute = self.diagonal_lookups
        bounds.append(compute.clean_ancillas + b - 1)
        bounds.append(uncompute.clean_ancillas)
        return b + max(bounds)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("data", QAny(self.register_bits)),
                          Register("phase_gradient", QAny(self.phase_bits))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        for calls in self.layer_lookups:
            for call in calls:
                count[call] += 1
            count[self.rotation] += 2
        shift = shift_toffolis(self.register_bits)
        if shift and self.max_extent > 1:
            count[And()] += (self.max_extent - 1) * shift
        compute, uncompute = self.diagonal_lookups
        count[compute] += 1
        count[self.rotation] += 1
        count[uncompute] += 1
        return dict(count)
