r"""P-20 -- block-boundary split and merge.

A densely packed label :math:`x=t_a+r` splits into its block index :math:`a` and
its within-block index :math:`r`.  The circuit finds :math:`a` with a coherent
balanced binary search and then removes the block's starting offset.  At search
depth :math:`q` the :math:`q` bits already written form a prefix that addresses a
:math:`2^q`-entry threshold table; one P-14 load, one P-19 comparison, and one
P-14 unload write the next bit of :math:`a`.  After :math:`h=\lceil\log_2K\rceil`
levels the block register holds :math:`a`, and one further load/subtract/unload of
:math:`t_a` leaves :math:`r` in the low :math:`c` qubits with the high :math:`h`
qubits clean.  Merge reverses the construction, so the two directions cost the
same.

The loaded data are the :math:`2^h-1<2K` threshold words and the :math:`K` start
words -- fewer than :math:`3K` in total, and never a function of
:math:`T=\sum_a B_a`.  If every block already occupies one aligned
:math:`B`-slot, both directions reduce to Clifford wire relabeling.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, Register, Signature

try:
    from .arithmetic import RippleCompare, ModularAddSubtract
    from .qroam import RangeSafeQROAM
except ImportError:  # pragma: no cover - direct execution
    from arithmetic import RippleCompare, ModularAddSubtract
    from qroam import RangeSafeQROAM

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class BlockStaircase(Bloq):
    r"""``|t_a + r>|0^h> -> |r>|a>`` (split), or its inverse (merge).

    ``block_sizes`` lists the live sizes :math:`B_a` in packing order.  The common
    capacity is :math:`B=2^c` with :math:`c` the smallest exponent admitting every
    block, unless ``capacity_exponent`` overrides it.
    """

    block_sizes: Tuple[int, ...] = attrs.field(converter=lambda v: tuple(int(x) for x in v))
    capacity_exponent: int = -1
    split: bool = True

    @cached_property
    def n_blocks(self) -> int:
        return len(self.block_sizes)

    @cached_property
    def block_bits(self) -> int:
        """``c``, the smallest exponent with ``B = 2^c >= max_a B_a``.

        ``bit_length`` of the maximum itself is one too large when that maximum is
        an exact power of two, which is the common case once (BSE.8) has rounded
        every block up.
        """
        if self.capacity_exponent >= 0:
            return int(self.capacity_exponent)
        largest = max(self.block_sizes)
        return max(1, (largest - 1).bit_length()) if largest > 1 else 1

    @cached_property
    def index_bits(self) -> int:
        """``h = ceil(log2 K)``."""
        return max(1, (self.n_blocks - 1).bit_length())

    @cached_property
    def width(self) -> int:
        """``w = c + h``."""
        return self.block_bits + self.index_bits

    @cached_property
    def aligned(self) -> bool:
        """True when every block already fills one aligned capacity slot."""
        return all(size == 1 << self.block_bits for size in self.block_sizes)

    @cached_property
    def threshold_lookups(self):
        """One compute/uncompute pair per non-singleton search level."""
        pairs = []
        for depth in range(self.index_bits):
            entries = 1 << depth
            if entries <= 1:
                continue
            pairs.append((
                RangeSafeQROAM(region=((0, entries - 1),), word=self.width, compute=True),
                RangeSafeQROAM(region=((0, entries - 1),), word=self.width, compute=False),
            ))
        return tuple(pairs)

    @cached_property
    def offset_lookups(self):
        return (RangeSafeQROAM(region=((0, self.n_blocks - 1),), word=self.width, compute=True),
                RangeSafeQROAM(region=((0, self.n_blocks - 1),), word=self.width, compute=False))

    @cached_property
    def comparator(self) -> RippleCompare:
        return RippleCompare(width=self.width)

    @cached_property
    def subtractor(self) -> ModularAddSubtract:
        return ModularAddSubtract(width=self.width)

    @cached_property
    def boundary_search_toffolis(self) -> int:
        total = 0
        for depth in range(self.index_bits):
            entries = 1 << depth
            if entries > 1:
                compute, uncompute = [
                    RangeSafeQROAM(region=((0, entries - 1),), word=self.width, compute=flag)
                    for flag in (True, False)]
                total += compute.toffolis + uncompute.toffolis
            total += self.comparator.toffolis
        return total

    @cached_property
    def block_offset_toffolis(self) -> int:
        compute, uncompute = self.offset_lookups
        return compute.toffolis + self.subtractor.toffolis + uncompute.toffolis

    @cached_property
    def toffolis(self) -> int:
        if self.aligned:
            return 0
        return self.boundary_search_toffolis + self.block_offset_toffolis

    @cached_property
    def clean_ancillas(self) -> int:
        if self.aligned:
            return 0
        bounds = [self.width + 1]
        for compute, uncompute in self.threshold_lookups:
            bounds.extend([compute.clean_ancillas, uncompute.clean_ancillas])
        compute, uncompute = self.offset_lookups
        bounds.extend([compute.clean_ancillas, uncompute.clean_ancillas])
        return max(bounds)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("fused", QAny(self.width)),
                          Register("block", QAny(self.index_bits))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        if self.aligned:
            return dict(count)
        for compute, uncompute in self.threshold_lookups:
            count[compute] += 1
            count[uncompute] += 1
        count[self.comparator] += self.index_bits
        compute, uncompute = self.offset_lookups
        count[compute] += 1
        count[self.subtractor] += 1
        count[uncompute] += 1
        return dict(count)
