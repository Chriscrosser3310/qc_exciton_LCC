r"""P-08 -- indexed load-all state preparation.

The load-all option trades a wider loaded word for a single data lookup.  One
range-safe P-14 call loads the complete angle bank of the addressed block: all
$P_{\max}-1$ magnitude words and all $P_{\max}$ phase words, giving
$W_{\max}=(2P_{\max}-1)b$ bits.  Controlled-swap networks then route the word
selected at each tree level into one active $b$-qubit slot, P-05 applies the
$p_{\max}+1$ rotations, the networks reverse, and one clean uncompute removes the
bank.  Routing costs $2b(2P_{\max}-p_{\max}-2)$ Toffolis and no ancillas.

Both directions cost the same, because neither performs per-layer lookup erasure:
the inverse surrounds the reversed preparation with the same load and unload.

The BSE exchange term uses this child in the *physical* basis.  Its target is a
normalized flagged completion whose good amplitudes are the column entries over
the column scale; the remaining norm sits in a predefined failure state, so the
tree holds $N_s+1$ live amplitudes rather than $N_s$.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, Register, Signature
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

try:
    from .qroam import RangeSafeQROAM, Region, address_bitsize, normalize_region
except ImportError:  # pragma: no cover - direct execution
    from qroam import RangeSafeQROAM, Region, address_bitsize, normalize_region

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def load_all_routing_toffolis(tree_capacity: int, phase_bits: int) -> int:
    """(P8.19): ``2b(2P - p - 2)``."""
    p = int(tree_capacity).bit_length() - 1
    return 2 * int(phase_bits) * (2 * int(tree_capacity) - p - 2)


@attrs.frozen
class IndexedLoadAllStatePreparation(Bloq):
    r"""Indexed load-all P-08, in either direction.

    ``block_region`` is the live block address region $R_B^I$; ``live_amplitudes``
    is the number of live entries of one prepared state, already including the
    failure component when the target is a flagged completion.
    """

    block_region: Region = attrs.field(converter=normalize_region)
    live_amplitudes: int
    phase_bits: int = 32
    ancilla_budget: Optional[int] = None

    @cached_property
    def tree_bits(self) -> int:
        """``p_max`` with ``2^{p_max} >= live amplitudes``."""
        return max(1, (max(1, int(self.live_amplitudes)) - 1).bit_length())

    @cached_property
    def tree_capacity(self) -> int:
        """``P_max``."""
        return 1 << self.tree_bits

    @cached_property
    def bank_bits(self) -> int:
        """``W_max = (2 P_max - 1) b``."""
        return (2 * self.tree_capacity - 1) * self.phase_bits

    @cached_property
    def lookups(self):
        return (RangeSafeQROAM(region=self.block_region, word=self.bank_bits, compute=True,
                               ancilla_budget=self.ancilla_budget),
                RangeSafeQROAM(region=self.block_region, word=self.bank_bits, compute=False,
                               ancilla_budget=self.ancilla_budget))

    @cached_property
    def routing_toffolis(self) -> int:
        return load_all_routing_toffolis(self.tree_capacity, self.phase_bits)

    @cached_property
    def rotation(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bits, self.phase_bits)

    @cached_property
    def toffolis(self) -> int:
        """(P8.20); equal in both directions."""
        compute, uncompute = self.lookups
        return (compute.toffolis + self.routing_toffolis
                + (self.tree_bits + 1) * (self.phase_bits - 2) + uncompute.toffolis)

    @cached_property
    def clean_ancillas(self) -> int:
        """(P8.21)."""
        compute, uncompute = self.lookups
        b = self.phase_bits
        return b + max(compute.clean_ancillas + b - 1, uncompute.clean_ancillas)

    @cached_property
    def signature(self) -> Signature:
        registers = [Register(f"block{index}", QAny(max(1, address_bitsize(high))))
                     for index, (_, high) in enumerate(self.block_region)]
        registers.append(Register("target", QAny(self.tree_bits)))
        registers.append(Register("phase_gradient", QAny(self.phase_bits)))
        return Signature(registers)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        compute, uncompute = self.lookups
        count: "Counter[Bloq]" = Counter()
        count[compute] += 1
        count[CSwap(self.phase_bits)] += self.routing_toffolis // self.phase_bits
        count[self.rotation] += self.tree_bits + 1
        count[uncompute] += 1
        return dict(count)
