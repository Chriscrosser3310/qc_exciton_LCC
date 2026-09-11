r"""P-17 -- table-defined reversible relabeling.

A classical permutation family :math:`\rho_u` is applied to a label register while
the register that held the old label finishes clean.  The circuit copies the old
label with CNOTs, adds the forward difference
:math:`d_u(x)=\rho_u(x)\oplus x` from a lookup, XORs the two registers, and clears
the old register with a clean uncompute of the inverse difference
:math:`d_u^{-1}`.  Both register XORs are Clifford, so the Toffoli cost is one
P-14 compute plus one P-14 clean uncompute.  Reversing the circuit exchanges the
two tables, so the two directions need not cost the same.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, Register, Signature

try:
    from .qroam import RangeSafeQROAM, Region, address_bitsize, normalize_region
except ImportError:  # pragma: no cover - direct execution
    from qroam import RangeSafeQROAM, Region, address_bitsize, normalize_region

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class ReversibleRelabel(Bloq):
    r"""``|u>|x> -> |u>|rho_u(x)>`` by difference lookup and inverse uncompute.

    ``region`` is the Cartesian live region of the difference table, including any
    family axes; its last axis is the label axis of width ``label_bits``.  Forward
    and inverse share the same region because the two difference tables have the
    same shape.
    """

    region: Region = attrs.field(converter=normalize_region)
    label_bits: int
    forward: bool = True

    @cached_property
    def children(self):
        return (RangeSafeQROAM(region=self.region, word=self.label_bits, compute=True),
                RangeSafeQROAM(region=self.region, word=self.label_bits, compute=False))

    @cached_property
    def toffolis(self) -> int:
        compute, uncompute = self.children
        return compute.toffolis + uncompute.toffolis

    @cached_property
    def clean_ancillas(self) -> int:
        compute, uncompute = self.children
        return max(compute.clean_ancillas, uncompute.clean_ancillas)

    @cached_property
    def signature(self) -> Signature:
        registers = [
            Register(f"family{index}", QAny(max(1, address_bitsize(high))))
            for index, (_, high) in enumerate(self.region[:-1])
        ]
        registers.append(Register("label", QAny(max(1, int(self.label_bits)))))
        registers.append(Register("output", QAny(max(1, int(self.label_bits)))))
        return Signature(registers)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        compute, uncompute = self.children
        return {compute: 1, uncompute: 1}
