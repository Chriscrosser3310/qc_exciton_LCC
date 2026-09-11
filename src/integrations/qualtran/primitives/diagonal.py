r"""P-09 -- diagonal block encoding.

P-09 is a load-rotate-unload circuit.  One range-safe lookup writes the $b$-bit
angle of the addressed diagonal entry, one P-05 call rotates the signal qubit by
it, and one clean uncompute removes the word.  Outside the declared live region
the lookup writes zero, so the circuit acts as the identity there.

The indexed variant addresses a Cartesian tuple through P-14; the fused variant
addresses one contiguous interval through P-02/P-03/P-04.  Berry's combined
controlled-lookup and sign-fixup saving of three Toffolis is booked against the
uncompute, matching the final-diagonal convention in P-16.

A real diagonal is made involutory for free by a Clifford $Z$ on the signal
qubit, since $ZR_y(\theta)Z=R_y(\theta)^\dagger$.  That $Z$ changes neither the
Toffoli count nor the ancilla bound, so it appears in the signature and not in
the cost.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import ZGate
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

try:
    from .qroam import RangeSafeQROAM, Region, address_bitsize, normalize_region
except ImportError:  # pragma: no cover - direct execution
    from qroam import RangeSafeQROAM, Region, address_bitsize, normalize_region

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class DiagonalBlockEncoding(Bloq):
    r"""``(<0|_a (x) I) O_D (|0>_a (x) I) = D/alpha + (I - Pi)``.

    ``region`` is the live diagonal region: a tuple of intervals for the indexed
    variant, one interval for the fused variant.  ``involutory`` adds the Clifford
    $Z$ that makes a real diagonal self-inverse.
    """

    region: Region = attrs.field(converter=normalize_region)
    phase_bits: int = 32
    involutory: bool = True
    ancilla_budget: Optional[int] = None

    @cached_property
    def lookups(self):
        return (RangeSafeQROAM(region=self.region, word=self.phase_bits, compute=True,
                               ancilla_budget=self.ancilla_budget),
                RangeSafeQROAM(region=self.region, word=self.phase_bits, compute=False,
                               adjustment=-3, ancilla_budget=self.ancilla_budget))

    @cached_property
    def rotation(self) -> AddIntoPhaseGrad:
        return AddIntoPhaseGrad(self.phase_bits, self.phase_bits)

    @cached_property
    def toffolis(self) -> int:
        compute, uncompute = self.lookups
        return compute.toffolis + (self.phase_bits - 2) + uncompute.toffolis

    @cached_property
    def clean_ancillas(self) -> int:
        """(P9.9): signal qubit, gradient state, and the wider child bound."""
        compute, uncompute = self.lookups
        return 1 + self.phase_bits + max(compute.clean_ancillas + self.phase_bits - 1,
                                         uncompute.clean_ancillas)

    @cached_property
    def signature(self) -> Signature:
        registers = [Register(f"address{index}", QAny(max(1, address_bitsize(high))))
                     for index, (_, high) in enumerate(self.region)]
        registers.append(Register("signal", QBit()))
        registers.append(Register("phase_gradient", QAny(self.phase_bits)))
        return Signature(registers)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        compute, uncompute = self.lookups
        count: "Counter[Bloq]" = Counter()
        count[compute] += 1
        count[self.rotation] += 1
        count[uncompute] += 1
        if self.involutory:
            count[ZGate()] += 1
        return dict(count)
