"""Resource leaves for the isometry sign frame (dense execution in its sibling).

No dynamic-table hardware decomposition is claimed. In particular, a final
sign correction is a diagonal phase-fixup circuit, not a fresh angle lookup
followed by another deferred measurement.
"""

from functools import cached_property

import attrs
from qualtran import Bloq, QAny, QBit, Register, Signature

try:
    from .primitives.qroam import RangeSafeQROAM
except ImportError:
    from primitives.qroam import RangeSafeQROAM


@attrs.frozen
class IsometrySignCorrection(Bloq):
    """One P-04 phase fixup over the full binary address register.

    This conservative domain includes all junk-induced signs, even outside
    live isometry rows. The phase-fixup child includes its own exact cleanup;
    it is not another open sign frame. Classical sign bits carry no rounding
    error. Leaving ``tradeoff`` unset minimizes this child's Toffolis.
    """

    address_bits: int
    tradeoff: int | None = None
    external_control: bool = False

    @cached_property
    def lookup(self):
        return RangeSafeQROAM(region=((0, (1 << self.address_bits) - 1),), word=1,
                              compute=False,
                              tradeoffs=None if self.tradeoff is None else (self.tradeoff,),
                              address_widths=(self.address_bits,),
                              external_control=self.external_control)

    @cached_property
    def signature(self):
        registers = [Register("data", QAny(max(1, self.address_bits)))]
        if self.external_control:
            registers.insert(0, Register("ctrl", QBit()))
        return Signature(registers)

    @cached_property
    def toffolis(self):
        return self.lookup.toffolis

    @cached_property
    def clean_ancillas(self):
        return self.lookup.clean_ancillas

    def build_call_graph(self, ssa):
        return {self.lookup: 1}

    def adjoint(self):
        return self
