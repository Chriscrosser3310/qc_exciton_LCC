r"""P-12 -- eigendecomposition block encoding.

A Hermitian block family $A_r=V_rD_rV_r^\dagger$ becomes a block encoding by
running the eigenvector synthesis, then the diagonal block encoding of the
eigenvalues, then the inverse synthesis.  Nothing else is required: the three
children run in sequence and reuse their clean workspace, so the peak is their
maximum and the cost is their sum.  With matching forward and inverse schedules
the two syntheses cost the same, which is the form used here.

The fused variant places the padded blocks in adjacent intervals of one register,
so no block address enters any lookup.  Negating the stored eigenvalues supplies a
minus sign at no cost, which is how the Fock term's occupied sector is built.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QBit, Register, Signature

try:
    from .diagonal import DiagonalBlockEncoding
    from .unitary import FusedBlockUnitarySynthesis, even_padded_sizes
except ImportError:  # pragma: no cover - direct execution
    from diagonal import DiagonalBlockEncoding
    from unitary import FusedBlockUnitarySynthesis, even_padded_sizes

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class FusedEigendecompositionBlockEncoding(Bloq):
    r"""Fused P-12: ``V``, the diagonal encoding of ``D``, then ``V^dagger``."""

    block_sizes: Tuple[int, ...] = attrs.field(converter=lambda v: tuple(int(x) for x in v))
    phase_bits: int = 32
    involutory: bool = True

    @cached_property
    def padded_sizes(self) -> Tuple[int, ...]:
        return even_padded_sizes(self.block_sizes)

    @cached_property
    def live_extent(self) -> int:
        return sum(self.padded_sizes)

    @cached_property
    def synthesis(self) -> FusedBlockUnitarySynthesis:
        return FusedBlockUnitarySynthesis(live_sizes=self.block_sizes,
                                          phase_bits=self.phase_bits)

    @cached_property
    def diagonal(self) -> DiagonalBlockEncoding:
        return DiagonalBlockEncoding(region=((0, self.live_extent - 1),),
                                     phase_bits=self.phase_bits,
                                     involutory=self.involutory)

    @cached_property
    def toffolis(self) -> int:
        """(P12.5): two syntheses plus one diagonal."""
        return 2 * self.synthesis.toffolis + self.diagonal.toffolis

    @cached_property
    def clean_ancillas(self) -> int:
        return max(self.synthesis.clean_ancillas, self.diagonal.clean_ancillas)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("data", QAny(self.synthesis.register_bits)),
                          Register("signal", QBit()),
                          Register("phase_gradient", QAny(self.phase_bits))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        count[self.synthesis] += 2
        count[self.diagonal] += 1
        return dict(count)
