r"""P-19 -- Cuccaro ripple-carry comparison and modular addition.

Cuccaro's majority/unmajority ripple gives both operations on ``w``-qubit words.
Dropping its final carry leaves addition modulo :math:`2^w` at :math:`2w-3`
Toffolis; keeping only that carry gives a comparison at :math:`2w-1`.  Reversing
either circuit costs the same, so subtraction and uncomparison need no separate
row.  One clean carry qubit serves both.  Both operand words are excluded from the
workspace count, and a classically initialized operand is prepared with Pauli
:math:`X` gates alone.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QBit, Register, Signature
from qualtran.bloqs.mcmt import And

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def modular_add_toffolis(width: int) -> int:
    """``C_modular-add/subtract(w) = 2w - 3``."""
    return max(0, 2 * int(width) - 3)


def compare_toffolis(width: int) -> int:
    """``C_compare(w) = 2w - 1``."""
    return max(0, 2 * int(width) - 1)


@attrs.frozen
class ModularAddSubtract(Bloq):
    """``|a>|b> -> |a>|a+b mod 2^w>``, or its reversal.  One clean carry qubit."""

    width: int

    @cached_property
    def toffolis(self) -> int:
        return modular_add_toffolis(self.width)

    @cached_property
    def clean_ancillas(self) -> int:
        return 1

    @cached_property
    def signature(self) -> Signature:
        w = max(1, int(self.width))
        return Signature([Register("a", QAny(w)), Register("b", QAny(w))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        if self.toffolis:
            count[And()] += self.toffolis
        return dict(count)


@attrs.frozen
class RippleCompare(Bloq):
    """``|a>|b>|z> -> |a>|b>|z XOR [a>=b]>``.  One clean carry qubit."""

    width: int

    @cached_property
    def toffolis(self) -> int:
        return compare_toffolis(self.width)

    @cached_property
    def clean_ancillas(self) -> int:
        return 1

    @cached_property
    def signature(self) -> Signature:
        w = max(1, int(self.width))
        return Signature([Register("a", QAny(w)), Register("b", QAny(w)),
                          Register("target", QBit())])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        if self.toffolis:
            count[And()] += self.toffolis
        return dict(count)


def ctrl_modular_add_toffolis(phase_bits: int) -> int:
    """``2b - 1``: one controlled modular add or subtract without carry."""
    return max(0, 2 * int(phase_bits) - 1)


@attrs.frozen
class CtrlModularAddNoCarry(Bloq):
    r"""Controlled ``b``-bit modular add or subtract without carry, at ``2b-1`` Toffolis.

    P-18's Bloch phase adds a shifted phase coordinate into the gradient state once
    per non-sign fold bit and subtracts it for the sign bit.  Litinski states the
    :math:`2b-1` count for that controlled carry-free addition; the sign choice is
    Clifford and does not change it.
    """

    phase_bits: int

    @cached_property
    def toffolis(self) -> int:
        return ctrl_modular_add_toffolis(self.phase_bits)

    @cached_property
    def clean_ancillas(self) -> int:
        return 0

    @cached_property
    def signature(self) -> Signature:
        b = max(1, int(self.phase_bits))
        return Signature([Register("ctrl", QBit()), Register("x", QAny(b)),
                          Register("phase_gradient", QAny(b))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        if self.toffolis:
            count[And()] += self.toffolis
        return dict(count)
