r"""P-13, P-02/P-03/P-04, and P-14 -- range-safe QROAM, exactly as specified.

This module charges the lookup costs written in ``context/primitives.md`` rather
than Qualtran's ``QROAMClean``.  The two differ: Qualtran's unary iteration is
*unrestricted*, so it omits the P-13 range-safety surcharge

.. math::  \delta_a = h_a + \operatorname{popcount}(q_a^-)
                      - \operatorname{popcount}(q_a^+),

and it flattens a Cartesian address tuple instead of chaining the Selects.  Both
choices change the count, and the walk operator's reflection populates the whole
address register, so the restricted form is the one this project needs.

The single-address entry (P-02/P-03/P-04) is the ``p = 1`` case of the
multi-address entry (P-14); one implementation therefore serves both.  For an
address tuple with live region :math:`R=\prod_i[l_i,r_i]`, tradeoffs
:math:`\Lambda_i`, and Select order :math:`\pi`,

.. math::
    C_{\rm compute} &= G + \Delta_\pi + b(K-1), \\
    C_{\rm uncompute} &= G' + \Delta'_{\pi'} + (K'-1),

with :math:`G=\prod_i g_i`, :math:`K=\prod_i\Lambda_i`, and
:math:`\Delta_\pi=\sum_t(\prod_{s<t}g_{\pi_s})\delta_{\pi_t}`.  The order rule is
nonincreasing :math:`\delta_i/(g_i-1)`.

Every Toffoli is emitted as a Qualtran ``And``, so ``toffoli_count`` reproduces
the formula exactly and a composite's total remains machine-checkable.
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import cached_property
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QBit, Register, Signature
from qualtran.bloqs.mcmt import And

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


Region = Tuple[Tuple[int, int], ...]


def address_bitsize(high: int) -> int:
    """Smallest ``n`` with ``high < 2**n``."""
    return int(high).bit_length()


def normalize_region(region: Sequence[Sequence[int]]) -> Region:
    """Accept ``[L1, L2, ...]`` or ``[(l1, r1), (l2, r2), ...]`` and return pairs."""
    out = []
    for item in region:
        if isinstance(item, (tuple, list)):
            low, high = int(item[0]), int(item[1])
        else:
            low, high = 0, int(item) - 1
        if low < 0 or high < low:
            raise ValueError(f"empty or negative interval {(low, high)}")
        out.append((low, high))
    return tuple(out)


def region_size(region: Region) -> int:
    total = 1
    for low, high in region:
        total *= high - low + 1
    return total


def _axis_terms(low: int, high: int, tradeoff: int, bits: Optional[int] = None) -> Tuple[int, int]:
    """``(g_i, delta_i)`` of (P14.1)-(P14.2) for one address axis."""
    bits = address_bitsize(high) if bits is None else bits
    if tradeoff > 1 << bits:
        raise ValueError("tradeoff exceeds the address capacity")
    minus, plus = low // tradeoff, high // tradeoff
    g = plus - minus + 1
    h = bits - tradeoff.bit_length() + 1
    delta = h + bin(minus).count("1") - bin(plus).count("1")
    return g, delta


def _order_by_rule(terms: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    """Nonincreasing ``delta_i / (g_i - 1)``; ties keep the declared axis order."""
    def key(index: int):
        g, delta = terms[index]
        return (0, 0) if g <= 1 else (-delta / (g - 1), index)
    return tuple(sorted(range(len(terms)), key=key))


def lookup_toffolis(region: Region, word: int, tradeoffs: Sequence[int],
                    compute: bool, order: Optional[Sequence[int]] = None,
                    address_widths: Optional[Sequence[int]] = None) -> int:
    """(P14.5)-(P14.6): ``G + Delta_pi + b(K-1)`` or ``G' + Delta' + (K'-1)``."""
    if region_size(region) <= 1 and (address_widths is None or not any(address_widths)):
        return 0
    widths = (tuple(address_bitsize(high) for _, high in region)
              if address_widths is None else tuple(address_widths))
    terms = [_axis_terms(low, high, t, bits)
             for (low, high), t, bits in zip(region, tradeoffs, widths)]
    sequence = tuple(order) if order is not None else _order_by_rule(terms)
    total_g, delta_sum, prefix = 1, 0, 1
    for axis in sequence:
        g, delta = terms[axis]
        delta_sum += prefix * delta
        prefix *= g
        total_g *= g
    swaps = 1
    for t in tradeoffs:
        swaps *= int(t)
    routed = int(word) * (swaps - 1) if compute else (swaps - 1)
    return total_g + delta_sum + routed


def lookup_ancillas(region: Region, word: int, tradeoffs: Sequence[int],
                    compute: bool, address_widths: Optional[Sequence[int]] = None) -> int:
    """P-14 clean workspace: ``b(K-1)+H`` plus the output, or ``K'+H'``."""
    if region_size(region) <= 1 and (address_widths is None or not any(address_widths)):
        return 0
    swaps, height = 1, 0
    widths = (tuple(address_bitsize(high) for _, high in region)
              if address_widths is None else tuple(address_widths))
    for bits, t in zip(widths, tradeoffs):
        swaps *= int(t)
        height += bits - int(t).bit_length() + 1
    if compute:
        return int(word) * (swaps - 1) + height + int(word)
    return swaps + height


def _power_of_two_choices(high: int) -> Tuple[int, ...]:
    bits = address_bitsize(high)
    return tuple(1 << e for e in range(bits + 1))


def best_schedule(region: Region, word: int, compute: bool,
                  ancilla_budget: Optional[int] = None,
                  address_widths: Optional[Sequence[int]] = None,
                  max_log_block_sizes: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    """The Toffoli-minimizing tradeoff tuple, by exhaustive search over powers of two.

    A declared ``ancilla_budget`` restricts the search to schedules whose clean
    workspace fits, which is P-14's ``A_max`` condition. Explicit-width callers
    raise if none fits. Implicit-width callers retain their legacy all-ones
    fallback for compatibility.
    """
    region = normalize_region(region)
    if region_size(region) <= 1 and (address_widths is None or not any(address_widths)):
        return tuple(1 for _ in region)
    widths = (tuple(address_bitsize(high) for _, high in region)
              if address_widths is None else tuple(address_widths))
    caps = widths if max_log_block_sizes is None else tuple(max_log_block_sizes)
    grids = [tuple(1 << e for e in range(cap + 1)) for cap in caps]
    best, best_cost = None, None
    for candidate in itertools.product(*grids):
        if ancilla_budget is not None and lookup_ancillas(
                region, word, candidate, compute, address_widths) > ancilla_budget:
            continue
        cost = lookup_toffolis(region, word, candidate, compute, address_widths=address_widths)
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    if best is None:
        if address_widths is not None:
            raise ValueError("no lookup schedule fits the declared ancilla budget")
        best = tuple(1 for _ in region)
    return best


@attrs.frozen
class RangeSafeQROAM(Bloq):
    r"""One range-safe QROAM call: P-14 for ``p>1`` addresses, P-03/P-04 for one.

    ``region`` lists the live interval of every address axis, ``word`` is the
    output width :math:`b`, and ``compute`` selects the clean compute (P-03) or the
    measurement-based clean uncompute (P-04).  Leaving ``tradeoffs`` unset selects
    the Toffoli-minimizing schedule. ``address_widths`` retains fixed leading
    zeros and restricted singleton windows. ``max_log_block_sizes`` can keep
    fixed address bits in Select instead of the swap bank. ``external_control``
    uses the catalog's controlled lookup convention, with zero words off-control.
    """

    region: Region = attrs.field(converter=normalize_region)
    word: int
    compute: bool = True
    tradeoffs: Optional[Tuple[int, ...]] = attrs.field(
        default=None, converter=lambda v: tuple(int(x) for x in v) if v is not None else None)
    order: Optional[Tuple[int, ...]] = attrs.field(
        default=None, converter=lambda v: tuple(int(x) for x in v) if v is not None else None)
    #: A parent's source-specific saving, subtracted from this call's Toffoli count.
    #: P-16 uses it to book Berry's ``-1`` layer-control and ``-3`` sign-fixup
    #: credits against the lookup that would otherwise pay them.
    adjustment: int = 0
    #: Optional ``A_max``; restricts the schedule search to affordable tradeoffs.
    ancilla_budget: Optional[int] = None
    #: Physical widths, including fixed leading-zero bits. A singleton window
    #: inside a nonzero-width register still needs its address checks.
    address_widths: Optional[Tuple[int, ...]] = attrs.field(
        default=None, converter=lambda v: tuple(int(x) for x in v) if v is not None else None)
    #: Restrict each swap bank to these low bits of its address. P-07 packs
    #: fixed lower system bits above the upper address and keeps them in Select.
    max_log_block_sizes: Optional[Tuple[int, ...]] = attrs.field(
        default=None, converter=lambda v: tuple(int(x) for x in v) if v is not None else None)
    external_control: bool = False

    def __attrs_post_init__(self):
        if self.address_widths is None and self.max_log_block_sizes is None:
            return  # Preserve the existing implicit-width callers' convention.
        widths = self.selection_bitsizes
        if len(widths) != len(self.region) or any(
                bits < 0 or high >= 1 << bits
                for (_, high), bits in zip(self.region, widths)):
            raise ValueError("address widths do not contain the lookup region")
        caps = widths if self.max_log_block_sizes is None else self.max_log_block_sizes
        if len(caps) != len(widths) or any(cap < 0 or cap > bits for cap, bits in zip(caps, widths)):
            raise ValueError("invalid lookup block-size caps")
        if self.tradeoffs is not None and (len(self.tradeoffs) != len(widths) or any(
                t < 1 or t & (t - 1) or t > 1 << cap for t, cap in zip(self.tradeoffs, caps))):
            raise ValueError("tradeoff must be a power of two within the declared cap")

    @cached_property
    def selection_bitsizes(self) -> Tuple[int, ...]:
        return (self.address_widths if self.address_widths is not None
                else tuple(address_bitsize(high) for _, high in self.region))

    def contains_address(self, *values: int) -> bool:
        """The actual joint lookup's live window; all other words are zero."""
        if len(values) != len(self.region):
            raise ValueError("one address value is required per axis")
        return all(low <= value <= high for value, (low, high) in zip(values, self.region))

    @cached_property
    def schedule(self) -> Tuple[int, ...]:
        if self.tradeoffs is not None:
            return self.tradeoffs
        return best_schedule(self.region, self.word, self.compute, self.ancilla_budget,
                             self.address_widths, self.max_log_block_sizes)

    @cached_property
    def toffolis(self) -> int:
        base = lookup_toffolis(self.region, self.word, self.schedule, self.compute, self.order,
                              self.address_widths)
        return base + int(self.adjustment)

    @cached_property
    def clean_ancillas(self) -> int:
        return lookup_ancillas(self.region, self.word, self.schedule, self.compute, self.address_widths)

    @cached_property
    def signature(self) -> Signature:
        registers = [
            Register(f"address{index}", QAny(max(1, bits)))
            for index, bits in enumerate(self.selection_bitsizes)
            if bits or self.address_widths is None
        ]
        if self.external_control:
            registers.insert(0, Register("ctrl", QBit()))
        registers.append(Register("target", QAny(max(1, int(self.word)))))
        return Signature(registers)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        if self.toffolis:
            count[And()] += self.toffolis
        return dict(count)


def qroam_pair(region, word: int) -> Tuple[RangeSafeQROAM, RangeSafeQROAM]:
    """A compute/uncompute pair on one region, each at its own optimal schedule."""
    region = normalize_region(region)
    return (RangeSafeQROAM(region=region, word=word, compute=True),
            RangeSafeQROAM(region=region, word=word, compute=False))
