r"""P-18 -- the space-group transform ``U_S``, in five stages.

``U_S`` exchanges a symmetry label for a physical one.  Its forward direction maps
the fused irrep/copy label :math:`L=(\lambda,c)` and the partner label
:math:`a=(\kappa,\nu)` to the momentum :math:`\mathbf k` and the interpolating
label :math:`\mu`,

.. math::
    U_S^{\to}|L\rangle|a\rangle
    =|\mathbf k\rangle\sum_\mu S^\dagger_{(\lambda,a,c),(\mathbf k,\mu)}|\mu\rangle,
    \qquad U_S^{\leftarrow}=(U_S^{\to})^\dagger .

The construction never tabulates the momentum-multiplexed matrix.  It exploits the
factorization :math:`S^{(\mathbf k)}=P(g)\Lambda(g,\mathbf k)S^{(f)}`: one
representative matrix per momentum star, induced to the star's other members by a
fiber permutation :math:`P(g)` and a diagonal Bloch phase
:math:`\Lambda(g,\mathbf k)`.  Only :math:`N_f` representatives carry unitary data,
and each is block diagonal over the little group's orbits on the interpolating
points, so the synthesized data are the orbit blocks alone.

The five stages run in order:

1. split the fused labels into :math:`(\lambda, c)` and then :math:`(f,\alpha)`;
2. construct the physical momentum :math:`\mathbf k = A_g\mathbf k_f`;
3. apply the representative unitary, one fused synthesis over all orbit blocks;
4. induce the result across the orbit with :math:`\overline\Lambda` then
   :math:`P(g)`;
5. erase the temporary star labels, retaining :math:`\mathbf k`.

The inverse runs them backwards, exchanging the directional children: the forward
transform ends with a measurement-based clean uncompute of the star labels, while
the inverse begins with a clean compute of the same table.

Every count comes from the child primitives -- P-20 for both label splits, P-14
for every lookup, P-16 for the representative synthesis, P-17 for the fiber
permutation, and the sourced controlled carry-free adder for the Bloch phase.  No
count is asserted here that a child does not supply.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, Register, Signature

try:
    from .arithmetic import CtrlModularAddNoCarry
    from .qroam import RangeSafeQROAM, address_bitsize
    from .relabel import ReversibleRelabel
    from .staircase import BlockStaircase
    from .unitary import FusedBlockUnitarySynthesis
except ImportError:  # pragma: no cover - direct execution
    from arithmetic import CtrlModularAddNoCarry
    from qroam import RangeSafeQROAM, address_bitsize
    from relabel import ReversibleRelabel
    from staircase import BlockStaircase
    from unitary import FusedBlockUnitarySynthesis

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SpaceGroupData:
    """Every classical input P-18 declares, for one material, mesh, and rank.

    ``orbit_sizes`` concatenates the little-group orbits of all stars in the
    representative-block order P-16 consumes.  ``irrep_copies`` lists the copy
    dimensions :math:`n_\\lambda` in the contiguous star-family order of (P18.1a),
    so that the first P-20 call splits :math:`L` and the second splits
    :math:`\\lambda`.
    """

    material: str
    mesh: Tuple[int, int, int]
    n_thc: int
    n_ops: int
    star_sizes: Tuple[int, ...]
    little_orders: Tuple[int, ...]
    irreps_per_star: Tuple[int, ...]
    irrep_copies: Tuple[int, ...]
    orbit_sizes: Tuple[int, ...]
    fold_widths: Tuple[int, int, int]
    fold_l1_max: int
    signed_permutation_ops: int

    # ---------------------------------------------------------------- derived
    @property
    def n_k(self) -> int:
        return self.mesh[0] * self.mesh[1] * self.mesh[2]

    @property
    def n_stars(self) -> int:
        return len(self.star_sizes)

    @property
    def n_irreps(self) -> int:
        return len(self.irrep_copies)

    @property
    def axis_bits(self) -> Tuple[int, ...]:
        return tuple(max(1, (length - 1).bit_length()) for length in self.mesh)

    @property
    def momentum_bits(self) -> int:
        """``n_k = sum_j n_j``."""
        return sum(self.axis_bits)

    @property
    def fiber_bits(self) -> int:
        """``n_mu = ceil(log2 N_THC)``."""
        return max(1, (self.n_thc - 1).bit_length())

    @property
    def star_label_bits(self) -> int:
        """``n_fg = ceil(log2 N_f) + ceil(log2 G)``."""
        return (max(1, (self.n_stars - 1).bit_length())
                + max(1, (self.n_ops - 1).bit_length()))

    @property
    def fold_bits(self) -> int:
        """``ell = sum_j ell_j``."""
        return sum(self.fold_widths)

    @property
    def all_signed_permutations(self) -> bool:
        return self.signed_permutation_ops == self.n_ops

    def check(self) -> Dict[str, bool]:
        return {
            "sum star sizes == N_k": sum(self.star_sizes) == self.n_k,
            "sum J_f == K_irr": sum(self.irreps_per_star) == self.n_irreps,
            "sum orbit sizes == N_f * N_THC":
                sum(self.orbit_sizes) == self.n_stars * self.n_thc,
            "every orbit fits the point group": max(self.orbit_sizes) <= self.n_ops,
        }

    @classmethod
    def from_json(cls, path) -> "SpaceGroupData":
        record = json.loads(Path(path).read_text())
        stars = record["stars"]
        copies: list[int] = []
        orbits: list[int] = []
        for star in stars:
            copies.extend(int(multiplicity) for _, multiplicity in star["little_irreps"])
            orbits.extend(int(size) for size in star["orbit_sizes"])
        mesh = tuple(int(value) for value in record["mesh"].split("x"))
        return cls(
            material=record["material"],
            mesh=mesh,
            n_thc=int(record["N_THC"]),
            n_ops=int(record["G"]),
            star_sizes=tuple(int(star["star_size"]) for star in stars),
            little_orders=tuple(int(star["little_group"]) for star in stars),
            irreps_per_star=tuple(int(count) for count in record["J_f"]),
            irrep_copies=tuple(copies),
            orbit_sizes=tuple(orbits),
            fold_widths=tuple(int(width) for width in record["fold_widths"]),
            fold_l1_max=int(record["fold_L1_max"]),
            signed_permutation_ops=int(record["n_signed_permutations"]),
        )


def diamond_6x6x6_c8() -> SpaceGroupData:
    """The recorded diamond instance: ``N_k = 216``, ``N_THC = 208``, ``G = 48``."""
    return SpaceGroupData.from_json(HERE / "diamond_space_group_6x6x6_c8.json")


@attrs.frozen
class SpaceGroupTransform(Bloq):
    r"""One direction of :math:`U_S`, assembled from the five P-18 stages.

    ``momentum_word`` names the declared selected-point schedule
    :math:`\mathcal A_P`.  ``"star-index"`` keeps the star label itself as the
    representative word, so the representative table is a wire identity and the
    schedule is one lookup on the Cartesian region
    :math:`[0,G-1]\times[0,N_f-1]`; it needs no property of :math:`A_g` beyond the
    label bijection.  ``"momentum-label"`` instead loads :math:`\mathbf k_f` and
    applies :math:`A_g` as a signed coordinate permutation, which is legal only
    when every point operation acts that way on the mesh.
    """

    data: SpaceGroupData
    phase_bits: int = 32
    forward: bool = True
    block_extents: Optional[Tuple[int, ...]] = attrs.field(
        default=None, converter=lambda v: tuple(int(x) for x in v) if v is not None else None)
    momentum_word: str = "star-index"

    def __attrs_post_init__(self):
        if self.momentum_word not in ("star-index", "momentum-label"):
            raise ValueError(f"unknown selected-point schedule {self.momentum_word!r}")
        if self.momentum_word == "momentum-label" and not self.data.all_signed_permutations:
            raise ValueError(
                "the momentum-label schedule needs every A_g to be a signed coordinate "
                f"permutation on the mesh; only {self.data.signed_permutation_ops} of "
                f"{self.data.n_ops} are")

    # ------------------------------------------------------ stage 1: labels --
    @cached_property
    def extents(self) -> Tuple[int, ...]:
        """``B_lambda``; densely packed live sizes unless a padded layout is declared."""
        return self.block_extents or self.data.irrep_copies

    @cached_property
    def label_split(self) -> BlockStaircase:
        """``L <-> (lambda, c)`` over ``K_irr`` blocks."""
        return BlockStaircase(block_sizes=self.extents, split=self.forward)

    @cached_property
    def family_split(self) -> BlockStaircase:
        """``lambda <-> (f, alpha)`` over ``N_f`` contiguous star families."""
        return BlockStaircase(block_sizes=self.data.irreps_per_star, split=self.forward)

    @cached_property
    def label_toffolis(self) -> int:
        return self.label_split.toffolis + self.family_split.toffolis

    # ---------------------------------------------------- stage 2: momentum --
    @cached_property
    def star_table(self) -> Tuple[RangeSafeQROAM, RangeSafeQROAM]:
        """``f -> k_f``, loaded and then unloaded around the selected point operation."""
        if self.momentum_word == "star-index":
            empty = ((0, 0),)
            return (RangeSafeQROAM(region=empty, word=self.data.momentum_bits, compute=True),
                    RangeSafeQROAM(region=empty, word=self.data.momentum_bits, compute=False))
        region = ((0, self.data.n_stars - 1),)
        return (RangeSafeQROAM(region=region, word=self.data.momentum_bits, compute=True),
                RangeSafeQROAM(region=region, word=self.data.momentum_bits, compute=False))

    @cached_property
    def selected_point(self) -> RangeSafeQROAM:
        """``O_A``: XOR ``A_g`` applied to the representative into a clean register."""
        region = ((0, self.data.n_ops - 1), (0, self.data.n_stars - 1))
        return RangeSafeQROAM(region=region, word=self.data.momentum_bits, compute=True)

    @cached_property
    def representative_momentum_toffolis(self) -> int:
        load, unload = self.star_table
        return load.toffolis + self.selected_point.toffolis + unload.toffolis

    @cached_property
    def star_label_table(self) -> RangeSafeQROAM:
        """``k -> z(k)``: a measurement-based unload forward, a clean load inverse.

        (P18.6a) gives the region as the Cartesian product
        ``R_k^I = prod_j [0, K_j - 1]``, not a flat ``N_k``-entry prefix.  The
        momentum register is ``d`` axis words, so the live labels do not occupy a
        contiguous window of it and a flat prefix would not be a correct live
        region.
        """
        region = tuple((0, length - 1) for length in self.data.mesh)
        return RangeSafeQROAM(region=region,
                              word=self.data.star_label_bits,
                              compute=not self.forward)

    @cached_property
    def momentum_toffolis(self) -> int:
        return self.representative_momentum_toffolis + self.star_label_table.toffolis

    # ---------------------------------------------- stage 3: representative --
    @cached_property
    def representative(self) -> FusedBlockUnitarySynthesis:
        """One fused P-16 synthesis over every star's little-group orbit blocks."""
        return FusedBlockUnitarySynthesis(live_sizes=self.data.orbit_sizes,
                                          phase_bits=self.phase_bits)

    # --------------------------------------------------- stage 4: induction --
    @cached_property
    def phase_coordinates(self) -> Tuple[Tuple[RangeSafeQROAM, RangeSafeQROAM], ...]:
        """One ``q_j(k_j)`` load/unload pair per mesh axis."""
        pairs = []
        for length in self.data.mesh:
            region = ((0, length - 1),)
            pairs.append((
                RangeSafeQROAM(region=region, word=self.phase_bits, compute=True),
                RangeSafeQROAM(region=region, word=self.phase_bits, compute=False),
            ))
        return tuple(pairs)

    @cached_property
    def fold(self) -> Tuple[RangeSafeQROAM, RangeSafeQROAM]:
        """The signed fold vector over ``[0, G-1] x [0, N_THC-1]``."""
        region = ((0, self.data.n_ops - 1), (0, self.data.n_thc - 1))
        return (RangeSafeQROAM(region=region, word=self.data.fold_bits, compute=True),
                RangeSafeQROAM(region=region, word=self.data.fold_bits, compute=False))

    @cached_property
    def bloch_adder(self) -> CtrlModularAddNoCarry:
        return CtrlModularAddNoCarry(phase_bits=self.phase_bits)

    @cached_property
    def fiber_permutation(self) -> ReversibleRelabel:
        """The family-controlled ``p_g`` acting on the interpolating label."""
        region = ((0, self.data.n_ops - 1), (0, self.data.n_thc - 1))
        return ReversibleRelabel(region=region, label_bits=self.data.fiber_bits,
                                 forward=self.forward)

    @cached_property
    def induced_toffolis(self) -> int:
        total = sum(compute.toffolis + uncompute.toffolis
                    for compute, uncompute in self.phase_coordinates)
        compute, uncompute = self.fold
        total += compute.toffolis + uncompute.toffolis
        total += self.data.fold_bits * self.bloch_adder.toffolis
        total += self.fiber_permutation.toffolis
        return total

    # -------------------------------------------------------------- totals --
    @cached_property
    def toffolis(self) -> int:
        """(P18.8)."""
        return (self.label_toffolis + self.momentum_toffolis
                + self.representative.toffolis + self.induced_toffolis)

    @cached_property
    def stage_toffolis(self) -> Dict[str, int]:
        return {
            "1 labels": self.label_toffolis,
            "2 momentum": self.momentum_toffolis,
            "3 representative": self.representative.toffolis,
            "4 induced": self.induced_toffolis,
        }

    @cached_property
    def child_toffolis(self) -> Dict[str, int]:
        load, unload = self.star_table
        fold_compute, fold_uncompute = self.fold
        coordinates = sum(compute.toffolis + uncompute.toffolis
                          for compute, uncompute in self.phase_coordinates)
        return {
            "P-20 label split/merge": self.label_split.toffolis,
            "P-20 star-family split/merge": self.family_split.toffolis,
            "P-14 representative table": load.toffolis + unload.toffolis,
            "selected point operation": self.selected_point.toffolis,
            "P-03/P-04 star-label table": self.star_label_table.toffolis,
            "P-16 representative synthesis": self.representative.toffolis,
            "P-14 phase coordinates": coordinates,
            "P-14 fold vector": fold_compute.toffolis + fold_uncompute.toffolis,
            "Bloch phase adders": self.data.fold_bits * self.bloch_adder.toffolis,
            "P-17 fiber permutation": self.fiber_permutation.toffolis,
        }

    @cached_property
    def clean_ancillas(self) -> int:
        """(P18.9)-(P18.10), the sequential no-overlap bound."""
        b = self.phase_bits
        label = max(self.label_split.clean_ancillas, self.family_split.clean_ancillas)
        load, unload = self.star_table
        representative_momentum = max(load.clean_ancillas,
                                      self.data.momentum_bits + self.selected_point.clean_ancillas,
                                      unload.clean_ancillas)
        momentum = max(representative_momentum, self.star_label_table.clean_ancillas)
        fold_compute, fold_uncompute = self.fold
        fold_ancillas = max(fold_compute.clean_ancillas, fold_uncompute.clean_ancillas)
        coordinate_ancillas = max(
            max(compute.clean_ancillas, uncompute.clean_ancillas)
            for compute, uncompute in self.phase_coordinates)
        induced = max(b + fold_ancillas,
                      b + self.data.fold_bits + coordinate_ancillas,
                      self.data.fold_bits + 3 * b,
                      b + self.fiber_permutation.clean_ancillas)
        return max(b + label, b + momentum, self.representative.clean_ancillas, induced)

    @cached_property
    def phase_error_bound(self) -> float:
        """``pi * L_1max / 2^b``, the diagonal error of compiling the Bloch phase."""
        return 3.141592653589793 * self.data.fold_l1_max / float(1 << self.phase_bits)

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register("fused_label", QAny(self.label_split.width)),
            Register("partner", QAny(max(1, (max(self.data.star_sizes) - 1).bit_length()))),
            Register("momentum", QAny(self.data.momentum_bits)),
            Register("fiber", QAny(self.data.fiber_bits)),
            Register("phase_gradient", QAny(self.phase_bits)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        count: "Counter[Bloq]" = Counter()
        count[self.label_split] += 1
        count[self.family_split] += 1
        load, unload = self.star_table
        count[load] += 1
        count[self.selected_point] += 1
        count[unload] += 1
        count[self.star_label_table] += 1
        count[self.representative] += 1
        for compute, uncompute in self.phase_coordinates:
            count[compute] += 1
            count[uncompute] += 1
        fold_compute, fold_uncompute = self.fold
        count[fold_compute] += 1
        count[fold_uncompute] += 1
        count[self.bloch_adder] += self.data.fold_bits
        count[self.fiber_permutation] += 1
        return dict(count)
