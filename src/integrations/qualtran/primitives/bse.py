r"""BSE-01--BSE-06, symmetry-adapted branch.

The symmetry-adapted branch stores every external particle register in its
symmetry basis, so the external basis change $R_s$ is a convention and costs
nothing.  What remains is a choice of layout for each data family, and the
interfaces that move between layouts:

============================  ==========================  ==========================
data family                   layout                      child
============================  ==========================  ==========================
$f^o$, $f^v$                  fused, even padded          P-12
$x^{W,s}$, $x^{V,s}$          fused, power-of-two padded  P-07
$z^V$                         fused, even padded          P-12
$\widetilde\zeta^W$           indexed, physical           P-09
$S^{V,s}$ (prepared species)  indexed, physical           P-08
$U_{S_X}$, $U_{S_Z}$          --                          P-18
============================  ==========================  ==========================

Two families deliberately stay in the physical basis.  The screened center is a
real-space diagonal over $(\mathbf R,\mu,\nu)$, which symmetry adaptation would
not compress; and the state-prepared exchange factor uses standard
physical-address load-all preparation, which needs no P-18 interface at all.
Everything else moves to the irrep basis, where the partner index $a$ carries an
identity and addresses no lookup --- that is the whole compression.

Equation (BSE.8) gives the $X$ factors of both species one common padded extent,
and P-18 acts on the interpolating label alone.  The $X$ interface is therefore
species independent, and the four screened $X$ applications serve $oo$, $vv$, and
the direct $ov$ term unchanged.  Exchange borrows that pair on its symmetry-
adapted leg and adds only the two interfaces around its center, for six shared
applications rather than ten.

The Fourier transforms, momentum subtraction, the $T$ isometry, term routing, and
the weighted PREPARE remain uncatalogued leaves; `open_leaves` names them and
their multiplicities, and no total here silently sets them to zero.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import json

from qualtran import Bloq, QAny, QBit, Register, Signature

try:
    from .diagonal import DiagonalBlockEncoding
    from .eigendecomposition import FusedEigendecompositionBlockEncoding
    from .isometry import FusedColumnIsometry, power_of_two_at_least
    from .space_group import SpaceGroupData, SpaceGroupTransform, diamond_6x6x6_c8, HERE
    from .state_prep import IndexedLoadAllStatePreparation
    from .unitary import even_padded_sizes
except ImportError:  # pragma: no cover - direct execution
    from diagonal import DiagonalBlockEncoding
    from eigendecomposition import FusedEigendecompositionBlockEncoding
    from isometry import FusedColumnIsometry, power_of_two_at_least
    from space_group import SpaceGroupData, SpaceGroupTransform, diamond_6x6x6_c8, HERE
    from state_prep import IndexedLoadAllStatePreparation
    from unitary import even_padded_sizes

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@dataclass(frozen=True)
class IrrepBlocks:
    """The per-irrep dimensions of (BSE.5), aligned across the three representations."""

    partner: Tuple[int, ...]
    auxiliary: Tuple[int, ...]
    occupied: Tuple[int, ...]
    virtual: Tuple[int, ...]
    n_k: int
    n_thc: int
    n_occ: int
    n_virt: int

    def copies(self, species: str) -> Tuple[int, ...]:
        return self.occupied if species == "o" else self.virtual

    def check(self) -> Dict[str, bool]:
        return {
            "sum m n^A == N_k N_THC":
                sum(m * n for m, n in zip(self.partner, self.auxiliary))
                == self.n_k * self.n_thc,
            "sum m n^o == N_k N_o":
                sum(m * n for m, n in zip(self.partner, self.occupied))
                == self.n_k * self.n_occ,
            "sum m n^v == N_k N_v":
                sum(m * n for m, n in zip(self.partner, self.virtual))
                == self.n_k * self.n_virt,
            "equal irrep counts":
                len({len(self.partner), len(self.auxiliary),
                     len(self.occupied), len(self.virtual)}) == 1,
        }

    @cached_property
    def x_extents(self) -> Tuple[int, ...]:
        """(BSE.8): one power-of-two row extent per irrep, shared by both species."""
        return tuple(
            power_of_two_at_least(max(2, aux + occ, aux + virt))
            for aux, occ, virt in zip(self.auxiliary, self.occupied, self.virtual))

    @cached_property
    def z_extents(self) -> Tuple[int, ...]:
        """The $V$ layout of the exchange center: even-padded auxiliary blocks.

        Kept in irrep order, because P-18's second P-20 call needs the contiguous
        star-family ordering of (P18.1a).  The P-16 child inside P-12 sorts its
        own copy into descending size order.
        """
        return tuple(max(2, n + (n % 2)) for n in self.auxiliary)

    @classmethod
    def from_json(cls, path) -> "IrrepBlocks":
        record = json.loads(Path(path).read_text())
        return cls(
            partner=tuple(int(v) for v in record["partner"]),
            auxiliary=tuple(int(v) for v in record["auxiliary"]),
            occupied=tuple(int(v) for v in record["occupied"]),
            virtual=tuple(int(v) for v in record["virtual"]),
            n_k=int(record["N_k"]), n_thc=int(record["N_THC"]),
            n_occ=int(record["N_o"]), n_virt=int(record["N_v"]))


def diamond_blocks_6x6x6_c8() -> IrrepBlocks:
    return IrrepBlocks.from_json(HERE / "diamond_irrep_blocks_6x6x6_c8.json")


@attrs.frozen
class SymmetryAdaptedBSE(Bloq):
    r"""The symmetry-adapted BSE block encoding, assembled from BSE-01--BSE-06.

    ``prepared`` selects the exchange species that uses standard physical-basis
    state preparation; the other species keeps its symmetry-adapted $X$ factor.
    ``allocation`` selects the modular account, in which every term is a complete
    standalone circuit, or the draft's shared account, in which one compiled
    factor serves several terms.
    """

    blocks: IrrepBlocks
    group: SpaceGroupData
    phase_bits: int = 32
    prepared: str = "v"
    allocation: str = "shared"
    #: Optional ``A_max`` for the two physical-basis children whose banks dominate
    #: the peak: the indexed screened diagonal and the indexed load-all state
    #: preparation.  Every symmetry-adapted child already sits far below them.
    ancilla_budget: Optional[int] = None

    def __attrs_post_init__(self):
        if self.prepared not in ("o", "v"):
            raise ValueError("prepared species must be 'o' or 'v'")
        if self.allocation not in ("modular", "shared"):
            raise ValueError("allocation must be 'modular' or 'shared'")

    # ------------------------------------------------------------ children --
    def fock(self, species: str) -> FusedEigendecompositionBlockEncoding:
        """BSE-01: fused P-12 on the $f_\\lambda^s$ blocks; no P-18 is needed."""
        return FusedEigendecompositionBlockEncoding(
            block_sizes=self.blocks.copies(species), phase_bits=self.phase_bits)

    def x_factor(self, family: str, species: str, forward: bool) -> FusedColumnIsometry:
        """Fused P-07 on the (BSE.9) dilations, padded to the common (BSE.8) extent.

        ``family`` is ``"W"`` for the screened factors and ``"V"`` for the exchange
        factor.  The two families carry different data but identical shapes, so
        their costs coincide while their lookups do not; they stay distinct
        children and are never shared.
        """
        if family not in ("W", "V"):
            raise ValueError("family must be 'W' or 'V'")
        copies = self.blocks.copies(species)
        shapes = tuple((aux + own, own) for aux, own in zip(self.blocks.auxiliary, copies))
        return FusedColumnIsometry(blocks=shapes, row_capacities=self.blocks.x_extents,
                                   phase_bits=self.phase_bits, forward=forward,
                                   tag=f"x^{family},{species}")

    @cached_property
    def screened_center(self) -> DiagonalBlockEncoding:
        """Indexed P-09 on the real-space $\\widetilde\\zeta^W(\\mathbf R,\\mu,\\nu)$."""
        return DiagonalBlockEncoding(
            region=((0, self.blocks.n_k - 1), (0, self.blocks.n_thc - 1),
                    (0, self.blocks.n_thc - 1)),
            phase_bits=self.phase_bits, involutory=True,
            ancilla_budget=self.ancilla_budget)

    @cached_property
    def exchange_center(self) -> FusedEigendecompositionBlockEncoding:
        """Fused P-12 on the $z_\\lambda^V$ Hermitian blocks."""
        return FusedEigendecompositionBlockEncoding(
            block_sizes=self.blocks.auxiliary, phase_bits=self.phase_bits)

    def state_preparation(self, species: str) -> IndexedLoadAllStatePreparation:
        """Indexed load-all P-08 in the physical basis, over $(\\mathbf k,\\mu)$.

        The tree holds ``N_s`` live amplitudes.  The flagged completion's failure
        component sits on the flag qubit, not as an extra amplitude in the same
        tree, so the tree is not sized for ``N_s + 1``.
        """
        live = self.blocks.n_occ if species == "o" else self.blocks.n_virt
        return IndexedLoadAllStatePreparation(
            block_region=((0, self.blocks.n_k * self.blocks.n_thc - 1),),
            live_amplitudes=live, phase_bits=self.phase_bits,
            ancilla_budget=self.ancilla_budget)

    def s_x(self, forward: bool) -> SpaceGroupTransform:
        """P-18 on the (BSE.8) layout: one implementation serves both species."""
        return SpaceGroupTransform(data=self.group, phase_bits=self.phase_bits,
                                   forward=forward, block_extents=self.blocks.x_extents)

    def s_z(self, forward: bool) -> SpaceGroupTransform:
        """P-18 on the $V$ layout of the exchange center."""
        return SpaceGroupTransform(data=self.group, phase_bits=self.phase_bits,
                                   forward=forward, block_extents=self.blocks.z_extents)

    @property
    def nonprepared(self) -> str:
        return "o" if self.prepared == "v" else "v"

    # --------------------------------------------------------------- terms --
    @cached_property
    def open_leaves(self) -> Dict[str, int]:
        """Uncatalogued arithmetic and control leaves, with their multiplicities.

        The modular account gives each screened term its own Fourier and
        subtraction bundle; the shared account charges one bundle to BSE-04.
        """
        modular = self.allocation == "modular"
        return {
            "momentum Fourier transform C_F(N_k)": 12 if modular else 4,
            "momentum subtraction C_minus(N_k)": 3 if modular else 1,
            "exchange momentum map C_T": 2,
            "term and particle routing C_route^SG": 1,
            "weighted PREPARE and its inverse": 2,
            "walk reflection (n_R^SG - 2)": 1,
        }

    @cached_property
    def child_applications(self) -> Counter:
        """Every charged application of a compiled child, under this allocation."""
        other = self.nonprepared
        count: Counter = Counter()
        count["fock o"] = 1
        count["fock v"] = 1
        if self.allocation == "modular":
            count[f"x^W,o forward"] = 3
            count[f"x^W,o inverse"] = 3
            count[f"x^W,v forward"] = 3
            count[f"x^W,v inverse"] = 3
            count["screened center"] = 3
            count["S_X forward"] = 7
            count["S_X inverse"] = 7
        else:
            count[f"x^W,o forward"] = 2
            count[f"x^W,o inverse"] = 2
            count[f"x^W,v forward"] = 2
            count[f"x^W,v inverse"] = 2
            count["screened center"] = 1
            count["S_X forward"] = 2
            count["S_X inverse"] = 2
        count[f"x^V,{other} forward"] = 1
        count[f"x^V,{other} inverse"] = 1
        count[f"state preparation {self.prepared}"] = 2
        count["exchange center"] = 1
        count["S_Z forward"] = 1
        count["S_Z inverse"] = 1
        return count

    @cached_property
    def child_toffolis(self) -> Dict[str, int]:
        """One application of each compiled child."""
        other = self.nonprepared
        return {
            "fock o": self.fock("o").toffolis,
            "fock v": self.fock("v").toffolis,
            "x^W,o forward": self.x_factor("W", "o", True).toffolis,
            "x^W,o inverse": self.x_factor("W", "o", False).toffolis,
            "x^W,v forward": self.x_factor("W", "v", True).toffolis,
            "x^W,v inverse": self.x_factor("W", "v", False).toffolis,
            f"x^V,{other} forward": self.x_factor("V", other, True).toffolis,
            f"x^V,{other} inverse": self.x_factor("V", other, False).toffolis,
            "screened center": self.screened_center.toffolis,
            "exchange center": self.exchange_center.toffolis,
            f"state preparation {self.prepared}":
                self.state_preparation(self.prepared).toffolis,
            "S_X forward": self.s_x(True).toffolis,
            "S_X inverse": self.s_x(False).toffolis,
            "S_Z forward": self.s_z(True).toffolis,
            "S_Z inverse": self.s_z(False).toffolis,
        }

    @cached_property
    def term_toffolis(self) -> Dict[str, int]:
        """Closed Toffoli cost of each term, under the selected allocation."""
        unit = self.child_toffolis
        other = self.nonprepared
        o_pair = unit["x^W,o forward"] + unit["x^W,o inverse"]
        v_pair = unit["x^W,v forward"] + unit["x^W,v inverse"]
        exchange_pair = unit[f"x^V,{other} forward"] + unit[f"x^V,{other} inverse"]
        s_x = unit["S_X forward"] + unit["S_X inverse"]
        s_z = unit["S_Z forward"] + unit["S_Z inverse"]
        preparation = 2 * unit[f"state preparation {self.prepared}"]
        fock = unit["fock o"] + unit["fock v"]
        exchange = (exchange_pair + preparation + unit["exchange center"] + s_z)
        if self.allocation == "modular":
            return {
                "BSE-01 Fock": fock,
                "BSE-02 oo screened": 2 * o_pair + 2 * s_x + unit["screened center"],
                "BSE-03 vv screened": 2 * v_pair + 2 * s_x + unit["screened center"],
                "BSE-04 ov direct": o_pair + v_pair + 2 * s_x + unit["screened center"],
                "BSE-05 ov exchange": exchange + s_x,
            }
        return {
            "BSE-01 Fock": fock,
            "BSE-02 oo screened": 2 * o_pair + 2 * s_x + unit["screened center"],
            "BSE-03 vv screened": 2 * v_pair,
            "BSE-04 ov direct": 0,
            "BSE-05 ov exchange": exchange,
        }

    @cached_property
    def transform_applications(self) -> Counter:
        """P-18 applications charged, by interface and direction."""
        counts = self.child_applications
        return Counter(x_forward=counts["S_X forward"], x_inverse=counts["S_X inverse"],
                       z_forward=counts["S_Z forward"], z_inverse=counts["S_Z inverse"])

    @cached_property
    def toffolis(self) -> int:
        """Closed part of ``C_SELECT,SG``; the leaves in ``open_leaves`` are extra."""
        return sum(self.term_toffolis.values())

    @cached_property
    def clean_ancillas(self) -> int:
        """Sequential no-overlap bound over every child, plus the live selector bits."""
        bounds = [
            self.fock("o").clean_ancillas, self.fock("v").clean_ancillas,
            self.screened_center.clean_ancillas, self.exchange_center.clean_ancillas,
            self.state_preparation(self.prepared).clean_ancillas,
            self.s_x(True).clean_ancillas, self.s_x(False).clean_ancillas,
            self.s_z(True).clean_ancillas, self.s_z(False).clean_ancillas,
        ]
        for family in ("W", "V"):
            for species in ("o", "v"):
                for forward in (True, False):
                    bounds.append(self.x_factor(family, species, forward).clean_ancillas)
        return max(bounds)

    @cached_property
    def signature(self) -> Signature:
        aux_bits = max(1, (sum(self.blocks.x_extents) - 1).bit_length())
        return Signature([
            Register("electron", QAny(aux_bits)),
            Register("hole", QAny(aux_bits)),
            Register("term_selector", QAny(3)),
            Register("signal", QBit()),
            Register("spin", QAny(2)),
            Register("phase_gradient", QAny(self.phase_bits)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        other = self.nonprepared
        counts = self.child_applications
        nodes = {
            "fock o": self.fock("o"),
            "fock v": self.fock("v"),
            "x^W,o forward": self.x_factor("W", "o", True),
            "x^W,o inverse": self.x_factor("W", "o", False),
            "x^W,v forward": self.x_factor("W", "v", True),
            "x^W,v inverse": self.x_factor("W", "v", False),
            f"x^V,{other} forward": self.x_factor("V", other, True),
            f"x^V,{other} inverse": self.x_factor("V", other, False),
            "screened center": self.screened_center,
            "exchange center": self.exchange_center,
            f"state preparation {self.prepared}": self.state_preparation(self.prepared),
            "S_X forward": self.s_x(True),
            "S_X inverse": self.s_x(False),
            "S_Z forward": self.s_z(True),
            "S_Z inverse": self.s_z(False),
        }
        count: "Counter[Bloq]" = Counter()
        for name, times in counts.items():
            count[nodes[name]] += times
        return dict(count)
