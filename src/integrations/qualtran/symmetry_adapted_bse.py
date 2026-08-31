r"""Point-group symmetry-adapted BSE templates (`SYMMETRY_ADAPTED_BSE_SPEC.md` §9).

Extends the translation-only :mod:`bse_block_encoding` by inserting the space-group
transform ``Q : |k>|I> -> |i>|a>|c>`` so that the THC factors ``X`` and the central ``W``
act in **irrep-block** form:

    Q_grid^dag W Q_grid = (+)_i 1_{m_i} (x) w_i        (spec §2)
    Q_grid^dag X Q_orb  = (+)_i 1_{m_i} (x) x_i        (spec §3)

The partner index ``a`` (dimension ``m_i``) carries an identity, so **no lookup is
addressed by it** -- the tables see only ``(irrep, copy)``. That is the compression.

Term structures are §9.3 / §9.4 verbatim:

  exchange (6 Q):  X_o (x) X_v -> Q^dag (x) Q^dag -> [momentum transfer]
                   -> Q W Q^dag  (W = U D U^dag, U synthesized)
                   -> [reversed transfer] -> Q (x) Q -> X_o^dag (x) X_v^dag
  direct   (4 Q):  X_o (x) X_v -> Q^dag (x) Q^dag -> F^dag (x) F^dag
                   -> [compute R = R1-R2; diagonal W(R,mu,nu); uncompute R]
                   -> F (x) F -> Q (x) Q -> X_o^dag (x) X_v^dag

Design choices, each following the spec:

* **X_o / X_v are column selection inside the copy index** (§9.1), not a projector bloq:
  ``x_i^o = x_i[:, :n_occ_i]``. The occ/virt choice is absorbed into the synthesized
  column count. A :class:`CopyMaskProjector` is still emitted for the *padding* mask
  (§9.6), which is a different object.
* **Ragged blocks are grouped by size** rather than uniform-padded (§9.6). Each distinct
  ``(rows, cols)`` gets one block-indexed synthesis with ``n_blocks`` = how many blocks
  share that size, so the cost is the true ``Sum_i``, not ``#irreps * max^2``. This is the
  same pattern §9.8 prescribes for the orbit synthesis.
* **`Q`'s blocks are orbits, `X`/`W`'s are irreps, and they do not line up** (§9.7), so a
  reindex QROAM sits between them -- it is not optional and not free.
* **`U` is synthesized** (§2, §7 item 3): the exchange central is ``U D U^dag`` with
  ``U = (+)_i 1_{m_i} (x) u_i``, and ``u_i`` costs a block unitary synthesis.

Cost convention: see :mod:`toffoli_cost`. **NOTE** the spec's §7 formula
(``ts_per_rotation=0``) does *not* reproduce the spec's own §8 numbers; ``n_ccz + n_t/4``
does, exactly. This module uses the latter -- see the module-level note in the cost report.

Data-free: only sizes/counts; all Toffoli counts come from Qualtran's resource counter.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from math import ceil, log2 as np_log2
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QUInt, Register, Signature
from qualtran.bloqs.arithmetic import LessThanConstant, Subtract
from qualtran.bloqs.basic_gates import CSwap, ZGate
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.arithmetic import AddK
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.qft import QFTTextBook
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length

try:
    from .berry_isometry_synthesis_QROAM import BerryIsometrySynthesisQROAM
    from .block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from .block_isometry_column_synthesis_QROAM import (
        num_mcgs, RealMultiControlledRotationQROAM,
        _phase_layer_log_block_sizes as _phase_layer_lbs)
    from .block_unitary_interferometer_QROAM import (
        BlockInterferometerPhaseLayerQROAM, BlockInterferometerFinalPhasesQROAM,
        optimal_interferometer_log_block_sizes)
    from .block_unitary_interferometer_QROAM import BlockUnitaryInterferometerSynthesisQROAM
    from .diagonal_kernel_block_encoding import DiagonalCoulombKernelBlockEncoding
    from .qroam_block_sizes import optimal_log_block_sizes_measured
    from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
    from .range_safe_qroam import emit_range_safety
    from .toffoli_cost import toffoli_count
    from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
except ImportError:  # pragma: no cover
    from berry_isometry_synthesis_QROAM import BerryIsometrySynthesisQROAM
    from block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from block_isometry_column_synthesis_QROAM import (
        num_mcgs, RealMultiControlledRotationQROAM,
        _phase_layer_log_block_sizes as _phase_layer_lbs)
    from block_unitary_interferometer_QROAM import (
        BlockInterferometerPhaseLayerQROAM, BlockInterferometerFinalPhasesQROAM,
        optimal_interferometer_log_block_sizes)
    from block_unitary_interferometer_QROAM import BlockUnitaryInterferometerSynthesisQROAM
    from diagonal_kernel_block_encoding import DiagonalCoulombKernelBlockEncoding
    from qroam_block_sizes import optimal_log_block_sizes_measured
    from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
    from range_safe_qroam import emit_range_safety
    from toffoli_cost import toffoli_count
    from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

JSUN3 = "/central/groups/changroup/members/jsun3/xprize/THC_general/log_irreps"
_HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# Material / group data (spec §6)
# ===========================================================================


@dataclass(frozen=True)
class IrrepData:
    """The irrep block tables and orbit structure for one mesh.

    ``W``: ``(m_i, n_i)`` per irrep -- grid-grid.  ``X``: ``(m_i, n_grid_i, n_orb_i)``.
    ``n_occ``: the per-irrep occupied copy counts, satisfying the spec §9.2 constraint
    ``Sum_i m_i n_occ_i = n_occ * N_k`` exactly.
    """

    mesh: str
    N_k: int
    n_IP: int
    n_orb: int
    n_occ_tot: int
    W: Tuple[Tuple[int, int], ...]
    X: Tuple[Tuple[int, int, int], ...]
    orbit_sizes: Tuple[int, ...]
    n_families: int

    # ---- derived ----
    @property
    def n_i(self) -> List[int]:
        return [n for _, n in self.W]

    @property
    def m_i(self) -> List[int]:
        return [m for m, _ in self.W]

    @cached_property
    def n_occ(self) -> List[int]:
        """Per-irrep occupied copy counts (spec §9.2).

        Distributed proportionally to ``n_orb_i`` then corrected to hit
        ``Sum_i m_i n_occ_i = n_occ_tot * N_k`` exactly.  The spec states the exact split
        barely moves the cost, so any plausible one obeying the sum is acceptable.
        """
        target = self.n_occ_tot * self.N_k
        frac = self.n_occ_tot / self.n_orb
        occ = [max(0, min(no, int(round(no * frac)))) for _, _, no in self.X]
        cur = sum(m * o for (m, _, _), o in zip(self.X, occ))
        # greedy correction, largest m_i first so few steps are needed
        order = sorted(range(len(occ)), key=lambda j: -self.X[j][0])
        guard = 0
        while cur != target and guard < 100000:
            guard += 1
            moved = False
            for j in order:
                m, _, no = self.X[j]
                if cur < target and occ[j] < no:
                    occ[j] += 1; cur += m; moved = True
                elif cur > target and occ[j] > 0:
                    occ[j] -= 1; cur -= m; moved = True
                if cur == target:
                    break
            if not moved:
                break
        return occ

    @property
    def n_virt(self) -> List[int]:
        return [no - o for (_, _, no), o in zip(self.X, self.n_occ)]

    def check(self) -> Dict[str, bool]:
        return {
            "sum m*n == n_IP*N_k": sum(m * n for m, n in self.W) == self.n_IP * self.N_k,
            "sum m*n_orb == n_orb*N_k": sum(m * no for m, _, no in self.X)
            == self.n_orb * self.N_k,
            "sum m*n_occ == n_occ*N_k": sum(m * o for (m, _, _), o in zip(self.X, self.n_occ))
            == self.n_occ_tot * self.N_k,
        }


RECORDED = os.path.normpath(os.path.join(_HERE, "..", "..", "chem", "fftisdf", "irrep_tables"))


def load_irrep_data(mesh: str = "6x6x6", n_orb: int = 26, n_occ: int = 4,
                    n_IP: Optional[int] = None, root: Optional[str] = None,
                    c: int = 5) -> IrrepData:
    """Load the grid-grid / grid-ao block tables and the orbit distribution.

    ``c`` selects the ISDF rank factor (``c_isdf``); tables live in
    ``src/chem/fftisdf/irrep_tables`` (see its README for provenance).  ``n_IP`` is read
    off the tables rather than assumed -- at ``c=5`` the target ``5*n_orb=130`` rounds up
    to 136 because the selected set must be a union of whole point-group orbits, while
    ``c=8`` hits ``8*n_orb=208`` exactly.

    For ``c != 5`` the per-family orbit decomposition is not available, so the ``c=5``
    orbit list is rescaled to sum to the new ``n_IP`` keeping its size distribution.  That
    is exact in the two figures ``Q``'s cost actually depends on -- the depth
    ``max_o |o| <= |P|`` and the fused address ``n_families * n_IP`` -- and approximate
    only in the mid-layer table sizes.
    """
    import numpy as np

    root = root or RECORDED
    gg = np.loadtxt(f"{root}/data_ov_diamond_{mesh}_c{c}_grid-grid.txt", dtype=int).reshape(-1, 3)
    ga = np.loadtxt(f"{root}/data_ov_diamond_{mesh}_c{c}_grid-ao.txt", dtype=int).reshape(-1, 3)
    orb = json.load(open(os.path.join(_HERE, "diamond_symm_orbit_data.json")))[mesh]
    if n_IP is None:
        n_IP = int((gg[:, 0] * gg[:, 1]).sum()) // int(orb["nk"])
    if n_IP != int(orb["nIP"]):
        scaled = []
        for fam in orb["families"]:
            base = list(fam["orbit_sizes"]); out = []; i = 0
            while sum(out) + base[i % len(base)] <= n_IP:
                out.append(base[i % len(base)]); i += 1
                if i > 100000: break
            rem = n_IP - sum(out)
            for sz in sorted(set(base), reverse=True):
                while rem >= sz: out.append(sz); rem -= sz
            fam["orbit_sizes"] = out
    sizes: List[int] = []
    for fam in orb["families"]:
        sizes.extend(fam["orbit_sizes"])
    return IrrepData(
        mesh=mesh, N_k=int(orb["nk"]), n_IP=n_IP, n_orb=n_orb, n_occ_tot=n_occ,
        W=tuple((int(a), int(b)) for a, b, _ in gg),
        X=tuple((int(a), int(b), int(c)) for a, b, c in ga),
        orbit_sizes=tuple(sizes), n_families=int(orb["n_families"]),
    )


# ===========================================================================
# helpers
# ===========================================================================


def _qroam_pair(shape: Sequence[int], word: int) -> List[Bloq]:
    """A load/erase pair over ``shape`` with a ``word``-bit output, Lambda by measured optimum."""
    shape = tuple(int(x) for x in shape)
    lf = optimal_log_block_sizes_measured(shape, (word,))
    la = optimal_log_block_sizes_measured(shape, (word,), adjoint=True)
    return [QROAMClean.build_from_bitsize(shape, target_bitsizes=(word,), log_block_sizes=lf),
            QROAMCleanAdjoint.build_from_bitsize(shape, target_bitsizes=(word,),
                                                 log_block_sizes=la)]


def _group_by_size(pairs: Sequence[Tuple[int, ...]]) -> Dict[Tuple[int, ...], int]:
    """``{size tuple: how many blocks have it}`` -- the ragged-to-Sum_i device (§9.6)."""
    c: Counter = Counter()
    for p in pairs:
        c[tuple(int(x) for x in p)] += 1
    return dict(c)


# ===========================================================================
# Q -- the space-group symmetry adaptation transform (spec §9.8)
# ===========================================================================


@attrs.frozen
class QSymmetryAdaptation(Bloq):
    r"""``Q : |k>|I> -> |i>|a>|c>``, built to be ``N_k``-independent (spec §9.8).

    Three stages, none addressed by ``k``:

    **A. Orbit-unitary synthesis** -- ``Q^{(f)} = (+)_o U_o`` over little-group orbits of
    interpolation points.  Orbits are **grouped by size**: each distinct ``|o|`` gets one
    block-indexed interferometer with ``n_blocks`` = the number of orbits of that size, so
    the data cost is ``Sum_o |o|^2`` and the address is ``(family, orbit)`` -- *never* ``k``.

    **B. Member reconstruction** -- ``Q^{(k)} = P(g_k) Lambda(g_k,k) Q^{(f)}``: a select over
    the ``|P| = 48`` point ops loading ``perm[g,I]`` and the fold-back vector ``L_I(g)`` from
    fixed ``|P| x n_IP`` geometry tables, then the Bloch phase ``e^{-i k.L}`` accumulated into
    the phase gradient.  ``|P|`` and ``n_IP`` are fixed; only the ``k.L`` multiply touches
    ``k``, at ``~log2 N_k`` width.

    **C. Reindex** ``(k, I') -> (i, a, c)`` -- a QROAM over ``(n_families, n_IP)``.  This is
    the stage §9.7 insists on: ``Q``'s blocks are **orbits**, ``X``/``W``'s are **irreps**,
    and the two partitions are transverse, so the regrouping is a real cost.

    Family/coset determination is ``|P|`` comparisons of the ``k`` register against fixed
    representatives (``k* = min_g A_g k``), i.e. ``O(|P| log N_k)`` and no ``N_k`` table.
    """

    N_k: int
    n_IP: int
    n_families: int
    orbit_sizes: Tuple[int, ...] = attrs.field(converter=tuple)
    n_ops: int = 48
    phase_bitsize: int = 32
    optimal_T: bool = True

    @cached_property
    def k_bitsize(self) -> int:
        return int(bit_length(self.N_k - 1))

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('k', QAny(self.k_bitsize)),
            Register('fiber', QAny(int(bit_length(self.n_IP - 1)))),
            Register('phase_gradient', QAny(self.phase_bitsize)),
        ])

    # ---- stage A ----
    @cached_property
    def orbit_synthesis(self) -> Dict[int, Bloq]:
        """One block-indexed interferometer per distinct orbit size."""
        out: Dict[int, Bloq] = {}
        for size, count in _group_by_size([(s,) for s in self.orbit_sizes]).items():
            d = max(2, size[0])
            if d % 2:
                d += 1                     # the Givens layer pairs modes: even dimension
            out[size[0]] = BlockUnitaryInterferometerSynthesisQROAM(
                n_blocks=count, n_rows=d, phase_bitsize=self.phase_bitsize,
                optimal_T=self.optimal_T)
        return out

    @cached_property
    def orbit_counts(self) -> Dict[int, int]:
        return {k[0]: v for k, v in _group_by_size([(s,) for s in self.orbit_sizes]).items()}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        b = int(self.phase_bitsize)

        # A -- orbit unitaries, addressed by (family, orbit-of-this-size); NOT by k
        for size, bloq in self.orbit_synthesis.items():
            ret[bloq] += 1

        # B -- |P|-op select: perm table and fold-back table, then the Bloch phase
        for word in (int(bit_length(self.n_IP - 1)), 3 * self.k_bitsize):
            load, erase = _qroam_pair((self.n_ops, self.n_IP), word)
            ret[load] += 1
            emit_range_safety(ret, (self.n_ops, self.n_IP))
            ret[erase] += 1
        ret[SignedCtrlAddIntoPhaseGrad(b)] += 3              # k.L, one per axis (App. A: b-2 each)

        # C -- reindex (k, I') -> (i, a, c), transverse to Q's orbit blocks (§9.7)
        load, erase = _qroam_pair((self.n_families, self.n_IP),
                                  2 * int(bit_length(self.n_IP - 1)))
        ret[load] += 1
        emit_range_safety(ret, (self.n_families, self.n_IP))
        ret[erase] += 1

        # family / coset op: |P| comparisons against fixed representatives
        ret[LessThanConstant(bitsize=self.k_bitsize, less_than_val=self.N_k - 1)] += self.n_ops
        return ret


# ===========================================================================
# ragged block primitives
# ===========================================================================


@attrs.frozen
class CopyMaskProjector(Bloq):
    r"""Block-encoded diagonal mask ``keep c < valid_i`` (spec §9.6).

    The padding projector, not the occ/virt split -- that is absorbed into the synthesized
    column count (§9.1).  Registers are allocated at the max width and the surplus
    dimensions are zeroed by a diagonal ``[1..1,0..0]``: load ``valid_i`` from a small QROM
    indexed by the irrep, compare against ``c``, and apply ``Z R_y`` on the flag.  Never pad
    the *data*.
    """

    n_irreps: int
    max_copy: int
    phase_bitsize: int = 32

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('irrep', QAny(int(bit_length(self.n_irreps - 1)))),
            Register('copy', QAny(int(bit_length(self.max_copy - 1)))),
            Register('flag', QAny(1)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        w = int(bit_length(self.max_copy - 1))
        load, erase = _qroam_pair((self.n_irreps,), w)
        ret[load] += 1
        emit_range_safety(ret, (self.n_irreps,))
        ret[erase] += 1
        ret[LessThanConstant(bitsize=w, less_than_val=self.max_copy - 1)] += 1
        ret[ZGate()] += 1
        return ret


@attrs.frozen
class RaggedBlockIsometry(Bloq):
    r"""Block-indexed isometry over ragged blocks ``{(rows_i, cols_i)}`` (spec §9.6).

    Blocks are **grouped by size**; each distinct ``(rows, cols)`` gets one block-indexed
    column-by-column synthesis with ``n_blocks`` = the count sharing that size.  So the cost
    is the true ``Sum_i`` over blocks rather than ``#blocks x max^2`` -- the ``Sum_r``-not-
    ``k max_r`` property of P-15, obtained by reusing the verified equal-block synthesis
    rather than writing a new ragged one.

    Also emits the padding mask, once, since the register is allocated at the max width.
    """

    blocks: Tuple[Tuple[int, int], ...] = attrs.field(converter=tuple)
    phase_bitsize: int = 32
    optimal_T: bool = True
    #: "iten" (column-by-column) or "berry" (Sec. III B) -- Berry wins for many columns.
    synthesis: str = "iten"

    @cached_property
    def groups(self) -> Dict[Tuple[int, int], int]:
        return _group_by_size(self.blocks)

    @cached_property
    def signature(self) -> Signature:
        rows = max(r for r, _ in self.blocks)
        return Signature([
            Register('irrep', QAny(int(bit_length(len(self.blocks) - 1)))),
            Register('system', QAny(int(bit_length(rows - 1)))),
            Register('phase_gradient', QAny(self.phase_bitsize)),
        ])

    @cached_property
    def mask(self) -> CopyMaskProjector:
        return CopyMaskProjector(n_irreps=len(self.blocks),
                                 max_copy=max(max(c, 2) for _, c in self.blocks),
                                 phase_bitsize=self.phase_bitsize)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        for (rows, cols), count in self.groups.items():
            r, c = max(2, int(rows)), max(1, int(cols))
            if c > r:
                r = c
            if self.synthesis == "berry":
                ret[BerryIsometrySynthesisQROAM(
                    n_blocks=count, n_rows=r, n_cols=c,
                    phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)] += 1
            else:
                ret[ColumnIsometryRectangularBlockEncoding(
                    n_blocks=count, n_rows=r + c, n_reflections=c,   # dilated completion
                    phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)] += 1
        ret[self.mask] += 1
        return ret


@attrs.frozen
class WExchangeIrrep(Bloq):
    r"""Exchange central in the irrep basis: ``W = U D U^dag`` (spec §2, §7 item 3).

    ``U = (+)_i 1_{m_i} (x) u_i`` -- the eigenvector unitary **must be synthesized, twice**;
    it is not free.  ``D`` is a diagonal over the ``Sum_i n_i`` eigenvalues.  Copy blocks are
    grouped by size so ragged ``{n_i}`` costs ``Sum_i``.
    """

    n_i: Tuple[int, ...] = attrs.field(converter=tuple)
    phase_bitsize: int = 32
    optimal_T: bool = True

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('irrep', QAny(int(bit_length(len(self.n_i) - 1)))),
            Register('copy', QAny(int(bit_length(max(self.n_i) - 1)))),
            Register('phase_gradient', QAny(self.phase_bitsize)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        for size, count in _group_by_size([(n,) for n in self.n_i]).items():
            d = max(2, size[0])
            if d % 2:
                d += 1
            u = BlockUnitaryInterferometerSynthesisQROAM(
                n_blocks=count, n_rows=d, phase_bitsize=self.phase_bitsize,
                optimal_T=self.optimal_T)
            ret[u] += 2                                     # U and U^dag -- NOT free
        # D: one real eigenvalue per (irrep, copy)
        load, erase = _qroam_pair((len(self.n_i), max(self.n_i)), int(self.phase_bitsize))
        ret[load] += 1
        emit_range_safety(ret, (len(self.n_i), max(self.n_i)))
        ret[SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)] += 1   # App. A: b-2
        ret[ZGate()] += 1
        ret[erase] += 1
        return ret


# ===========================================================================
# the two symmetry-adapted templates
# ===========================================================================


@attrs.frozen
class SymmetryAdaptedExchangeTemplate(Bloq):
    r"""``C_ov^ex`` symmetry-adapted -- **6 Q's** (spec §9.3)."""

    data: IrrepData
    phase_bitsize: int = 32
    optimal_T: bool = True

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(2 * self.data.n_IP.bit_length())),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    @cached_property
    def Q(self) -> QSymmetryAdaptation:
        d = self.data
        return QSymmetryAdaptation(N_k=d.N_k, n_IP=d.n_IP, n_families=d.n_families,
                                   orbit_sizes=d.orbit_sizes,
                                   phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)

    @cached_property
    def X_o(self) -> RaggedBlockIsometry:
        d = self.data
        return RaggedBlockIsometry(blocks=tuple((ng, max(1, o)) for (_, ng, _), o
                                                in zip(d.X, d.n_occ)),
                                   phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)

    @cached_property
    def X_v(self) -> RaggedBlockIsometry:
        d = self.data
        return RaggedBlockIsometry(blocks=tuple((ng, max(1, v)) for (_, ng, _), v
                                                in zip(d.X, d.n_virt)),
                                   phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)

    @cached_property
    def W(self) -> WExchangeIrrep:
        return WExchangeIrrep(n_i=tuple(self.data.n_i), phase_bitsize=self.phase_bitsize,
                              optimal_T=self.optimal_T)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        kb = self.Q.k_bitsize
        ret[self.X_o] += 2                       # X_o and X_o^dag
        ret[self.X_v] += 2
        ret[self.Q] += 6                         # §9.3: Q^dag(x)2, Q..Q^dag around W, Q(x)2
        ret[self.W] += 1                         # U D U^dag
        ret[Subtract(QUInt(kb))] += 2            # momentum transfer, forward and reversed
        return ret


@attrs.frozen
class SymmetryAdaptedDirectTemplate(Bloq):
    r"""``C_ov^dir`` symmetry-adapted -- **4 Q's** (spec §9.4).

    No ``W``-side ``Q``'s: after the Fourier transform ``W`` is a real diagonal over
    ``(R = R1-R2, mu, nu)``, ``alpha = 1`` (the DFT consumes the ``1/N_k``).  ``same_spin``
    marks the ``oo``/``vv`` siblings, which add only two isometries each -- everything else
    is routed in (§9.5).
    """

    data: IrrepData
    phase_bitsize: int = 32
    optimal_T: bool = True
    same_spin: Optional[str] = None

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(2 * self.data.n_IP.bit_length())),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    @cached_property
    def _ex(self) -> SymmetryAdaptedExchangeTemplate:
        return SymmetryAdaptedExchangeTemplate(data=self.data,
                                               phase_bitsize=self.phase_bitsize,
                                               optimal_T=self.optimal_T)

    @cached_property
    def central(self) -> DiagonalCoulombKernelBlockEncoding:
        """Real-space diagonal ``W(R, mu, nu)``; real data -> one rotation, b-bit word."""
        d = self.data
        return DiagonalCoulombKernelBlockEncoding(
            N_k=d.N_k, N_IP=max(d.n_i), phase_bitsize=self.phase_bitsize,
            optimal_T=self.optimal_T, complex_data=False)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ex = self._ex
        kb = ex.Q.k_bitsize
        if self.same_spin is not None:
            X = ex.X_o if self.same_spin == "oo" else ex.X_v
            ret[X] += 2                          # §9.5: only 2 isometries; Q and W routed in
            return ret
        ret[ex.X_o] += 2
        ret[ex.X_v] += 2
        ret[ex.Q] += 4                           # §9.4: only the X-attached Q's
        ret[QFTTextBook(bitsize=kb)] += 4        # both momentum registers, fwd + inv
        ret[Subtract(QUInt(kb))] += 1            # R = R1 - R2
        ret[self.central] += 1
        return ret


# ===========================================================================
# Fused variants (spec §9.7's "match the coordinates" -- P-16)
# ===========================================================================


@attrs.frozen
class FusedBlockSynthesis(Bloq):
    r"""Block-diagonal synthesis on **one fused register** of dimension ``l = Sum_i n_i``.

    The block label is the high part of a single register rather than a register of its own
    (P-16).  Layers run to ``max_i n_i``; at layer ``r`` the table holds only the blocks
    still live, ``a_r = Sum_{n_i > r} n_i``; finished blocks are held inert by the
    ``ceil(log2(l/a_r))`` surplus-address control -- which is P-13 range safety costed
    *inside* the construction, the one place the 2026 literature does that.

    Cost shape, quoted (arXiv:2605.28489 §2, ledger P-16):

    .. math::
        C_V = \sum_{r}\Big(\lceil a_r/2\Lambda_r\rceil + \lceil\log(l/a_r)\rceil
              + 2b\Lambda_r\Big) + \lceil l/\Lambda\rceil + b\Lambda + s\lceil\log l\rceil

    Fusing removes the block register but forces a **coordinate match**: the orbit-organised
    and irrep-organised fused indices differ by a permutation, which is itself a QROAM
    (``reindex_permutation``).  Indexed got that for free from having a block register.
    """

    sizes: Tuple[int, ...] = attrs.field(converter=tuple)
    phase_bitsize: int = 32
    n_cols: Optional[Tuple[int, ...]] = attrs.field(default=None,
                                                   converter=lambda x: tuple(x) if x else None)
    with_reindex: bool = True

    @cached_property
    def l(self) -> int:
        return int(sum(self.sizes))

    @cached_property
    def s(self) -> int:
        """Merged layer count = max block size, +1 for the alignment filler (P-16)."""
        return int(max(self.sizes)) + 1

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('fused', QAny(int(bit_length(self.l - 1)))),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    def _live(self, r: int) -> int:
        """``a_r`` -- indices still live at layer ``r``."""
        cols = self.n_cols
        if cols is None:
            return sum(n for n in self.sizes if n > r)
        return sum(n for n, c in zip(self.sizes, cols) if c > r)

    @cached_property
    def reindex_permutation(self) -> List[Bloq]:
        """``|i> -> |P(i)>`` over the fused index: load ``P(i)``, swap, erase (P-16)."""
        w = int(bit_length(self.l - 1))
        return _qroam_pair((self.l,), w)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        b = int(self.phase_bitsize)
        nlayer = int(max(self.n_cols)) if self.n_cols else int(max(self.sizes))
        for r in range(nlayer):
            a_r = self._live(r)
            if a_r <= 0:
                continue
            load, erase = _qroam_pair((a_r,), 2 * b)
            ret[load] += 1
            emit_range_safety(ret, (a_r,))                 # the ceil(log(l/a_r)) inertness
            ret[SignedCtrlAddIntoPhaseGrad(b)] += 2         # App. A: b-2 each
        # final diagonal phase layer over the whole fused register
        load, erase = _qroam_pair((self.l,), b)
        ret[load] += 1
        ret[AddIntoPhaseGrad(b, b)] += 1
        ret[erase] += 1
        # s * ceil(log l) increments
        ret[Subtract(QUInt(max(2, int(bit_length(self.l - 1)))))] += self.s
        if self.with_reindex:
            p_load, p_erase = self.reindex_permutation
            ret[p_load] += 1
            ret[p_erase] += 1
        return ret


@attrs.frozen
class FusedExchangeTemplate(Bloq):
    """`C_ov^ex`, every block-diagonal object on a fused register."""

    data: IrrepData
    phase_bitsize: int = 32

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(2 * self.data.n_IP.bit_length())),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    @cached_property
    def Q_fused(self) -> FusedBlockSynthesis:
        return FusedBlockSynthesis(sizes=self.data.orbit_sizes, phase_bitsize=self.phase_bitsize)

    @cached_property
    def X_o_fused(self) -> FusedBlockSynthesis:
        d = self.data
        return FusedBlockSynthesis(sizes=tuple(ng for _, ng, _ in d.X),
                                   n_cols=tuple(max(1, o) for o in d.n_occ),
                                   phase_bitsize=self.phase_bitsize)

    @cached_property
    def X_v_fused(self) -> FusedBlockSynthesis:
        d = self.data
        return FusedBlockSynthesis(sizes=tuple(ng for _, ng, _ in d.X),
                                   n_cols=tuple(max(1, v) for v in d.n_virt),
                                   phase_bitsize=self.phase_bitsize)

    @cached_property
    def W_fused(self) -> FusedBlockSynthesis:
        return FusedBlockSynthesis(sizes=tuple(self.data.n_i), phase_bitsize=self.phase_bitsize)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        kb = int(bit_length(self.data.N_k - 1))
        ret[self.X_o_fused] += 2
        ret[self.X_v_fused] += 2
        ret[self.Q_fused] += 6
        ret[self.W_fused] += 2                          # U and U^dag
        load, erase = _qroam_pair((sum(self.data.n_i),), int(self.phase_bitsize))
        ret[load] += 1                                  # D over Sum n_i eigenvalues
        ret[SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)] += 1   # App. A: b-2
        ret[ZGate()] += 1
        ret[erase] += 1
        ret[Subtract(QUInt(kb))] += 2
        return ret


@attrs.frozen
class FusedDirectTemplate(Bloq):
    """`C_ov^dir` (and oo/vv) with every block-diagonal object fused."""

    data: IrrepData
    phase_bitsize: int = 32
    same_spin: Optional[str] = None

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(2 * self.data.n_IP.bit_length())),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    @cached_property
    def _ex(self) -> FusedExchangeTemplate:
        return FusedExchangeTemplate(data=self.data, phase_bitsize=self.phase_bitsize)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ex = self._ex
        kb = int(bit_length(self.data.N_k - 1))
        if self.same_spin is not None:
            ret[ex.X_o_fused if self.same_spin == "oo" else ex.X_v_fused] += 2
            return ret
        ret[ex.X_o_fused] += 2
        ret[ex.X_v_fused] += 2
        ret[ex.Q_fused] += 4
        ret[QFTTextBook(bitsize=kb)] += 4
        ret[Subtract(QUInt(kb))] += 1
        ret[DiagonalCoulombKernelBlockEncoding(
            N_k=self.data.N_k, N_IP=max(self.data.n_i),
            phase_bitsize=self.phase_bitsize, optimal_T=True, complex_data=False)] += 1
        return ret


# ===========================================================================
# Fused synthesis, corrected -- the real Iten (column, control-count) layer grid
# ===========================================================================
#
# Implementation features, all established by measurement earlier in this work:
#
#  F1  Per-block power-of-two ROW padding, 2^ceil(log2 n_i), NOT a common power of two
#      (per-block l = 2,576 vs uniform 6,272 at 6x6x6, 2.43x cheaper).  Padded rows carry
#      zero amplitude; [V;0] still has orthonormal columns so it is still an isometry.
#  F2  Blocks laid out LARGEST FIRST, so s_i = Sum_{j<i} 2^{a_j} is a multiple of 2^{a_i}
#      (buddy allocation).  Padding alone is NOT enough: file order leaves 34/49 blocks
#      misaligned and 1,024 bit-s pairs straddling a block boundary; descending gives 0.
#  F3  Secondary sort by COLUMN count descending within each padded-size class.  Free
#      (alignment only depends on the padded size, which is constant inside a class) and
#      it cuts the live set from up to 10 disjoint runs to 1-3, which is what makes the
#      range restriction valid.
#  F4  Column c of block i targets e_{s_i + c}, never e_k.  With F2 the offset s_i has
#      zeros in the low a_i bits, so this is "write c into the low bits" -- free.
#      Naive e_k targeting still reconstructs V but makes the elimination depth scale
#      with the FUSED dimension instead of the block dimension (quadratic in block count).
#  F5  No compaction of the padding holes.  The permutation is cheap in principle
#      (proportional to block count, not address space) but nets only +8 Toffolis per
#      layer out of 768 at 6x6x6 and -48 at 2x2x2.  Not worth the decode path.
#  F6  Layer grid is (column c, control count s), K*(n-1) layers, table (n_live, 2^{s+1}).
#      Fusing merges all live blocks into ONE layer per (c,s) instead of one stack per
#      size group.  This is the actual saving; the address space is not.
#  F7  W (ov exchange) is NOT an isometry -- it is the central kernel, block-diagonal and
#      Hermitian.  It goes through U D U^dag with U a block-diagonal UNITARY (chi_i = n_i,
#      the full square block), synthesized twice, plus one diagonal over Sum_i n_i.
#  F8  Complex data throughout (word 2b for a phase-pair layer, b for the final diagonal).


def _fused_layout(sizes: Sequence[int], n_cols: Sequence[int]) -> List[Tuple[int, int, int]]:
    """F1+F2+F3: ``[(padded_rows, a_i, chi_i)]`` sorted by ``(-padded, -chi)``."""
    p2 = lambda x: 1 << (int(x) - 1).bit_length()
    out = [(p2(n), int(p2(n)).bit_length() - 1, int(k)) for n, k in zip(sizes, n_cols)]
    return sorted(out, key=lambda t: (-t[0], -t[2]))


def _n_runs(live_flags: Sequence[bool]) -> int:
    """Number of contiguous live intervals (F3); each needs its own range restriction."""
    r, prev = 0, False
    for x in live_flags:
        if x and not prev:
            r += 1
        prev = x
    return max(1, r)


@attrs.frozen
class FusedColumnSynthesis(Bloq):
    r"""Fused column-by-column synthesis on the real Iten layer grid (F1-F6, F8).

    ``sizes`` are the TRUE block row counts; padding, ordering and alignment are applied
    internally.  ``n_cols=None`` means every block is a full square unitary (F7's ``U``).
    """

    sizes: Tuple[int, ...] = attrs.field(converter=tuple)
    n_cols: Optional[Tuple[int, ...]] = attrs.field(
        default=None, converter=lambda x: tuple(int(v) for v in x) if x else None)
    phase_bitsize: int = 32
    with_reindex: bool = True
    phase_per_column: bool = False

    @cached_property
    def _cols(self) -> Tuple[int, ...]:
        p2 = lambda x: 1 << (int(x) - 1).bit_length()
        return self.n_cols if self.n_cols else tuple(p2(n) for n in self.sizes)

    @cached_property
    def layout(self) -> List[Tuple[int, int, int]]:
        return _fused_layout(self.sizes, self._cols)

    @cached_property
    def l_pad(self) -> int:
        return sum(pr for pr, _, _ in self.layout)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('fused', QAny(int(bit_length(self.l_pad - 1)))),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        b = int(self.phase_bitsize)
        lay = self.layout
        maxc = max(k for _, _, k in lay)
        for c in range(maxc):
            live = [(pr, a, k) for pr, a, k in lay if k > c]
            if not live:
                continue
            amax = max(a for _, a, _ in live)
            for s in range(1, amax):                       # F6: control counts 1..n-1
                nb = sum(1 for _, a, _ in live if a > s)
                if nb == 0:
                    continue
                lbs = _phase_layer_lbs(nb, 1 << (s + 1), b, True, None)
                ret[BlockInterferometerPhaseLayerQROAM(
                    n_blocks=nb, n_rows=1 << (s + 1), phase_bitsize=b,
                    log_block_sizes=lbs)] += 1
            # F3: one range restriction per contiguous live run
            runs = _n_runs([k > c for _, _, k in lay])
            for _ in range(runs):
                emit_range_safety(ret, (max(2, sum(pr for pr, _, _ in live)),))
        # Iten fix-up gates, per block, plus their block-dependent angle table
        # F15 -- Iten fix-up gates are shared across a SIZE CLASS, not per block.  The
        # pair address (i0, i1) = (a_{s+1} 2^{s+1} + f, +2^s) depends only on the column
        # and the bit position, so with F2 alignment it is the same LOW bits in every
        # block of that padded size and one multi-controlled gate fires in all of them.
        # Charging one per block over-counts 8.8x at 6x6x6 (1,081 vs 123).
        # F16 -- and the gate is one App.-A signed rotation (b-2), not a naively
        # controlled adder (2(b-1)); RealMultiControlledRotationQROAM predates that fix.
        cls_max: Dict[int, int] = {}
        for pr, _, kk in lay:
            cls_max[pr] = max(cls_max.get(pr, 0), kk)
        n_mcg = 0
        for pr, kmax in cls_max.items():
            n = num_mcgs(pr, min(kmax, pr))
            if not n:
                continue
            n_mcg += n
            nctrl = max(1, (int(pr).bit_length() - 1) - 1)
            # F19 -- NO class flag and NO held decode are needed.  The angle QROM is
            # range-restricted to this class's window of upper-bit values, so outside it
            # the angle register stays |0> and the rotation is the identity: the spurious
            # low-bit matches in other classes are harmless.  Controls are just the a-1
            # non-target low bits.  (Author's correction, 2026-08-20.)
            ret[And()] += n * max(0, nctrl - 1)
            ret[And().adjoint()] += n * max(0, nctrl - 1)
            ret[SignedCtrlAddIntoPhaseGrad(b)] += n
            # ...but the ANGLE is per block: one QROM over the blocks of THIS class per
            # fix-up gate.  The gate is shared, the data is not.
            nblk = sum(1 for q, _, _ in lay if q == pr)
            ld, er = _qroam_pair((max(2, nblk),), b)
            ret[ld] += n
            ret[er] += n
        # F17 -- optional: rotation-only sublayers with the phase stripped per column
        # ("remove phase first"), which needs one diagonal over the live rows per column
        # instead of a 2-angle word per sublayer entry.
        if self.phase_per_column:
            for c in range(maxc):
                live = sum(pr for pr, _, kk in lay if kk > c)
                if live <= 0:
                    continue
                pl, pe = _qroam_pair((live,), b)
                ret[pl] += 1
                ret[AddIntoPhaseGrad(b, b)] += 1
                ret[pe] += 1
        # one merged final diagonal over the whole fused index
        ret[BlockInterferometerFinalPhasesQROAM(
            n_blocks=1, n_rows=self.l_pad, phase_bitsize=b,
            log_block_sizes=optimal_interferometer_log_block_sizes(1, self.l_pad, b),
            adjoint_log_block_sizes=optimal_interferometer_log_block_sizes(1, self.l_pad, b))] += 1
        if self.with_reindex:
            pl, pe = _qroam_pair((self.l_pad,), int(bit_length(self.l_pad - 1)))
            ret[pl] += 1
            ret[pe] += 1
        return ret


@attrs.frozen
class FusedInterferometerSynthesis(Bloq):
    r"""Fused synthesis for **square** blocks -- layers of disjoint 2x2 rotations (F7, F9).

    F9  A square block is synthesized by the interferometer mesh, NOT column-by-column.
        Column-by-column on an ``n x n`` unitary runs ``n(log2 n - 1)`` layer invocations;
        the mesh runs ``n``.  Same data, fewer loads.  The mesh also needs no power-of-two
        padding: its pairs are ADJACENT, so they are block-local as long as each block
        starts at an even offset -- pad each block to an even size, not to a power of two.

    Layer ``r`` holds the still-live blocks (``n_i > r``); a finished block contributes
    ``1_{n_i}`` and drops out of the table.
    """

    sizes: Tuple[int, ...] = attrs.field(converter=tuple)
    phase_bitsize: int = 32
    with_reindex: bool = True

    @cached_property
    def layout(self) -> List[int]:
        return sorted((n + (n % 2) for n in self.sizes), reverse=True)   # even, largest first

    @cached_property
    def l(self) -> int:
        return sum(self.layout)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('fused', QAny(int(bit_length(self.l - 1)))),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        b = int(self.phase_bitsize)
        lay = self.layout
        for r in range(max(lay)):
            live = [n for n in lay if n > r]
            pairs = sum(n // 2 for n in live)
            if pairs <= 0:
                continue
            n_rows = max(2, 2 * pairs)
            ret[BlockInterferometerPhaseLayerQROAM(
                n_blocks=1, n_rows=n_rows, phase_bitsize=b,
                log_block_sizes=optimal_interferometer_log_block_sizes(1, n_rows, b))] += 1
            for _ in range(_n_runs([n > r for n in lay])):
                emit_range_safety(ret, (max(2, sum(live)),))
        ret[BlockInterferometerFinalPhasesQROAM(
            n_blocks=1, n_rows=self.l, phase_bitsize=b,
            log_block_sizes=optimal_interferometer_log_block_sizes(1, self.l, b),
            adjoint_log_block_sizes=optimal_interferometer_log_block_sizes(1, self.l, b))] += 1
        if self.with_reindex:
            pl, pe = _qroam_pair((self.l,), int(bit_length(self.l - 1)))
            ret[pl] += 1
            ret[pe] += 1
        return ret


@attrs.frozen
class QAdaptationFused(Bloq):
    r"""``Q : |k>|I> -> |i>|a>|c>`` -- the space-group adaptation, done correctly.

    Structure, every part checked against ``diamond_symm_orbit_data.json`` (F10-F13):

    F10 ``Q`` is **family-multiplexed and acts on the grid index only**.  The 16 families
        are the 16 k-stars (``Sum_f star_size = 216 = N_k`` exactly); within a family the
        little group acts on the ``n_IP = 136`` grid points, giving orbits whose sizes
        divide ``|LG|`` and sum to 136 for every family.  Address is ``16 x 136 = 2,176``,
        with **no** ``N_k`` factor -- ``Q`` does not touch the momentum register.
    F11 The index regrouping ``(f, mu, kappa, nu, c) -> (f, kappa | mu, nu, c)`` is pure
        rewiring: ``f`` is shared, ``kappa`` moves into ``k``, and ``(mu,nu,c)`` collapse
        into ``I``.  Only two things cost anything: prying ``(f, mu, c)`` out of the fused
        padded ``L``, and the 136-value irrep->orbit permutation inside a family.
    F12 Prying ``(f, mu, c)`` out of ``L`` is cheap **because of F2/F3**: largest-first
        ordering puts the size classes in contiguous ranges, so it is
        ``ceil(log2 n_cls)`` comparisons + a staged controlled shift + a mask, not a
        49-way decode.  The partner register is inert (Schur), so widening it into fixed
        ``kappa|nu`` subfields makes that split free in Toffolis -- qubits, not gates.
    F13 Diamond is **nonsymmorphic** (``Fd-3m``), so ``R_kappa`` carries a fractional
        translation and ``Q = P_kappa D_kappa q_f`` with ``D_kappa`` a phase
        ``e^{-i k.tau}``.  It is a diagonal over ``k`` (``N_k`` entries), not over the grid
        index.  It **cancels** in the direct term (diagonal middle, diagonals commute) but
        not obviously against ``W``'s block structure in the exchange term, so it is
        charged here.  ``with_phase=False`` recovers the direct-term case.

    Free labelling choices that turn lookups into wires: label the mesh **star-major** so
    ``k = (f,kappa)`` is rewiring, and the grid index **orbit-major** so ``J = I``.
    """

    orbit_sizes: Tuple[int, ...] = attrs.field(converter=tuple)
    n_families: int = 16
    n_IP: int = 136
    N_k: int = 216
    n_blocks_X: int = 49
    l_pad_X: int = 2576
    n_size_classes: int = 5
    phase_bitsize: int = 32
    with_phase: bool = True

    @cached_property
    def synthesis(self) -> FusedInterferometerSynthesis:
        """F10: the orbit-block unitary; reindex is emitted explicitly below, not here."""
        return FusedInterferometerSynthesis(
            sizes=self.orbit_sizes, phase_bitsize=self.phase_bitsize, with_reindex=False)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('grid', QAny(int(bit_length(self.n_IP - 1)))),
                          Register('momentum', QAny(int(bit_length(self.N_k - 1)))),
                          Register('phase_gradient', QAny(self.phase_bitsize))])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        b = int(self.phase_bitsize)
        wL = int(bit_length(self.l_pad_X - 1))
        # F12 -- size-class decode, then split L into (block-in-class, copy)
        n_cmp = max(1, int(ceil(np_log2(self.n_size_classes))))
        ret[LessThanConstant(wL, self.l_pad_X // 2)] += n_cmp
        ret[Subtract(QUInt(wL))] += 1                      # subtract class start
        ret[Subtract(QUInt(wL))] += n_cmp                  # staged controlled shift
        ld, er = _qroam_pair((self.n_blocks_X,), 8)        # block -> (f, mu)
        ret[ld] += 1
        ret[er] += 1
        # F11 -- irrep-major -> orbit-major inside a family; 16 x 136 entries
        pl, pe = _qroam_pair((self.n_families * self.n_IP,), int(bit_length(self.n_IP - 1)))
        ret[pl] += 1
        ret[pe] += 1
        # F10 -- the orbit-block unitary itself
        ret[self.synthesis] += 1
        # F13 -- nonsymmorphic phase, a diagonal over k
        if self.with_phase:
            dl, de = _qroam_pair((self.N_k,), b)
            ret[dl] += 1
            ret[AddIntoPhaseGrad(b, b)] += 1
            ret[de] += 1
        return ret


@attrs.frozen
class StaircaseShift(Bloq):
    r"""**P-17.** The ragged<->padded index map, as one primitive (F14).

    A monotone piecewise-constant shift: with segment starts ``t_1<...<t_{k-1}`` and
    per-segment shifts ``delta_1..delta_{k-1}``,

    .. math::  x \;\longmapsto\; x + \sum_{j\,:\,x \ge t_j} \delta_j .

    As a matrix it is the identity on the diagonal up to the first boundary, then the
    diagonal *jumps* to a further column and continues, jumps again, and so on -- a
    staircase of unit diagonals.  It is injective for any non-negative shifts, so it is a
    permutation onto its image and is its own inverse structure with ``-delta``.

    **The one subtlety.** Applied in place and in increasing ``j``, the register no longer
    holds ``x`` after the first shift.  But the map is monotone, so ``[x >= t_j]`` equals
    ``[x' >= t_j + Delta_{j-1}]`` -- the boundary measured on the *output* side.  Those are
    classical constants (they are exactly the padded-side block starts ``s_j``), so no
    arithmetic on the boundary is needed.  Verified against the definition for all 1,720
    inputs of the 6x6x6 ``X`` grid.

    **Cost.** Per boundary: one ``w``-bit compare-against-constant (``w-1`` ANDs), one
    controlled add of a constant (``w-1``), and the compare uncomputed by AND-ladder
    measurement (**free**).  Total

    .. math::  C = 2(w-1)(k-1) .

    Qualtran's ``LessThanConstant.adjoint()`` is charged in full rather than measured out,
    which would give ``(3w-1)(k-1)``; the AND form below is what the construction actually
    admits.

    Used three times in this module: compact<->padded on the fused ``X`` index, the
    size-class decode inside ``Q``'s alignment, and the irrep->orbit offset.
    """

    bitsize: int
    boundaries: Tuple[int, ...] = attrs.field(converter=tuple)   # output-side starts s_j
    shifts: Tuple[int, ...] = attrs.field(converter=tuple)
    free_uncompute: bool = True

    def __attrs_post_init__(self):
        assert len(self.boundaries) == len(self.shifts)
        assert list(self.boundaries) == sorted(self.boundaries)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', QAny(self.bitsize))])

    @cached_property
    def n_boundaries(self) -> int:
        return len(self.boundaries)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        w = int(self.bitsize)
        for s, dl in zip(self.boundaries, self.shifts):
            if dl == 0:
                continue
            ret[And()] += w - 1                                   # compare x >= s
            ret[AddK(QUInt(w), int(dl)).controlled()] += 1         # conditional shift
            if self.free_uncompute:
                ret[And().adjoint()] += w - 1                      # measured out: free
            else:
                ret[And()] += w - 1
        return ret

    @staticmethod
    def from_block_sizes(sizes: Sequence[int], pad=None) -> "StaircaseShift":
        """Build the compact->padded map for blocks laid out in the given order."""
        p2 = lambda x: 1 << (int(x) - 1).bit_length()
        pad = pad or p2
        bnds, shifts, acc = [], [], 0
        for n in list(sizes)[:-1]:
            acc += pad(n)                       # output-side boundary = padded start
            bnds.append(acc)
            shifts.append(pad(n) - n)
        w = int(bit_length(sum(pad(n) for n in sizes) - 1))
        return StaircaseShift(bitsize=w, boundaries=tuple(bnds), shifts=tuple(shifts))
