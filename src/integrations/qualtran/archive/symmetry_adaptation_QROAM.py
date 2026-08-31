r"""Symmetry-adaptation (Clebsch--Gordan) transform ``Q`` as an :math:`n_k`-independent bloq.

This is a *resource model* for the quantum circuit that implements the space-group
symmetry-adaptation transform

    ``Q : |k>|I>  ->  |i>|a>|c>``

mapping an ISDF grid basis state (crystal momentum ``k``, interpolation point ``I``) to the
irrep-adapted basis (irrep ``i``, partner ``a``, copy/multiplicity ``c``).  The whole point of the
construction -- and the **hard requirement** of this module -- is that the dominant cost does **not**
scale with the number of ``k``-points ``n_k``.  It scales with the *unique unitary data*
``Sum_{family reps} Sum_{orbits} |o|^2``, which saturates (is flat) in ``n_k``.

Structure exploited (this is the entire design)
------------------------------------------------

1.  ``Q`` is **block-diagonal in ``k``**: ``Q = (+)_k Q^{(k)}``, each ``Q^{(k)}`` an ``n_IP x n_IP``
    unitary acting on the ``I`` register.

2.  ``k``-points fall into **families / stars** under the crystal point group ``P`` (order
    ``|P|``, ``= 48`` for diamond).  Within a family the member blocks are **phased-permutation**
    images of the family representative ``k_0``:

    .. math::

        Q^{(k_j)}_{perm[g_j, I],\,c}
            \;=\; e^{-i\,k_j \cdot L_I(g_j)}\; Q^{(k_0)}_{I,\,c},
        \qquad k_j = g_j\cdot k_0,\; g_j \in P .

    So **only one block per family is independent**.  Every other member is reconstructed from the
    representative by (a) a permutation ``perm[g_j]`` of the ``I`` register selected by the
    point-group op index ``g \in \{|P|\}``, and (b) a Bloch phase ``exp(-i k \cdot L)`` where
    ``L_I(g)`` is a fold-back lattice vector read from a ``|P| x n_IP`` geometry table.  Both the
    permutation select (over ``|P|``) and the geometry table are **independent of ``n_k``**.

3.  Within a family representative, ``Q^{(k_0)}`` is itself block-diagonal over interpolation-point
    **orbits** under the little group: ``Q^{(k_0)} = (+)_{orbits} U_orbit``, each ``U_orbit`` a small
    ``d x d`` unitary with ``d <= |P| = 48``.

The genuinely independent unitary data is therefore ``Sum_{family reps} Sum_{orbits} |o|^2`` -- the
"unique data".  For diamond: ``2x2x2 -> 6800`` (3 families, 38 orbit-blocks, ``n_k = 8``) and
``6x6x6 -> 12912`` (16 families, 730 orbit-blocks, ``n_k = 216``).  ``n_k`` grows ``27x`` between the
two meshes while the unique data grows only ``~1.9x``; the dominant Toffoli count tracks the latter.

How ``n_k``-independence is achieved
------------------------------------

* **(A) Orbit-unitary synthesis (dominant, ``~ Sum|o|^2``).**  The per-family-representative orbit
  unitaries are grouped by orbit size ``d``.  For each distinct rounded size ``N = 2^ceil(log2 d)``
  the ``count_N`` orbits of that size form one block-diagonal ``(+)_a U_a`` synthesized by a single
  :class:`BlockUnitaryInterferometerSynthesisQROAM` (multiplexed nearest-neighbour beamsplitter
  layers + phase QROAM) with ``n_blocks = count_N`` and ``n_rows = N``.  The QROAM that holds the
  orbit-unitary phase data is addressed by ``(orbit-of-size-N index, beamsplitter-pair index)`` --
  i.e. by *(family, orbit)*, **never by ``k``**.  The QROAM **select-swap tradeoff**
  (``log_block_sizes = lambda``) is chosen per size group to *minimize the real Toffoli count*
  (:func:`optimal_interferometer_log_block_sizes_by_toffoli`), which reduces the load term from
  ``~ entries`` toward ``~ sqrt(entries * b)``.  The data driving the cost is ``Sum_N count_N * N``
  beamsplitter phases ``~ Sum|o|`` (per-layer) over ``N`` layers, ``~ Sum|o|^2`` total -- flat in
  ``n_k``.

* **(B) Member reconstruction (select over ``|P|``, ``n_k``-independent).**  A permutation of the
  ``I`` register selected by the point-op index ``g`` (a ``|P|``-way select realized as a
  ``|P| x n_IP`` QROAM load of ``perm[g, I]`` plus its inverse for in-place routing), followed by the
  Bloch phase: load ``L_I(g)`` from the ``|P| x n_IP`` geometry table and accumulate ``k . L`` into a
  phase-gradient register with :class:`AddIntoPhaseGrad`.  ``|P| = 48`` and ``n_IP`` are fixed; the
  only ``k``-touching quantity is the per-axis width of ``k`` in the ``k . L`` multiply, which is
  ``~ (1/3) log2(n_k)`` -- **logarithmic**, hence subdominant.

* **(C) Reindex + ``k``-map (modeled ``n_k``-independent).**  The representative reindex
  ``I' -> (i, a, c)`` is a QROAM over ``(family, I')`` (``n_families`` bounded, ``n_IP`` fixed).  The
  only step that reads ``k`` is ``k -> (family index, coset op g)``; this is modeled as
  ``n_k``-independent arithmetic -- ``|P|`` equality comparisons of the ``k`` register against a fixed
  set of high-symmetry representatives / coset ops (``EqualsAConstant`` on ``k_bitsize`` bits).  Its
  only ``n_k`` dependence is the ``log2(n_k)`` width of the comparator, kept subdominant.  **This is
  an assumption**: it presumes the star/coset of ``k`` is determined by a fixed-size decision rule
  rather than an ``O(n_k)`` table lookup.

Data-free convention
---------------------
Following ``AGENTS.md`` and the sibling ``*QROAM`` modules, the bloq carries only **sizes/counts**
(the orbit-size multiset, ``n_IP``, ``|P|``, phase/lattice bit widths, register widths) -- never
concrete ROM data.  ``build_call_graph`` composes the real sub-bloqs
(:class:`BlockUnitaryInterferometerSynthesisQROAM`, :class:`QROAMClean`, :class:`AddIntoPhaseGrad`,
:class:`EqualsAConstant`), so ``QECGatesCost`` / ``QubitCount`` report costs from genuine gate
counting, not an analytic assertion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil, log2
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import (
    Bloq,
    DecomposeTypeError,
    GateWithRegisters,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.arithmetic.comparison import EqualsAConstant
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting.generalizers import generalize_cswap_approx
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        split_interferometer_log_block_sizes,
    )
    from .state_prep_QROAM import _to_tuple_or_none
except ImportError:  # pragma: no cover - script / direct execution
    from block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        split_interferometer_log_block_sizes,
    )
    from state_prep_QROAM import _to_tuple_or_none

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


# ============================================================================
# helpers
# ============================================================================


def _clog2(x: int) -> int:
    """``ceil(log2(x))`` clamped to ``>= 1`` for register widths (never a 0-width register)."""
    return max(1, int(ceil(log2(max(2, int(x))))))


def _round_pow2(d: int) -> int:
    """Synthesized dimension ``N`` of a size-``d`` orbit: smallest power of two ``>= d``, floored at 4.

    The floor at 4 (``system_bitsize >= 2``) keeps the multiplexed-interferometer synthesis and its
    ``QubitCount`` well-defined for the abundant tiny orbits (``d = 1, 2``); a ``d <= 2`` block is a
    phase / ``2 x 2`` unitary, so padding it to ``4 x 4`` is a modest, ``n_k``-independent over-count.
    """
    return 1 << max(2, int(ceil(log2(max(2, int(d))))))


def group_orbit_sizes(orbit_sizes: Tuple[int, ...]) -> Dict[int, int]:
    """Group an orbit-size multiset by rounded power-of-two size ``N`` -> ``count_N``.

    All orbits that round up to the same ``N = 2^ceil(log2 d)`` are synthesized together as one
    block-diagonal ``(+)_a U_a`` of ``count_N`` blocks (a single :class:`RecursiveCSDSynthesisQROAM`).
    """
    groups: "Counter[int]" = Counter()
    for d in orbit_sizes:
        groups[_round_pow2(int(d))] += 1
    return dict(sorted(groups.items()))


def rounded_unique_data(orbit_sizes: Tuple[int, ...]) -> int:
    """``Sum_orbits N^2`` with ``N = 2^ceil(log2 d)`` -- the QROAM-realized proxy for ``Sum|o|^2``."""
    return sum(_round_pow2(int(d)) ** 2 for d in orbit_sizes)


def orbit_one_qubit_gates(orbit_sizes: Tuple[int, ...]) -> int:
    """``Sum_orbits (N^2 - 3N/2)`` -- a size proxy for the orbit data (``~ Sum|o|^2``).

    Flat in ``n_k`` because the orbit multiset comes from the *family representatives*, whose count
    saturates with mesh refinement.
    """
    total = 0
    for d in orbit_sizes:
        N = _round_pow2(int(d))
        total += N * N - 3 * N // 2
    return total


def _interferometer_toffoli(
    n_blocks: int, n_rows: int, phase_bitsize: int, lbs: Tuple[int, ...]
) -> int:
    """Real ``QECGatesCost`` Toffoli of one interferometer synthesis with a given QROAM split."""
    bloq = BlockUnitaryInterferometerSynthesisQROAM.from_shape(
        n_blocks=n_blocks,
        n_rows=n_rows,
        phase_bitsize=phase_bitsize,
        log_block_sizes=lbs,
        final_log_block_sizes=lbs,
        final_adjoint_log_block_sizes=lbs,
    )
    return _toffoli(bloq)


def optimal_interferometer_log_block_sizes_by_toffoli(
    n_blocks: int, n_rows: int, phase_bitsize: int
) -> Tuple[int, ...]:
    """Select-swap QROAM split ``(log_block, log_pair)`` that **minimizes the real Toffoli count**.

    Sweeps the total block size ``lambda`` over powers of two across the QROAM address range
    ``[1, n_blocks * n_rows // 2]``, forms the 2-D split with
    :func:`split_interferometer_log_block_sizes`, applies it to the phase-layer, final, and
    final-adjoint lookups, and returns the split with the smallest
    ``get_cost_value(..., QECGatesCost())`` Toffoli count (ties broken toward the smaller
    ``lambda`` -- fewer ancillae).  This is an exact argmin over the constructed circuit, not the
    heuristic ``optimal_interferometer_log_block_sizes`` closed form.
    """
    n_blocks = int(n_blocks)
    n_rows = int(n_rows)
    max_entries = max(1, n_blocks * (n_rows // 2))
    best: Optional[Tuple[int, int, Tuple[int, ...]]] = None  # (toffoli, lam, lbs)
    for log_lam in range(int(bit_length(max_entries - 1)) + 1):
        lam = 1 << log_lam
        lbs = split_interferometer_log_block_sizes(lam, n_blocks, n_rows)
        t = _interferometer_toffoli(n_blocks, n_rows, phase_bitsize, lbs)
        cand = (t, lam, lbs)
        if best is None or cand[:2] < best[:2]:
            best = cand
    assert best is not None
    return best[2]


# ============================================================================
# the bloq
# ============================================================================


@attrs.frozen
class SymmetryAdaptationQROAM(GateWithRegisters):
    r"""Space-group symmetry-adaptation transform ``Q : |k>|I> -> |i>|a>|c>`` (resource model).

    Data-free: constructed from *sizes only*.  See the module docstring for the algorithm and the
    ``n_k``-independence argument.

    Args:
        n_k: number of ``k``-points (only sets the ``k`` register width ``ceil(log2 n_k)`` and the
            ``log2 n_k`` widths of the subdominant ``k``-map / ``k.L`` arithmetic -- never a table size).
        n_IP: number of ISDF interpolation points (``I`` register width ``ceil(log2 n_IP)``).
        point_group_order: ``|P|`` (``= 48`` for diamond) -- the permutation-select cardinality and the
            leading dimension of the ``perm`` / geometry QROAM tables.
        orbit_sizes: multiset (tuple) of interpolation-point orbit sizes ``|o|`` across **all family
            representatives** (``d <= |P|``).  This -- not ``n_k`` -- is the dominant cost driver.
        n_families: number of ``k``-star family representatives (leading dim of the reindex QROAM).
        phase_bitsize: fixed-point bit width ``b`` of stored angles / the phase-gradient register.
        i_bitsize, a_bitsize, c_bitsize: output register widths for irrep / partner / copy.  Default
            to a partition sized from ``point_group_order`` and ``n_IP``.
        lattice_bits: bit width of one fold-back lattice-vector component ``L`` (small signed int).
        n_lattice_components: number of components of ``L`` (``= 3`` in 3D).
        k_axis_bits: per-axis width of ``k`` used in the ``k.L`` multiply.  Default
            ``ceil(log2(round(n_k**(1/3))))`` (``~ (1/3) log2 n_k``); the only ``n_k`` dependence of (B).
        minimize_toffoli: when ``True`` (default), each orbit :class:`BlockUnitaryInterferometerSynthesisQROAM`
            uses the QROAM select-swap split ``log_block_sizes`` that minimizes its real Toffoli count
            (:func:`optimal_interferometer_log_block_sizes_by_toffoli` -- an exact argmin sweep over
            ``lambda``).  When ``False`` the interferometer's default ``(0, 0)`` split (``lambda = 1``,
            no select-swap amortization) is used.  Inspect the chosen splits via
            :meth:`orbit_synthesis_plan`.
    """

    n_k: SymbolicInt
    n_IP: SymbolicInt
    point_group_order: SymbolicInt
    orbit_sizes: Tuple[int, ...] = attrs.field(converter=lambda x: tuple(int(v) for v in x))
    n_families: SymbolicInt = 1
    phase_bitsize: SymbolicInt = 24
    i_bitsize: Optional[int] = None
    a_bitsize: Optional[int] = None
    c_bitsize: Optional[int] = None
    lattice_bits: int = 4
    n_lattice_components: int = 3
    k_axis_bits: Optional[int] = None
    minimize_toffoli: bool = True

    def __attrs_post_init__(self):
        if not is_symbolic(self.phase_bitsize):
            assert self.phase_bitsize > 1
        if not is_symbolic(self.point_group_order):
            assert self.point_group_order >= 1
        assert len(self.orbit_sizes) >= 1, "need at least one orbit"

    # -- register widths -----------------------------------------------------

    @property
    def k_bitsize(self) -> int:
        return _clog2(self.n_k)

    @property
    def I_bitsize(self) -> int:
        return _clog2(self.n_IP)

    @property
    def op_bitsize(self) -> int:
        """Width of the point-op index ``g`` (select over ``|P|``)."""
        return _clog2(self.point_group_order)

    @property
    def family_bitsize(self) -> int:
        return _clog2(self.n_families)

    @property
    def _i_bits(self) -> int:
        return self.i_bitsize if self.i_bitsize is not None else _clog2(max(2, len(set(self.orbit_sizes)) + 1))

    @property
    def _a_bits(self) -> int:
        return self.a_bitsize if self.a_bitsize is not None else _clog2(self.point_group_order)

    @property
    def _c_bits(self) -> int:
        # copy/multiplicity: width to index all n_IP points of a k-block (generous default).
        if self.c_bitsize is not None:
            return self.c_bitsize
        return max(1, self.I_bitsize)

    @property
    def _kaxis_bits(self) -> int:
        if self.k_axis_bits is not None:
            return int(self.k_axis_bits)
        per_axis = max(2, int(round(int(self.n_k) ** (1.0 / 3.0))))
        return _clog2(per_axis)

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('k', QUInt(self.k_bitsize)),
                Register('I', QUInt(self.I_bitsize), side=Side.LEFT),
                Register('i', QUInt(self._i_bits), side=Side.RIGHT),
                Register('a', QUInt(self._a_bits), side=Side.RIGHT),
                Register('c', QUInt(self._c_bits), side=Side.RIGHT),
                Register('phase_gradient', QUInt(self.phase_bitsize)),
            ]
        )

    # -- component sub-bloqs -------------------------------------------------

    def _orbit_log_block_sizes(self, n_blocks: int, N: int) -> Tuple[int, ...]:
        """The QROAM select-swap split for a size group -- Toffoli-argmin when ``minimize_toffoli``."""
        if self.minimize_toffoli:
            return optimal_interferometer_log_block_sizes_by_toffoli(
                n_blocks, N, int(self.phase_bitsize)
            )
        return (0, 0)

    def orbit_synthesizers(self) -> List[Tuple[BlockUnitaryInterferometerSynthesisQROAM, int]]:
        """(A) One :class:`BlockUnitaryInterferometerSynthesisQROAM` per distinct rounded orbit size ``N``.

        Returns ``[(bloq, multiplicity=1), ...]``; ``n_blocks = count_N`` batches all size-``N`` orbits
        into one block-diagonal ``(+)_a U_a`` synthesis addressed by *(family, orbit)*, never by ``k``.
        The QROAM ``log_block_sizes`` are the select-swap split chosen by :meth:`_orbit_log_block_sizes`
        (the Toffoli-minimizing tradeoff by default).
        """
        out: List[Tuple[BlockUnitaryInterferometerSynthesisQROAM, int]] = []
        for N, count in group_orbit_sizes(self.orbit_sizes).items():
            lbs = self._orbit_log_block_sizes(count, N)
            out.append(
                (
                    BlockUnitaryInterferometerSynthesisQROAM.from_shape(
                        n_blocks=count,
                        n_rows=N,
                        phase_bitsize=self.phase_bitsize,
                        log_block_sizes=lbs,
                        final_log_block_sizes=lbs,
                        final_adjoint_log_block_sizes=lbs,
                    ),
                    1,
                )
            )
        return out

    def orbit_synthesis_plan(self) -> List[Dict[str, object]]:
        """Inspect stage (A): per size group, the chosen split and its default-vs-optimized Toffoli.

        Each row has ``N``, ``count`` (``= n_blocks``), ``log_block_sizes`` (the chosen split),
        ``toffoli`` (with the chosen split) and ``toffoli_default`` (the ``(0, 0)`` / ``lambda = 1``
        split), exposing exactly how much the select-swap tradeoff lowered the count.
        """
        rows: List[Dict[str, object]] = []
        for N, count in group_orbit_sizes(self.orbit_sizes).items():
            lbs = self._orbit_log_block_sizes(count, N)
            b = int(self.phase_bitsize)
            rows.append(
                {
                    'N': N,
                    'count': count,
                    'log_block_sizes': lbs,
                    'lambda': 1 << sum(int(x) for x in lbs),
                    'toffoli': _interferometer_toffoli(count, N, b, lbs),
                    'toffoli_default': _interferometer_toffoli(count, N, b, (0, 0)),
                }
            )
        return rows

    @property
    def perm_qroam(self) -> QROAMClean:
        """(B)(a) ``perm[g, I] -> I'`` load, table ``(|P|, n_IP)`` of ``I_bitsize``-bit indices."""
        return QROAMClean.build_from_bitsize(
            (int(self.point_group_order), int(self.n_IP)),
            target_bitsizes=(self.I_bitsize,),
            selection_bitsizes=(self.op_bitsize, self.I_bitsize),
        )

    @property
    def geom_qroam(self) -> QROAMClean:
        """(B)(b) fold-back lattice vector ``L[g, I]`` load, table ``(|P|, n_IP)`` of ``n_components``
        words of ``lattice_bits`` bits each."""
        return QROAMClean.build_from_bitsize(
            (int(self.point_group_order), int(self.n_IP)),
            target_bitsizes=(self.lattice_bits,) * self.n_lattice_components,
            selection_bitsizes=(self.op_bitsize, self.I_bitsize),
        )

    @property
    def reindex_qroam(self) -> QROAMClean:
        """(C) representative reindex ``(family, I') -> (i, a, c)`` load, table ``(n_families, n_IP)``."""
        return QROAMClean.build_from_bitsize(
            (int(self.n_families), int(self.n_IP)),
            target_bitsizes=(self._i_bits + self._a_bits + self._c_bits,),
            selection_bitsizes=(self.family_bitsize, self.I_bitsize),
        )

    @property
    def _ctrl_add(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize).controlled()

    @property
    def _kmap_compare(self) -> Bloq:
        """One equality comparison of the ``k`` register against a fixed representative (``k``-map)."""
        return EqualsAConstant(self.k_bitsize, 0)

    # -- cost ----------------------------------------------------------------

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.n_IP, self.point_group_order, self.phase_bitsize):
            raise DecomposeTypeError(f"cannot enumerate a data-free symbolic {self}")
        ret: 'Counter[Bloq]' = Counter()

        # (A) orbit-unitary synthesis -- dominant, ~ Sum|o|^2, n_k-INDEPENDENT.
        for bloq, mult in self.orbit_synthesizers():
            ret[bloq] += mult

        # (B) member reconstruction -- select over |P|, n_k-independent (log2 n_k only in k.L).
        ret[self.perm_qroam] += 2  # perm[g] forward + inverse-perm for in-place routing
        ret[self.geom_qroam] += 1  # load fold-back lattice vector L[g, I]
        # accumulate k . L into the phase gradient: n_components multiplies, each k_axis_bits adds.
        ret[self._ctrl_add] += self.n_lattice_components * self._kaxis_bits

        # (C) reindex I' -> (i, a, c); k-map modeled n_k-independent (|P| fixed-width comparisons).
        ret[self.reindex_qroam] += 1
        ret[self._kmap_compare] += int(self.point_group_order)

        return ret

    def resource_estimate(self) -> 'SymmetryAdaptationEstimate':
        return estimate_symmetry_adaptation_resources(
            n_k=int(self.n_k),
            n_IP=int(self.n_IP),
            point_group_order=int(self.point_group_order),
            orbit_sizes=self.orbit_sizes,
            n_families=int(self.n_families),
            phase_bitsize=int(self.phase_bitsize),
            i_bitsize=self._i_bits,
            a_bitsize=self._a_bits,
            c_bitsize=self._c_bits,
            lattice_bits=self.lattice_bits,
            n_lattice_components=self.n_lattice_components,
            k_axis_bits=self._kaxis_bits,
            minimize_toffoli=self.minimize_toffoli,
        )


# ============================================================================
# closed-form estimate
# ============================================================================


@dataclass(frozen=True)
class SymmetryAdaptationEstimate:
    """Aggregate resource estimate for one :class:`SymmetryAdaptationQROAM` configuration.

    ``orbit_synthesis_toffoli`` is the **dominant** term; it tracks ``Sum|o|^2`` and is flat in ``n_k``.
    ``reconstruction_toffoli`` / ``reindex_toffoli`` / ``kmap_toffoli`` are the subdominant
    ``n_k``-independent (at most ``log2 n_k``) reconstruction, reindex, and ``k``-map contributions.
    """

    n_k: int
    n_IP: int
    point_group_order: int
    unique_data_sum_o2: int          # true Sum|o|^2 (from the orbit multiset)
    rounded_unique_data: int         # Sum N^2 (power-of-two-rounded) -- the QROAM-realized proxy
    orbit_synthesis_toffoli: int     # (A) DOMINANT, ~ Sum|o|^2
    reconstruction_toffoli: int      # (B) perm select + Bloch phase
    reindex_toffoli: int             # (C) I' -> (i,a,c)
    kmap_toffoli: int                # (C) k -> (family, g)   [modeled n_k-independent]
    toffoli: int                     # total
    clifford: int                    # total Clifford
    qubits: int                      # peak logical qubits (register footprint + peak ancilla)
    breakdown: Dict[str, int] = field(default_factory=dict)


def _toffoli(bloq: Bloq) -> int:
    cost = get_cost_value(bloq, QECGatesCost(), generalizer=generalize_cswap_approx)
    return int(cost.total_t_and_ccz_count(ts_per_rotation=0)['n_ccz'])


def _clifford(bloq: Bloq) -> int:
    cost = get_cost_value(bloq, QECGatesCost(), generalizer=generalize_cswap_approx)
    return int(cost.clifford)


def _qubits(bloq: Bloq) -> int:
    return int(get_cost_value(bloq, QubitCount()))


def estimate_symmetry_adaptation_resources(
    *,
    n_k: int,
    n_IP: int,
    point_group_order: int,
    orbit_sizes: Tuple[int, ...],
    n_families: int = 1,
    phase_bitsize: int = 24,
    i_bitsize: Optional[int] = None,
    a_bitsize: Optional[int] = None,
    c_bitsize: Optional[int] = None,
    lattice_bits: int = 4,
    n_lattice_components: int = 3,
    k_axis_bits: Optional[int] = None,
    minimize_toffoli: bool = True,
) -> SymmetryAdaptationEstimate:
    """Closed-form-ish estimate: build the real component bloqs and aggregate their ``QECGatesCost``.

    Returns a :class:`SymmetryAdaptationEstimate` whose dominant ``orbit_synthesis_toffoli`` term
    tracks ``Sum|o|^2`` and is flat in ``n_k`` (see the module docstring).
    """
    bloq = SymmetryAdaptationQROAM(
        n_k=n_k,
        n_IP=n_IP,
        point_group_order=point_group_order,
        orbit_sizes=orbit_sizes,
        n_families=n_families,
        phase_bitsize=phase_bitsize,
        i_bitsize=i_bitsize,
        a_bitsize=a_bitsize,
        c_bitsize=c_bitsize,
        lattice_bits=lattice_bits,
        n_lattice_components=n_lattice_components,
        k_axis_bits=k_axis_bits,
        minimize_toffoli=minimize_toffoli,
    )

    # (A) orbit synthesis -- dominant.
    orbit_t = 0
    orbit_cl = 0
    peak_child_qubits = 0
    for sub, mult in bloq.orbit_synthesizers():
        orbit_t += mult * _toffoli(sub)
        orbit_cl += mult * _clifford(sub)
        peak_child_qubits = max(peak_child_qubits, _qubits(sub))

    # (B) reconstruction: perm (x2) + geometry load + k.L phase adds.
    perm_t = 2 * _toffoli(bloq.perm_qroam)
    perm_cl = 2 * _clifford(bloq.perm_qroam)
    geom_t = _toffoli(bloq.geom_qroam)
    geom_cl = _clifford(bloq.geom_qroam)
    add_t = bloq.n_lattice_components * bloq._kaxis_bits * _toffoli(bloq._ctrl_add)
    add_cl = bloq.n_lattice_components * bloq._kaxis_bits * _clifford(bloq._ctrl_add)
    recon_t = perm_t + geom_t + add_t
    recon_cl = perm_cl + geom_cl + add_cl

    # (C) reindex + k-map.
    reindex_t = _toffoli(bloq.reindex_qroam)
    reindex_cl = _clifford(bloq.reindex_qroam)
    kmap_t = int(point_group_order) * _toffoli(bloq._kmap_compare)
    kmap_cl = int(point_group_order) * _clifford(bloq._kmap_compare)

    total_t = orbit_t + recon_t + reindex_t + kmap_t
    total_cl = orbit_cl + recon_cl + reindex_cl + kmap_cl

    # qubits: peak child ancilla + the persistent register footprint that lives across the whole Q.
    register_footprint = (
        bloq.k_bitsize
        + bloq.I_bitsize
        + bloq._i_bits
        + bloq._a_bits
        + bloq._c_bits
        + int(phase_bitsize)
    )
    peak_recon_qubits = max(_qubits(bloq.perm_qroam), _qubits(bloq.geom_qroam), _qubits(bloq.reindex_qroam))
    qubits = register_footprint + max(peak_child_qubits, peak_recon_qubits)

    return SymmetryAdaptationEstimate(
        n_k=int(n_k),
        n_IP=int(n_IP),
        point_group_order=int(point_group_order),
        unique_data_sum_o2=int(sum(int(d) ** 2 for d in orbit_sizes)),
        rounded_unique_data=int(rounded_unique_data(orbit_sizes)),
        orbit_synthesis_toffoli=int(orbit_t),
        reconstruction_toffoli=int(recon_t),
        reindex_toffoli=int(reindex_t),
        kmap_toffoli=int(kmap_t),
        toffoli=int(total_t),
        clifford=int(total_cl),
        qubits=int(qubits),
        breakdown={
            'orbit_synthesis': int(orbit_t),
            'perm_select': int(perm_t),
            'geometry_load': int(geom_t),
            'bloch_phase_add': int(add_t),
            'reindex': int(reindex_t),
            'k_map': int(kmap_t),
            'n_orbit_blocks': int(len(orbit_sizes)),
            'n_one_qubit_gates_proxy': int(orbit_one_qubit_gates(orbit_sizes)),
        },
    )


# ============================================================================
# material data + demo
# ============================================================================


def load_material(path: Optional[str] = None):
    """Load the diamond symmetry-orbit data JSON (sizes only)."""
    import json
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diamond_symm_orbit_data.json')
    with open(path) as fh:
        return json.load(fh)


def bloq_for_mesh(mesh: str, data=None, **overrides) -> SymmetryAdaptationQROAM:
    """Construct :class:`SymmetryAdaptationQROAM` for a mesh key (``"2x2x2"`` / ``"6x6x6"``)."""
    if data is None:
        data = load_material()
    m = data[mesh]
    orbit_sizes = tuple(o for fam in m['families'] for o in fam['orbit_sizes'])
    kwargs = dict(
        n_k=int(m['nk']),
        n_IP=int(m['nIP']),
        point_group_order=int(m['nops']),
        orbit_sizes=orbit_sizes,
        n_families=int(m['n_families']),
    )
    kwargs.update(overrides)
    return SymmetryAdaptationQROAM(**kwargs)


def _demo() -> None:
    data = load_material()
    rows = []
    for mesh in ('2x2x2', '6x6x6'):
        est = bloq_for_mesh(mesh, data).resource_estimate()
        rows.append((mesh, est))

    print("Symmetry-adaptation transform Q  |k>|I> -> |i>|a>|c>  (diamond)")
    print("=" * 92)
    hdr = f"{'mesh':>7} {'n_k':>5} {'Sum|o|^2':>9} {'orbit-Tof(A)':>13} {'recon(B)':>9} " \
          f"{'reindex':>8} {'kmap':>7} {'TOTAL Tof':>11} {'Clifford':>11} {'qubits':>7}"
    print(hdr)
    print("-" * 92)
    for mesh, e in rows:
        print(f"{mesh:>7} {e.n_k:>5} {e.unique_data_sum_o2:>9} {e.orbit_synthesis_toffoli:>13} "
              f"{e.reconstruction_toffoli:>9} {e.reindex_toffoli:>8} {e.kmap_toffoli:>7} "
              f"{e.toffoli:>11} {e.clifford:>11} {e.qubits:>7}")
    print("-" * 92)

    e0, e1 = rows[0][1], rows[1][1]
    nk_ratio = e1.n_k / e0.n_k
    data_ratio = e1.unique_data_sum_o2 / e0.unique_data_sum_o2
    dom_ratio = e1.orbit_synthesis_toffoli / max(1, e0.orbit_synthesis_toffoli)
    tot_ratio = e1.toffoli / max(1, e0.toffoli)
    naive_ratio = (e1.n_k * e1.n_IP ** 2) / (e0.n_k * e0.n_IP ** 2)
    print(f"n_k grows          {nk_ratio:6.1f}x  (8 -> 216)")
    print(f"Sum|o|^2 grows     {data_ratio:6.2f}x  (6800 -> 12912)")
    print(f"dominant (A) Tof   {dom_ratio:6.2f}x   <-- tracks Sum|o|^2, NOT n_k")
    print(f"total Toffoli      {tot_ratio:6.2f}x")
    print(f"naive n_k*n_IP^2   {naive_ratio:6.1f}x  (per-k dense model, for contrast)")

    # Select-swap tradeoff: default (lambda=1) vs Toffoli-minimizing split, per size group (6x6x6).
    print()
    print("Stage (A) select-swap tradeoff -- interferometer synthesis per orbit-size group (6x6x6)")
    print("-" * 92)
    print(f"{'N':>4} {'count':>6} {'log_block_sizes':>16} {'lambda':>7} {'Tof(opt)':>10} "
          f"{'Tof(default)':>13} {'saved':>8}")
    plan = bloq_for_mesh('6x6x6', data).orbit_synthesis_plan()
    tot_opt = tot_def = 0
    for r in sorted(plan, key=lambda x: -x['toffoli_default']):
        saved = r['toffoli_default'] - r['toffoli']
        tot_opt += r['toffoli']
        tot_def += r['toffoli_default']
        print(f"{r['N']:>4} {r['count']:>6} {str(r['log_block_sizes']):>16} {r['lambda']:>7} "
              f"{r['toffoli']:>10} {r['toffoli_default']:>13} {saved:>8}")
    print("-" * 92)
    print(f"stage (A) total: optimized {tot_opt}  vs default {tot_def}  "
          f"(reduction {tot_def / max(1, tot_opt):.2f}x)")


if __name__ == '__main__':
    _demo()
