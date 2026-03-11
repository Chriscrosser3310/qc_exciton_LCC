from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import ceil, log2

import numpy as np
from attrs import frozen

try:
    from qualtran import Bloq, Register, Side, Signature
    from qualtran.bloqs.block_encoding.sparse_matrix import RowColumnOracle
    from qualtran.resource_counting import GateCounts, QECGatesCost
    from qualtran.symbolics import SymbolicInt
    from qualtran import BQUInt, QUInt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for qualtran_lattice_index_oracles. Install with '[qualtran]'."
    ) from exc


def _ceil_log2_int(x: int) -> int:
    if x <= 1:
        return 1
    return int(ceil(log2(x)))


def _mixed_radix_digits(value: int, radices: list[int]) -> list[int]:
    out: list[int] = []
    v = int(value)
    for r in radices:
        out.append(v % r)
        v //= r
    return out


def _estimate_single_oracle_gate_counts(system_bitsize: int, ndim: int, radix_bits: int) -> GateCounts:
    """Rough arithmetic gate proxy for mixed-radix decode + modular coordinate updates.

    This is a scaling-oriented estimate intended for benchmarking trends.
    """
    toffoli = ndim * (4 * system_bitsize + 2 * radix_bits + 4)
    clifford = ndim * (12 * system_bitsize + 4 * radix_bits + 8)
    return GateCounts(toffoli=toffoli, clifford=clifford)


def _estimate_two_oracle_gate_counts(
    system_bitsize: int, ndim: int, radix_c_bits: int, radix_loc_bits: int
) -> GateCounts:
    # Two neighbor expansions (i->j and j->j'), plus pair packing overhead.
    toffoli = (8 * ndim * system_bitsize) + (2 * ndim * (radix_c_bits + radix_loc_bits)) + (6 * ndim)
    clifford = (24 * ndim * system_bitsize) + (4 * ndim * (radix_c_bits + radix_loc_bits)) + (16 * ndim)
    # Small additional packing cost for (j, j')
    toffoli += 2 * system_bitsize
    clifford += 6 * system_bitsize
    return GateCounts(toffoli=toffoli, clifford=clifford)


@dataclass(frozen=True)
class _LatticeGeometry:
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n_sites(self) -> int:
        n = 1
        for s in self.shape:
            if s <= 0:
                raise ValueError("All lattice extents must be positive.")
            n *= s
        return n

    @property
    def index_bitsize(self) -> int:
        return _ceil_log2_int(self.n_sites)

    def decode_index(self, idx: int) -> tuple[int, ...]:
        idx_mod = int(idx) % self.n_sites
        return tuple(np.unravel_index(idx_mod, self.shape))

    def encode_index(self, coord: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(coord, self.shape))


def _offset_table_1d(length: int, radius: int) -> list[int]:
    """Unique periodic offsets preserving the natural [-R, ..., R] ordering."""
    if radius < 0:
        raise ValueError("radius must be >= 0.")
    seen: set[int] = set()
    offsets: list[int] = []
    for d in range(-radius, radius + 1):
        d_mod = d % length
        if d_mod not in seen:
            seen.add(d_mod)
            offsets.append(d_mod)
    return offsets


def _offset_tables(shape: tuple[int, ...], radius: int) -> list[list[int]]:
    return [_offset_table_1d(length=s, radius=radius) for s in shape]


@frozen
class SingleParticleSparseIndexOracle(RowColumnOracle):
    """Lattice neighbor oracle: (l, i) -> (j, i).

    Given site index `i` and neighbor rank `l`, returns the `l`-th lattice site `j` whose
    per-coordinate periodic offset from `i` lies within radius `r_loc`.

    Ordering:
      `l` is interpreted in mixed radix with radices `span_d = len(offset_table_d)` and
      each digit picks one per-dimension offset from the ordered list
      `[-r_loc, ..., r_loc]` (deduplicated modulo lattice size).
    """

    lattice_shape: tuple[int, ...]
    r_loc: int

    @cached_property
    def _geom(self) -> _LatticeGeometry:
        return _LatticeGeometry(self.lattice_shape)

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self._geom.index_bitsize

    @cached_property
    def _offsets(self) -> list[list[int]]:
        return _offset_tables(self.lattice_shape, self.r_loc)

    @cached_property
    def _radices(self) -> list[int]:
        return [len(v) for v in self._offsets]

    @cached_property
    def num_nonzero(self) -> SymbolicInt:
        n = 1
        for r in self._radices:
            n *= r
        return n

    def lth_neighbor(self, i: int, l: int) -> int:
        if l < 0 or l >= int(self.num_nonzero):
            raise IndexError("l out of range for SingleParticleSparseIndexOracle.")
        coord = self._geom.decode_index(i)
        digits = _mixed_radix_digits(l, self._radices)
        out_coord = []
        for d, digit in enumerate(digits):
            shift = self._offsets[d][digit]
            out_coord.append((coord[d] + shift) % self.lattice_shape[d])
        return self._geom.encode_index(tuple(out_coord))

    def call_classically(self, l, i):
        assert not isinstance(l, np.ndarray) and not isinstance(i, np.ndarray)
        j = self.lth_neighbor(int(i), int(l))
        return (j, int(i))

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QECGatesCost):
            radix_bits = _ceil_log2_int(max(self._radices))
            return _estimate_single_oracle_gate_counts(
                system_bitsize=int(self.system_bitsize),
                ndim=self._geom.ndim,
                radix_bits=radix_bits,
            )
        return NotImplemented


@frozen
class TwoParticleSparseIndexOracle(Bloq):
    r"""Two-particle lattice oracle.

    Maps `(l, i, i_prime)` to `(packed(j, j_prime), i, i_prime)`, where:
    - `j` is within radius `r_c` of `i` (per-coordinate periodic max distance).
    - `j_prime` is within radius `r_loc` of `j`.

    The input condition `dist_max(i, i_prime) <= r_loc` can be checked externally as a domain
    restriction for meaningful two-particle configurations.

    Ordering:
    - first `D` mixed-radix digits of `l` choose offsets for `j` around `i`
    - next `D` digits choose offsets for `j_prime` around `j`
    """

    lattice_shape: tuple[int, ...]
    r_loc: int
    r_c: int

    @cached_property
    def _geom(self) -> _LatticeGeometry:
        return _LatticeGeometry(self.lattice_shape)

    @cached_property
    def system_bitsize(self) -> int:
        return self._geom.index_bitsize

    @cached_property
    def pair_bitsize(self) -> int:
        return 2 * self.system_bitsize

    @cached_property
    def _offsets_c(self) -> list[list[int]]:
        return _offset_tables(self.lattice_shape, self.r_c)

    @cached_property
    def _offsets_loc(self) -> list[list[int]]:
        return _offset_tables(self.lattice_shape, self.r_loc)

    @cached_property
    def _radices_c(self) -> list[int]:
        return [len(v) for v in self._offsets_c]

    @cached_property
    def _radices_loc(self) -> list[int]:
        return [len(v) for v in self._offsets_loc]

    @cached_property
    def _radices_all(self) -> list[int]:
        return self._radices_c + self._radices_loc

    @cached_property
    def num_nonzero(self) -> int:
        n = 1
        for r in self._radices_all:
            n *= r
        return n

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("l", BQUInt(self.pair_bitsize, self.num_nonzero), side=Side.LEFT),
                Register("l", QUInt(self.pair_bitsize), side=Side.RIGHT),
                Register("i", QUInt(self.system_bitsize)),
                Register("i_prime", QUInt(self.system_bitsize)),
            ]
        )

    def _encode_pair(self, j: int, j_prime: int) -> int:
        return int(j) * self._geom.n_sites + int(j_prime)

    def decode_pair(self, packed: int) -> tuple[int, int]:
        n = self._geom.n_sites
        return (int(packed) // n, int(packed) % n)

    def lth_pair(self, i: int, i_prime: int, l: int) -> tuple[int, int]:
        _ = i_prime  # kept for interface symmetry and future constrained variants
        if l < 0 or l >= self.num_nonzero:
            raise IndexError("l out of range for TwoParticleSparseIndexOracle.")

        digits = _mixed_radix_digits(l, self._radices_all)
        d = self._geom.ndim
        digits_c = digits[:d]
        digits_loc = digits[d:]

        i_coord = self._geom.decode_index(i)
        j_coord = []
        for k, dig in enumerate(digits_c):
            shift = self._offsets_c[k][dig]
            j_coord.append((i_coord[k] + shift) % self.lattice_shape[k])
        j_coord_t = tuple(j_coord)

        jp_coord = []
        for k, dig in enumerate(digits_loc):
            shift = self._offsets_loc[k][dig]
            jp_coord.append((j_coord_t[k] + shift) % self.lattice_shape[k])

        j = self._geom.encode_index(j_coord_t)
        j_prime = self._geom.encode_index(tuple(jp_coord))
        return j, j_prime

    def call_classically(self, l, i, i_prime):
        assert not isinstance(l, np.ndarray)
        assert not isinstance(i, np.ndarray)
        assert not isinstance(i_prime, np.ndarray)
        j, j_prime = self.lth_pair(int(i), int(i_prime), int(l))
        packed = self._encode_pair(j, j_prime)
        return (packed, int(i), int(i_prime))

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QECGatesCost):
            radix_c_bits = _ceil_log2_int(max(self._radices_c))
            radix_loc_bits = _ceil_log2_int(max(self._radices_loc))
            return _estimate_two_oracle_gate_counts(
                system_bitsize=self.system_bitsize,
                ndim=self._geom.ndim,
                radix_c_bits=radix_c_bits,
                radix_loc_bits=radix_loc_bits,
            )
        return NotImplemented


def build_lattice_sparse_index_oracles(
    lattice_shape: tuple[int, ...],
    r_loc: int,
    r_c: int,
) -> tuple[SingleParticleSparseIndexOracle, TwoParticleSparseIndexOracle]:
    """Factory returning single- and two-particle sparse index oracles."""
    single = SingleParticleSparseIndexOracle(lattice_shape=lattice_shape, r_loc=r_loc)
    two = TwoParticleSparseIndexOracle(lattice_shape=lattice_shape, r_loc=r_loc, r_c=r_c)
    return single, two
