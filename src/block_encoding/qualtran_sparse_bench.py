from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import ceil, log2
from typing import Literal

import numpy as np
from attrs import field, frozen

from exciton.benchmark_tensors import Metric, generate_f_tensor, generate_v_tensor

try:
    from qualtran import BloqBuilder, DecomposeTypeError, QUInt
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.bloqs.basic_gates import Swap
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        EntryOracle,
        ExplicitEntryOracle,
        RowColumnOracle,
        SparseMatrix,
    )
    from qualtran.bloqs.data_loading import QROM
    from qualtran.symbolics import SymbolicInt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for qualtran_sparse_bench. Install with extra '[qualtran]'."
    ) from exc


Layout = Literal["pq_rs", "pr_qs"]


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << int(ceil(log2(x)))


def _threshold(x: np.ndarray, epsilon: float) -> np.ndarray:
    y = np.array(x, copy=True)
    y[np.abs(y) < epsilon] = 0.0
    return y


@dataclass(frozen=True)
class PolylogSparseIndex:
    """Rank/select index for O(log N) l-th nonzero lookup in rows/columns."""

    row_prefix: np.ndarray  # shape (n, n+1), cumulative counts over columns
    col_prefix: np.ndarray  # shape (n, n+1), cumulative counts over rows

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, epsilon: float) -> "PolylogSparseIndex":
        nz = np.abs(matrix) >= epsilon
        row_prefix = np.concatenate(
            [np.zeros((matrix.shape[0], 1), dtype=int), np.cumsum(nz, axis=1)], axis=1
        )
        col_prefix = np.concatenate(
            [np.zeros((matrix.shape[0], 1), dtype=int), np.cumsum(nz, axis=0).T], axis=1
        )
        return cls(row_prefix=row_prefix, col_prefix=col_prefix)

    def row_lth_nonzero(self, row: int, l: int) -> int:
        # 0-based l among nonzeros in the row.
        target_rank = l + 1
        pref = self.row_prefix[row]
        if target_rank > pref[-1]:
            raise IndexError("l exceeds number of nonzero entries in row.")
        return int(np.searchsorted(pref, target_rank, side="left") - 1)

    def col_lth_nonzero(self, col: int, l: int) -> int:
        # 0-based l among nonzeros in the column.
        target_rank = l + 1
        pref = self.col_prefix[col]
        if target_rank > pref[-1]:
            raise IndexError("l exceeds number of nonzero entries in column.")
        return int(np.searchsorted(pref, target_rank, side="left") - 1)


def _pad_square_to_pow2(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("Expected square 2-D array.")
    n = x.shape[0]
    n2 = _next_pow2(n)
    if n2 == n:
        return x
    out = np.zeros((n2, n2), dtype=x.dtype)
    out[:n, :n] = x
    return out


def _build_sparse_index_lists(matrix: np.ndarray, epsilon: float) -> tuple[list[list[int]], list[list[int]], int]:
    n = matrix.shape[0]
    row_lists: list[list[int]] = []
    col_lists: list[list[int]] = []

    nz_mask = np.abs(matrix) >= epsilon
    for i in range(n):
        row_lists.append(np.flatnonzero(nz_mask[i]).tolist())
        col_lists.append(np.flatnonzero(nz_mask[:, i]).tolist())

    row_max = max((len(v) for v in row_lists), default=0)
    col_max = max((len(v) for v in col_lists), default=0)
    s = max(1, row_max, col_max)
    s = min(s, n)
    return row_lists, col_lists, s


def _complete_prefix_to_permutation(prefix: list[int], n: int) -> np.ndarray:
    seen = set(prefix)
    if len(prefix) != len(seen):
        raise ValueError("Prefix indices must be unique.")
    if any((v < 0 or v >= n) for v in prefix):
        raise ValueError("Prefix contains out-of-range indices.")

    tail = [j for j in range(n) if j not in seen]
    perm = np.array(prefix + tail, dtype=int)
    if perm.shape[0] != n:
        raise ValueError("Internal error while building permutation.")
    return perm


def _padded_unique_prefix(indices: list[int], n: int, s: int) -> list[int]:
    uniq = list(dict.fromkeys(indices))
    if len(uniq) > s:
        uniq = uniq[:s]
    if len(uniq) < s:
        need = s - len(uniq)
        extras = [j for j in range(n) if j not in set(uniq)]
        uniq.extend(extras[:need])
    return uniq


def _build_permutation_tables(
    lists: list[list[int]], n: int, s: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-row full permutations and their inverses.

    For each row i:
    - first s images encode l-th nonzero candidate indices.
    - remaining images complete a full permutation for reversibility.
    """
    perm = np.zeros((n, n), dtype=int)
    inv = np.zeros((n, n), dtype=int)
    for i in range(n):
        prefix = _padded_unique_prefix(lists[i], n=n, s=s)
        p = _complete_prefix_to_permutation(prefix, n=n)
        pinv = np.zeros(n, dtype=int)
        pinv[p] = np.arange(n, dtype=int)
        perm[i] = p
        inv[i] = pinv
    return perm, inv


@frozen
class QROMPermutationRowColumnOracle(RowColumnOracle):
    """Row/column oracle via reversible QROM-backed permutation tables.

    For fixed row/column index i, this oracle implements:
      l -> perm_table[i, l]
    where the first `num_nonzero` l values correspond to the l-th candidate nonzero index.
    """

    system_bitsize: SymbolicInt
    num_nonzero_value: SymbolicInt
    perm_table: np.ndarray = field(eq=lambda d: tuple(np.asarray(d).flat))
    perm_inv_table: np.ndarray = field(eq=lambda d: tuple(np.asarray(d).flat))

    @cached_property
    def num_nonzero(self) -> SymbolicInt:
        return self.num_nonzero_value

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if isinstance(self.system_bitsize, int):
            n = 2**self.system_bitsize
            if self.perm_table.shape != (n, n) or self.perm_inv_table.shape != (n, n):
                raise ValueError("Permutation tables must have shape (2^n, 2^n).")

    @cached_property
    def _qrom_fwd(self):
        # Store l XOR perm(i,l) so one XOR with l recovers perm(i,l).
        n = self.perm_table.shape[0]
        lvals = np.arange(n, dtype=int)[None, :]
        data = np.bitwise_xor(self.perm_table, np.repeat(lvals, n, axis=0))
        return QROM.build_from_data(data, target_bitsizes=(self.system_bitsize,))

    @cached_property
    def _qrom_inv(self):
        return QROM.build_from_data(self.perm_inv_table, target_bitsizes=(self.system_bitsize,))

    def build_composite_bloq(self, bb: BloqBuilder, l, i):
        if not isinstance(self.system_bitsize, int):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        n = self.system_bitsize
        tmp = bb.allocate(n)

        # tmp <- l XOR perm(i,l)
        i, l, tmp = bb.add(self._qrom_fwd, selection0=i, selection1=l, target0_=tmp)
        # tmp <- perm(i,l)
        l, tmp = bb.add(Xor(dtype=QUInt(self.system_bitsize)), x=l, y=tmp)
        # place output in l; tmp now holds old l
        l, tmp = bb.add(Swap(n), x=l, y=tmp)
        # tmp <- old_l XOR inv(i,l_out) = 0
        i, l, tmp = bb.add(self._qrom_inv, selection0=i, selection1=l, target0_=tmp)
        bb.free(tmp)
        return {"l": l, "i": i}


@dataclass(frozen=True)
class SparseOracleBundle:
    label: str
    matrix: np.ndarray
    matrix_unpadded_dim: int
    epsilon: float
    alpha: float
    polylog_index: PolylogSparseIndex
    row_oracle: RowColumnOracle
    col_oracle: RowColumnOracle
    entry_oracle: EntryOracle
    block_encoding: SparseMatrix


def build_thresholded_benchmark_tensors(
    shape: tuple[int, ...],
    epsilon: float,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    r_loc: int | None = None,
    r_c: int | None = None,
    periodic_cutoff: bool = False,
    metric: Metric = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate thresholded F and V tensors."""
    f_cut = r_c if r_c is not None else r_loc
    f = _threshold(
        generate_f_tensor(
            shape=shape,
            a=a,
            metric=metric,
            r_cut=f_cut,
            periodic_cutoff=periodic_cutoff,
        ),
        epsilon=epsilon,
    )
    v = _threshold(
        generate_v_tensor(
            shape=shape,
            a=a,
            b=b,
            c=c,
            metric=metric,
            r_loc=r_loc,
            r_c=r_c,
            periodic_cutoff=periodic_cutoff,
        ),
        epsilon=epsilon,
    )
    return f, v


def build_product_matrices_from_tensors(
    f: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two flattened matrices from F and V product F_{pq}V_{pqrs}.

    Returns:
      m_pq_rs: rows=(p,q), cols=(r,s)
      m_pr_qs: rows=(p,r), cols=(q,s)
    """
    if f.ndim != 2 or v.ndim != 4:
        raise ValueError("Expected f shape (N,N) and v shape (N,N,N,N).")
    n = f.shape[0]
    if f.shape[1] != n or v.shape != (n, n, n, n):
        raise ValueError("Incompatible F/V shapes.")

    prod = f[:, :, None, None] * v
    m_pq_rs = prod.reshape(n * n, n * n)
    m_pr_qs = np.transpose(prod, (0, 2, 1, 3)).reshape(n * n, n * n)
    return m_pq_rs, m_pr_qs


def _normalize_for_entry_oracle(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    alpha = float(np.max(np.abs(matrix)))
    if alpha <= 0:
        return np.zeros_like(matrix, dtype=np.float64), 1.0
    return (np.abs(matrix) / alpha).astype(np.float64), alpha


def build_sparse_oracle_from_matrix(
    matrix: np.ndarray,
    epsilon: float,
    entry_bitsize: int = 10,
    label: str = "benchmark",
) -> SparseOracleBundle:
    """Build Qualtran sparse block-encoding oracles from a thresholded matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square 2-D.")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0.")

    matrix = np.array(matrix, dtype=np.float64, copy=True)
    matrix[np.abs(matrix) < epsilon] = 0.0
    unpadded_n = matrix.shape[0]
    padded = _pad_square_to_pow2(matrix)
    n = padded.shape[0]
    n_bits = int(log2(n))

    row_lists, col_lists, s = _build_sparse_index_lists(padded, epsilon=epsilon)
    row_perm, row_inv = _build_permutation_tables(row_lists, n=n, s=s)
    col_perm, col_inv = _build_permutation_tables(col_lists, n=n, s=s)

    row_oracle = QROMPermutationRowColumnOracle(
        system_bitsize=n_bits,
        num_nonzero_value=s,
        perm_table=row_perm,
        perm_inv_table=row_inv,
    )
    col_oracle = QROMPermutationRowColumnOracle(
        system_bitsize=n_bits,
        num_nonzero_value=s,
        perm_table=col_perm,
        perm_inv_table=col_inv,
    )

    normalized, alpha = _normalize_for_entry_oracle(padded)
    entry_oracle = ExplicitEntryOracle(
        system_bitsize=n_bits,
        data=normalized,
        entry_bitsize=entry_bitsize,
    )
    block = SparseMatrix(
        row_oracle=row_oracle,
        col_oracle=col_oracle,
        entry_oracle=entry_oracle,
        eps=0.0,
    )
    return SparseOracleBundle(
        label=label,
        matrix=padded,
        matrix_unpadded_dim=unpadded_n,
        epsilon=epsilon,
        alpha=alpha,
        polylog_index=PolylogSparseIndex.from_matrix(padded, epsilon=epsilon),
        row_oracle=row_oracle,
        col_oracle=col_oracle,
        entry_oracle=entry_oracle,
        block_encoding=block,
    )


def build_all_sparse_oracles(
    shape: tuple[int, ...],
    epsilon: float,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    r_loc: int | None = None,
    r_c: int | None = None,
    periodic_cutoff: bool = False,
    metric: Metric = "euclidean",
    entry_bitsize: int = 10,
) -> dict[str, SparseOracleBundle]:
    """Build sparse-oracle bundles for:
    - F_{pq}
    - (pq,rs) layout of F_{pq}V_{pqrs}
    - (pr,qs) layout of F_{pq}V_{pqrs}
    """
    f, v = build_thresholded_benchmark_tensors(
        shape=shape,
        epsilon=epsilon,
        a=a,
        b=b,
        c=c,
        r_loc=r_loc,
        r_c=r_c,
        periodic_cutoff=periodic_cutoff,
        metric=metric,
    )
    m_pq_rs, m_pr_qs = build_product_matrices_from_tensors(f, v)
    return {
        "F_pq": build_sparse_oracle_from_matrix(
            f, epsilon=epsilon, entry_bitsize=entry_bitsize, label="F_pq"
        ),
        "FV_pq_rs": build_sparse_oracle_from_matrix(
            m_pq_rs, epsilon=epsilon, entry_bitsize=entry_bitsize, label="FV_pq_rs"
        ),
        "FV_pr_qs": build_sparse_oracle_from_matrix(
            m_pr_qs, epsilon=epsilon, entry_bitsize=entry_bitsize, label="FV_pr_qs"
        ),
    }
