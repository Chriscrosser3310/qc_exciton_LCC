from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from exciton.benchmark_tensors import (
    LatticeSpec,
    Metric,
    generate_f_tensor,
    generate_v_tensor,
    lattice_max_dist,
)

from .qualtran_sparse_bench import SparseOracleBundle, build_sparse_oracle_from_matrix


def _flatten_v_direct(v: np.ndarray) -> np.ndarray:
    """(p, q) -> (r, s) orientation."""
    n = v.shape[0]
    return v.reshape(n * n, n * n)


def _flatten_v_exchange(v: np.ndarray) -> np.ndarray:
    """(p, r) -> (q, s) orientation."""
    n = v.shape[0]
    return np.transpose(v, (0, 2, 1, 3)).reshape(n * n, n * n)


@dataclass(frozen=True)
class ExcitonLCUTerm:
    coefficient: float
    op_kind: str  # "F", "V_direct", "V_exchange"
    acting_on: tuple[int, ...]
    set_label: str  # "first", "second", "cross"


@dataclass(frozen=True)
class ExcitonHamiltonianEncoding:
    """Structured LCU specification for the exciton Hamiltonian.

    `first` register set has m registers: indices [0, ..., m-1]
    `second` register set has m registers: indices [m, ..., 2m-1]
    """

    m: int
    lattice_shape: tuple[int, ...]
    f_bundle: SparseOracleBundle
    v_direct_bundle: SparseOracleBundle
    v_exchange_bundle: SparseOracleBundle
    terms: tuple[ExcitonLCUTerm, ...]


def build_f_and_v_block_encodings(
    lattice_shape: tuple[int, ...],
    r_loc: int | None = None,
    r_c: int | None = None,
    *,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    metric: Metric = "chebyshev",
    periodic_cutoff: bool = False,
    epsilon: float = 0.0,
    entry_bitsize: int = 10,
) -> tuple[SparseOracleBundle, SparseOracleBundle, SparseOracleBundle]:
    """Build sparse block-encodings for F, V_direct, and V_exchange.

    - F_{pq} uses hard cutoff `r_cut` (defaults to lattice max-distance).
    - V_{pqrs} uses hard cutoffs `r_loc` and `r_c` (both default to lattice max-distance).
    - `epsilon` here is additional numerical thresholding (default 0).
    """
    spec = LatticeSpec(shape=lattice_shape)
    default_max = lattice_max_dist(spec, periodic=periodic_cutoff)
    r_loc_eff = default_max if r_loc is None else int(r_loc)
    r_c_eff = default_max if r_c is None else int(r_c)

    f = generate_f_tensor(
        shape=lattice_shape,
        a=a,
        metric=metric,
        r_cut=r_c_eff,
        periodic_cutoff=periodic_cutoff,
    )
    v = generate_v_tensor(
        shape=lattice_shape,
        a=a,
        b=b,
        c=c,
        metric=metric,
        r_loc=r_loc_eff,
        r_c=r_c_eff,
        periodic_cutoff=periodic_cutoff,
    )

    f[np.abs(f) < epsilon] = 0.0
    v[np.abs(v) < epsilon] = 0.0

    v_direct = _flatten_v_direct(v)
    v_exchange = _flatten_v_exchange(v)

    f_bundle = build_sparse_oracle_from_matrix(
        f, epsilon=epsilon, entry_bitsize=entry_bitsize, label="F_pq"
    )
    v_direct_bundle = build_sparse_oracle_from_matrix(
        v_direct, epsilon=epsilon, entry_bitsize=entry_bitsize, label="V_direct_pq_rs"
    )
    v_exchange_bundle = build_sparse_oracle_from_matrix(
        v_exchange, epsilon=epsilon, entry_bitsize=entry_bitsize, label="V_exchange_pr_qs"
    )
    return f_bundle, v_direct_bundle, v_exchange_bundle


def build_exciton_hamiltonian_encoding(
    m: int,
    lattice_shape: tuple[int, ...],
    r_loc: int | None = None,
    r_c: int | None = None,
    *,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    metric: Metric = "chebyshev",
    periodic_cutoff: bool = False,
    epsilon: float = 0.0,
    entry_bitsize: int = 10,
) -> ExcitonHamiltonianEncoding:
    """Build structured LCU terms and block encodings for the exciton Hamiltonian.

    Hamiltonian terms:
    1. +F on each register in first set.
    2. -F on each register in second set.
    3. +V_direct within each pair of first-set registers.
    4. +V_direct within each pair of second-set registers.
    5. +(V_exchange - V_direct) on each cross pair (first-set, second-set).
    """
    if m <= 0:
        raise ValueError("m must be >= 1.")

    f_bundle, v_direct_bundle, v_exchange_bundle = build_f_and_v_block_encodings(
        lattice_shape=lattice_shape,
        r_loc=r_loc,
        r_c=r_c,
        a=a,
        b=b,
        c=c,
        metric=metric,
        periodic_cutoff=periodic_cutoff,
        epsilon=epsilon,
        entry_bitsize=entry_bitsize,
    )

    first = list(range(m))
    second = list(range(m, 2 * m))
    terms: list[ExcitonLCUTerm] = []

    # +F on first set, -F on second set.
    for i in first:
        terms.append(ExcitonLCUTerm(coefficient=+1.0, op_kind="F", acting_on=(i,), set_label="first"))
    for a_idx in second:
        terms.append(
            ExcitonLCUTerm(coefficient=-1.0, op_kind="F", acting_on=(a_idx,), set_label="second")
        )

    # direct V within first and within second.
    for i, j in combinations(first, 2):
        terms.append(
            ExcitonLCUTerm(
                coefficient=+1.0, op_kind="V_direct", acting_on=(i, j), set_label="first"
            )
        )
    for a_idx, b_idx in combinations(second, 2):
        terms.append(
            ExcitonLCUTerm(
                coefficient=+1.0,
                op_kind="V_direct",
                acting_on=(a_idx, b_idx),
                set_label="second",
            )
        )

    # cross terms: V_exchange - V_direct.
    for i in first:
        for a_idx in second:
            terms.append(
                ExcitonLCUTerm(
                    coefficient=+1.0,
                    op_kind="V_exchange",
                    acting_on=(i, a_idx),
                    set_label="cross",
                )
            )
            terms.append(
                ExcitonLCUTerm(
                    coefficient=-1.0,
                    op_kind="V_direct",
                    acting_on=(i, a_idx),
                    set_label="cross",
                )
            )

    return ExcitonHamiltonianEncoding(
        m=m,
        lattice_shape=lattice_shape,
        f_bundle=f_bundle,
        v_direct_bundle=v_direct_bundle,
        v_exchange_bundle=v_exchange_bundle,
        terms=tuple(terms),
    )
