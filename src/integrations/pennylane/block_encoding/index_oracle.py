from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


Wire = int | str


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def _as_dim_wires(wires: Sequence[Wire] | Sequence[Sequence[Wire]]) -> list[list[Wire]]:
    if len(wires) == 0:
        raise ValueError("Wire collection must be non-empty.")

    first = wires[0]
    if isinstance(first, (list, tuple)):
        out = [list(w) for w in wires]  # type: ignore[arg-type]
    else:
        out = [list(wires)]  # type: ignore[list-item]

    if any(len(w) == 0 for w in out):
        raise ValueError("Each dimension wire list must be non-empty.")
    return out


def one_particle_index_oracle(
    *,
    i_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    m_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    L: int,
    R_loc: int,
    work_wires: Sequence[Wire],
) -> None:
    r"""Apply one-particle index oracle:

      |m>|i> -> |m>|i'>
      i'_k = i_k - R_loc + m_k (mod L)

    This implementation is exact for power-of-two `L`.
    """
    import pennylane as qml

    if not _is_power_of_two(int(L)):
        raise ValueError("This implementation currently requires L to be a power of two.")
    if int(R_loc) < 0:
        raise ValueError("R_loc must be >= 0.")
    i_dims = _as_dim_wires(i_wires)
    m_dims = _as_dim_wires(m_wires)
    if len(i_dims) != len(m_dims):
        raise ValueError("i_wires and m_wires must have the same number of dimensions.")
    n_bits = len(i_dims[0])
    if any(len(x) != n_bits for x in i_dims):
        raise ValueError("All i-register dimensions must have identical bit-width.")
    if any(len(x) < n_bits for x in m_dims):
        raise ValueError("Each m-register dimension must have bit-width >= i-register bit-width.")
    if len(work_wires) < max(0, n_bits - 1):
        raise ValueError(
            f"SemiAdder requires at least {max(0, n_bits - 1)} work_wires for {n_bits}-bit registers."
        )

    k_shift = int((-int(R_loc)) % int(L))
    for m_d, i_d in zip(m_dims, i_dims):
        m_use = m_d[-n_bits:] if len(m_d) > n_bits else m_d
        qml.SemiAdder(x_wires=m_use, y_wires=i_d, work_wires=work_wires)
        qml.Adder(k_shift, x_wires=i_d, mod=int(L), work_wires=work_wires)


def two_particle_index_oracle(
    *,
    i_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    j_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    m_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    l_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    L: int,
    R_c: int,
    R_loc: int,
    work_wires: Sequence[Wire],
) -> None:
    r"""Apply two-particle index oracle:

      |m>|l>|i>|j> -> |m>|l>|i'>|j'>

      i'_k = i_k - R_c + m_k                    (mod L)
      j'_k = j_k - R_c + m_k - R_loc + l_k      (mod L)

    This implementation is exact for power-of-two `L`.
    """
    import pennylane as qml

    if not _is_power_of_two(int(L)):
        raise ValueError("This implementation currently requires L to be a power of two.")
    if int(R_c) < 0 or int(R_loc) < 0:
        raise ValueError("R_c and R_loc must be >= 0.")
    i_dims = _as_dim_wires(i_wires)
    j_dims = _as_dim_wires(j_wires)
    m_dims = _as_dim_wires(m_wires)
    l_dims = _as_dim_wires(l_wires)

    D = len(i_dims)
    if not (len(j_dims) == len(m_dims) == len(l_dims) == D):
        raise ValueError("All register groups must have the same number of dimensions.")
    n_bits = len(i_dims[0])
    if any(len(x) != n_bits for grp in (i_dims, j_dims) for x in grp):
        raise ValueError("All i/j-register dimensions must have identical bit-width.")
    if any(len(x) < n_bits for grp in (m_dims, l_dims) for x in grp):
        raise ValueError("All m/l-register dimensions must have bit-width >= i/j bit-width.")
    if len(work_wires) < max(0, n_bits - 1):
        raise ValueError(
            f"SemiAdder requires at least {max(0, n_bits - 1)} work_wires for {n_bits}-bit registers."
        )

    i_shift = int((-int(R_c)) % int(L))
    j_shift = int((-(int(R_c) + int(R_loc))) % int(L))

    for m_d, l_d, i_d, j_d in zip(m_dims, l_dims, i_dims, j_dims):
        m_use = m_d[-n_bits:] if len(m_d) > n_bits else m_d
        l_use = l_d[-n_bits:] if len(l_d) > n_bits else l_d
        # i <- i + m - R_c
        qml.SemiAdder(x_wires=m_use, y_wires=i_d, work_wires=work_wires)
        qml.Adder(i_shift, x_wires=i_d, mod=int(L), work_wires=work_wires)

        # j <- j + m + l - R_c - R_loc
        qml.SemiAdder(x_wires=m_use, y_wires=j_d, work_wires=work_wires)
        qml.SemiAdder(x_wires=l_use, y_wires=j_d, work_wires=work_wires)
        qml.Adder(j_shift, x_wires=j_d, mod=int(L), work_wires=work_wires)


__all__ = ["one_particle_index_oracle", "two_particle_index_oracle"]
