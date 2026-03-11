from __future__ import annotations

from typing import Sequence

import numpy as np

from .entry_oracle import controlled_entry_oracle_2d, entry_oracle_2d
from .index_oracle import one_particle_index_oracle, two_particle_index_oracle


Wire = int | str


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


def _uniform_prefix_state(num_states: int, n_qubits: int) -> np.ndarray:
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    dim = 2**n_qubits
    if num_states > dim:
        raise ValueError(
            f"num_states={num_states} exceeds register capacity 2^{n_qubits}={dim}."
        )
    vec = np.zeros(dim, dtype=np.complex128)
    vec[:num_states] = 1.0 / np.sqrt(float(num_states))
    return vec


def _prepare_uniform_prefix(wires: Sequence[Wire], num_states: int) -> None:
    import pennylane as qml

    if num_states <= 1:
        return
    state = _uniform_prefix_state(num_states, len(wires))
    qml.MottonenStatePreparation(state, wires=wires)


def _unprepare_uniform_prefix(wires: Sequence[Wire], num_states: int) -> None:
    import pennylane as qml

    if num_states <= 1:
        return
    state = _uniform_prefix_state(num_states, len(wires))
    qml.adjoint(qml.MottonenStatePreparation)(state, wires=wires)


def one_particle_sparse_block_encoding(
    *,
    table: np.ndarray | Sequence[Sequence[float]],
    i_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    m_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    ancilla_wire: Wire,
    angle_wires: Sequence[Wire],
    L: int,
    R_loc: int,
    index_work_wires: Sequence[Wire],
    entry_work_wires: Sequence[Wire] | None = None,
    control_wires: Sequence[Wire] | None = None,
    control_values: Sequence[int] | None = None,
) -> None:
    r"""Construct one-particle sparse block-encoding template in PennyLane.

    Sequence:
    1) Prepare uniform superposition on m-registers (each dimension over 0..2R_loc).
    2) Apply adjoint one-particle index oracle on (m, i).
    3) Apply entry oracle using 2D table over flattened (i, m).
    4) Unprepare m-registers.
    """
    import pennylane as qml

    i_dims = _as_dim_wires(i_wires)
    m_dims = _as_dim_wires(m_wires)
    if len(i_dims) != len(m_dims):
        raise ValueError("i_wires and m_wires must have same number of dimensions.")
    D = len(i_dims)

    n_i = int(L**D)
    n_m = int((2 * int(R_loc) + 1) ** D)
    arr = np.asarray(table, dtype=float)
    if arr.shape != (n_i, n_m):
        raise ValueError(f"table must have shape ({n_i}, {n_m}), got {arr.shape}.")
    ext_controls = tuple(control_wires or ())
    if control_values is None:
        control_values = (1,) * len(ext_controls)
    if len(control_values) != len(ext_controls):
        raise ValueError("control_values must have same length as control_wires.")

    local_states = 2 * int(R_loc) + 1
    for m_d in m_dims:
        _prepare_uniform_prefix(m_d, local_states)

    index_kwargs = dict(
        i_wires=i_dims,
        m_wires=m_dims,
        L=int(L),
        R_loc=int(R_loc),
        work_wires=index_work_wires,
    )
    if ext_controls:
        qml.ctrl(
            qml.adjoint(one_particle_index_oracle),
            control=ext_controls,
            control_values=tuple(control_values),
        )(**index_kwargs)
    else:
        qml.adjoint(one_particle_index_oracle)(**index_kwargs)

    i_flat = [w for reg in i_dims for w in reg]
    m_flat = [w for reg in m_dims for w in reg]
    if ext_controls:
        controlled_entry_oracle_2d(
            table=arr,
            i_wires=i_flat,
            j_wires=m_flat,
            ancilla_wire=ancilla_wire,
            angle_wires=angle_wires,
            control_wires=ext_controls,
            control_values=tuple(control_values),
            work_wires=entry_work_wires or (),
        )
    else:
        entry_oracle_2d(
            table=arr,
            i_wires=i_flat,
            j_wires=m_flat,
            ancilla_wire=ancilla_wire,
            angle_wires=angle_wires,
            work_wires=entry_work_wires or (),
        )

    for m_d in reversed(m_dims):
        _unprepare_uniform_prefix(m_d, local_states)


def two_particle_sparse_block_encoding(
    *,
    table: np.ndarray | Sequence[Sequence[float]],
    i_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    j_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    m_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    l_wires: Sequence[Wire] | Sequence[Sequence[Wire]],
    ancilla_wire: Wire,
    angle_wires: Sequence[Wire],
    L: int,
    R_c: int,
    R_loc: int,
    index_work_wires: Sequence[Wire],
    entry_work_wires: Sequence[Wire] | None = None,
    control_wires: Sequence[Wire] | None = None,
    control_values: Sequence[int] | None = None,
) -> None:
    r"""Construct two-particle sparse block-encoding template in PennyLane.

    Sequence:
    1) Prepare uniform superposition on m and l registers.
    2) Apply adjoint two-particle index oracle on (m, l, i, j).
    3) Apply entry oracle using 2D table over flattened ((i,j), (m,l)).
    4) Unprepare m and l registers.
    """
    import pennylane as qml

    i_dims = _as_dim_wires(i_wires)
    j_dims = _as_dim_wires(j_wires)
    m_dims = _as_dim_wires(m_wires)
    l_dims = _as_dim_wires(l_wires)

    D = len(i_dims)
    if not (len(j_dims) == len(m_dims) == len(l_dims) == D):
        raise ValueError("All wire groups must have the same number of dimensions.")

    n_row = int((L**D) * (L**D))
    n_col = int(((2 * int(R_c) + 1) ** D) * ((2 * int(R_loc) + 1) ** D))
    arr = np.asarray(table, dtype=float)
    if arr.shape != (n_row, n_col):
        raise ValueError(f"table must have shape ({n_row}, {n_col}), got {arr.shape}.")
    ext_controls = tuple(control_wires or ())
    if control_values is None:
        control_values = (1,) * len(ext_controls)
    if len(control_values) != len(ext_controls):
        raise ValueError("control_values must have same length as control_wires.")

    m_states = 2 * int(R_c) + 1
    l_states = 2 * int(R_loc) + 1

    for m_d in m_dims:
        _prepare_uniform_prefix(m_d, m_states)
    for l_d in l_dims:
        _prepare_uniform_prefix(l_d, l_states)

    index_kwargs = dict(
        i_wires=i_dims,
        j_wires=j_dims,
        m_wires=m_dims,
        l_wires=l_dims,
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        work_wires=index_work_wires,
    )
    if ext_controls:
        qml.ctrl(
            qml.adjoint(two_particle_index_oracle),
            control=ext_controls,
            control_values=tuple(control_values),
        )(**index_kwargs)
    else:
        qml.adjoint(two_particle_index_oracle)(**index_kwargs)

    row_wires = [w for reg in i_dims for w in reg] + [w for reg in j_dims for w in reg]
    col_wires = [w for reg in m_dims for w in reg] + [w for reg in l_dims for w in reg]
    if ext_controls:
        controlled_entry_oracle_2d(
            table=arr,
            i_wires=row_wires,
            j_wires=col_wires,
            ancilla_wire=ancilla_wire,
            angle_wires=angle_wires,
            control_wires=ext_controls,
            control_values=tuple(control_values),
            work_wires=entry_work_wires or (),
        )
    else:
        entry_oracle_2d(
            table=arr,
            i_wires=row_wires,
            j_wires=col_wires,
            ancilla_wire=ancilla_wire,
            angle_wires=angle_wires,
            work_wires=entry_work_wires or (),
        )

    for l_d in reversed(l_dims):
        _unprepare_uniform_prefix(l_d, l_states)
    for m_d in reversed(m_dims):
        _unprepare_uniform_prefix(m_d, m_states)


__all__ = ["one_particle_sparse_block_encoding", "two_particle_sparse_block_encoding"]
