from __future__ import annotations

from typing import Sequence

import numpy as np


def _required_index_bits(size: int) -> int:
    if int(size) <= 0:
        raise ValueError("Size must be positive.")
    if int(size) == 1:
        return 0
    return int(np.ceil(np.log2(int(size))))


def _theta_bitstrings_from_table(
    table: np.ndarray,
    i_width: int,
    j_width: int,
    angle_bits: int,
) -> list[str]:
    """Encode theta(i,j)=2*arccos(A_ij) into fixed-point bitstrings over [0, 2pi)."""
    n_rows, n_cols = int(table.shape[0]), int(table.shape[1])
    addr_width = i_width + j_width
    mod = 2**angle_bits

    bitstrings: list[str] = []
    for addr in range(2**addr_width):
        i = addr >> j_width if j_width > 0 else addr
        j = addr & ((1 << j_width) - 1) if j_width > 0 else 0
        a = float(table[i, j]) if (i < n_rows and j < n_cols) else 0.0
        a = float(np.clip(a, -1.0, 1.0))
        theta = 2.0 * float(np.arccos(a))  # in [0, 2pi]
        t_int = int(np.rint(theta / (2.0 * np.pi) * mod)) % mod
        bitstrings.append(format(t_int, f"0{angle_bits}b"))
    return bitstrings


def entry_oracle_2d(
    table: np.ndarray | Sequence[Sequence[float]],
    i_wires: Sequence[int | str],
    j_wires: Sequence[int | str],
    ancilla_wire: int | str,
    angle_wires: Sequence[int | str],
    work_wires: Sequence[int | str] | None = None,
) -> None:
    r"""Entry oracle using two QROM calls and angle-controlled rotations.

    For basis input |i>|j>|0>_a (with angle register initialized to |0...0>), this applies:
      |i>|j>|0>_a  ->  |i>|j>(A_ij |0>_a + sqrt(1 - A_ij^2) |1>_a)
    for real entries A_ij in [-1, 1].

    Implementation:
    1) QROM-load fixed-point theta_ij = 2 arccos(A_ij) into angle register.
    2) Apply controlled Ry ladder on ancilla from angle bits.
    3) QROM-uncompute angle register (second QROM call).
    """
    import pennylane as qml

    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"`table` must be 2D. Received shape={arr.shape}.")
    if np.any(np.abs(arr) > 1.0):
        raise ValueError("All table entries must satisfy |A_ij| <= 1.")

    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    req_i = _required_index_bits(n_rows)
    req_j = _required_index_bits(n_cols)
    if len(i_wires) < req_i:
        raise ValueError(f"Need at least {req_i} i_wires for {n_rows} rows.")
    if len(j_wires) < req_j:
        raise ValueError(f"Need at least {req_j} j_wires for {n_cols} cols.")
    if len(angle_wires) < 1:
        raise ValueError("Need at least one angle wire.")

    i_width = len(i_wires)
    j_width = len(j_wires)
    b = len(angle_wires)
    controls = tuple(i_wires) + tuple(j_wires)
    work = tuple(work_wires or ())

    bitstrings = _theta_bitstrings_from_table(arr, i_width=i_width, j_width=j_width, angle_bits=b)

    # QROM call 1: load theta bits into angle register.
    qml.QROM(
        bitstrings=bitstrings,
        control_wires=controls,
        target_wires=tuple(angle_wires),
        work_wires=work,
    )

    # Angle-controlled Ry synthesis:
    # If theta_int has bits b_k (MSB->LSB), then theta ~= sum_k b_k * 2pi * 2^{-(k+1)}.
    for k, w in enumerate(angle_wires):
        qml.ctrl(qml.RY, control=w)(2.0 * np.pi * (2.0 ** (-(k + 1))), wires=ancilla_wire)

    # QROM call 2: uncompute theta register.
    qml.QROM(
        bitstrings=bitstrings,
        control_wires=controls,
        target_wires=tuple(angle_wires),
        work_wires=work,
    )


def controlled_entry_oracle_2d(
    table: np.ndarray | Sequence[Sequence[float]],
    i_wires: Sequence[int | str],
    j_wires: Sequence[int | str],
    ancilla_wire: int | str,
    angle_wires: Sequence[int | str],
    control_wires: Sequence[int | str],
    control_values: Sequence[int] | None = None,
    work_wires: Sequence[int | str] | None = None,
) -> None:
    r"""Controlled entry oracle with unconditional QROM and controlled rotation ladder.

    This uses the same two QROM calls as :func:`entry_oracle_2d` to load/uncompute the
    angle register, but only the angle-synthesis rotations are additionally controlled
    by ``control_wires``. The net effect is that the data-loading cancels while the
    ancilla rotation is applied iff the external controls match ``control_values``.
    """
    import pennylane as qml

    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"`table` must be 2D. Received shape={arr.shape}.")
    if np.any(np.abs(arr) > 1.0):
        raise ValueError("All table entries must satisfy |A_ij| <= 1.")
    if len(control_wires) == 0:
        entry_oracle_2d(
            table=arr,
            i_wires=i_wires,
            j_wires=j_wires,
            ancilla_wire=ancilla_wire,
            angle_wires=angle_wires,
            work_wires=work_wires,
        )
        return

    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    req_i = _required_index_bits(n_rows)
    req_j = _required_index_bits(n_cols)
    if len(i_wires) < req_i:
        raise ValueError(f"Need at least {req_i} i_wires for {n_rows} rows.")
    if len(j_wires) < req_j:
        raise ValueError(f"Need at least {req_j} j_wires for {n_cols} cols.")
    if len(angle_wires) < 1:
        raise ValueError("Need at least one angle wire.")

    if control_values is None:
        control_values = (1,) * len(control_wires)
    if len(control_values) != len(control_wires):
        raise ValueError("control_values must have the same length as control_wires.")

    i_width = len(i_wires)
    j_width = len(j_wires)
    b = len(angle_wires)
    qrom_controls = tuple(i_wires) + tuple(j_wires)
    ext_controls = tuple(control_wires)
    work = tuple(work_wires or ())

    bitstrings = _theta_bitstrings_from_table(arr, i_width=i_width, j_width=j_width, angle_bits=b)

    qml.QROM(
        bitstrings=bitstrings,
        control_wires=qrom_controls,
        target_wires=tuple(angle_wires),
        work_wires=work,
    )

    for k, w in enumerate(angle_wires):
        qml.ctrl(
            qml.RY,
            control=ext_controls + (w,),
            control_values=tuple(control_values) + (1,),
        )(2.0 * np.pi * (2.0 ** (-(k + 1))), wires=ancilla_wire)

    qml.QROM(
        bitstrings=bitstrings,
        control_wires=qrom_controls,
        target_wires=tuple(angle_wires),
        work_wires=work,
    )


__all__ = ["entry_oracle_2d", "controlled_entry_oracle_2d"]
