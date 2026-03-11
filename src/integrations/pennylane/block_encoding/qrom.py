from __future__ import annotations

from typing import Sequence

import numpy as np


def _required_bits(size: int) -> int:
    if int(size) <= 0:
        raise ValueError("Size must be positive.")
    return max(1, int(np.ceil(np.log2(int(size)))))


def _required_index_bits(size: int) -> int:
    if int(size) <= 0:
        raise ValueError("Size must be positive.")
    # For singleton index sets, allow zero wires.
    if int(size) == 1:
        return 0
    return int(np.ceil(np.log2(int(size))))


def qrom_table_2d(
    table: np.ndarray | Sequence[Sequence[int]],
    i_wires: Sequence[int | str],
    j_wires: Sequence[int | str],
    data_wires: Sequence[int | str],
    work_wires: Sequence[int | str] | None = None,
) -> None:
    r"""Apply a 2D table QROM in PennyLane.

    Implements:
      |i>|j>|x> -> |i>|j>|x xor A[i,j]>

    If the data register starts in |0...0>, this realizes:
      |i>|j>|0> -> |i>|j>|A[i,j]>

    Notes:
    - Out-of-range basis states for i/j (when 2^n > N or M) are untouched.
    - Table values must be non-negative integers.
    """
    import pennylane as qml

    arr = np.asarray(table)
    if arr.ndim != 2:
        raise ValueError(f"`table` must be 2D. Received shape={arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    if np.any(arr < 0):
        raise ValueError("All table entries must be non-negative integers.")

    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    req_i = _required_index_bits(n_rows)
    req_j = _required_index_bits(n_cols)
    if len(i_wires) < req_i:
        raise ValueError(f"Need at least {req_i} i_wires for {n_rows} rows.")
    if len(j_wires) < req_j:
        raise ValueError(f"Need at least {req_j} j_wires for {n_cols} cols.")

    max_value = int(arr.max(initial=0))
    req_data = _required_bits(max_value + 1)
    if len(data_wires) < req_data:
        raise ValueError(
            f"Need at least {req_data} data_wires for max table value {max_value}."
        )

    i_width = len(i_wires)
    j_width = len(j_wires)
    d_width = len(data_wires)

    # PennyLane QROM indexes bitstrings by the integer value of control wires.
    # We pack address bits as [i_bits, j_bits] and pad out-of-range indices with 0.
    addr_width = i_width + j_width
    bitstrings: list[str] = []
    for addr in range(2**addr_width):
        i = addr >> j_width
        j = addr & ((1 << j_width) - 1)
        val = int(arr[i, j]) if (i < n_rows and j < n_cols) else 0
        bitstrings.append(format(val, f"0{d_width}b"))

    qml.QROM(
        bitstrings=bitstrings,
        control_wires=tuple(i_wires) + tuple(j_wires),
        target_wires=tuple(data_wires),
        work_wires=tuple(work_wires or ()),
    )


__all__ = ["qrom_table_2d"]
