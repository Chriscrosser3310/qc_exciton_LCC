from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from qualtran.bloqs.data_loading.qrom import QROM
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for qrom_sparse utilities. Install with extra '[qualtran]'."
    ) from exc


@dataclass(frozen=True)
class SparseTensorCOO:
    """Simple COO sparse tensor representation for QROM data loading.

    Attributes:
        shape: Full tensor shape for the dense data table.
        coords: Integer coordinates of nonzero entries, shape (nnz, ndim).
        values: Integer values for each nonzero coordinate, shape (nnz,).
    """

    shape: tuple[int, ...]
    coords: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.shape) == 0:
            raise ValueError("shape must be non-empty.")
        if any(s <= 0 for s in self.shape):
            raise ValueError("all shape extents must be positive.")
        if self.coords.ndim != 2:
            raise ValueError("coords must have shape (nnz, ndim).")
        if self.coords.shape[1] != len(self.shape):
            raise ValueError("coords second dimension must match len(shape).")
        if self.values.ndim != 1:
            raise ValueError("values must be 1-D.")
        if self.coords.shape[0] != self.values.shape[0]:
            raise ValueError("coords and values must have matching nnz.")
        if not np.issubdtype(self.coords.dtype, np.integer):
            raise ValueError("coords must be integer type.")
        if not np.issubdtype(self.values.dtype, np.integer):
            raise ValueError("values must be integer type.")
        for dim, bound in enumerate(self.shape):
            c = self.coords[:, dim]
            if np.any(c < 0) or np.any(c >= bound):
                raise ValueError(f"coords out of range in dimension {dim}.")

    @classmethod
    def from_dense(cls, data: np.ndarray, atol: float = 0.0) -> "SparseTensorCOO":
        if data.ndim == 0:
            raise ValueError("data must be at least 1-D.")
        dense = np.asarray(data)
        mask = np.abs(dense) > float(atol)
        coords = np.argwhere(mask).astype(np.int64, copy=False)
        values = dense[mask].astype(np.int64, copy=False)
        return cls(shape=tuple(int(s) for s in dense.shape), coords=coords, values=values)

    def to_dense(self, fill_value: int = 0) -> np.ndarray:
        out = np.full(self.shape, int(fill_value), dtype=np.int64)
        if self.coords.shape[0] > 0:
            out[tuple(self.coords.T)] = self.values
        return out


def _to_dense_data_item(item: SparseTensorCOO | np.ndarray, fill_value: int) -> np.ndarray:
    if isinstance(item, SparseTensorCOO):
        return item.to_dense(fill_value=fill_value)
    return np.asarray(item, dtype=np.int64)


def qrom_build_from_sparse_data(
    *data: SparseTensorCOO | np.ndarray,
    target_bitsizes: int | tuple[int, ...] | None = None,
    target_shapes: tuple[tuple[int, ...], ...] = (),
    num_controls: int = 0,
    fill_value: int = 0,
) -> QROM:
    """Build a Qualtran QROM from sparse tensors (COO) and/or dense arrays.

    This mirrors `QROM.build_from_data(...)` but accepts `SparseTensorCOO`
    elements directly.
    """
    if len(data) == 0:
        raise ValueError("At least one data tensor is required.")
    dense_data = tuple(_to_dense_data_item(item, fill_value=fill_value) for item in data)
    return QROM.build_from_data(
        *dense_data,
        target_bitsizes=target_bitsizes,
        target_shapes=target_shapes,
        num_controls=num_controls,
    )


def sparse_coo_from_items(
    shape: Sequence[int],
    items: Iterable[tuple[Sequence[int], int]],
) -> SparseTensorCOO:
    """Helper to construct SparseTensorCOO from iterable (coord, value) pairs."""
    coords: list[list[int]] = []
    values: list[int] = []
    for coord, val in items:
        coords.append([int(x) for x in coord])
        values.append(int(val))
    if len(coords) == 0:
        c = np.zeros((0, len(shape)), dtype=np.int64)
        v = np.zeros((0,), dtype=np.int64)
    else:
        c = np.asarray(coords, dtype=np.int64)
        v = np.asarray(values, dtype=np.int64)
    return SparseTensorCOO(shape=tuple(int(s) for s in shape), coords=c, values=v)

