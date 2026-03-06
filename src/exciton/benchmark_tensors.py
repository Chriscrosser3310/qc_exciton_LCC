from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Metric = Literal["euclidean", "manhattan", "chebyshev"]
VConvention = Literal["pqrs", "direct", "exchange"]
FConvention = Literal["pq", "row_oracle"]


@dataclass(frozen=True)
class LatticeSpec:
    """D-dimensional lattice specification.

    `shape` gives lattice extents along each dimension, e.g.:
    - 1D chain with 8 sites: (8,)
    - 2D grid 4x4: (4, 4)
    - 3D grid 3x3x3: (3, 3, 3)
    """

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


def lattice_coordinates(spec: LatticeSpec) -> np.ndarray:
    """Return coordinates for all lattice points as shape (n_sites, D)."""
    grids = np.indices(spec.shape, dtype=float)
    coords = grids.reshape(spec.ndim, -1).T
    return coords


def pairwise_lattice_distances(spec: LatticeSpec, metric: Metric = "euclidean") -> np.ndarray:
    """Compute pairwise distances R_pq between all lattice points."""
    coords = lattice_coordinates(spec)
    diffs = coords[:, None, :] - coords[None, :, :]
    if metric == "euclidean":
        return np.linalg.norm(diffs, axis=-1)
    if metric == "manhattan":
        return np.abs(diffs).sum(axis=-1)
    if metric == "chebyshev":
        return np.abs(diffs).max(axis=-1)
    raise ValueError(f"Unsupported metric: {metric}")


def _distance_from_diffs(diffs: np.ndarray, metric: Metric) -> np.ndarray:
    """Compute distance from coordinate differences along last axis."""
    if metric == "euclidean":
        return np.linalg.norm(diffs, axis=-1)
    if metric == "manhattan":
        return np.abs(diffs).sum(axis=-1)
    if metric == "chebyshev":
        return np.abs(diffs).max(axis=-1)
    raise ValueError(f"Unsupported metric: {metric}")


def pairwise_lattice_max_distances(spec: LatticeSpec, periodic: bool = False) -> np.ndarray:
    """Compute Chebyshev/max distance between lattice points.

    If `periodic=True`, each coordinate difference uses periodic shortest distance.
    """
    coords = lattice_coordinates(spec)
    diffs = np.abs(coords[:, None, :] - coords[None, :, :])
    if periodic:
        shape_arr = np.asarray(spec.shape, dtype=float)
        diffs = np.minimum(diffs, shape_arr[None, None, :] - diffs)
    return diffs.max(axis=-1)


def lattice_max_dist(spec: LatticeSpec, periodic: bool = False) -> int:
    """Maximum possible Chebyshev distance on the lattice."""
    if periodic:
        return int(max(s // 2 for s in spec.shape))
    return int(max(s - 1 for s in spec.shape))


def generate_f_tensor(
    shape: tuple[int, ...] | None = None,
    *,
    L: int | None = None,
    D: int | None = None,
    a: float = 1.0,
    metric: Metric = "chebyshev",
    r_cut: int | None = None,
    oracle_convention: FConvention = "pq",
    periodic_cutoff: bool = False,
    dtype: type[np.floating] = np.float64,
) -> np.ndarray:
    r"""Generate F tensor on a D-dimensional lattice.

    Conventions:
    - ``oracle_convention="pq"`` (default): return full matrix ``F_{pq}`` with shape
      ``(n_sites, n_sites)`` where ``n_sites = L^D`` for uniform lattices.
    - ``oracle_convention="row_oracle"``: return compact row-oracle table with shape
      ``(L,)*D + (2*r_cut+1,)*D`` and mapping
      ``q_k = p_k - r_cut + m_k (mod L)``.
    """
    shape = _resolve_shape(shape=shape, L=L, D=D)
    if oracle_convention == "row_oracle":
        return _generate_f_tensor_row_oracle_convention(
            shape=shape,
            a=a,
            metric=metric,
            r_loc=r_cut,
            periodic_cutoff=periodic_cutoff,
            dtype=dtype,
        )
    if oracle_convention != "pq":
        raise ValueError(f"Unsupported F oracle convention: {oracle_convention}")

    spec = LatticeSpec(shape=shape)
    r_pq = pairwise_lattice_distances(spec, metric=metric)
    f = np.exp(-float(a) * r_pq).astype(dtype, copy=False)
    if r_cut is None:
        r_cut = lattice_max_dist(spec, periodic=periodic_cutoff)
    dmax = pairwise_lattice_max_distances(spec, periodic=periodic_cutoff)
    f = np.where(dmax <= float(r_cut), f, 0.0)
    return f


def generate_v_tensor(
    shape: tuple[int, ...] | None = None,
    *,
    L: int | None = None,
    D: int | None = None,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    metric: Metric = "chebyshev",
    r_loc: int | None = None,
    r_c: int | None = None,
    zero_distance_value: float | None = None,
    oracle_convention: VConvention = "pqrs",
    periodic_cutoff: bool = False,
    dtype: type[np.floating] = np.float64,
) -> np.ndarray:
    r"""Generate V_{pqrs} with:

    V_{pqrs} =
      exp(-a R_{pr}) * exp(-b R_{qs}) * (c R_{(p,r),(q,s)})^(-(3 - delta_{pr} - delta_{qs}))

    where delta is a Kronecker delta over site indices and:
    - R_{xy} is site-to-site lattice distance,
    - R_{(p,r),(q,s)} is the distance between the centers of pairs (p,r) and (q,s).

    Optional hard cutoffs (independent of tolerance thresholding):
    - if r_c is set: zero out when max_dist(p, r) > r_c
    - if r_loc is set: zero out when max_dist(p, q) > r_loc or max_dist(r, s) > r_loc

    Zero center-distance handling:
    - if R_{(p,r),(q,s)} = 0, set entry to `zero_distance_value`.
    - default is `2*c` when `zero_distance_value` is not provided.
    """
    shape = _resolve_shape(shape=shape, L=L, D=D)
    if oracle_convention in ("direct", "exchange"):
        return _generate_v_tensor_oracle_convention(
            shape=shape,
            a=a,
            b=b,
            c=c,
            metric=metric,
            r_loc=r_loc,
            r_c=r_c,
            zero_distance_value=zero_distance_value,
            periodic_cutoff=periodic_cutoff,
            dtype=dtype,
            convention=oracle_convention,
        )

    spec = LatticeSpec(shape=shape)
    n = spec.n_sites
    coords = lattice_coordinates(spec).astype(dtype, copy=False)
    r = pairwise_lattice_distances(spec, metric=metric).astype(dtype, copy=False)

    r_pr = r[:, None, :, None]
    r_qs = r[None, :, None, :]
    # Distance between pair centers:
    # center(p,r) = (coords[p] + coords[r]) / 2
    # center(q,s) = (coords[q] + coords[s]) / 2
    center_pr = 0.5 * (coords[:, None, :] + coords[None, :, :])  # (n, n, D)
    center_qs = center_pr  # same index space
    center_diffs = center_pr[:, :, None, None, :] - center_qs[None, None, :, :, :]
    # center_diffs axes are (p, r, q, s, D); reorder to (p, q, r, s) to match tensor indexing.
    r_center = np.transpose(
        _distance_from_diffs(center_diffs, metric=metric).astype(dtype, copy=False),
        (0, 2, 1, 3),
    )

    eye = np.eye(n, dtype=dtype)
    delta_pr = eye[:, None, :, None]
    delta_qs = eye[None, :, None, :]

    exponent = -(3.0 - delta_pr - delta_qs)
    prefac = np.exp(-float(a) * r_pr) * np.exp(-float(b) * r_qs)
    base = float(c) * r_center
    with np.errstate(divide="ignore", invalid="ignore"):
        coulomb_like = np.power(base, exponent)
    zval = float(2.0 * c if zero_distance_value is None else zero_distance_value)
    coulomb_like = np.where(base == 0.0, zval, coulomb_like)
    coulomb_like = np.where(np.isfinite(coulomb_like), coulomb_like, 0.0)
    v = (prefac * coulomb_like).astype(dtype, copy=False)

    if r_loc is None:
        r_loc = lattice_max_dist(spec, periodic=periodic_cutoff)
    if r_c is None:
        r_c = lattice_max_dist(spec, periodic=periodic_cutoff)

    dmax = pairwise_lattice_max_distances(spec, periodic=periodic_cutoff).astype(dtype, copy=False)
    mask = np.ones((n, n, n, n), dtype=bool)
    mask &= dmax[:, None, :, None] <= float(r_c)  # max_dist(p, r)
    mask &= dmax[:, :, None, None] <= float(r_loc)  # max_dist(p, q)
    mask &= dmax[None, None, :, :] <= float(r_loc)  # max_dist(r, s)
    v = np.where(mask, v, 0.0)
    return v


def _resolve_shape(
    shape: tuple[int, ...] | None,
    L: int | None,
    D: int | None,
) -> tuple[int, ...]:
    if shape is not None:
        out = tuple(int(s) for s in shape)
        if len(out) == 0 or any(s <= 0 for s in out):
            raise ValueError("shape must be a non-empty tuple of positive ints.")
        if L is not None and D is not None and out != (int(L),) * int(D):
            raise ValueError("shape and (L, D) are inconsistent.")
        return out
    if L is None or D is None:
        raise ValueError("Provide either `shape`, or both `L` and `D`.")
    if int(L) <= 0 or int(D) <= 0:
        raise ValueError("L and D must be positive.")
    return (int(L),) * int(D)


def _require_uniform_l(shape: tuple[int, ...]) -> int:
    if len(shape) == 0:
        raise ValueError("shape must be non-empty.")
    l = int(shape[0])
    if any(int(s) != l for s in shape):
        raise ValueError(
            "oracle_convention output requires a uniform lattice shape (L,)*D."
        )
    return l


def _coords_to_flat_index(coords: np.ndarray, l: int) -> np.ndarray:
    d = coords.shape[0]
    flat = np.ravel_multi_index(
        coords.reshape(d, -1),
        dims=(l,) * d,
    )
    return flat.reshape(coords.shape[1:])


def _generate_v_tensor_oracle_convention(
    shape: tuple[int, ...],
    a: float,
    b: float,
    c: float,
    metric: Metric,
    r_loc: int | None,
    r_c: int | None,
    zero_distance_value: float | None,
    periodic_cutoff: bool,
    dtype: type[np.floating],
    convention: VConvention,
) -> np.ndarray:
    if convention not in ("direct", "exchange"):
        raise ValueError(f"Unsupported oracle convention: {convention}")

    spec = LatticeSpec(shape=shape)
    l = _require_uniform_l(shape)
    d = len(shape)
    coords = lattice_coordinates(spec).astype(dtype, copy=False)
    r = pairwise_lattice_distances(spec, metric=metric).astype(dtype, copy=False)
    dmax = pairwise_lattice_max_distances(spec, periodic=periodic_cutoff).astype(dtype, copy=False)
    if r_loc is None:
        r_loc = lattice_max_dist(spec, periodic=periodic_cutoff)
    if r_c is None:
        r_c = lattice_max_dist(spec, periodic=periodic_cutoff)

    i_shape = (l,) * d
    j_shape = (l,) * d
    m_shape = (2 * int(r_c) + 1,) * d
    l_shape = (2 * int(r_loc) + 1,) * d

    # Coordinate grids.
    i_coord = np.indices(i_shape, dtype=int)  # (d, *i_shape)
    j_coord = np.indices(j_shape, dtype=int)  # (d, *j_shape)
    m_coord = np.indices(m_shape, dtype=int)  # (d, *m_shape), values in [0, 2*r_c]
    l_coord = np.indices(l_shape, dtype=int)  # (d, *l_shape), values in [0, 2*r_loc]

    # i' and j' coordinates from oracle convention:
    # i'_k = i_k - r_c + m_k (mod l)
    # j'_k = j_k - r_c + m_k - r_loc + l_k (mod l)
    i_b = i_coord.reshape((d,) + i_shape + (1,) * d)
    m_b = m_coord.reshape((d,) + (1,) * d + m_shape)
    ip_coord = (i_b - int(r_c) + m_b) % l  # (d, *i_shape, *m_shape)

    j_b = j_coord.reshape((d,) + j_shape + (1,) * d + (1,) * d)
    m_b2 = m_coord.reshape((d,) + (1,) * d + m_shape + (1,) * d)
    l_b = l_coord.reshape((d,) + (1,) * d + (1,) * d + l_shape)
    jp_coord = (j_b - int(r_c) + m_b2 - int(r_loc) + l_b) % l  # (d, *j_shape, *m_shape, *l_shape)

    # Flat site indices for V[p,q,r,s] addressing.
    i_idx = _coords_to_flat_index(i_coord, l=l)  # *i_shape
    j_idx = _coords_to_flat_index(j_coord, l=l)  # *j_shape
    ip_idx = _coords_to_flat_index(ip_coord, l=l)  # *i_shape,*m_shape
    jp_idx = _coords_to_flat_index(jp_coord, l=l)  # *j_shape,*m_shape,*l_shape

    # Broadcast all indices to target output shape:
    # (i_dims, j_dims, m_dims, l_dims)
    p_idx = i_idx.reshape(i_shape + (1,) * (3 * d))
    s_idx = jp_idx.reshape((1,) * d + j_shape + m_shape + l_shape)

    if convention == "direct":
        # direct: V_{i j i' j'}
        q_idx = j_idx.reshape((1,) * d + j_shape + (1,) * (2 * d))
        r_idx = ip_idx.reshape(i_shape + (1,) * d + m_shape + (1,) * d)
    else:
        # exchange: V_{i i' j j'}
        q_idx = ip_idx.reshape(i_shape + (1,) * d + m_shape + (1,) * d)
        r_idx = j_idx.reshape((1,) * d + j_shape + (1,) * (2 * d))

    # Distances and deltas for the mapped indices.
    r_pr = r[p_idx, r_idx]
    r_qs = r[q_idx, s_idx]
    center_pr = 0.5 * (coords[p_idx] + coords[r_idx])
    center_qs = 0.5 * (coords[q_idx] + coords[s_idx])
    r_center = _distance_from_diffs(center_pr - center_qs, metric=metric).astype(dtype, copy=False)
    delta_pr = (p_idx == r_idx).astype(dtype, copy=False)
    delta_qs = (q_idx == s_idx).astype(dtype, copy=False)

    exponent = -(3.0 - delta_pr - delta_qs)
    prefac = np.exp(-float(a) * r_pr) * np.exp(-float(b) * r_qs)
    base = float(c) * r_center
    with np.errstate(divide="ignore", invalid="ignore"):
        coulomb_like = np.power(base, exponent)
    zval = float(2.0 * c if zero_distance_value is None else zero_distance_value)
    coulomb_like = np.where(base == 0.0, zval, coulomb_like)
    coulomb_like = np.where(np.isfinite(coulomb_like), coulomb_like, 0.0)
    out = (prefac * coulomb_like).astype(dtype, copy=False)

    # Same hard cutoffs, mapped to compact oracle indices.
    mask = np.ones_like(out, dtype=bool)
    mask &= dmax[p_idx, r_idx] <= float(r_c)    # max_dist(p, r)
    mask &= dmax[p_idx, q_idx] <= float(r_loc)  # max_dist(p, q)
    mask &= dmax[r_idx, s_idx] <= float(r_loc)  # max_dist(r, s)
    return np.where(mask, out, 0.0)


def _generate_f_tensor_row_oracle_convention(
    shape: tuple[int, ...],
    a: float,
    metric: Metric,
    r_loc: int | None,
    periodic_cutoff: bool,
    dtype: type[np.floating],
) -> np.ndarray:
    spec = LatticeSpec(shape=shape)
    l = _require_uniform_l(shape)
    d = len(shape)
    r = pairwise_lattice_distances(spec, metric=metric).astype(dtype, copy=False)
    dmax = pairwise_lattice_max_distances(spec, periodic=periodic_cutoff).astype(dtype, copy=False)

    if r_loc is None:
        r_loc = lattice_max_dist(spec, periodic=periodic_cutoff)

    i_shape = (l,) * d
    m_shape = (2 * int(r_loc) + 1,) * d

    i_coord = np.indices(i_shape, dtype=int)  # (d, *i_shape)
    m_coord = np.indices(m_shape, dtype=int)  # (d, *m_shape)

    # q_k = p_k - r_loc + m_k (mod l)
    i_b = i_coord.reshape((d,) + i_shape + (1,) * d)
    m_b = m_coord.reshape((d,) + (1,) * d + m_shape)
    q_coord = (i_b - int(r_loc) + m_b) % l  # (d, *i_shape, *m_shape)

    p_idx = _coords_to_flat_index(i_coord, l=l).reshape(i_shape + (1,) * d)
    q_idx = _coords_to_flat_index(q_coord, l=l).reshape(i_shape + m_shape)

    out = np.exp(-float(a) * r[p_idx, q_idx]).astype(dtype, copy=False)
    mask = dmax[p_idx, q_idx] <= float(r_loc)  # max_dist(p, q)
    return np.where(mask, out, 0.0)
