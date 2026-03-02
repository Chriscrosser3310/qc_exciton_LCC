from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Metric = Literal["euclidean", "manhattan", "chebyshev"]


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
    shape: tuple[int, ...],
    a: float = 1.0,
    metric: Metric = "chebyshev",
    r_cut: int | None = None,
    periodic_cutoff: bool = False,
    dtype: type[np.floating] = np.float64,
) -> np.ndarray:
    r"""Generate F_{pq} = exp(-a R_{pq}) on a D-dimensional lattice."""
    spec = LatticeSpec(shape=shape)
    r_pq = pairwise_lattice_distances(spec, metric=metric)
    f = np.exp(-float(a) * r_pq).astype(dtype, copy=False)
    if r_cut is None:
        r_cut = lattice_max_dist(spec, periodic=periodic_cutoff)
    dmax = pairwise_lattice_max_distances(spec, periodic=periodic_cutoff)
    f = np.where(dmax <= float(r_cut), f, 0.0)
    return f


def generate_v_tensor(
    shape: tuple[int, ...],
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    metric: Metric = "chebyshev",
    r_loc: int | None = None,
    r_c: int | None = None,
    periodic_cutoff: bool = False,
    dtype: type[np.floating] = np.float64,
) -> np.ndarray:
    r"""Generate V_{pqrs} with:

    V_{pqrs} =
      exp(-a R_{pr}) * exp(-b R_{qs}) * (c R_{pq})^(-(3 - delta_{pr} - delta_{qs}))

    where delta is a Kronecker delta over site indices and R_{xy} is lattice distance.

    Optional hard cutoffs (independent of tolerance thresholding):
    - if r_c is set: zero out when max_dist(p, r) > r_c
    - if r_loc is set: zero out when max_dist(p, q) > r_loc or max_dist(r, s) > r_loc
    """
    spec = LatticeSpec(shape=shape)
    n = spec.n_sites
    r = pairwise_lattice_distances(spec, metric=metric).astype(dtype, copy=False)

    r_pr = r[:, None, :, None]
    r_qs = r[None, :, None, :]
    r_pq = r[:, :, None, None]

    eye = np.eye(n, dtype=dtype)
    delta_pr = eye[:, None, :, None]
    delta_qs = eye[None, :, None, :]

    exponent = -(3.0 - delta_pr - delta_qs)
    prefac = np.exp(-float(a) * r_pr) * np.exp(-float(b) * r_qs)
    base = float(c) * r_pq
    with np.errstate(divide="ignore", invalid="ignore"):
        coulomb_like = np.power(base, exponent)
    # Remove singular self-distance terms from this benchmark tensor model.
    coulomb_like = np.where(base == 0.0, 0.0, coulomb_like)
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
