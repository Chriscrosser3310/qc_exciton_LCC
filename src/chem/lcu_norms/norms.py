"""LCU and block-encoding norms for THC/ISDF factorizations of the ov ERI block.

Five scalars are defined here. Three are *LCU* norms: they weight every entry
``|W^Q_IJ|`` by per-``(I, J)`` row-norm factors built from the collocation matrices.
Two are *embed* norms: a product of five extremal scalars. The two families are
different constructions, not variants of one another, and none of the five is a
validated lambda for a specific circuit -- in particular the ``4 / n_k`` prefactor
appearing in ``bse_thc.py:apply_V_thc`` is deliberately not folded in.

Notation, with ``I, J`` indexing THC interpolation points and ``p, q`` orbitals::

    A^Q_I(l_inf) = max_k  ||X^{o,k}_{I,:}||_2 ||X^{v,k-Q}_{I,:}||_2
    A^Q_I(l_2)   = sqrt( sum_k ( ||X^{o,k}_{I,:}||_2 ||X^{v,k-Q}_{I,:}||_2 )^2 )
    B^Q_J(...)   = same with o <-> v swapped
    S_Q(...)     = sum_{I,J} |W^Q_IJ| A^Q_I B^Q_J

    alpha_lcu_dir      = (1 / n_k) sum_Q S_Q(l_inf)
    alpha_lcu_exch     = (1 / n_k) sum_Q S_Q(l_2)
    alpha_lcu_exch_max = (1 / n_k) max_Q S_Q(l_2)

    P                  = (max_k ||X^{o,k}||_op)^2 (max_k ||X^{v,k}||_op)^2
    alpha_embed_op     = P * max_Q ||W^Q||_op
    alpha_embed_fro    = P * max_Q ||W^Q||_F

The ``2 -> inf`` norm of a matrix is its largest row 2-norm,
``||A||_{2->inf} = max_I ||A_{I,:}||_2``, maximized here over ``k`` as well. Since
``||A||_{2->inf} <= ||A||_op``, substituting it shrinks any embed norm::

    xv_2inf            = max_{k,I} ||X^{v,k}_{I,:}||_2
    xo_2inf            = max_{k,I} ||X^{o,k}_{I,:}||_2
    alpha_embed_op_mod = (max_k ||X^{o,k}||_op)^2 * xv_2inf^2 * max_Q ||W^Q||_op
    alpha_embed_op_mod_o= xo_2inf^2 * (max_k ||X^{v,k}||_op)^2 * max_Q ||W^Q||_op
    alpha_embed_op_2inf= xo_2inf^2 * xv_2inf^2 * max_Q ||W^Q||_op
    alpha_embed_l1_2inf= xo_2inf^2 * xv_2inf^2 * (1 / n_k) sum_{Q,I,J} |W^Q_IJ|

``2 -> inf`` is the *natural* norm for bounding the LCU brackets, because those are
built from row 2-norms: ``A^Q_I <= xo_2inf * xv_2inf`` holds directly, whereas going
through the operator norm overshoots. Consequently ``alpha_embed_l1_2inf`` is a
rigorous upper bound on ``alpha_lcu_dir`` roughly two orders of magnitude tighter
than ``alpha_embed_l1`` (measured median tightness 2.4e-2 against 2.7e-4).

The ``l_2`` form is often written with an explicit double orbital sum,
``sqrt( sum_{k,p,q} |X^{o,k}_{Ip} X^{v,k-Q}_{Iq}|^2 )``. That is the same quantity:
the orbital sums factorize into ``||X^{o,k}_{I,:}||_2^2 ||X^{v,k-Q}_{I,:}||_2^2``.
``sum_{k,p,q}`` is therefore never formed explicitly.

Exact relations, all of which :func:`check_relations` asserts:

- ``alpha_lcu_dir <= alpha_lcu_exch <= n_k * alpha_lcu_dir`` (from
  ``||v||_inf <= ||v||_2 <= sqrt(n_k) ||v||_inf``, applied to two brackets).
- ``alpha_lcu_exch / n_k <= alpha_lcu_exch_max <= alpha_lcu_exch`` (a maximum is at
  least the mean, and never exceeds the sum). Retaining the ``1 / n_k`` makes
  ``alpha_lcu_exch_max`` the *smallest* of the three LCU norms.
- ``alpha_lcu_dir <= P * (1 / n_k) sum_{Q,I,J} |W^Q_IJ|``, because a row 2-norm is
  bounded by the operator norm. This is a rigorous upper bound on the direct norm
  and is exposed as ``alpha_embed_l1``; it is very loose (2-4 orders of magnitude)
  but exact, so it makes a cheap unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class THCNorms:
    """The five norms plus the intermediates needed to interpret them."""

    n_aux: int          # M, number of interpolation points
    n_k: int
    n_occ: int
    n_vir: int
    xo_op: float        # max_k ||X^{o,k}||_op
    xv_op: float        # max_k ||X^{v,k}||_op
    xo_2inf: float      # max_{k,I} ||X^{o,k}_{I,:}||_2
    xv_2inf: float      # max_{k,I} ||X^{v,k}_{I,:}||_2
    w_op_max: float     # max_Q ||W^Q||_op
    w_fro_max: float    # max_Q ||W^Q||_F
    w_l1_mean: float    # (1 / n_k) sum_{Q,I,J} |W^Q_IJ|
    alpha_lcu_dir: float
    alpha_lcu_exch: float
    alpha_lcu_exch_max: float
    alpha_embed_op: float
    alpha_embed_fro: float
    alpha_embed_l1: float
    alpha_embed_op_mod: float    # X^v operator norm -> 2->inf
    alpha_embed_op_mod_o: float  # X^o operator norm -> 2->inf
    alpha_embed_op_2inf: float   # both -> 2->inf
    alpha_embed_l1_2inf: float   # both -> 2->inf, entrywise-1 W factor

    @property
    def prefactor(self) -> float:
        """``P``, the quartic collocation prefactor shared by the embed norms."""
        return self.xo_op**2 * self.xv_op**2

    @property
    def q_concentration(self) -> float:
        """``max_Q S_Q / mean_Q S_Q``. Equals 1 if the Q-weight is flat, n_k if it
        all sits on a single Q. Insensitive to the overall scale of W."""
        return self.alpha_lcu_exch_max * self.n_k / self.alpha_lcu_exch

    @property
    def k_participation(self) -> float:
        """``(alpha_lcu_exch / alpha_lcu_dir) / n_k``: the effective fraction of
        k-points carrying weight. Also insensitive to the scale of W."""
        return self.alpha_lcu_exch / self.alpha_lcu_dir / self.n_k

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["q_concentration"] = self.q_concentration
        d["k_participation"] = self.k_participation
        return d


def momentum_difference_table(kmesh: tuple[int, int, int]) -> np.ndarray:
    """``table[k, Q]`` is the flat index of ``k - Q``.

    The k index is the C-order index over the 3-D mesh, which is the convention
    ``cell.make_kpts`` produces and which ``utils.is_k_ordered`` asserts. This
    reproduces ``utils.add_k`` composed with ``utils.negative_k`` exactly.
    """
    extent = np.asarray(kmesh)
    n_k = int(np.prod(extent))
    triples = np.stack(np.unravel_index(np.arange(n_k), tuple(kmesh)), axis=1)
    table = np.zeros((n_k, n_k), dtype=int)
    for q in range(n_k):
        shifted = (triples - triples[q]) % extent
        table[:, q] = np.ravel_multi_index(tuple(shifted.T), tuple(kmesh))
    return table


def leading_singular_value(mat: np.ndarray, tol: float = 1e-13, max_iter: int = 5000) -> float:
    """Largest singular value by power iteration, with an exact-SVD fallback.

    Used because the W^Q of standard ISDF fits are strongly rank-1 dominated, so
    this converges in a handful of steps where a full SVD would dominate runtime.
    The fallback keeps the result exact when that assumption fails.
    """
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(mat.shape[1]) + 1j * rng.standard_normal(mat.shape[1])
    vec /= np.linalg.norm(vec)
    previous = 0.0
    for step in range(max_iter):
        left = mat @ vec
        left_norm = np.linalg.norm(left)
        if left_norm == 0.0:
            return 0.0
        vec = mat.conj().T @ (left / left_norm)
        current = float(np.linalg.norm(vec))
        if current == 0.0:
            return 0.0
        vec /= current
        if step > 2 and abs(current - previous) <= tol * max(current, 1e-300):
            return current
        previous = current
    return float(np.linalg.svd(mat, compute_uv=False)[0])


def compute_norms(
    x_occ: np.ndarray,
    x_vir: np.ndarray,
    coul: np.ndarray,
    kmesh: tuple[int, int, int],
) -> THCNorms:
    """Evaluate all five norms.

    Args:
        x_occ: ``X^o``, shape ``(n_k, M, n_occ)``. Build as ``inpv_kpt @ C_occ``.
        x_vir: ``X^v``, shape ``(n_k, M, n_vir)``. Build as ``inpv_kpt @ C_vir``.
        coul: ``W``, shape ``(n_k, M, M)``, i.e. ``coul_kpt`` indexed ``[Q, I, J]``.
        kmesh: the k-point mesh, e.g. ``(6, 6, 6)``.

    ``x_occ`` and ``x_vir`` must come from the *matching* mean field: symmetry-adapted
    checkpoints pair with ``DFT_<mesh>_symm.pkl`` and the rest with ``DFT_<mesh>.pkl``.
    Mixing them yields no error, only a silently different MO gauge.
    """
    n_k, n_aux, n_occ = x_occ.shape
    if x_vir.shape[:2] != (n_k, n_aux) or coul.shape != (n_k, n_aux, n_aux):
        raise ValueError(
            f"shape mismatch: x_occ={x_occ.shape} x_vir={x_vir.shape} coul={coul.shape}"
        )
    if int(np.prod(kmesh)) != n_k:
        raise ValueError(f"kmesh {kmesh} does not match n_k={n_k}")

    row_occ = np.linalg.norm(x_occ, axis=2)   # (n_k, M)
    row_vir = np.linalg.norm(x_vir, axis=2)
    xo_op = max(leading_singular_value(x_occ[k]) for k in range(n_k))
    xv_op = max(leading_singular_value(x_vir[k]) for k in range(n_k))

    minus = momentum_difference_table(kmesh)
    per_q_inf = np.empty(n_k)
    per_q_l2 = np.empty(n_k)
    w_op_max = 0.0
    w_fro_max = 0.0
    w_l1_total = 0.0
    for q in range(n_k):
        block = np.asarray(coul[q]).astype(np.complex128)
        magnitude = np.abs(block)
        prod_a = row_occ * row_vir[minus[:, q]]   # (n_k, M)
        prod_b = row_vir * row_occ[minus[:, q]]
        per_q_inf[q] = prod_a.max(axis=0) @ magnitude @ prod_b.max(axis=0)
        per_q_l2[q] = (
            np.sqrt((prod_a**2).sum(axis=0)) @ magnitude @ np.sqrt((prod_b**2).sum(axis=0))
        )
        w_op_max = max(w_op_max, leading_singular_value(block))
        w_fro_max = max(w_fro_max, float(np.linalg.norm(magnitude)))
        w_l1_total += float(magnitude.sum())

    prefactor = xo_op**2 * xv_op**2
    xo_2inf = float(row_occ.max())
    xv_2inf = float(row_vir.max())
    prefactor_mod = xo_op**2 * xv_2inf**2
    prefactor_mod_o = xo_2inf**2 * xv_op**2
    prefactor_2inf = xo_2inf**2 * xv_2inf**2
    w_l1_mean = w_l1_total / n_k
    return THCNorms(
        n_aux=int(n_aux),
        n_k=int(n_k),
        n_occ=int(n_occ),
        n_vir=int(x_vir.shape[2]),
        xo_op=float(xo_op),
        xv_op=float(xv_op),
        xo_2inf=xo_2inf,
        xv_2inf=xv_2inf,
        w_op_max=float(w_op_max),
        w_fro_max=float(w_fro_max),
        w_l1_mean=float(w_l1_mean),
        alpha_lcu_dir=float(per_q_inf.mean()),
        alpha_lcu_exch=float(per_q_l2.mean()),
        alpha_lcu_exch_max=float(per_q_l2.max() / n_k),
        alpha_embed_op=float(prefactor * w_op_max),
        alpha_embed_fro=float(prefactor * w_fro_max),
        alpha_embed_l1=float(prefactor * w_l1_mean),
        alpha_embed_op_mod=float(prefactor_mod * w_op_max),
        alpha_embed_op_mod_o=float(prefactor_mod_o * w_op_max),
        alpha_embed_op_2inf=float(prefactor_2inf * w_op_max),
        alpha_embed_l1_2inf=float(prefactor_2inf * w_l1_mean),
    )


def check_relations(norms: THCNorms, rtol: float = 1e-9) -> None:
    """Assert the exact inequalities among the norms. Raises AssertionError.

    Cheap enough to call after every evaluation; any violation is an implementation
    bug rather than a property of the data.
    """
    slack = 1.0 + rtol
    assert norms.alpha_lcu_dir <= norms.alpha_lcu_exch * slack, "dir <= exch violated"
    assert norms.alpha_lcu_exch <= norms.n_k * norms.alpha_lcu_dir * slack, (
        "exch <= n_k * dir violated"
    )
    assert norms.alpha_lcu_exch / norms.n_k <= norms.alpha_lcu_exch_max * slack, (
        "exch / n_k <= exch_max violated"
    )
    assert norms.alpha_lcu_exch_max <= norms.alpha_lcu_exch * slack, (
        "exch_max <= exch violated"
    )
    assert norms.alpha_lcu_dir <= norms.alpha_embed_l1 * slack, (
        "dir <= embed_l1 violated"
    )
    assert norms.w_op_max <= norms.w_fro_max * slack, "||W||_op <= ||W||_F violated"
    assert norms.xo_2inf <= norms.xo_op * slack, "||X^o||_{2->inf} <= ||X^o||_op violated"
    assert norms.xv_2inf <= norms.xv_op * slack, "||X^v||_{2->inf} <= ||X^v||_op violated"
    assert norms.alpha_embed_op_2inf <= norms.alpha_embed_op_mod * slack, (
        "embed_op_2inf <= embed_op_mod violated"
    )
    assert norms.alpha_embed_op_2inf <= norms.alpha_embed_op_mod_o * slack, (
        "embed_op_2inf <= embed_op_mod_o violated"
    )
    assert norms.alpha_embed_op_mod_o <= norms.alpha_embed_op * slack, (
        "embed_op_mod_o <= embed_op violated"
    )
    assert norms.alpha_embed_op_mod <= norms.alpha_embed_op * slack, (
        "embed_op_mod <= embed_op violated"
    )
    assert norms.alpha_lcu_dir <= norms.alpha_embed_l1_2inf * slack, (
        "dir <= embed_l1_2inf violated"
    )
    assert norms.alpha_embed_l1_2inf <= norms.alpha_embed_l1 * slack, (
        "embed_l1_2inf <= embed_l1 violated"
    )
