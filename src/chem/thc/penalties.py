"""Pluggable norm penalties for THC collocation-matrix optimization.

Upstream (``optimize_X_ov.py:50``) hardcodes the operator norm for both collocation
factors and offers a single switch (``--use_Fnorm``) for the central tensor::

    norm_Xo = torch.linalg.svdvals(Xo).max()
    norm_Xv = torch.linalg.svdvals(Xv).max()
    norm_W  = (torch.linalg.norm(W, dim=(1, 2)).max() if use_Fnorm
               else torch.linalg.eigvalsh(W).abs().max()) / nkpts
    return norm_Xo**2 * norm_Xv**2 * norm_W

That product is ``alpha_embed / n_k``. This module generalizes it so the norm used
for each factor is selectable, while :func:`abs_norm_loss` with its default kinds
reproduces the upstream value bit-for-bit.

The motivating case is ``xv_kind="2inf"``. The LCU brackets are built from *row*
2-norms of the collocation matrices, and ``||A||_{2->inf} = max_I ||A_{I,:}||_2`` is
the tight bound on them, whereas the operator norm overshoots. Substituting it is
also cheaper -- a norm reduction instead of a batched SVD.

One caveat on the gradient. ``svdvals(X).max()`` has dense support: the leading
singular vectors touch every row, so every interpolation point receives gradient at
every step. ``norm(X, dim=2).max()`` is supported on the single maximizing row only
(measured: 22 of 183040 elements). The optimizer flattens that row, the maximum jumps
elsewhere, and so on. Both are non-smooth maxima, which is what upstream already
relies on, but the sparser signal may converge more slowly. ``"p<N>"`` provides a
smooth upper bound on the ``2->inf`` norm with support on all rows, weighted toward
the largest, which is the intended remedy if plain ``"2inf"`` stalls.

Note also that the gap between the two cannot be closed by optimization. With
``M > n_vir`` the rows are overcomplete, ``rank(X X^H) <= n_vir``, and
``lambda_max >= tr / n_vir`` forces ``||X||_op > ||X||_{2->inf}`` strictly.
Optimizing the ``2->inf`` norm targets the quantity of interest directly; it does not
make the two coincide.
"""

from __future__ import annotations

import torch

X_KINDS = ("op", "2inf", "fro")
W_KINDS = ("op", "fro", "absmax", "l1")


def x_norm(x: torch.Tensor, kind: str = "op") -> torch.Tensor:
    """Norm of a batched collocation matrix ``x`` of shape ``(n_k, M, n_orb)``.

    - ``"op"``: ``max_k ||X^k||_op``, the largest singular value. Upstream default.
    - ``"2inf"``: ``max_{k,I} ||X^k_{I,:}||_2``, the largest row 2-norm.
    - ``"fro"``: ``max_k ||X^k||_F``.
    - ``"p<N>"``: ``max_k ( sum_I ||X^k_{I,:}||_2^N )^(1/N)``, a smooth upper bound on
      ``"2inf"`` that tends to it as ``N -> inf``. Computed in a max-scaled form so
      large ``N`` does not overflow.
    """
    if kind == "op":
        return torch.linalg.svdvals(x).max()
    if kind == "2inf":
        return torch.linalg.norm(x, dim=2).max()
    if kind == "fro":
        return torch.linalg.norm(x, dim=(1, 2)).max()
    if kind.startswith("p"):
        power = float(kind[1:])
        if power <= 0.0:
            raise ValueError(f"p-norm exponent must be positive, got {kind!r}")
        rows = torch.linalg.norm(x, dim=2)                 # (n_k, M)
        scale = rows.max().detach().clamp_min(1e-300)
        return scale * ((rows / scale) ** power).sum(dim=1).max() ** (1.0 / power)
    raise ValueError(f"unknown x norm kind {kind!r}; expected one of {X_KINDS} or 'p<N>'")


def w_norm(w: torch.Tensor, n_k: int, kind: str = "op") -> torch.Tensor:
    """Norm of the central tensor ``w`` of shape ``(n_k, M, M)``, divided by ``n_k``.

    The ``1 / n_k`` matches upstream, so the product in :func:`abs_norm_loss` is
    ``alpha_embed / n_k`` rather than ``alpha_embed``.

    - ``"op"``: ``max_Q ||W^Q||_op`` via ``eigvalsh``, as upstream does. Assumes each
      ``W^Q`` is Hermitian, which the solver produces up to rounding.
    - ``"fro"``: ``max_Q ||W^Q||_F``. Upstream ``--use_Fnorm``.
    - ``"absmax"``: ``max_{Q,I,J} |W^Q_IJ|``. What ``optimize_X_full.py:65`` uses.
    - ``"l1"``: ``(1 / n_k) sum_{Q,I,J} |W^Q_IJ|``, the entrywise 1-norm averaged
      over ``Q``.
    """
    if kind == "op":
        return torch.linalg.eigvalsh(w).abs().max() / n_k
    if kind == "fro":
        return torch.linalg.norm(w, dim=(1, 2)).max() / n_k
    if kind == "absmax":
        return torch.abs(w).max() / n_k
    if kind == "l1":
        return torch.abs(w).sum() / (n_k * n_k)
    raise ValueError(f"unknown w norm kind {kind!r}; expected one of {W_KINDS}")


def abs_norm_loss(
    x_occ: torch.Tensor,
    x_vir: torch.Tensor,
    w: torch.Tensor,
    n_k: int,
    xo_kind: str = "op",
    xv_kind: str = "op",
    w_kind: str = "op",
) -> torch.Tensor:
    """``||X^o||^2 ||X^v||^2 ||W||``, i.e. ``alpha_embed / n_k``.

    With the default kinds this is numerically identical to upstream
    ``optimize_X_ov.get_abs_norm_loss``.
    """
    return (
        x_norm(x_occ, xo_kind) ** 2
        * x_norm(x_vir, xv_kind) ** 2
        * w_norm(w, n_k, w_kind)
    )


def norm_tag(xo_kind: str, xv_kind: str, w_kind: str) -> str:
    """Filename suffix identifying a non-default choice of norms.

    Empty for the upstream default, so default runs keep upstream's naming.
    """
    parts = []
    if xo_kind != "op":
        parts.append(f"xo{xo_kind}")
    if xv_kind != "op":
        parts.append(f"xv{xv_kind}")
    if w_kind != "op":
        parts.append(f"w{w_kind}")
    return ("_" + "_".join(parts)) if parts else ""
