"""Load THC/ISDF checkpoints and rebuild the collocation matrices.

``X^o`` and ``X^v`` are not stored in the checkpoints. Only ``X^{ao}`` is
(as ``inpv_kpt``), so they must be rebuilt by contracting with the mean-field MO
coefficients::

    X^{o,k} = X^{ao,k} C^{occ,k},    X^{v,k} = X^{ao,k} C^{vir,k}

matching ``generate_isdf_gdf.py:158`` and ``generate_isdf_gdf_symm.py:271``. All
generator families verified to write AO-basis ``X`` (``generate_isdf_gdf.py:241``,
``optimize_X_ov.py:286``).

The pairing between checkpoint and mean field is load-bearing: symmetry-adapted
checkpoints were produced against ``DFT_<mesh>_symm.pkl`` and the others against
``DFT_<mesh>.pkl`` (``generate_isdf_gdf_symm.py:180,182``). Mixing them raises no
error and produces no obviously wrong number -- the two mean fields differ by
unitary mixing inside degenerate bands, so the result is a valid tensor in the
wrong MO gauge. :func:`dft_checkpoint_for` reproduces the rule so it cannot be
got wrong by hand.
"""

from __future__ import annotations

import os
import pickle
import re

import numpy as np


def dft_checkpoint_for(chkfile: str) -> str:
    """Return the DFT pickle that a given ISDF checkpoint was produced against."""
    name = os.path.basename(chkfile)
    match = re.search(r"_(\d+x\d+x\d+)_", name)
    if match is None:
        raise ValueError(f"cannot parse a k-mesh out of {name!r}")
    suffix = "_symm" if "_symm" in name else ""
    return os.path.join(os.path.dirname(chkfile), f"DFT_{match.group(1)}{suffix}.pkl")


def kmesh_of(chkfile: str) -> tuple[int, int, int]:
    """Parse the k-mesh from an ISDF checkpoint filename."""
    match = re.search(r"_(\d+)x(\d+)x(\d+)_", os.path.basename(chkfile))
    if match is None:
        raise ValueError(f"cannot parse a k-mesh out of {chkfile!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def load_thc_factors(
    chkfile: str, dft_pkl: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Return ``(x_occ, x_vir, coul, kmesh)`` for an ISDF checkpoint.

    ``dft_pkl`` defaults to :func:`dft_checkpoint_for`. The k-point count and AO
    count from the mean field are asserted against the checkpoint's own array
    shapes, so a mismatched pair fails loudly instead of silently.
    """
    import h5py

    if dft_pkl is None:
        dft_pkl = dft_checkpoint_for(chkfile)
    with open(dft_pkl, "rb") as handle:
        mean_field = pickle.load(handle)
    cell = mean_field.cell
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_k, n_ao, _ = mo_coeff.shape
    n_occ = cell.nelectron // 2

    with h5py.File(chkfile, "r") as handle:
        x_ao = np.asarray(handle["inpv_kpt"])
        coul = np.asarray(handle["coul_kpt"])
    if x_ao.shape[0] != n_k or coul.shape[0] != n_k:
        raise ValueError(
            f"k-point mismatch: mean field has {n_k}, checkpoint has "
            f"{x_ao.shape[0]} / {coul.shape[0]} -- wrong DFT pickle?"
        )
    if x_ao.shape[2] != n_ao:
        raise ValueError(
            f"AO mismatch: mean field has {n_ao}, checkpoint has {x_ao.shape[2]}"
        )

    x_occ = x_ao @ mo_coeff[:, :, :n_occ]
    x_vir = x_ao @ mo_coeff[:, :, n_occ:]
    return x_occ, x_vir, coul, kmesh_of(chkfile)
