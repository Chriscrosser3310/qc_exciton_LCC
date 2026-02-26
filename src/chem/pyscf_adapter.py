from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LMOData:
    """Container for localized orbital quantities required by exciton builders."""

    mo_coeff: np.ndarray
    lmo_coeff: np.ndarray
    fock_lmo: np.ndarray
    hcore_lmo: np.ndarray
    eri_lmo: np.ndarray
    orbital_centers: np.ndarray
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]


class PySCFExcitonDataBuilder:
    """Build molecule -> LMO-domain quantities with PySCF.

    Implemented capabilities:
    - run SCF in PySCF (RHF/UHF/RKS/UKS)
    - localize occupied and virtual spaces separately (Boys/Pipek-Mezey)
    - transform one-/two-electron quantities to LMO basis
    - compute a static screened Coulomb tensor from bare 2e integrals
    """

    def build(
        self,
        mol: Any,
        method: str = "RHF",
        xc: str = "PBE",
        localization: str = "boys",
        conv_tol: float = 1e-9,
    ) -> LMOData:
        mf = run_scf(mol=mol, method=method, xc=xc, conv_tol=conv_tol)
        lmo_coeff, occupied, virtual = localize_orbitals(
            mol=mol,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            scheme=localization,
        )
        hcore_lmo, fock_lmo = compute_one_electron_integrals_lmo(mf=mf, lmo_coeff=lmo_coeff)
        eri_lmo = compute_two_electron_integrals_lmo(mol=mol, lmo_coeff=lmo_coeff)
        orbital_centers = compute_orbital_centers(mol=mol, lmo_coeff=lmo_coeff)
        return LMOData(
            mo_coeff=np.asarray(mf.mo_coeff),
            lmo_coeff=lmo_coeff,
            fock_lmo=fock_lmo,
            hcore_lmo=hcore_lmo,
            eri_lmo=eri_lmo,
            orbital_centers=orbital_centers,
            occupied=occupied,
            virtual=virtual,
        )


def build_mos2_molecule(
    mo_s_bond_angstrom: float = 2.41,
    basis: str = "def2-svp",
    charge: int = 0,
    spin: int = 0,
    symmetry: bool = True,
    verbose: int = 0,
) -> Any:
    """Build a minimal molecular MoS2 geometry (triatomic cluster) in PySCF."""
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("PySCF is required for build_mos2_molecule(). Install with [chem].") from exc

    # Planar triatomic MoS2 reference geometry.
    half_angle = np.deg2rad(60.0)
    x = mo_s_bond_angstrom * np.cos(half_angle)
    y = mo_s_bond_angstrom * np.sin(half_angle)
    atom = f"Mo 0.0 0.0 0.0; S {x:.12f} {y:.12f} 0.0; S {x:.12f} {-y:.12f} 0.0"
    return gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=verbose,
    )


def run_scf(mol: Any, method: str = "RHF", xc: str = "PBE", conv_tol: float = 1e-9) -> Any:
    """Run a PySCF SCF driver and return the converged mean-field object."""
    method_u = method.upper()
    if method_u == "RHF":
        mf = mol.RHF()
    elif method_u == "UHF":
        mf = mol.UHF()
    elif method_u == "RKS":
        mf = mol.RKS()
        mf.xc = xc
    elif method_u == "UKS":
        mf = mol.UKS()
        mf.xc = xc
    else:
        raise ValueError(f"Unsupported SCF method: {method}. Use RHF/UHF/RKS/UKS.")

    mf.conv_tol = conv_tol
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge.")
    return mf


def localize_orbitals(
    mol: Any,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    scheme: str = "boys",
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Localize occupied and virtual spaces separately and return full LMO coefficients."""
    try:
        from pyscf import lo
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("PySCF is required for localize_orbitals(). Install with [chem].") from exc

    occ_idx = tuple(np.where(np.asarray(mo_occ) > 0)[0].tolist())
    vir_idx = tuple(np.where(np.asarray(mo_occ) == 0)[0].tolist())

    c_occ = np.asarray(mo_coeff)[:, occ_idx]
    c_vir = np.asarray(mo_coeff)[:, vir_idx]

    scheme_l = scheme.strip().lower()
    if scheme_l == "boys":
        c_occ_loc = lo.Boys(mol, c_occ).kernel()
        c_vir_loc = lo.Boys(mol, c_vir).kernel() if c_vir.shape[1] > 0 else c_vir
    elif scheme_l in ("pipek", "pipek-mezey", "pipek_mezey"):
        c_occ_loc = lo.PM(mol, c_occ).kernel()
        c_vir_loc = lo.PM(mol, c_vir).kernel() if c_vir.shape[1] > 0 else c_vir
    else:
        raise ValueError(f"Unsupported localization scheme: {scheme}. Use boys or pipek-mezey.")

    lmo_coeff = np.concatenate([c_occ_loc, c_vir_loc], axis=1)
    occupied = tuple(range(len(occ_idx)))
    virtual = tuple(range(len(occ_idx), len(occ_idx) + len(vir_idx)))
    return lmo_coeff, occupied, virtual


def compute_one_electron_integrals_lmo(mf: Any, lmo_coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute hcore and Fock matrices in the LMO basis."""
    c = np.asarray(lmo_coeff)
    h_ao = np.asarray(mf.get_hcore())
    f_ao = np.asarray(mf.get_fock())
    h_lmo = c.T @ h_ao @ c
    f_lmo = c.T @ f_ao @ c
    return h_lmo, f_lmo


def compute_two_electron_integrals_lmo(mol: Any, lmo_coeff: np.ndarray) -> np.ndarray:
    """Compute chemist-notation two-electron integrals (pq|rs) in LMO basis."""
    try:
        from pyscf import ao2mo
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "PySCF is required for compute_two_electron_integrals_lmo(). Install with [chem]."
        ) from exc

    c = np.asarray(lmo_coeff)
    n_orb = c.shape[1]
    eri = ao2mo.kernel(mol, c, compact=False)
    return np.asarray(eri).reshape(n_orb, n_orb, n_orb, n_orb)


def compute_orbital_centers(mol: Any, lmo_coeff: np.ndarray) -> np.ndarray:
    """Compute LMO charge-centroid approximations <phi_p|r|phi_p>."""
    c = np.asarray(lmo_coeff)
    n_orb = c.shape[1]
    r_ints = np.asarray(mol.intor_symmetric("int1e_r", comp=3))  # (x,y,z, nao, nao)
    centers = np.zeros((n_orb, 3), dtype=float)
    for p in range(n_orb):
        vec = c[:, p]
        for xyz in range(3):
            centers[p, xyz] = float(vec.T @ r_ints[xyz] @ vec)
    return centers


def compute_static_screened_coulomb_lmo(
    eri_lmo: np.ndarray,
    epsilon_r: float = 4.0,
    orbital_centers: np.ndarray | None = None,
    kappa: float | None = None,
) -> np.ndarray:
    """Compute static screened Coulomb interaction W_pqrs from bare (pq|rs).

    Base model:
    W_pqrs = (pq|rs) / epsilon_r

    Optional distance damping:
    W_pqrs *= exp(-kappa * (d_pr + d_qs) / 2)
    where d_ab = |R_a - R_b| using LMO centers.
    """
    eri = np.asarray(eri_lmo, dtype=float)
    if eri.ndim != 4:
        raise ValueError("eri_lmo must have shape (n_orb, n_orb, n_orb, n_orb).")
    if epsilon_r <= 0:
        raise ValueError("epsilon_r must be > 0.")

    screened = eri / float(epsilon_r)
    if kappa is None:
        return screened
    if orbital_centers is None:
        raise ValueError("orbital_centers must be provided when kappa is set.")

    centers = np.asarray(orbital_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("orbital_centers must have shape (n_orb, 3).")

    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    damping = np.exp(-float(kappa) * (dmat[:, None, :, None] + dmat[None, :, None, :]) / 2.0)
    return screened * damping
