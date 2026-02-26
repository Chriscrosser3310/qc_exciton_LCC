from __future__ import annotations

import numpy as np

from chem.pyscf_adapter import compute_static_screened_coulomb_lmo


def test_static_screening_scalar_epsilon():
    eri = np.zeros((2, 2, 2, 2), dtype=float)
    eri[0, 0, 0, 0] = 2.0
    w = compute_static_screened_coulomb_lmo(eri_lmo=eri, epsilon_r=4.0)
    assert np.isclose(w[0, 0, 0, 0], 0.5)


def test_static_screening_with_distance_damping():
    eri = np.ones((2, 2, 2, 2), dtype=float)
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    w = compute_static_screened_coulomb_lmo(
        eri_lmo=eri, epsilon_r=2.0, orbital_centers=centers, kappa=0.5
    )
    assert w.shape == (2, 2, 2, 2)
    assert w[0, 0, 1, 1] < w[0, 0, 0, 0]
