from chem import (
    PySCFExcitonDataBuilder,
    build_mos2_molecule,
    compute_static_screened_coulomb_lmo,
)


def main() -> None:
    mol = build_mos2_molecule(
        mo_s_bond_angstrom=2.41,
        basis="def2-svp",
        charge=0,
        spin=0,
        symmetry=True,
        verbose=4,
    )
    builder = PySCFExcitonDataBuilder()
    data = builder.build(mol=mol, method="RHF", localization="boys")

    w_static = compute_static_screened_coulomb_lmo(
        eri_lmo=data.eri_lmo,
        epsilon_r=6.0,
        orbital_centers=data.orbital_centers,
        kappa=0.2,
    )
    print("n_lmo =", data.lmo_coeff.shape[1])
    print("hcore_lmo shape =", data.hcore_lmo.shape)
    print("eri_lmo shape =", data.eri_lmo.shape)
    print("W(0,0,0,0) =", w_static[0, 0, 0, 0])


if __name__ == "__main__":
    main()
