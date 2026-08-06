import pickle
import sys

import h5py
import numpy as np

basis = "gth-dzvp"
ke_cutoff = 40.0
nk = int(sys.argv[1])
kmesh = np.array([nk, nk, nk])
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
cisdf = sys.argv[2]
scf_pkl = f"data/SCF_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"
norm_ratio = sys.argv[3]
isdf_chk = f"data/ISDFopt_diamond_{klabel}_{basis}_c{cisdf}_norm{norm_ratio}.chk"
isdf_chk_ref = f"data/ISDF_diamond_{klabel}_{basis}_c12_ref.chk"
if len(sys.argv) > 4:
    isdf_chk = sys.argv[4]
#isdf_chk = f"data/ISDF_diamond_{klabel}_{basis}_c{cisdf}_ref.chk"

with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
kpts = cell.make_kpts(kmesh)

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
nvir = nmo - nocc
Cocc = np.array(C[:, :, :nocc], order="C", copy=True)
Cvir = np.array(C[:, :, nocc:], order="C", copy=True)

with h5py.File(isdf_chk, "r") as f:
    X = np.asarray(f["inpv_kpt"])
    W = np.asarray(f["coul_kpt"])

is_ao_tensor = X.shape[2] == nao
if is_ao_tensor:
    Xo = X @ Cocc
    Xv = X @ Cvir

    Xo_2norm = np.linalg.svdvals(Xo).max()
    Xv_2norm = np.linalg.svdvals(Xv).max()
    W_2norm_by_q = np.linalg.svdvals(W)
    W_2norm = W_2norm_by_q.max()
    Xo_frob = np.linalg.norm(Xo, axis=(1, 2)).max()
    Xv_frob = np.linalg.norm(Xv, axis=(1, 2)).max()
    W_frob = np.linalg.norm(W, axis=(1, 2)).max()
    Xo_max = np.abs(Xo).max()
    Xv_max = np.abs(Xv).max()
    total_2norm = Xo_2norm**2 * Xv_2norm**2 * W_2norm
    total_frob = Xo_frob**2 * Xv_frob**2 * W_frob
    total_max = Xo_2norm**2 * Xv_2norm**2 * np.abs(W).max()
else:
    X_2norm = np.linalg.svdvals(X).max()
    W_2norm_by_q = np.linalg.svdvals(W)
    W_2norm = W_2norm_by_q.max()
    X_frob = np.linalg.norm(X, axis=(1, 2)).max()
    W_frob = np.linalg.norm(W, axis=(1, 2)).max()
    total_2norm = X_2norm**2 * W_2norm
    total_frob = X_frob**2 * W_frob
    total_max = X_2norm**2 * np.abs(W).max()

W_max = np.abs(W).max()

e = np.array(mf.mo_energy)
eocc = e[:, :nocc]
evir = e[:, nocc:]

rel_error = None
if is_ao_tensor:
    import fft
    import utils
    isdf = fft.ISDF(cell, kpts)
    isdf._isdf = isdf_chk
    isdf.build()
    isdf_ref = fft.ISDF(cell, kpts)
    isdf_ref._isdf = isdf_chk_ref
    isdf_ref.build()
    rel_error = utils.compare_two_isdf(isdf_ref, isdf, Cocc, Cvir, kmesh)


print("isdf_chk =", isdf_chk)
print("scf_pkl =", scf_pkl)
print("nkpts =", nkpts)
print("nao =", nao)
print("nocc =", nocc)
print("nvir =", nvir)
print("X.shape =", X.shape)
print("W.shape =", W.shape)
if is_ao_tensor:
    print("Xo_2norm =", Xo_2norm)
    print("Xv_2norm =", Xv_2norm)
    print("Xo_frob =", Xo_frob)
    print("Xv_frob =", Xv_frob)
else:
    print("X_2norm =", X_2norm)
    print("X_frob =", X_frob)
print("W_2norm =", W_2norm)
print("W_frob =", W_frob)
print("W_max =", W_max)
print("total_2norm =", total_2norm)
print("total_frob =", total_frob)
print("total_max =", total_max)
print("eocc min", eocc.min())
print("eocc max", eocc.max())
print("evir min", evir.min())
print("evir max", evir.max())
print("eov min", evir.min() - eocc.max())
print("eov max", evir.max() - eocc.min())
if rel_error is not None:
    print(f"ovvo ||ERI - ERI_ref|| / ||ERI_ref|| = %16.8e" % rel_error, flush=True)
