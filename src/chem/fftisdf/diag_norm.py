#!/usr/bin/env python3
"""
Diagnostic: figure out why GDF ||T||_2/N_k differs from expected 0.4525 (2x2x2).

Tests:
1. Correct T_Q construction from eri_7d (ki,ka,kb format, not Q,ki,kj)
2. ISDF vs GDF comparison for 1x1x1 (single k-point, simpler to check)
3. Whether Xo conjugation matters in A_Q
"""
import pickle, h5py, os, sys
import numpy as np

os.chdir('/resnick/home/jchen9/fftisdf')
sys.path.insert(0, '/resnick/home/jchen9/fftisdf')

def specnorm(M):
    return float(np.linalg.svd(M, compute_uv=False)[0])

# ─────────────────────────────────────────────────────────────────────────────
# Part A: Check the eri_7d format & correct T_Q construction for 2x2x2
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Part A: eri_7d format check & correct T_Q (2x2x2)")
print("=" * 60)

eri_7d = np.load('data/eri_7d_cache/GDF_2x2x2_gth-dzvp_ke40.0_ovvo_7d.npy')
print(f"  eri_7d.shape = {eri_7d.shape}")  # expected (8,8,8,4,22,22,4)

# The raw ao2mo_7d output for C_ovvo = [Cocc,Cvir,Cvir,Cocc] is:
#   eri_7d[k1, k2, k3, i, a, b, j] = ERI(i_{k1}, a_{k2} | b_{k3}, j_{k4})
# where k4 = kconserv3[k1,k2,k3].
# For ovvo: k1=ki(occ), k2=ka(vir)=ki+Q, k3=kb(vir)=kj+Q, k4=kj(occ)
# So format is (ki, ka, kb, i, a, b, j).

# Load SCF for kpt info
with open('data/SCF_diamond_2x2x2_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf2 = pickle.load(f)
cell2 = mf2.cell
nQ2 = 8
nocc2 = cell2.nelectron // 2    # 4
C2 = np.asarray(mf2.mo_coeff)
nvir2 = C2.shape[2] - nocc2     # 22
n_block2 = nQ2 * nocc2 * nvir2  # 704

# Build kconserv2: given ki, Q → ka = (ki+Q)%N
# For our cubic mesh, kconserv2[ki, ka] = Q such that ki + Q = ka (mod N)
# We just use modular arithmetic: ka = (ki + Q) % nQ2, kb = (kj + Q) % nQ2

print(f"  nQ2={nQ2}, nocc2={nocc2}, nvir2={nvir2}, n_block2={n_block2}")

# Current (buggy) method: eri_7d[q, ki, kj, i, a, b, j] treated as Q-indexed
# Correct method: eri_7d[ki, ka, kb, i, a, b, j] with ka=(ki+Q)%N, kb=(kj+Q)%N

sv_buggy = []
sv_correct = []

for q in range(nQ2):
    # --- BUGGY construction (what gen_verify_2norm.py currently does) ---
    # Treats eri_7d[q] as if q is the Q index
    # Actually gets: eri_7d[ki=q, ka, kb, i, a, b, j]
    T_buggy = eri_7d[q].transpose(0, 2, 3, 1, 5, 4).reshape(n_block2, n_block2)
    sv_buggy.append(specnorm(T_buggy))

    # --- CORRECT construction: build T_Q for fixed Q=q ---
    T_correct = np.zeros((n_block2, n_block2), dtype=np.complex128)
    for ki in range(nQ2):
        ka = (ki + q) % nQ2
        rs = ki * nocc2 * nvir2
        re = (ki + 1) * nocc2 * nvir2
        for kj in range(nQ2):
            kb = (kj + q) % nQ2
            cs = kj * nocc2 * nvir2
            ce = (kj + 1) * nocc2 * nvir2
            # eri_7d[ki, ka, kb, i, a, b, j] shape (nocc2, nvir2, nvir2, nocc2)
            blk = eri_7d[ki, ka, kb]  # (nocc2, nvir2, nvir2, nocc2)
            # T[(ki,i,a),(kj,j,b)] = blk[i,a,b,j]
            # Reorder to (i,a,j,b) then reshape
            T_correct[rs:re, cs:ce] = blk.transpose(0, 1, 3, 2).reshape(nocc2 * nvir2, nocc2 * nvir2)
    sv_correct.append(specnorm(T_correct))

norm2_buggy   = max(sv_buggy)
norm2_correct = max(sv_correct)
print(f"\n  BUGGY   ||T||_2 = {norm2_buggy:.6e}  /N_k = {norm2_buggy/nQ2:.6e}")
print(f"  CORRECT ||T||_2 = {norm2_correct:.6e}  /N_k = {norm2_correct/nQ2:.6e}")
print(f"  Expected /N_k ≈ 0.4525")
print()
for q in range(nQ2):
    print(f"  Q={q}: buggy σ_max={sv_buggy[q]:.6e}  correct σ_max={sv_correct[q]:.6e}")

# ─────────────────────────────────────────────────────────────────────────────
# Part B: 1x1x1 — single k-point, direct verification
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Part B: 1x1x1 ISDF ref — compute ||T||_2 (expected 0.4726)")
print("=" * 60)

with open('data/SCF_diamond_1x1x1_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf1 = pickle.load(f)
cell1 = mf1.cell
nocc1 = cell1.nelectron // 2
C1 = np.asarray(mf1.mo_coeff)  # (1, nao, nmo)
nvir1 = C1.shape[2] - nocc1

with h5py.File('data/ISDF_diamond_1x1x1_gth-dzvp_c12_ref.chk', 'r') as f:
    X1 = np.asarray(f['inpv_kpt'])   # (1, naux, nao)
    W1 = np.asarray(f['coul_kpt'])   # (1, naux, naux)

Xo1 = X1 @ C1[:, :, :nocc1]   # (1, naux, nocc1)
Xv1 = X1 @ C1[:, :, nocc1:]   # (1, naux, nvir1)

print(f"  1x1x1: nocc={nocc1}, nvir={nvir1}, naux={X1.shape[1]}")
print(f"  W1.shape={W1.shape}, Xo1.shape={Xo1.shape}")

# Only one Q block (Q=0, only k=0)
# A_0[(0,i,a), P] = Xo1[0,P,i] * Xv1[0,P,a]
# T_0 = A_0 W[0] A_0†
# Method 1: without conjugating Xo
A1_noconj = np.einsum('Pi,Pa->iaP', Xo1[0], Xv1[0], optimize=True).reshape(nocc1*nvir1, X1.shape[1])
T1_noconj = A1_noconj @ W1[0] @ A1_noconj.conj().T
sv1_noconj = specnorm(T1_noconj)

# Method 2: with conjugating Xo (physical ERI)
A1_conj = np.einsum('Pi,Pa->iaP', Xo1[0].conj(), Xv1[0], optimize=True).reshape(nocc1*nvir1, X1.shape[1])
T1_conj = A1_conj @ W1[0] @ A1_conj.conj().T
sv1_conj = specnorm(T1_conj)

print(f"  σ_max(T), A with Xo (no conj): {sv1_noconj:.6e}  (= ||T||_2/N_k since N_k=1)")
print(f"  σ_max(T), A with Xo* (conj):   {sv1_conj:.6e}")
print(f"  Expected: 0.4726")

# ─────────────────────────────────────────────────────────────────────────────
# Part C: 2x2x2 ISDF ref — both with and without Xo conjugation
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Part C: 2x2x2 ISDF ref — Xo conj vs no-conj, compare with correct GDF")
print("=" * 60)

with h5py.File('data/ISDF_diamond_2x2x2_gth-dzvp_c12_ref.chk', 'r') as f:
    X2r = np.asarray(f['inpv_kpt'])   # (8, naux, nao)
    W2r = np.asarray(f['coul_kpt'])   # (8, naux, naux)

Xo2r = X2r @ C2[:, :, :nocc2]   # (8, naux, nocc2)
Xv2r = X2r @ C2[:, :, nocc2:]   # (8, naux, nvir2)

sv_isdf_noconj = []
sv_isdf_conj = []

for q in range(nQ2):
    kq_idx = (np.arange(nQ2) + q) % nQ2
    # No conjugation of Xo
    A_nc = np.einsum('kPi,kPa->kiaP', Xo2r, Xv2r[kq_idx], optimize=True
                     ).reshape(n_block2, X2r.shape[1])
    sv_isdf_noconj.append(specnorm(A_nc @ W2r[q] @ A_nc.conj().T))

    # With conjugation of Xo
    A_c = np.einsum('kPi,kPa->kiaP', Xo2r.conj(), Xv2r[kq_idx], optimize=True
                    ).reshape(n_block2, X2r.shape[1])
    sv_isdf_conj.append(specnorm(A_c @ W2r[q] @ A_c.conj().T))

norm2_isdf_noconj = max(sv_isdf_noconj)
norm2_isdf_conj   = max(sv_isdf_conj)
print(f"  ISDF (no conj) ||T||_2={norm2_isdf_noconj:.6e}  /N_k={norm2_isdf_noconj/nQ2:.6e}")
print(f"  ISDF (conj)    ||T||_2={norm2_isdf_conj:.6e}  /N_k={norm2_isdf_conj/nQ2:.6e}")
print(f"  GDF correct    ||T||_2={norm2_correct:.6e}  /N_k={norm2_correct/nQ2:.6e}")
print(f"  Expected /N_k ≈ 0.4525")
print()
for q in range(nQ2):
    print(f"  Q={q}: GDF={sv_correct[q]:.6e}  ISDF_nc={sv_isdf_noconj[q]:.6e}  ISDF_c={sv_isdf_conj[q]:.6e}")

# ─────────────────────────────────────────────────────────────────────────────
# Part D: 3x3x3 GDF — correct T_Q construction, compare with expected 0.4525
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Part D: 3x3x3 GDF — correct T_Q (expected /N_k ≈ 0.4525)")
print("=" * 60)

eri_3 = np.load('data/eri_7d_cache/GDF_3x3x3_gth-dzvp_ke40.0_ovvo_7d.npy')
print(f"  eri_3.shape = {eri_3.shape}")

with open('data/SCF_diamond_3x3x3_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf3 = pickle.load(f)
cell3 = mf3.cell
nQ3 = 27
nocc3 = cell3.nelectron // 2
C3 = np.asarray(mf3.mo_coeff)
nvir3 = C3.shape[2] - nocc3
n_block3 = nQ3 * nocc3 * nvir3
print(f"  nQ3={nQ3}, nocc3={nocc3}, nvir3={nvir3}, n_block3={n_block3}")

sv_correct3 = []
for q in range(nQ3):
    T_q3 = np.zeros((n_block3, n_block3), dtype=np.complex128)
    for ki in range(nQ3):
        ka = (ki + q) % nQ3
        rs = ki * nocc3 * nvir3
        re = (ki + 1) * nocc3 * nvir3
        for kj in range(nQ3):
            kb = (kj + q) % nQ3
            cs = kj * nocc3 * nvir3
            ce = (kj + 1) * nocc3 * nvir3
            blk = eri_3[ki, ka, kb]  # (nocc3, nvir3, nvir3, nocc3)
            T_q3[rs:re, cs:ce] = blk.transpose(0, 1, 3, 2).reshape(nocc3*nvir3, nocc3*nvir3)
    sv_correct3.append(specnorm(T_q3))
    print(f"  Q={q:2d}: σ_max={sv_correct3[-1]:.6e}", flush=True)

norm2_correct3 = max(sv_correct3)
print(f"\n  3x3x3 GDF CORRECT ||T||_2={norm2_correct3:.6e}  /N_k={norm2_correct3/nQ3:.6e}")
print(f"  Expected /N_k ≈ 0.4525")

print("\nDone.")
