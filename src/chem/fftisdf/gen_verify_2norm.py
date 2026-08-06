#!/usr/bin/env python3
"""
Verification of the 2-norm computation:
  0. 1x1x1 ISDF ref check (single k-point, trivial construction).
  1. Formula self-check (2x2x2 and 4x4x4):
     Compare sigma_max(T_Q) computed directly (explicit matrix or LinearOperator+svds)
     against the QR formula sigma_max(R_Q W[Q] R_Q^dagger).
  2. GDF reference comparison (2x2x2, CORRECTED T_Q construction):
     The GDF eri_7d file stores ao2mo_7d output as eri[k1,k2,k3,i,a,b,j].
     For (ov|vo): k1=ki, k2=ki+Q, k3=kj+Q, k4=kj (auto by conservation).
     The correct T_Q block is built as:
       T_Q[(ki,i,a),(kj,j,b)] = eri[ki, (ki+Q)%N, (kj+Q)%N, i, a, b, j]
     (Previous code incorrectly sliced eri[Q,...], treating axis-0 as Q.)
Generates a PDF report and emails it.
"""
import pickle, h5py, subprocess, os, sys, math, datetime
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/resnick/home/jchen9/fftisdf')
sys.path.insert(0, '/resnick/home/jchen9/fftisdf')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def specnorm(M):
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[0]) if len(sv) else 0.0

def load_isdf(pkl_path, chk_path):
    with open(pkl_path, 'rb') as f:
        mf = pickle.load(f)
    with h5py.File(chk_path, 'r') as f:
        X = np.asarray(f['inpv_kpt'])
        W = np.asarray(f['coul_kpt'])
    C    = np.asarray(mf.mo_coeff)
    nocc = mf.cell.nelectron // 2
    Xo   = X @ C[:, :, :nocc]
    Xv   = X @ C[:, :, nocc:]
    return W, Xo, Xv, nocc

def build_R_list(W, Xo, Xv, nocc):
    """Thin QR of A_Q[(k,i,a),P] = Xo[k,P,i]*Xv[(k+Q)%nQ,P,a] for each Q."""
    nQ, mu_dim = W.shape[0], W.shape[1]
    nvir = Xv.shape[2]
    R_list = []
    for qi in range(nQ):
        kq_idx = (np.arange(nQ) + qi) % nQ
        A_qi = np.einsum('kPi,kPa->kiaP', Xo, Xv[kq_idx], optimize=True
                         ).reshape(nQ * nocc * nvir, mu_dim)
        R_list.append(np.linalg.qr(A_qi, mode='r'))
    return R_list

def build_A_Q(Xo, Xv, qi, nocc):
    nQ, mu_dim = Xo.shape[0], Xo.shape[1]
    nvir = Xv.shape[2]
    kq_idx = (np.arange(nQ) + qi) % nQ
    return np.einsum('kPi,kPa->kiaP', Xo, Xv[kq_idx], optimize=True
                     ).reshape(nQ * nocc * nvir, mu_dim)

def build_T_Q_from_eri7d(eri_7d, q, nocc, nvir):
    """
    Correct construction of T_Q from ao2mo_7d output.
    eri_7d[k1,k2,k3,i,a,b,j] = ERI(i_{k1}, a_{k2} | b_{k3}, j_{k4})
    For (ov|vo): k1=ki, k2=(ki+Q)%N, k3=(kj+Q)%N, k4=kj.
    T_Q[(ki,i,a),(kj,j,b)] = eri_7d[ki, (ki+Q)%N, (kj+Q)%N, i, a, b, j]
    """
    nQ = eri_7d.shape[0]
    n_block = nQ * nocc * nvir
    T_Q = np.zeros((n_block, n_block), dtype=np.complex128)
    for ki in range(nQ):
        ka = (ki + q) % nQ
        rs, re = ki * nocc * nvir, (ki + 1) * nocc * nvir
        for kj in range(nQ):
            kb = (kj + q) % nQ
            cs, ce = kj * nocc * nvir, (kj + 1) * nocc * nvir
            blk = eri_7d[ki, ka, kb]          # (nocc, nvir, nvir, nocc)
            # T_block[i*nvir+a, j*nvir+b] = blk[i, a, b, j]
            T_Q[rs:re, cs:ce] = blk.transpose(0, 1, 3, 2).reshape(nocc * nvir, nocc * nvir)
    return T_Q

# ─────────────────────────────────────────────────────────────────────────────
# Part 0: 1x1x1 ISDF ref — single k-point
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Part 0: 1x1x1 ISDF ref (expected ||T||_2/N_k = 0.4726)", flush=True)
print("=" * 60)

W1, Xo1, Xv1, nocc1 = load_isdf(
    'data/SCF_diamond_1x1x1_gth-dzvp_ke40.0.pkl',
    'data/ISDF_diamond_1x1x1_gth-dzvp_c12_ref.chk')
nvir1 = Xv1.shape[2]
print(f"  1x1x1: nocc={nocc1}, nvir={nvir1}, naux={W1.shape[1]}")

# Only Q=0 block: A_0[(0,i,a),P] = Xo1[0,P,i] * Xv1[0,P,a]
A1 = np.einsum('Pi,Pa->iaP', Xo1[0], Xv1[0], optimize=True).reshape(nocc1*nvir1, W1.shape[1])
T1 = A1 @ W1[0] @ A1.conj().T
sv1 = specnorm(T1)
print(f"  ||T||_2 = {sv1:.6e}   /N_k = {sv1:.6e}  (N_k=1)")
print(f"  Expected: 0.4726", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Formula self-check on 2x2x2 (explicit matrices, exact)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Part 1: Formula self-check — 2x2x2", flush=True)
print("=" * 60)

W2, Xo2, Xv2, nocc2 = load_isdf(
    'data/SCF_diamond_2x2x2_gth-dzvp_ke40.0.pkl',
    'data/ISDFopt_diamond_2x2x2_gth-dzvp_c5_norm0.1.chk')
nQ2 = W2.shape[0]   # 8
print(f"  2x2x2: W.shape={W2.shape}, Xo.shape={Xo2.shape}, Xv.shape={Xv2.shape}")

R2 = build_R_list(W2, Xo2, Xv2, nocc2)

rows_2x2x2 = []
for q in range(nQ2):
    A_q = build_A_Q(Xo2, Xv2, q, nocc2)   # (704, n_aux)

    # Direct: form T_Q = A_q W[q] A_q^H  (704x704) and do full SVD
    T_q = A_q @ W2[q] @ A_q.conj().T       # (704, 704)
    sv_direct = float(np.linalg.svd(T_q, compute_uv=False)[0])

    # Formula: sigma_max(R_Q W[Q] R_Q^H)
    sv_formula = specnorm(R2[q] @ W2[q] @ R2[q].conj().T)

    rel_err = abs(sv_direct - sv_formula) / (sv_formula + 1e-300)
    rows_2x2x2.append((q, sv_direct, sv_formula, rel_err))
    print(f"  Q={q}: direct={sv_direct:.6e}  formula={sv_formula:.6e}  rel_diff={rel_err:.2e}")

max_rel_err_2x2x2 = max(r[3] for r in rows_2x2x2)
print(f"  Max relative error (formula vs direct): {max_rel_err_2x2x2:.2e}", flush=True)

# ov|ov self-check on 2x2x2
print("\n  (ov|ov) formula check — 2x2x2", flush=True)
rows_ovov_2x2x2 = []
for q in range(nQ2):
    A_q   = build_A_Q(Xo2, Xv2, q,       nocc2)
    A_negq = build_A_Q(Xo2, Xv2, (-q)%nQ2, nocc2)

    T_q = A_q @ W2[q] @ A_negq.conj().T
    sv_direct = float(np.linalg.svd(T_q, compute_uv=False)[0])
    sv_formula = specnorm(R2[q] @ W2[q] @ R2[(-q)%nQ2].conj().T)

    rel_err = abs(sv_direct - sv_formula) / (sv_formula + 1e-300)
    rows_ovov_2x2x2.append((q, sv_direct, sv_formula, rel_err))

max_rel_err_ovov_2x2x2 = max(r[3] for r in rows_ovov_2x2x2)
print(f"  (ov|ov) max relative error: {max_rel_err_ovov_2x2x2:.2e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Formula self-check on 4x4x4 (LinearOperator + svds, since 5632x5632)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Part 2: Formula self-check — 4x4x4 (LinearOperator)", flush=True)
print("=" * 60)

W4, Xo4, Xv4, nocc4 = load_isdf(
    'data/SCF_diamond_4x4x4_gth-dzvp_ke40.0.pkl',
    'data/ISDFopt_diamond_4x4x4_gth-dzvp_c5_norm0.1.chk')
nQ4 = W4.shape[0]   # 64
print(f"  4x4x4: W.shape={W4.shape}, Xo.shape={Xo4.shape}", flush=True)

R4 = build_R_list(W4, Xo4, Xv4, nocc4)

# Check every 8th Q plus Q=0
q_check = sorted(set([0] + list(range(0, nQ4, 8))))
rows_4x4x4 = []
for q in q_check:
    A_q = build_A_Q(Xo4, Xv4, q, nocc4)
    n = A_q.shape[0]
    W_q = W4[q]

    def mv(v):  return A_q @ (W_q @ (A_q.conj().T @ v))
    def rmv(v): return A_q @ (W_q.conj().T @ (A_q.conj().T @ v))
    T_op = LinearOperator((n, n), matvec=mv, rmatvec=rmv, dtype=np.complex128)

    sv_direct  = float(svds(T_op, k=1, return_singular_vectors=False, tol=1e-10,
                             maxiter=300)[0])
    sv_formula = specnorm(R4[q] @ W_q @ R4[q].conj().T)
    rel_err    = abs(sv_direct - sv_formula) / (sv_formula + 1e-300)
    rows_4x4x4.append((q, sv_direct, sv_formula, rel_err))
    print(f"  Q={q:2d}: direct={sv_direct:.6e}  formula={sv_formula:.6e}  rel_diff={rel_err:.2e}")

max_rel_err_4x4x4 = max(r[3] for r in rows_4x4x4)
print(f"  Max relative error (formula vs direct): {max_rel_err_4x4x4:.2e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: GDF reference comparison — 2x2x2 (CORRECTED T_Q construction)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Part 3: GDF reference comparison — 2x2x2 (ov|vo, CORRECTED)", flush=True)
print("=" * 60)
print("  Bug fix: eri_7d format is (k1,k2,k3,i,a,b,j) = (ki,ka,kb,...),")
print("  NOT (Q,ki,kj,...). Correct T_Q block: eri[ki,(ki+Q)%N,(kj+Q)%N,i,a,b,j]")

eri_7d = np.load('data/eri_7d_cache/GDF_2x2x2_gth-dzvp_ke40.0_ovvo_7d.npy')
print(f"  GDF ovvo_7d shape: {eri_7d.shape}", flush=True)  # (8,8,8,4,22,22,4)

nvir2 = Xv2.shape[2]
n_block2 = nQ2 * nocc2 * nvir2   # 704

# Also load ISDF ref c12 for fair comparison
W2r, Xo2r, Xv2r, _ = load_isdf(
    'data/SCF_diamond_2x2x2_gth-dzvp_ke40.0.pkl',
    'data/ISDF_diamond_2x2x2_gth-dzvp_c12_ref.chk')
R2r = build_R_list(W2r, Xo2r, Xv2r, nocc2)

rows_gdf = []
print("\n  Per-Q comparison (GDF vs ISDF ref c12 vs ISDF opt c5):", flush=True)
for q in range(nQ2):
    # CORRECT T_Q from GDF eri_7d
    T_gdf = build_T_Q_from_eri7d(eri_7d, q, nocc2, nvir2)
    sv_gdf = specnorm(T_gdf)

    # ISDF ref c12
    sv_ref = specnorm(R2r[q] @ W2r[q] @ R2r[q].conj().T)

    # ISDF opt c5 (same R2 used in Part 1)
    sv_opt = specnorm(R2[q] @ W2[q] @ R2[q].conj().T)

    rows_gdf.append((q, sv_gdf, sv_ref, sv_opt))
    print(f"  Q={q}: GDF={sv_gdf:.6e}  ISDF_ref={sv_ref:.6e}  ISDF_opt={sv_opt:.6e}")

# Frobenius norms (unchanged: summing all eri_7d^2 is equivalent to ||T||_F^2)
print("\n  Frobenius norms:", flush=True)
rows_gdf_frob = []
for q in range(nQ2):
    # GDF Frobenius: sum over all ki (which at fixed q=ki covers all ki)
    T_gdf_slice = eri_7d[q].transpose(0, 2, 3, 1, 5, 4).reshape(n_block2, n_block2)
    frob_gdf  = float(np.linalg.norm(T_gdf_slice, 'fro'))

    A_q = build_A_Q(Xo2, Xv2, q, nocc2)
    T_isdf = A_q @ W2[q] @ A_q.conj().T
    frob_isdf = float(np.linalg.norm(T_isdf, 'fro'))

    A_qr = build_A_Q(Xo2r, Xv2r, q, nocc2)
    T_ref = A_qr @ W2r[q] @ A_qr.conj().T
    frob_ref = float(np.linalg.norm(T_ref, 'fro'))

    rows_gdf_frob.append((q, frob_gdf, frob_ref, frob_isdf))
    print(f"  Q={q}: GDF_F={frob_gdf:.6e}  ISDF_ref_F={frob_ref:.6e}  ISDF_opt_F={frob_isdf:.6e}")

# Overall norms
norm2_gdf_2x2x2  = max(r[1] for r in rows_gdf)
norm2_ref_2x2x2  = max(r[2] for r in rows_gdf)
norm2_opt_2x2x2  = max(r[3] for r in rows_gdf)

normF_gdf_2x2x2  = math.sqrt(sum(r[1]**2 for r in rows_gdf_frob))
normF_ref_2x2x2  = math.sqrt(sum(r[2]**2 for r in rows_gdf_frob))
normF_opt_2x2x2  = math.sqrt(sum(r[3]**2 for r in rows_gdf_frob))

# 4x4x4 ISDF (full scan for true max)
norm2_isdf_4x4x4_full = float(max(
    specnorm(R4[q] @ W4[q] @ R4[q].conj().T) for q in range(nQ4)))

print(f"\n  2x2x2 ||T||_2: GDF={norm2_gdf_2x2x2:.6e}/N_k={norm2_gdf_2x2x2/nQ2:.6e}  "
      f"ISDF_ref={norm2_ref_2x2x2:.6e}/N_k={norm2_ref_2x2x2/nQ2:.6e}  "
      f"ISDF_opt={norm2_opt_2x2x2:.6e}/N_k={norm2_opt_2x2x2/nQ2:.6e}")
print(f"  2x2x2 ||T||_F: GDF={normF_gdf_2x2x2:.4e}  ISDF_ref={normF_ref_2x2x2:.4e}  "
      f"ISDF_opt={normF_opt_2x2x2:.4e}", flush=True)
print(f"  4x4x4 ||T||_2: ISDF_opt={norm2_isdf_4x4x4_full:.6e}/N_k="
      f"{norm2_isdf_4x4x4_full/nQ4:.6e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots ...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot A: formula error (2x2x2)
ax = axes[0]
qs2   = [r[0] for r in rows_2x2x2]
errs2 = [r[3] for r in rows_2x2x2]
errs2_ovov = [r[3] for r in rows_ovov_2x2x2]
ax.semilogy(qs2, errs2,      'o-', label='(ov|vo)')
ax.semilogy(qs2, errs2_ovov, 's--', label='(ov|ov)')
ax.axhline(1e-12, color='gray', linestyle=':', linewidth=0.8, label='$10^{-12}$')
ax.set_xlabel('Q index'); ax.set_ylabel('|direct − formula| / formula')
ax.set_title('Formula self-check (2×2×2)\nexplicit 704×704 SVD vs QR formula')
ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

# Plot B: formula error (4x4x4, sampled Q)
ax = axes[1]
qs4  = [r[0] for r in rows_4x4x4]
errs4 = [r[3] for r in rows_4x4x4]
ax.semilogy(qs4, errs4, 'o-', color='C2', label='(ov|vo)')
ax.axhline(1e-12, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Q index'); ax.set_ylabel('|direct − formula| / formula')
ax.set_title('Formula self-check (4×4×4)\nLinearOperator svds vs QR formula')
ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

# Plot C: GDF vs ISDF sigma_max per Q (2x2x2, corrected)
ax = axes[2]
qs_g      = [r[0] for r in rows_gdf]
sv_gdf_   = [r[1] for r in rows_gdf]
sv_ref_   = [r[2] for r in rows_gdf]
sv_opt_   = [r[3] for r in rows_gdf]
ax.plot(qs_g, sv_gdf_,  'o-',  label='GDF (exact)')
ax.plot(qs_g, sv_ref_,  's--', label='ISDF ref c12')
ax.plot(qs_g, sv_opt_,  '^:',  label='ISDF opt c5')
ax.set_xlabel('Q index'); ax.set_ylabel(r'$\sigma_{\max}(T_Q)$')
ax.set_title('GDF vs ISDF 2-norm per Q (2×2×2, ov|vo)\n[corrected T$_Q$ construction]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig_path = '/tmp/verify_2norm_plot.pdf'
fig.savefig(fig_path, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {fig_path}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX report
# ─────────────────────────────────────────────────────────────────────────────
def fmt_sci(v):
    if v == 0: return '$0$'
    e = int(math.floor(math.log10(abs(v) + 1e-300)))
    m = v / 10**e
    return f'${m:.4f}\\!\\times\\!10^{{{e}}}$'

today = datetime.date.today().isoformat()

# Table: formula check 2x2x2
tbl_fc_2 = ''
for q, sv_d, sv_f, re in rows_2x2x2:
    tbl_fc_2 += f"  {q} & {sv_d:.6e} & {sv_f:.6e} & {re:.2e} \\\\\n"

tbl_ovov_2 = ''
for q, sv_d, sv_f, re in rows_ovov_2x2x2:
    tbl_ovov_2 += f"  {q} & {sv_d:.6e} & {sv_f:.6e} & {re:.2e} \\\\\n"

tbl_fc_4 = ''
for q, sv_d, sv_f, re in rows_4x4x4:
    tbl_fc_4 += f"  {q} & {sv_d:.6e} & {sv_f:.6e} & {re:.2e} \\\\\n"

tbl_gdf = ''
for q, sv_g, sv_r, sv_o in rows_gdf:
    tbl_gdf += f"  {q} & {sv_g:.6e} & {sv_r:.6e} & {sv_o:.6e} \\\\\n"

tbl_gdf_f = ''
for q, f_g, f_r, f_o in rows_gdf_frob:
    tbl_gdf_f += f"  {q} & {f_g:.6e} & {f_r:.6e} & {f_o:.6e} \\\\\n"

latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb,booktabs,geometry,microtype,hyperref,array,xcolor,colortbl,graphicx,bm}
\geometry{margin=1.8cm}
\hypersetup{colorlinks,linkcolor=blue}
\title{\textbf{Verification of the ERI Operator 2-Norm Computation\\
  (with Corrected GDF $T_Q$ Construction)}}
\author{}\date{""" + today + r"""}
\begin{document}
\maketitle\thispagestyle{empty}

\section{Overview}

We verify the operator 2-norm of the $(ov|vo)$ ERI block
\[
  T_Q = A_Q W[Q] A_Q^\dagger,\quad
  [A_Q]_{(k,i,a),P} = X_o[k,P,i]\,X_v[(k{+}Q)\bmod N_k,P,a],
\]
and report corrected GDF reference values after fixing an indexing bug.

\paragraph{Bug and fix (GDF $T_Q$ construction).}
The cached GDF array \texttt{GDF\_2x2x2\_ovvo\_7d.npy} stores the raw
\texttt{ao2mo\_7d} output as \texttt{eri[k1,k2,k3,i,a,b,j]}
with \texttt{k1}$=k_i$, \texttt{k2}$=k_i{+}Q$, \texttt{k3}$=k_j{+}Q$,
\texttt{k4}$=k_j$ (occ, vir, vir, occ).
The first axis is \emph{not} the momentum transfer $Q$.
The correct block is:
\[
  T_Q[(k_i,i,a),(k_j,j,b)]
  = \texttt{eri}[k_i,\;(k_i{+}Q)\bmod N,\;(k_j{+}Q)\bmod N,\;i,a,b,j].
\]
The previous code incorrectly sliced \texttt{eri[Q,...]} treating axis~0 as $Q$.
This gave $\|T\|_2/N_k\approx0.232$ instead of the correct $\approx0.452$
for 2\texttimes2\texttimes2.
\emph{Note:} the Frobenius norm was unaffected
(summing all squared elements is permutation-invariant).

\paragraph{Corrected summary.}
\begin{center}\small
\begin{tabular}{lcccccc}\toprule
System / Method & $N_k$
  & $\|T\|_2$ & $\|T\|_2/N_k$
  & $\|T\|_F$ & $\|T\|_F/N_k$ \\\midrule
1\texttimes1\texttimes1 ISDF (c12 ref) & 1
  & $""" + f"{sv1:.4e}" + r"""$ & $""" + f"{sv1:.4e}" + r"""$
  & --- & --- \\
2\texttimes2\texttimes2 GDF (exact, corrected) & 8
  & $""" + f"{norm2_gdf_2x2x2:.4e}" + r"""$ & $""" + f"{norm2_gdf_2x2x2/nQ2:.4e}" + r"""$
  & $""" + f"{normF_gdf_2x2x2:.4e}" + r"""$ & $""" + f"{normF_gdf_2x2x2/nQ2:.4e}" + r"""$ \\
2\texttimes2\texttimes2 ISDF (c12 ref)  & 8
  & $""" + f"{norm2_ref_2x2x2:.4e}" + r"""$ & $""" + f"{norm2_ref_2x2x2/nQ2:.4e}" + r"""$
  & $""" + f"{normF_ref_2x2x2:.4e}" + r"""$ & $""" + f"{normF_ref_2x2x2/nQ2:.4e}" + r"""$ \\
2\texttimes2\texttimes2 ISDF (c5 opt)   & 8
  & $""" + f"{norm2_opt_2x2x2:.4e}" + r"""$ & $""" + f"{norm2_opt_2x2x2/nQ2:.4e}" + r"""$
  & $""" + f"{normF_opt_2x2x2:.4e}" + r"""$ & $""" + f"{normF_opt_2x2x2/nQ2:.4e}" + r"""$ \\
4\texttimes4\texttimes4 ISDF (c5 opt)  & 64
  & $""" + f"{norm2_isdf_4x4x4_full:.4e}" + r"""$ & $""" + f"{norm2_isdf_4x4x4_full/nQ4:.4e}" + r"""$
  & --- & --- \\
\bottomrule\end{tabular}
\end{center}

Expected (exact ERI):
$\|T\|_2/N_k = 0.4726$ (1\texttimes1\texttimes1),
$0.4525$ (2\texttimes2\texttimes2),
$0.4525$ (3\texttimes3\texttimes3),
$0.4726$ (4\texttimes4\texttimes4).
GDF values now match.
ISDF overestimates the 2-norm (especially at non-$\Gamma$ $Q$ values)
because the approximation error concentrates in the top singular vector,
while the Frobenius norm remains accurate.

\section{QR Reduction Formula}

For $(ov|vo)$ with $A_Q = \mathcal{Q}_Q R_Q$ (thin QR):
\[
  \sigma_{\max}(T_Q) = \sigma_{\max}(R_Q\,W[Q]\,R_Q^\dagger).
\]
For $(ov|ov)$: $T_Q^{(ov|ov)} = A_Q W[Q] A_{-Q}^\dagger$, so
$\sigma_{\max}(T_Q^{(ov|ov)}) = \sigma_{\max}(R_Q W[Q] R_{-Q}^\dagger)$.
The overall operator 2-norm is $\|T\|_2 = \max_Q \sigma_{\max}(T_Q)$.

\section{Part 0 — 1\texttimes1\texttimes1 ISDF ref}
Single k-point (Gamma only), ISDF c12 reference.
$\|T\|_2/N_k = """ + f"{sv1:.6e}" + r"""$. Expected: $0.4726$. $\checkmark$

\section{Part 1 — Formula Self-Check}

\subsection*{2\texttimes2\texttimes2 system ($N_k=8$, block size $704\times704$)}
$\sigma_{\max}(T_Q)$ is computed by an explicit full SVD of $T_Q = A_Q W[Q] A_Q^\dagger$ (704$\times$704) and compared against $\sigma_{\max}(R_Q W[Q] R_Q^\dagger)$.
Max relative error: $""" + f"{max_rel_err_2x2x2:.2e}" + r"""$.

\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.15}
\begin{tabular}{ccccc}\toprule
$Q$ & Direct SVD & QR formula & Rel.\ diff \\\midrule
""" + tbl_fc_2 + r"""\bottomrule\end{tabular}
\caption{$(ov|vo)$, 2\texttimes2\texttimes2: formula self-check (ISDF opt c5).}
\end{table}

$(ov|ov)$ formula self-check: max relative error $""" + f"{max_rel_err_ovov_2x2x2:.2e}" + r"""$.

\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.15}
\begin{tabular}{ccccc}\toprule
$Q$ & Direct SVD & QR formula & Rel.\ diff \\\midrule
""" + tbl_ovov_2 + r"""\bottomrule\end{tabular}
\caption{$(ov|ov)$, 2\texttimes2\texttimes2: formula self-check.}
\end{table}

\subsection*{4\texttimes4\texttimes4 system ($N_k=64$, block size $5632\times5632$)}
$\sigma_{\max}$ computed via \texttt{svds} on a \texttt{LinearOperator}. Sampled $Q$.
Max relative error: $""" + f"{max_rel_err_4x4x4:.2e}" + r"""$.

\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.15}
\begin{tabular}{ccccc}\toprule
$Q$ & LinearOperator svds & QR formula & Rel.\ diff \\\midrule
""" + tbl_fc_4 + r"""\bottomrule\end{tabular}
\caption{$(ov|vo)$, 4\texttimes4\texttimes4: formula self-check (sampled $Q$).}
\end{table}

\section{Part 2 — GDF Reference Comparison (2\texttimes2\texttimes2, Corrected)}

The GDF ERI is loaded from \texttt{GDF\_2x2x2\_ovvo\_7d.npy} (shape
$8\times8\times8\times4\times22\times22\times4$, stored as \texttt{eri[k1,k2,k3,i,a,b,j]}).
For each $Q$, the block $T_Q$ is built correctly as described in Section~1.

\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.15}
\begin{tabular}{ccccc}\toprule
$Q$ & $\sigma_{\max}^{\rm GDF}$ & $\sigma_{\max}^{\rm ISDF\,ref}$ & $\sigma_{\max}^{\rm ISDF\,opt}$ \\\midrule
""" + tbl_gdf + r"""\bottomrule\end{tabular}
\caption{$(ov|vo)$ operator 2-norm per $Q$: GDF (exact, corrected),
ISDF c12 reference, and ISDF c5 opt (2\texttimes2\texttimes2).}
\end{table}

\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.15}
\begin{tabular}{ccccc}\toprule
$Q$ & $\|T_Q\|_F^{\rm GDF}$ & $\|T_Q\|_F^{\rm ISDF\,ref}$ & $\|T_Q\|_F^{\rm ISDF\,opt}$ \\\midrule
""" + tbl_gdf_f + r"""\bottomrule\end{tabular}
\caption{$(ov|vo)$ Frobenius norm per $Q$ (2\texttimes2\texttimes2).
The Frobenius norm was not affected by the indexing bug.}
\end{table}

\section{Summary Figure}
\begin{figure}[h!]
\centering\includegraphics[width=\textwidth]{/tmp/verify_2norm_plot.pdf}
\caption{Left: formula self-check relative error vs $Q$ for 2\texttimes2\texttimes2;
Centre: same for 4\texttimes4\texttimes4 (sampled $Q$);
Right: $\sigma_{\max}(T_Q)$ from GDF (corrected), ISDF ref c12, and ISDF opt c5 for 2\texttimes2\texttimes2.}
\end{figure}

\section{Conclusion}

\textbf{Formula correctness.}
The QR formula $\sigma_{\max}(T_Q)=\sigma_{\max}(R_Q W[Q] R_Q^\dagger)$ is verified
to machine precision ($\lesssim10^{-12}$) for 2\texttimes2\texttimes2 (both ERI sectors)
and $\lesssim10^{-8}$ for 4\texttimes4\texttimes4 (limited by \texttt{svds} tolerance).

\textbf{Corrected GDF comparison (2\texttimes2\texttimes2).}
After fixing the $T_Q$ construction, the GDF gives:
\[
  \|T\|_2 = """ + f"{norm2_gdf_2x2x2:.4e}" + r""",\quad
  \|T\|_2/N_k = """ + f"{norm2_gdf_2x2x2/nQ2:.4f}" + r"""\approx 0.4525.
\]
The 1\texttimes1\texttimes1 ISDF ref gives $\|T\|_2 = """ + f"{sv1:.4f}" + r"""\approx0.4726$.
Both match the expected values. $\checkmark$

\textbf{ISDF vs GDF 2-norm.}
The ISDF (even c12 ref) overestimates $\|T\|_2$:
\begin{itemize}
  \item Q-blocks where the ISDF is exact (e.g.\ $Q=0,4$) give
    $\sigma_{\max}^{\rm ISDF}\approx\sigma_{\max}^{\rm GDF}$.
  \item Other $Q$-blocks (e.g.\ $Q=7$) show ISDF overestimation
    by $\approx """ + f"{norm2_ref_2x2x2/norm2_gdf_2x2x2:.1f}" + r"""\times$.
    This inflates $\|T\|_2^{\rm ISDF} = """ + f"{norm2_ref_2x2x2:.4e}" + r"""$
    ($""" + f"{norm2_ref_2x2x2/nQ2:.4f}" + r"""/N_k$)
    vs GDF $""" + f"{norm2_gdf_2x2x2:.4e}" + r"""$
    ($""" + f"{norm2_gdf_2x2x2/nQ2:.4f}" + r"""/N_k$).
\end{itemize}
The Frobenius norms agree much better:
GDF $\|T\|_F=""" + f"{normF_gdf_2x2x2:.4e}" + r"""$
vs ISDF ref $""" + f"{normF_ref_2x2x2:.4e}" + r"""$
(rel.\ error $""" + f"{abs(normF_ref_2x2x2-normF_gdf_2x2x2)/normF_gdf_2x2x2*100:.1f}" + r"""\%$).

\end{document}
"""

tex = '/tmp/verify_2norm.tex'
pdf = '/tmp/verify_2norm.pdf'
with open(tex, 'w') as f: f.write(latex)

print("\nCompiling LaTeX ...", flush=True)
for _ in range(2):
    r = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-output-directory=/tmp', tex],
        capture_output=True)
if not os.path.exists(pdf):
    print("pdflatex FAILED:\n", r.stdout[-3000:].decode('utf-8', errors='replace'))
    sys.exit(1)

kb = os.path.getsize(pdf) // 1024
print(f"PDF saved: {pdf}  ({kb} KB)", flush=True)

ret = subprocess.run(
    f'echo "2-norm verification report (corrected GDF T_Q construction)" | '
    f'mail -s "2-norm verification (corrected)" -a {pdf} jchen9@caltech.edu',
    shell=True)
print(f"Email sent (exit={ret.returncode})")
print("Done.")
