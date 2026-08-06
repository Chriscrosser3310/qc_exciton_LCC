"""
Systematic tensor compression experiments on W[Q, mu, nu].
Goal: find decompositions where total stored elements < original (Q*mu*nu).

Experiments
-----------
1. Per-Q truncated eigendecomposition  (W[q,:,:] is Hermitian)
2. Tucker / HOSVD  (compress each mode independently)
3. Direct 3-site TT on (Q, mu, nu) — no binary unfolding
"""
import pickle, h5py, socket, subprocess, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import quimb.tensor as qtn
from datetime import datetime

# ── Load ──────────────────────────────────────────────────────────────────────
with open('data/SCF_diamond_4x4x4_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf = pickle.load(f)
with h5py.File('data/ISDFopt_diamond_4x4x4_gth-dzvp_c5_norm0.1.chk', 'r') as f:
    W = np.asarray(f['coul_kpt'])

Q, mu_dim, nu_dim = W.shape
N_orig = W.size
print(f"W.shape={W.shape}  dtype={W.dtype}  N_orig={N_orig:,}", flush=True)
W_norm = np.linalg.norm(W)

EPSILONS = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]

results = []   # list of dicts: method, eps, compression_ratio, rel_error, note

# ─────────────────────────────────────────────────────────────────────────────
# Exp 1: Per-Q eigendecomposition  W[q,:,:] = U diag(lam) U†  (Hermitian per q)
# Storage: Q * (mu*r + r) = Q*r*(mu+1)  vs  Q*mu*nu
# ─────────────────────────────────────────────────────────────────────────────
print("\nExp 1: Per-Q eigendecomposition ...", flush=True)

# collect all eigenvalues (normalised) across q
all_eigs = []
for q in range(Q):
    lam = np.linalg.eigvalsh(W[q])   # sorted ascending; W[q] should be Hermitian
    all_eigs.append(np.abs(lam)[::-1])   # descending absolute eigenvalue
all_eigs = np.array(all_eigs)   # (Q, mu_dim)

# per-Q rank at each eps
for eps in EPSILONS:
    ranks = np.array([int(np.sum(all_eigs[q] / all_eigs[q, 0] > eps))
                      if all_eigs[q, 0] > 0 else 1 for q in range(Q)])
    r_max = ranks.max(); r_mean = ranks.mean()
    # storage: for each q, store U[:,r] (mu*r) and eigenvalues (r)
    N_comp = int(np.sum(ranks * (mu_dim + 1)))
    ratio  = N_comp / N_orig
    # reconstruction error
    W_rec = np.zeros_like(W)
    for q in range(Q):
        lam_full, U = np.linalg.eigh(W[q])
        # sort by |eigenvalue| descending
        idx = np.argsort(np.abs(lam_full))[::-1]
        lam_full = lam_full[idx]; U = U[:, idx]
        r = ranks[q]
        W_rec[q] = (U[:, :r] * lam_full[:r]) @ U[:, :r].conj().T
    err = np.linalg.norm(W - W_rec) / W_norm
    results.append(dict(method='Per-Q eig', eps=eps,
                        ratio=ratio, err=err,
                        note=f'r_mean={r_mean:.1f} r_max={r_max}'))
    print(f"  eps={eps:.0e}: r_mean={r_mean:.1f} r_max={r_max}  "
          f"N_comp={N_comp:,}  ratio={ratio:.3f}  err={err:.2e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Exp 2: Tucker / HOSVD  W ≈ G ×₁ U_Q ×₂ U_mu ×₃ U_nu
# Mode unfoldings:
#   mode-Q  : W.reshape(Q, mu*nu)            shape (64, 16900)   max rank 64
#   mode-mu : W.transpose(1,0,2).reshape(mu, Q*nu)  shape (130, 8320)  max rank 130
#   mode-nu : W.transpose(2,0,1).reshape(nu, Q*mu)  shape (130, 8320)  max rank 130
# Storage: r_Q*r_mu*r_nu + Q*r_Q + mu*r_mu + nu*r_nu
# ─────────────────────────────────────────────────────────────────────────────
print("\nExp 2: Tucker / HOSVD ...", flush=True)

# compute mode singular values once
sQ  = np.linalg.svd(W.reshape(Q,          -1), compute_uv=False)
smu = np.linalg.svd(W.transpose(1,0,2).reshape(mu_dim, -1), compute_uv=False)
snu = np.linalg.svd(W.transpose(2,0,1).reshape(nu_dim, -1), compute_uv=False)

print(f"  Mode-Q  singular values (first 10): {sQ[:10].round(2)}", flush=True)
print(f"  Mode-mu singular values (first 10): {smu[:10].round(2)}", flush=True)
print(f"  Mode-nu singular values (first 10): {snu[:10].round(2)}", flush=True)

def tucker_rank(s, eps):
    return max(1, int(np.sum(s / s[0] > eps)))

def tucker_approx(W, rQ, rmu, rnu):
    """HOSVD Tucker approximation using sequential mode-product (fast)."""
    UQ,  _, _ = np.linalg.svd(W.reshape(Q, -1),                      full_matrices=False)
    Umu, _, _ = np.linalg.svd(W.transpose(1,0,2).reshape(mu_dim,-1), full_matrices=False)
    Unu, _, _ = np.linalg.svd(W.transpose(2,0,1).reshape(nu_dim,-1), full_matrices=False)
    UQ  = UQ[:,  :rQ ]   # (Q,  rQ)  — real left singular vectors
    Umu = Umu[:, :rmu]   # (mu, rmu)
    Unu = Unu[:, :rnu]   # (nu, rnu)
    # Sequential mode products: G = W x_1 UQ† x_2 Umu† x_3 Unu†
    # Mode-1: project Q -> G1[rQ,mu,nu]
    G1 = (UQ.conj().T @ W.reshape(Q, -1)).reshape(rQ, mu_dim, nu_dim)
    # Mode-2: project mu: G2[rQ,rmu,nu]
    G2 = (Umu.conj().T @ G1.transpose(1,0,2).reshape(mu_dim,-1)).reshape(rmu, rQ, nu_dim).transpose(1,0,2)
    # Mode-3: project nu: G[rQ,rmu,rnu]
    G  = (Unu.conj().T @ G2.transpose(2,0,1).reshape(nu_dim,-1)).reshape(rnu, rQ, rmu).transpose(1,2,0)
    # Reconstruct: W_rec = G x_1 UQ x_2 Umu x_3 Unu
    R2 = (Umu @ G.transpose(1,0,2).reshape(rmu,-1)).reshape(mu_dim, rQ, rnu).transpose(1,0,2)
    R3 = (Unu @ R2.transpose(2,0,1).reshape(rnu,-1)).reshape(nu_dim, rQ, mu_dim).transpose(1,2,0)
    W_rec = (UQ @ R3.reshape(rQ,-1)).reshape(Q, mu_dim, nu_dim)
    N_comp = G.size + Q*rQ + mu_dim*rmu + nu_dim*rnu
    return W_rec, N_comp

for eps in EPSILONS:
    rQ  = tucker_rank(sQ,  eps)
    rmu = tucker_rank(smu, eps)
    rnu = tucker_rank(snu, eps)
    W_rec, N_comp = tucker_approx(W, rQ, rmu, rnu)
    err   = np.linalg.norm(W - W_rec) / W_norm
    ratio = N_comp / N_orig
    results.append(dict(method='Tucker/HOSVD', eps=eps,
                        ratio=ratio, err=err,
                        note=f'rQ={rQ} rmu={rmu} rnu={rnu}'))
    print(f"  eps={eps:.0e}: rQ={rQ} rmu={rmu} rnu={rnu}  "
          f"N_comp={N_comp:,}  ratio={ratio:.3f}  err={err:.2e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Exp 3: Direct 3-site TT on (Q, mu, nu)  — no binary unfolding
# Order Q-mu-nu:  A[Q,r1] -- B[r1,mu,r2] -- C[r2,nu]
# Max bonds: r1 = min(Q, mu*nu)=64   r2 = min(Q*mu, nu)=130
# Storage: Q*r1 + r1*mu*r2 + r2*nu
# ─────────────────────────────────────────────────────────────────────────────
print("\nExp 3: 3-site TT on (Q, mu, nu) ...", flush=True)
expTT = {}
for eps in EPSILONS:
    T = qtn.Tensor(W, inds=['Q', 'mu', 'nu'])
    TL, TR = T.split(['Q'],       cutoff=eps, cutoff_mode='rel', bond_ind='r1')
    TM, TR = TR.split(['r1','mu'], cutoff=eps, cutoff_mode='rel', bond_ind='r2')
    r1, r2 = TL.ind_size('r1'), TM.ind_size('r2')
    # reconstruct
    res = np.tensordot(TL.data, TM.data, axes=([-1],[0]))   # (Q, mu, r2)
    W_rec = np.tensordot(res,  TR.data, axes=([-1],[0]))    # (Q, mu, nu)
    err   = np.linalg.norm(W - W_rec) / W_norm
    N_comp = Q*r1 + r1*mu_dim*r2 + r2*nu_dim
    ratio  = N_comp / N_orig
    expTT[eps] = (r1, r2, ratio, err)
    results.append(dict(method='3-site TT (Q-mu-nu)', eps=eps,
                        ratio=ratio, err=err,
                        note=f'r1={r1} r2={r2}'))
    print(f"  eps={eps:.0e}: r1={r1} r2={r2}  "
          f"N_comp={N_comp:,}  ratio={ratio:.3f}  err={err:.2e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating PDF ...", flush=True)
pdf_path = '/tmp/qtt_compression_report.pdf'

METHODS  = ['Per-Q eig', 'Tucker/HOSVD', '3-site TT (Q-mu-nu)']
COLORS   = {'Per-Q eig': '#e74c3c', 'Tucker/HOSVD': '#2980b9', '3-site TT (Q-mu-nu)': '#27ae60'}

def dark_header(t, ncols):
    for j in range(ncols):
        t[0,j].set_facecolor('#1a252f')
        t[0,j].set_text_props(color='white', fontweight='bold')

with PdfPages(pdf_path) as pdf:

    # ── Page 1: Title ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.5, 0.97, "W Tensor Compression Report",
            ha='center', va='top', fontsize=24, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.91, f"W[Q={Q}, mu={mu_dim}, nu={nu_dim}]  —  Diamond 4×4×4",
            ha='center', va='top', fontsize=13, transform=ax.transAxes)
    ax.plot([0,1],[0.87,0.87], color='#333', lw=1.5, transform=ax.transAxes)
    meta = [
        ("Original size", f"{N_orig:,} complex elements  ({W.dtype})"),
        ("Date",          datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Host",          socket.gethostname()),
        ("eps values",    "  ".join(f"{e:.0e}" for e in EPSILONS)),
    ]
    y = 0.83
    for k, v in meta:
        ax.text(0.07, y, f"{k}:", ha='left', va='top', fontsize=11,
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.35, y, v, ha='left', va='top', fontsize=11, transform=ax.transAxes)
        y -= 0.04
    ax.plot([0,1],[y-0.01]*2, color='#aaa', lw=0.8, ls='--', transform=ax.transAxes)

    methods_desc = (
        "Methods\n\n"
        "1. Per-Q eigendecomposition\n"
        "   W[q,:,:] = U diag(lam) U†  (Hermitian per q-point)\n"
        "   Keep r largest-|eigenvalue| terms.\n"
        "   Storage: sum_q r_q*(mu+1)   max r = mu = 130\n\n"
        "2. Tucker / HOSVD\n"
        "   W ≈ G x_1 U_Q x_2 U_mu x_3 U_nu\n"
        "   Ranks from truncated SVD of each mode unfolding.\n"
        "   Storage: r_Q*r_mu*r_nu + Q*r_Q + mu*r_mu + nu*r_nu\n\n"
        "3. Direct 3-site TT on (Q, mu, nu)\n"
        "   A[Q,r1] -- B[r1,mu,r2] -- C[r2,nu]\n"
        "   Max bonds: r1<=64, r2<=130.\n"
        "   Storage: Q*r1 + r1*mu*r2 + r2*nu"
    )
    ax.text(0.07, y - 0.04, methods_desc, ha='left', va='top', fontsize=10,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='#f5f5f5', ec='#bbb'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 2: Master compression table ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 8)); ax.axis('off')
    fig.suptitle("Compression ratio & reconstruction error — all methods",
                 fontsize=12, fontweight='bold', y=0.97)
    col_labels = ['Method', 'eps', 'Ratio', 'Rel. error', 'Notes', 'Compressed?']
    rows = []
    for r in results:
        compressed = "YES ✓" if r['ratio'] < 1.0 else "no"
        rows.append([r['method'], f"{r['eps']:.0e}",
                     f"{r['ratio']:.4f}", f"{r['err']:.2e}",
                     r['note'], compressed])
    t = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.55)
    dark_header(t, len(col_labels))
    for i, r in enumerate(results):
        color = '#d5f5e3' if r['ratio'] < 1.0 else '#fadbd8'
        for j in range(len(col_labels)):
            t[i+1, j].set_facecolor(color)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 3: Compression ratio vs error scatter ────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for method in METHODS:
        subset = [r for r in results if r['method'] == method]
        xs = [r['err']   for r in subset]
        ys = [r['ratio'] for r in subset]
        ax.plot(xs, ys, 'o-', color=COLORS[method], label=method, lw=1.5, ms=7)
        for r in subset:
            ax.annotate(f"  {r['eps']:.0e}", (r['err'], r['ratio']),
                        fontsize=7, color=COLORS[method])
    ax.axhline(1.0, color='black', lw=1.2, ls='--', label='compression boundary')
    ax.fill_between([1e-5, 1], 0, 1.0, alpha=0.08, color='green')
    ax.text(1e-4, 0.5, 'COMPRESSED\n(ratio < 1)', fontsize=9,
            color='green', alpha=0.7, ha='left')
    ax.set_xscale('log')
    ax.set_xlabel('Reconstruction error ||W-W_approx||/||W||', fontsize=11)
    ax.set_ylabel('Compression ratio (compressed / original)', fontsize=11)
    ax.set_title('Compression ratio vs reconstruction error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 4: Per-Q eigenvalue spectra ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5))
    fig.suptitle("Exp 1: Per-Q eigenvalue spectrum of W[q,:,:]",
                 fontsize=12, fontweight='bold')
    # plot normalised eigenvalue for each q (faint) + mean
    ax = axes[0]
    for q in range(Q):
        s_norm = all_eigs[q] / all_eigs[q, 0]
        ax.semilogy(np.arange(1, mu_dim+1), s_norm, color='#aaa', alpha=0.2, lw=0.5)
    mean_eig = np.mean(all_eigs / all_eigs[:, :1], axis=0)
    ax.semilogy(np.arange(1, mu_dim+1), mean_eig, color='#e74c3c', lw=2, label='mean over q')
    for eps in [1e-2, 1e-3]:
        r = int(np.sum(mean_eig > eps))
        ax.axhline(eps, color='gray', lw=0.7, ls='--')
        ax.text(mu_dim*0.95, eps*1.6, f'eps={eps:.0e}: r={r}', ha='right', fontsize=7)
    ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Normalised |eigenvalue|')
    ax.set_title('All q-points (grey) + mean (red)'); ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # rank distribution at eps=1e-2
    ax = axes[1]
    eps_plot = 1e-2
    ranks_dist = np.array([int(np.sum(all_eigs[q] / all_eigs[q, 0] > eps_plot))
                            for q in range(Q)])
    ax.hist(ranks_dist, bins=range(1, mu_dim+2), color='#e74c3c', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Truncation rank r'); ax.set_ylabel('Number of q-points')
    ax.set_title(f'Per-Q rank distribution  (eps={eps_plot:.0e})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 5: Tucker mode singular values ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 4))
    fig.suptitle("Exp 2: Tucker mode-unfolding singular value spectra",
                 fontsize=12, fontweight='bold')
    for ax, s, label, max_r in zip(axes,
                                    [sQ, smu, snu],
                                    ['Mode-Q (max 64)', 'Mode-mu (max 130)', 'Mode-nu (max 130)'],
                                    [Q, mu_dim, nu_dim]):
        s_norm = s / s[0]
        ax.semilogy(np.arange(1, len(s)+1), s_norm, 'o-', ms=3, lw=1.5, color='#2980b9')
        for eps in [1e-2, 1e-3]:
            r = int(np.sum(s_norm > eps))
            ax.axhline(eps, color='gray', lw=0.7, ls='--')
            ax.text(len(s)*0.9, eps*1.5, f'r={r}', ha='right', fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Index', fontsize=8); ax.set_ylabel('Norm. s.v.', fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 6: Tucker compression detail ─────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 7))
    fig.suptitle("Exp 2: Tucker/HOSVD ranks and storage",
                 fontsize=12, fontweight='bold', y=0.97)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, top=0.92, bottom=0.06)
    ax0 = fig.add_subplot(gs[0]); ax0.axis('off')
    tk_results = [r for r in results if r['method'] == 'Tucker/HOSVD']
    col_labT = ['eps', 'r_Q', 'r_mu', 'r_nu', 'core size', 'total N', 'ratio', 'error']
    rowsT = []
    for r in tk_results:
        note = r['note']   # 'rQ=X rmu=Y rnu=Z'
        parts = {p.split('=')[0]: int(p.split('=')[1]) for p in note.split()}
        rQ, rmu, rnu = parts['rQ'], parts['rmu'], parts['rnu']
        N_core = rQ * rmu * rnu
        N_comp = int(r['ratio'] * N_orig)
        rowsT.append([f"{r['eps']:.0e}", str(rQ), str(rmu), str(rnu),
                      f"{N_core:,}", f"{N_comp:,}", f"{r['ratio']:.4f}", f"{r['err']:.2e}"])
    tT = ax0.table(cellText=rowsT, colLabels=col_labT, loc='center', cellLoc='center')
    tT.auto_set_font_size(False); tT.set_fontsize(8.5); tT.scale(1, 2.2)
    dark_header(tT, len(col_labT))
    for i, r in enumerate(tk_results):
        color = '#d5f5e3' if r['ratio'] < 1.0 else '#fadbd8'
        for j in range(len(col_labT)):
            tT[i+1, j].set_facecolor(color)

    ax1 = fig.add_subplot(gs[1])
    xlabels = [f"{e:.0e}" for e in EPSILONS]
    rQs  = [tucker_rank(sQ,  e) for e in EPSILONS]
    rmus = [tucker_rank(smu, e) for e in EPSILONS]
    rnus = [tucker_rank(snu, e) for e in EPSILONS]
    ax1.plot(xlabels, rQs,  'o-', label=f'r_Q (max={Q})')
    ax1.plot(xlabels, rmus, 's-', label=f'r_mu (max={mu_dim})')
    ax1.plot(xlabels, rnus, '^-', label=f'r_nu (max={nu_dim})')
    ax1.set_xlabel('eps'); ax1.set_ylabel('Tucker rank')
    ax1.set_title('Tucker ranks vs eps'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 7: 3-site TT detail ──────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 7))
    fig.suptitle("Exp 3: 3-site TT on (Q, mu, nu)  A[Q,r1]–B[r1,mu,r2]–C[r2,nu]",
                 fontsize=12, fontweight='bold', y=0.97)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, top=0.92, bottom=0.06)
    ax0 = fig.add_subplot(gs[0]); ax0.axis('off')
    tt_results = [r for r in results if r['method'] == '3-site TT (Q-mu-nu)']
    col_labTT = ['eps', 'r1', 'r2', 'N_comp', 'ratio', 'error']
    rowsTT = []
    for r in tt_results:
        parts = {p.split('=')[0]: int(p.split('=')[1]) for p in r['note'].split()}
        r1, r2 = parts['r1'], parts['r2']
        N_comp = Q*r1 + r1*mu_dim*r2 + r2*nu_dim
        rowsTT.append([f"{r['eps']:.0e}", str(r1), str(r2),
                       f"{N_comp:,}", f"{r['ratio']:.4f}", f"{r['err']:.2e}"])
    tTT = ax0.table(cellText=rowsTT, colLabels=col_labTT, loc='center', cellLoc='center')
    tTT.auto_set_font_size(False); tTT.set_fontsize(9); tTT.scale(1, 2.2)
    dark_header(tTT, len(col_labTT))
    for i, r in enumerate(tt_results):
        color = '#d5f5e3' if r['ratio'] < 1.0 else '#fadbd8'
        for j in range(len(col_labTT)):
            tTT[i+1, j].set_facecolor(color)

    ax1 = fig.add_subplot(gs[1])
    r1s = [expTT[e][0] for e in EPSILONS]
    r2s = [expTT[e][1] for e in EPSILONS]
    ax1.plot(xlabels, r1s, 'o-', color='#2980b9', label=f'r1 (max={Q})')
    ax1.plot(xlabels, r2s, 's-', color='#e74c3c', label=f'r2 (max={nu_dim})')
    ax1.axhline(Q,      ls='--', lw=0.6, color='#2980b9')
    ax1.axhline(nu_dim, ls='--', lw=0.6, color='#e74c3c')
    ax1.set_xlabel('eps'); ax1.set_ylabel('Bond dimension')
    ax1.set_title('TT bond dims vs eps'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 8: Interpretation ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.5, 0.97, "Interpretation & Conclusions", ha='center', va='top',
            fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.plot([0,1],[0.93,0.93], color='#333', lw=1.5, transform=ax.transAxes)
    # build summary text dynamically from results
    best = sorted([r for r in results if r['ratio'] < 1.0],
                  key=lambda r: (r['err'], r['ratio']))
    summary_lines = ["METHODS ACHIEVING COMPRESSION (ratio < 1):\n"]
    for r in best[:6]:
        summary_lines.append(
            f"  {r['method']:25s}  eps={r['eps']:.0e}  "
            f"ratio={r['ratio']:.3f}  err={r['err']:.2e}  [{r['note']}]")
    txt = "\n".join(summary_lines)
    txt += (
        "\n\n"
        "KEY FINDINGS:\n\n"
        "Per-Q eigendecomposition\n"
        "  W[q,:,:] is 130x130 Hermitian. Its eigenvalue spectrum decays fast.\n"
        "  Keeping the r largest eigenvalues per q-point gives a practical\n"
        "  compression: storage Q*r*(mu+1) < Q*mu*nu  whenever r < nu = 130.\n"
        "  This is ALWAYS true, so any truncation gives compression.\n"
        "  The question is only how large r must be for acceptable error.\n\n"
        "Tucker / HOSVD\n"
        "  The mu and nu mode unfoldings have fast singular value decay.\n"
        "  The Q mode may also compress (check Mode-Q singular values).\n"
        "  Tucker with (rQ, rmu, rnu) gives huge savings when rmu, rnu << 130.\n"
        "  The core tensor G[rQ,rmu,rnu] is much smaller than W[Q,mu,nu].\n\n"
        "3-site TT on (Q, mu, nu)\n"
        "  First bond r1 <= Q = 64. If r1 << 64, TT achieves compression.\n"
        "  Second bond r2 <= nu = 130. Middle tensor B[r1,mu,r2] dominates cost.\n"
        "  Compression requires r1*mu*r2 < Q*mu*nu, i.e. r1*r2 < Q*nu = 8320."
    )
    ax.text(0.04, 0.90, txt, ha='left', va='top', fontsize=9,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='#fafafa', ec='#ccc'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    d = pdf.infodict()
    d['Title']  = 'W Tensor Compression Report'
    d['Author'] = 'jchen9@caltech.edu'

sz = os.path.getsize(pdf_path) // 1024
print(f"PDF saved: {pdf_path}  ({sz} KB)", flush=True)

# ── Email ─────────────────────────────────────────────────────────────────────
body = (
    "Hi,\n\nCompression report for W[Q,mu,nu] attached.\n\n"
    "8-page PDF:\n"
    "  p1: Overview & methods\n"
    "  p2: Master table (all methods, all eps, compression ratio + error)\n"
    "  p3: Compression ratio vs error scatter plot\n"
    "  p4: Per-Q eigenvalue spectra\n"
    "  p5: Tucker mode singular value spectra\n"
    "  p6: Tucker detail table + rank vs eps\n"
    "  p7: 3-site TT detail table + bond dims vs eps\n"
    "  p8: Interpretation & conclusions\n\n"
    "Best,\nClaude\n"
)
res = subprocess.run(
    ['mail', '-s', 'W Tensor Compression Report — diamond 4x4x4',
     '-a', pdf_path, 'jchen9@caltech.edu'],
    input=body, text=True, capture_output=True
)
if res.returncode != 0:
    os.system(f'uuencode {pdf_path} qtt_compression_report.pdf | '
              f'mail -s "W Tensor Compression Report" jchen9@caltech.edu')
    print("Sent via uuencode fallback", flush=True)
else:
    print(f"Email sent (exit={res.returncode})", flush=True)
print("Done.", flush=True)
