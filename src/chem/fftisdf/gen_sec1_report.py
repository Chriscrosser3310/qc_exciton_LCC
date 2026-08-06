#!/usr/bin/env python3
"""
Section 1 report: QTT on Q index (open mu/nu boundaries).
Outputs two PDFs: one for (ov|vo) ERI and one for (ov|ov) ERI.
Error metrics: relative Frobenius and operator 2-norm.
"""
import pickle, h5py, subprocess, os, sys, math, datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/resnick/home/jchen9/fftisdf')
sys.path.insert(0, '/resnick/home/jchen9/fftisdf')
from utils import thc_ovvo_build_Lbar
import fft as fft_pkg

# ── Load data ──────────────────────────────────────────────────────────────────
with open('data/SCF_diamond_4x4x4_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf = pickle.load(f)
with h5py.File('data/ISDFopt_diamond_4x4x4_gth-dzvp_c5_norm0.1.chk', 'r') as f:
    X = np.asarray(f['inpv_kpt'])   # (64, 130, 26)
    W = np.asarray(f['coul_kpt'])   # (64, 130, 130)

C    = np.asarray(mf.mo_coeff)     # (64, 26, 26)
nocc = mf.cell.nelectron // 2      # 4
Xo   = X @ C[:, :, :nocc]         # (64, 130, 4)
Xv   = X @ C[:, :, nocc:]         # (64, 130, 22)

nQ, mu_dim, nu_dim = W.shape       # 64, 130, 130
nvir = Xv.shape[2]                 # 22
N_orig = W.size
nk = 4; kmesh = (nk, nk, nk)

# kconserv3[:, 0, q] = ka for ki at momentum q (correct pairing)
_kpts = mf.cell.make_kpts([nk, nk, nk])
_isdf = fft_pkg.ISDF(mf.cell, _kpts)
ka_table = _isdf.kconserv3[:, 0, :]   # (64, 64)

# neg_q_idx[q] = index of -kpts[q] (time-reversal partner, for ov|ov)
_kpts_frac = mf.cell.get_scaled_kpts(_kpts)
neg_q_idx = np.zeros(nQ, dtype=int)
for _q in range(nQ):
    # find k s.t. kpts[k] + kpts[q] ≡ 0 (mod G): residual centered in [-0.5,0.5)
    _res = (_kpts_frac + _kpts_frac[_q] + 0.5) % 1.0 - 0.5
    neg_q_idx[_q] = int(np.argmin(np.linalg.norm(_res, axis=1)))
print(f"ka_table.shape={ka_table.shape}  neg_q_idx[:8]={neg_q_idx[:8].tolist()}", flush=True)

print(f"W.shape={W.shape}  Xo.shape={Xo.shape}  Xv.shape={Xv.shape}", flush=True)

# ── (ov|vo) Frobenius metric via L-bar ────────────────────────────────────────
print("Building Lbar for (ov|vo) Frobenius metric ...", flush=True)
W_t  = torch.from_numpy(W.astype(np.complex128))
Xo_t = torch.from_numpy(Xo.astype(np.complex128))
Xv_t = torch.from_numpy(Xv.astype(np.complex128))

L  = thc_ovvo_build_Lbar(Xo_t, Xv_t, Xo_t, Xv_t, kmesh)   # (64,130,130)
Lh = L.conj().transpose(-1, -2)

LW = L @ W_t
ref_norm2_ovvo = float(
    (W_t.conj().transpose(-1,-2) @ LW @ Lh).diagonal(dim1=-1,dim2=-2).sum().real)
ref_norm_ovvo = math.sqrt(ref_norm2_ovvo)
print(f"(ov|vo) ||T||_F = {ref_norm_ovvo:.6e}", flush=True)

def eri_frob_ovvo_err(W_rec_np):
    Wr  = torch.from_numpy(W_rec_np.astype(np.complex128))
    LWr = L @ Wr
    bb = float((Wr.conj().transpose(-1,-2) @ LWr @ Lh).diagonal(dim1=-1,dim2=-2).sum().real)
    ab = float((W_t.conj().transpose(-1,-2) @ LWr @ Lh).diagonal(dim1=-1,dim2=-2).sum().real)
    return float(np.sqrt(max(ref_norm2_ovvo + bb - 2.0*ab, 0.0) / ref_norm2_ovvo))

def specnorm(M):
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[0]) if len(sv) else 0.0

# ── QR precomputation (shared by all 2-norm and (ov|ov) metrics) ───────────────
# A_Q[(k,i,a), P] = conj(Xo[k,P,i]) * Xv[ka,P,a]  where ka=kconserv3[k,0,Q]
# T_Q^{(ov|vo)} = A_Q W[Q] A_Q†   →  sigma_max = sigma_max(R_Q W[Q] R_Q†)
# T_Q^{(ov|ov)} = A_Q W[Q] A_{-Q}†  →  sigma_max = sigma_max(R_Q W[Q] R_{neg_q}†)
# G_Q = A_Q† A_Q = R_Q† R_Q  (used for (ov|ov) Frobenius)
print("Precomputing QR factors A_Q = Q_Q R_Q ...", flush=True)
R_list = []   # R_list[q]: (130,130) upper-triangular R factor of A_q
for qi in range(nQ):
    kq_idx = ka_table[:, qi]   # correct ka for each ki at momentum qi
    A_qi = np.einsum('kPi,kPa->kiaP', Xo.conj(), Xv[kq_idx], optimize=True
                     ).reshape(nQ * nocc * nvir, mu_dim)   # (5632, 130)
    R_list.append(np.linalg.qr(A_qi, mode='r'))

# Gram matrices G_Q = R_Q† R_Q
G_list = [R.conj().T @ R for R in R_list]   # each (130,130), Hermitian

# ── (ov|vo) 2-norm ─────────────────────────────────────────────────────────────
print("Computing (ov|vo) ||T||_2 ...", flush=True)
ref_2norm_ovvo = float(max(
    specnorm(R_list[q] @ W[q] @ R_list[q].conj().T) for q in range(nQ)))
print(f"(ov|vo) ||T||_2 = {ref_2norm_ovvo:.6e}", flush=True)

def eri_2norm_ovvo_err(W_rec_np):
    dW = W - W_rec_np
    return float(max(
        specnorm(R_list[q] @ dW[q] @ R_list[q].conj().T) for q in range(nQ)
    )) / ref_2norm_ovvo

# ── (ov|ov) Frobenius via G matrices ──────────────────────────────────────────
# ⟨T_A,T_B⟩ = Σ_Q tr(W_A†[Q] G_Q W_B[Q] G_{-Q})
# (momentum conservation for bra=(ov): k_a=k+Q; for ket=(ov): k_b=k'-Q, same Q)
def inner_ovov(W_A, W_B):
    total = 0.0 + 0.0j
    for q in range(nQ):
        C = G_list[q] @ W_B[q] @ G_list[neg_q_idx[q]]   # G_Q W_B G_{-Q}
        total += np.sum(W_A[q].conj() * C)                # tr(W_A†[q] C)
    return total

print("Computing (ov|ov) ||T||_F ...", flush=True)
ref_norm2_ovov = float(inner_ovov(W, W).real)
ref_norm_ovov  = math.sqrt(ref_norm2_ovov)
print(f"(ov|ov) ||T||_F = {ref_norm_ovov:.6e}", flush=True)

def eri_frob_ovov_err(W_rec_np):
    aa = ref_norm2_ovov
    bb = float(inner_ovov(W_rec_np, W_rec_np).real)
    ab = inner_ovov(W, W_rec_np)
    return float(np.sqrt(max(aa + bb - 2.0*ab.real, 0.0) / aa))

# ── (ov|ov) 2-norm ────────────────────────────────────────────────────────────
print("Computing (ov|ov) ||T||_2 ...", flush=True)
ref_2norm_ovov = float(max(
    specnorm(R_list[q] @ W[q] @ R_list[neg_q_idx[q]].conj().T) for q in range(nQ)))
print(f"(ov|ov) ||T||_2 = {ref_2norm_ovov:.6e}", flush=True)

def eri_2norm_ovov_err(W_rec_np):
    dW = W - W_rec_np
    return float(max(
        specnorm(R_list[q] @ dW[q] @ R_list[neg_q_idx[q]].conj().T) for q in range(nQ)
    )) / ref_2norm_ovov

# ── W operator norm (for QTT norm bound) ──────────────────────────────────────
print("Computing W_op ...", flush=True)
W_op = float(max(specnorm(W[q]) for q in range(nQ)))

# ── QTT compression ────────────────────────────────────────────────────────────
def qtt_compress(phys_dim, cutoff):
    n_sites = math.ceil(math.log(nQ, phys_dim))
    Q_pad   = phys_dim ** n_sites
    Wp = np.zeros((Q_pad, mu_dim, nu_dim), dtype=W.dtype)
    Wp[:nQ] = W
    arr = Wp.transpose(1, 0, 2).reshape([mu_dim] + [phys_dim]*n_sites + [nu_dim])
    T = arr.copy()
    cores = []
    for k in range(n_sites - 1):
        nl = T.shape[0] * T.shape[1]; nr = T.size // nl
        U, s, Vh = np.linalg.svd(T.reshape(nl, nr), full_matrices=False)
        r = max(1, int(np.sum(s / s[0] > cutoff)))
        cores.append(U[:, :r].reshape(T.shape[0], T.shape[1], r))
        T = (np.diag(s[:r]) @ Vh[:r]).reshape([r] + list(T.shape[2:]))
    cores.append(T)

    W_rec = np.zeros_like(W)
    for q in range(nQ):
        tmp, digits = q, []
        for _ in range(n_sites): digits.append(tmp % phys_dim); tmp //= phys_dim
        digits.reverse()
        M = cores[0][:, digits[0], :]
        for k in range(1, n_sites): M = M @ cores[k][:, digits[k], :]
        W_rec[q] = M

    norm_bound = 1.0
    for c in cores:
        norm_bound *= max(specnorm(c[:, p, :]) for p in range(c.shape[1]))

    return dict(
        bond_dims  = [c.shape[-1] for c in cores[:-1]],
        N_comp     = sum(c.size for c in cores),
        ferr_ovvo  = eri_frob_ovvo_err(W_rec),
        err2_ovvo  = eri_2norm_ovvo_err(W_rec),
        ferr_ovov  = eri_frob_ovov_err(W_rec),
        err2_ovov  = eri_2norm_ovov_err(W_rec),
        norm_bound = float(norm_bound),
        norm_inc   = float(norm_bound / W_op),
    )

# ── Scan ───────────────────────────────────────────────────────────────────────
CUTOFFS   = [2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 1e-4]
PHYS_DIMS = [4, 2]
TARGET    = 1e-2

print("\nScanning ...", flush=True)
results = {}
for pd in PHYS_DIMS:
    results[pd] = []
    print(f"  phys_dim={pd}", flush=True)
    for eps in CUTOFFS:
        r = qtt_compress(pd, eps)
        results[pd].append((eps, r))
        print(f"    eps={eps:.0e}  ovvo_F={r['ferr_ovvo']:.3e} ovvo_2={r['err2_ovvo']:.3e}"
              f"  ovov_F={r['ferr_ovov']:.3e} ovov_2={r['err2_ovov']:.3e}"
              f"  ratio={r['N_comp']/N_orig:.3f}", flush=True)

# ── Bisect per ERI type ────────────────────────────────────────────────────────
def bisect(pd, ferr_key, lo=1e-5, hi=5e-1, target=TARGET, n_iter=30):
    if qtt_compress(pd, hi)[ferr_key] < target: return hi
    if qtt_compress(pd, lo)[ferr_key] >= target: return lo
    for _ in range(n_iter):
        mid = np.exp(0.5*(np.log(lo)+np.log(hi)))
        if qtt_compress(pd, mid)[ferr_key] < target: lo = mid
        else: hi = mid
        if hi/lo < 1.003: break
    return lo

print(f"\nBisecting ...", flush=True)
bisect_ovvo = {}; bisect_ovov = {}
for pd in PHYS_DIMS:
    print(f"  phys_dim={pd} (ov|vo) ...", flush=True)
    e = bisect(pd, 'ferr_ovvo')
    bisect_ovvo[pd] = (e, qtt_compress(pd, e))
    print(f"    eps={e:.3e}  F={bisect_ovvo[pd][1]['ferr_ovvo']:.3e}", flush=True)

    print(f"  phys_dim={pd} (ov|ov) ...", flush=True)
    e = bisect(pd, 'ferr_ovov')
    bisect_ovov[pd] = (e, qtt_compress(pd, e))
    print(f"    eps={e:.3e}  F={bisect_ovov[pd][1]['ferr_ovov']:.3e}", flush=True)

# ── Plots ──────────────────────────────────────────────────────────────────────
titles = {4: r'3-site QTT ($d=4$)', 2: r'6-site QTT ($d=2$)'}

def make_plot(ferr_key, err2_key, bisect_rows, ylabel, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pd in zip(axes, PHYS_DIMS):
        eps_vals  = [eps for eps, _ in results[pd]]
        ferr_vals = [r[ferr_key] for _, r in results[pd]]
        e2_vals   = [r[err2_key] for _, r in results[pd]]
        ax.loglog(eps_vals, ferr_vals, 'o-',  color='C0', label='Frobenius')
        ax.loglog(eps_vals, e2_vals,   's--', color='C1', label='Operator 2-norm')
        ax.axhline(TARGET, color='gray', linestyle=':', linewidth=1, label=r'$10^{-2}$ target')
        eps_b = bisect_rows[pd][0]
        ax.axvline(eps_b, color='gold', linestyle='--', linewidth=1.2,
                   label=r'bisected $\varepsilon$')
        ax.set_xlabel(r'SVD cutoff $\varepsilon$')
        if ax is axes[0]: ax.set_ylabel(ylabel)
        ax.set_title(titles[pd])
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {fig_path}", flush=True)

print("\nGenerating plots ...", flush=True)
plot_ovvo = '/tmp/sec1_qtt_plot_ovvo.pdf'
plot_ovov = '/tmp/sec1_qtt_plot_ovov.pdf'
make_plot('ferr_ovvo', 'err2_ovvo', bisect_ovvo,
          r'Relative ERI error $\|(ov|vo)\|$', plot_ovvo)
make_plot('ferr_ovov', 'err2_ovov', bisect_ovov,
          r'Relative ERI error $\|(ov|ov)\|$', plot_ovov)

# ── LaTeX helpers ──────────────────────────────────────────────────────────────
def sci_tex(v):
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    if abs(m - 1.0) < 0.06: return f'$10^{{{e}}}$'
    return f'${m:.4f}\\!\\times\\!10^{{{e}}}$'

def eps_tex(v):
    e = int(math.floor(math.log10(v)))
    m = v / 10**e
    if abs(m - 1.0) < 0.06: return f'$10^{{{e}}}$'
    return f'${m:.0f}\\!\\times\\!10^{{{e}}}$'

def make_table(pd, ferr_key, err2_key, bisect_rows):
    rows = ''
    for eps, r in results[pd]:
        ratio  = r['N_comp'] / N_orig
        saving = (1 - ratio) * 100
        bd_s   = ', '.join(str(b) for b in r['bond_dims'])
        save_s = f'{saving:.0f}\\%' if ratio < 1 else r'\textbf{none}'
        rows += (f"  {eps_tex(eps)} & [{bd_s}] & ${ratio:.3f}$ ({save_s})"
                 f" & ${r[ferr_key]:.2e}$ & ${r[err2_key]:.2e}$"
                 f" & ${r['norm_bound']:.3f}$ & ${r['norm_inc']:.3f}$ \\\\\n")
    eps_opt, r = bisect_rows[pd]
    ratio  = r['N_comp'] / N_orig
    saving = (1 - ratio) * 100
    bd_s   = ', '.join(str(b) for b in r['bond_dims'])
    save_s = f'{saving:.0f}\\%' if ratio < 1 else r'\textbf{none}'
    rows += r'\midrule' + '\n'
    rows += (r'\rowcolor{yellow!25}'
             f"  {eps_tex(eps_opt)} & [{bd_s}] & ${ratio:.3f}$ ({save_s})"
             f" & ${r[ferr_key]:.2e}$ & ${r[err2_key]:.2e}$"
             f" & ${r['norm_bound']:.3f}$ & ${r['norm_inc']:.3f}$ \\\\\n")
    return rows

today = datetime.date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# Note 1: (ov|vo)
# ─────────────────────────────────────────────────────────────────────────────
def make_latex_ovvo():
    tbl4 = make_table(4, 'ferr_ovvo', 'err2_ovvo', bisect_ovvo)
    tbl2 = make_table(2, 'ferr_ovvo', 'err2_ovvo', bisect_ovvo)
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb,booktabs,geometry,microtype,hyperref,array,xcolor,colortbl,bm,graphicx}
\geometry{margin=2cm}
\hypersetup{colorlinks,linkcolor=blue}
\title{\textbf{QTT on $Q$: ERI Error in the $(ov|vo)$ Sector}}
\author{}\date{""" + today + r"""}
\begin{document}
\maketitle\thispagestyle{empty}

\section*{Setup}
$W\!\in\!\mathbb{C}^{N_Q\times N_\mu\times N_\nu}$,\;
$N_Q=64$,\; $N_\mu=N_\nu=130$\; (diamond $4\!\times\!4\!\times\!4$, ISDF $c=5$).
$X_o=XC_{\rm occ}\in\mathbb{C}^{N_Q\times N_\mu\times 4}$,\;
$X_v=XC_{\rm vir}\in\mathbb{C}^{N_Q\times N_\mu\times 22}$.
$N_{\rm orig}=1{,}081{,}600$.
Reference norms:
\[
  \|T_{\rm ref}\|_F = """ + sci_tex(ref_norm_ovvo) + r""", \qquad
  \|T_{\rm ref}\|_2 = """ + sci_tex(ref_2norm_ovvo) + r""".
\]

\section{QTT Representation}
Write $Q=\sum_k s_k d^{n-1-k}$ (base-$d$, $n=\lceil\log_d N_Q\rceil$).
Reshape $W$ as $\mathcal{W}[\mu,s_0,\ldots,s_{n-1},\nu]$ and apply sequential
truncated SVD (relative cutoff $\varepsilon$):
\begin{equation}\label{eq:qtt}
  W[Q,\mu,\nu]\approx\hat W[Q,\mu,\nu]
  =\sum_{\bm\alpha}
  A_0[\mu,s_0,\alpha_0]\;A_1[\alpha_0,s_1,\alpha_1]\;\cdots\;
  A_{n-1}[\alpha_{n-2},s_{n-1},\nu].
\end{equation}
$N_{\rm comp}=\sum_k|A_k|$.\;
QTT norm bound:
$\|\hat W[Q]\|_2\le\prod_k\max_{s_k}\|A_k[\cdot,s_k,\cdot]\|_2$;\;
norm increase $=$ bound$\,/\,\|W\|_{\rm op}$,\;
$\|W\|_{\rm op}=""" + f"{W_op:.4f}" + r"""$.

\section{$(ov|vo)$ ERI Error Metrics}

\paragraph{THC factorization.}
\begin{equation}
(i_k\,a_{k+Q}\mid b_{k'+Q}\,j_{k'})
\approx\sum_{PP'}X_o[k,P,i]\,X_v[k{+}Q,P,a]\;W[Q,P,P']\;
X_v^*[k'{+}Q,P',b]\,X_o^*[k',P',j].
\end{equation}
Momentum conservation: $k_a-k_i=k_b-k_j=Q$.
For fixed $Q$ this is a linear map $(o_1,v_1)\mapsto(v_2,o_2)$ with block
\begin{equation}
  T_Q^{(ov|vo)} = A_Q\,W[Q]\,A_Q^\dagger,\qquad
  [A_Q]_{(k,i,a),P}=\bar{X}_o[k,P,i]\,X_v[k_a,P,a],\quad k_a=\mathrm{kconserv}(k,Q).
\end{equation}
The full ERI operator is block-diagonal in $Q$, so
$\|T^{(ov|vo)}\|=\max_Q\|T_Q^{(ov|vo)}\|$.

\paragraph{Frobenius norm.}
\begin{equation}
  \langle T_A,T_B\rangle_F=\sum_Q\mathrm{tr}
  \!\left(W_A^{(Q)\dagger}\bar L^{(Q)}W_B^{(Q)}\bar L^{(Q)\dagger}\right),\quad
  \bar L^{(Q)}=\mathrm{IFFT}_Q[\widetilde O(-Q)\cdot\widetilde V(Q)],
\end{equation}
with $O[k,I,J]=\sum_i X_o[k,I,i]X_o^*[k,J,i]$,\;
$V[k,I,J]=\sum_a X_v^*[k,I,a]X_v[k,J,a]$.
Relative error:
\begin{equation}
  \varepsilon_F=\sqrt{\frac{\|T_{\rm ref}\|_F^2+\|\hat T\|_F^2
    -2\operatorname{Re}\langle T_{\rm ref},\hat T\rangle_F}
    {\|T_{\rm ref}\|_F^2}}.
\end{equation}

\paragraph{Operator 2-norm and its computation.}
Since $T_Q^{(ov|vo)}=A_Q W[Q]A_Q^\dagger$ has rank $\le N_\mu=130$,
compute the thin QR $A_Q=\mathcal{Q}_Q R_Q$ (shape $N_k n_{\rm occ}n_{\rm vir}
\times N_\mu=5632\times130$).
Because $\mathcal{Q}_Q$ has orthonormal columns,
\begin{equation}\label{eq:2norm_ovvo}
  \|T_Q^{(ov|vo)}\|_2
  =\sigma_{\max}(\mathcal{Q}_Q R_Q W[Q]R_Q^\dagger\mathcal{Q}_Q^\dagger)
  =\sigma_{\max}(R_Q\,W[Q]\,R_Q^\dagger),
\end{equation}
a $130\times130$ problem.
Hence $\|T^{(ov|vo)}\|_2=\max_Q\sigma_{\max}(R_Q W[Q]R_Q^\dagger)$ and
\begin{equation}
  \varepsilon_2=
  \frac{\max_Q\,\sigma_{\max}\!\left(R_Q(W[Q]-\hat W[Q])R_Q^\dagger\right)}
       {\|T^{(ov|vo)}\|_2}.
\end{equation}
The $R_Q$ factors are computed once; each evaluation costs $64$ SVDs
of $130\times130$ matrices.

\section{Error vs.\ SVD Cutoff}
\begin{figure}[h!]
\centering\includegraphics[width=0.92\textwidth]{""" + plot_ovvo + r"""}
\caption{Relative Frobenius ($\varepsilon_F$) and 2-norm ($\varepsilon_2$) errors
for the $(ov|vo)$ ERI vs.\ SVD cutoff $\varepsilon$.
Vertical dashed: bisected $\varepsilon$ giving $\varepsilon_F<10^{-2}$ (highlighted row).}
\end{figure}

\section{Scan Results}
Highlighted row (\colorbox{yellow!25}{yellow}): bisected $\varepsilon$ with
$\varepsilon_F<10^{-2}$.

\subsection*{3-Site QTT\;($d=4$, $n=3$, $Q_{\rm pad}=64$)}
\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lllccccc}\toprule
$\varepsilon$ & Bond dims & $N_{\rm comp}$ (ratio,\,\%\,saved)
  & $\varepsilon_F$ & $\varepsilon_2$ & Norm bound & Norm incr.\\
\midrule
""" + tbl4 + r"""\bottomrule\end{tabular}
\caption{$3$-site QTT ($d=4$).}
\end{table}

\subsection*{6-Site QTT\;($d=2$, $n=6$, $Q_{\rm pad}=64$)}
\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lllccccc}\toprule
$\varepsilon$ & Bond dims & $N_{\rm comp}$ (ratio,\,\%\,saved)
  & $\varepsilon_F$ & $\varepsilon_2$ & Norm bound & Norm incr.\\
\midrule
""" + tbl2 + r"""\bottomrule\end{tabular}
\caption{$6$-site QTT ($d=2$).}
\end{table}
\end{document}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Note 2: (ov|ov)
# ─────────────────────────────────────────────────────────────────────────────
def make_latex_ovov():
    tbl4 = make_table(4, 'ferr_ovov', 'err2_ovov', bisect_ovov)
    tbl2 = make_table(2, 'ferr_ovov', 'err2_ovov', bisect_ovov)
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb,booktabs,geometry,microtype,hyperref,array,xcolor,colortbl,bm,graphicx}
\geometry{margin=2cm}
\hypersetup{colorlinks,linkcolor=blue}
\title{\textbf{QTT on $Q$: ERI Error in the $(ov|ov)$ Sector}}
\author{}\date{""" + today + r"""}
\begin{document}
\maketitle\thispagestyle{empty}

\section*{Setup}
Same system as the $(ov|vo)$ note. $N_{\rm orig}=1{,}081{,}600$.
Reference norms for the $(ov|ov)$ sector:
\[
  \|T_{\rm ref}\|_F = """ + sci_tex(ref_norm_ovov) + r""", \qquad
  \|T_{\rm ref}\|_2 = """ + sci_tex(ref_2norm_ovov) + r""".
\]

\section{QTT Representation}
Same QTT decomposition of $W$ as in the $(ov|vo)$ note (eq.~\eqref{eq:qtt}).
$\|W\|_{\rm op}=""" + f"{W_op:.4f}" + r"""$.
\begin{equation}\label{eq:qtt}
  W[Q,\mu,\nu]\approx\hat W[Q,\mu,\nu]
  =\sum_{\bm\alpha}
  A_0[\mu,s_0,\alpha_0]\;\cdots\;A_{n-1}[\alpha_{n-2},s_{n-1},\nu].
\end{equation}

\section{$(ov|ov)$ ERI Error Metrics}

\paragraph{THC factorization and block structure.}
\begin{equation}
(i_k\,a_{k+Q}\mid j_{k'}\,b_{k'-Q})
\approx\sum_{PP'}X_o[k,P,i]\,X_v[k{+}Q,P,a]\;W[Q,P,P']\;
X_o^*[k',P',j]\,X_v^*[k'{-}Q,P',b].
\end{equation}
Momentum conservation for the $(ov|ov)$ bracket requires $k_a-k_i=Q$ for the
bra and $k_b-k_j=-Q$ for the ket (so that $k_i+k_j=k_a+k_b$).
Defining
\begin{equation}
  [A_Q]_{(k,i,a),P}=\bar{X}_o[k,P,i]\,X_v[k_a,P,a],\quad k_a=\mathrm{kconserv}(k,Q), \qquad
  B_Q = A_{-Q},
\end{equation}
the $Q$-block of the ERI as a map $(o_1,v_1)\mapsto(o_2,v_2)$ is
\begin{equation}
  T_Q^{(ov|ov)} = A_Q\,W[Q]\,B_Q^\dagger = A_Q\,W[Q]\,A_{-Q}^\dagger.
\end{equation}
The full operator is block-diagonal in $Q$:
$\|T^{(ov|ov)}\|=\max_Q\|T_Q^{(ov|ov)}\|$.

\paragraph{Frobenius norm.}
\begin{equation}
  \langle T_A,T_B\rangle_F
  =\sum_Q\mathrm{tr}\!\left(W_A^{(Q)\dagger}\,G_Q\,W_B^{(Q)}\,G_{-Q}\right),
  \qquad G_Q = A_Q^\dagger A_Q = R_Q^\dagger R_Q,
\end{equation}
where $A_Q=\mathcal{Q}_Q R_Q$ is the thin QR factorisation (same $R_Q$ matrices
used for the 2-norm).
The norm satisfies $\|T^{(ov|ov)}\|_F^2=\sum_Q\mathrm{tr}(W^{(Q)\dagger}G_Q W^{(Q)}G_{-Q})$
and the relative Frobenius error is
\begin{equation}
  \varepsilon_F=\sqrt{\frac{\langle T_{\rm ref},T_{\rm ref}\rangle_F
    +\langle\hat T,\hat T\rangle_F-2\operatorname{Re}
    \langle T_{\rm ref},\hat T\rangle_F}{\langle T_{\rm ref},T_{\rm ref}\rangle_F}}.
\end{equation}

\paragraph{Operator 2-norm.}
Compute thin QR $A_Q=\mathcal{Q}_Q R_Q$ and $B_Q=A_{-Q}=\mathcal{Q}_{-Q}R_{-Q}$.
Since both factors are isometries,
\begin{equation}\label{eq:2norm_ovov}
  \|T_Q^{(ov|ov)}\|_2
  =\sigma_{\max}(\mathcal{Q}_Q R_Q W[Q]R_{-Q}^\dagger\mathcal{Q}_{-Q}^\dagger)
  =\sigma_{\max}(R_Q\,W[Q]\,R_{-Q}^\dagger),
\end{equation}
again a $130\times130$ problem using the same precomputed $R_Q$ factors.
\begin{equation}
  \varepsilon_2=
  \frac{\max_Q\,\sigma_{\max}\!\left(R_Q(W[Q]-\hat W[Q])R_{-Q}^\dagger\right)}
       {\|T^{(ov|ov)}\|_2}.
\end{equation}

\section{Error vs.\ SVD Cutoff}
\begin{figure}[h!]
\centering\includegraphics[width=0.92\textwidth]{""" + plot_ovov + r"""}
\caption{Relative Frobenius ($\varepsilon_F$) and 2-norm ($\varepsilon_2$) errors
for the $(ov|ov)$ ERI vs.\ SVD cutoff $\varepsilon$.
Vertical dashed: bisected $\varepsilon$ giving $\varepsilon_F<10^{-2}$ (highlighted row).}
\end{figure}

\section{Scan Results}
Highlighted row (\colorbox{yellow!25}{yellow}): bisected $\varepsilon$ with
$\varepsilon_F<10^{-2}$.

\subsection*{3-Site QTT\;($d=4$, $n=3$, $Q_{\rm pad}=64$)}
\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lllccccc}\toprule
$\varepsilon$ & Bond dims & $N_{\rm comp}$ (ratio,\,\%\,saved)
  & $\varepsilon_F$ & $\varepsilon_2$ & Norm bound & Norm incr.\\
\midrule
""" + tbl4 + r"""\bottomrule\end{tabular}
\caption{$3$-site QTT ($d=4$), $(ov|ov)$ errors.}
\end{table}

\subsection*{6-Site QTT\;($d=2$, $n=6$, $Q_{\rm pad}=64$)}
\begin{table}[h!]\centering\small\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lllccccc}\toprule
$\varepsilon$ & Bond dims & $N_{\rm comp}$ (ratio,\,\%\,saved)
  & $\varepsilon_F$ & $\varepsilon_2$ & Norm bound & Norm incr.\\
\midrule
""" + tbl2 + r"""\bottomrule\end{tabular}
\caption{$6$-site QTT ($d=2$), $(ov|ov)$ errors.}
\end{table}
\end{document}
"""

# ── Compile and email both PDFs ────────────────────────────────────────────────
print("\nCompiling LaTeX ...", flush=True)
jobs = [
    ('/tmp/sec1_ovvo.tex', '/tmp/sec1_ovvo.pdf', make_latex_ovvo(),
     'QTT on Q — (ov|vo) ERI error scan'),
    ('/tmp/sec1_ovov.tex', '/tmp/sec1_ovov.pdf', make_latex_ovov(),
     'QTT on Q — (ov|ov) ERI error scan'),
]
for tex, pdf, latex_src, subj in jobs:
    with open(tex, 'w') as f: f.write(latex_src)
    for _ in range(2):
        r = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', f'-output-directory={os.path.dirname(pdf)}', tex],
            capture_output=True, text=True)
    if not os.path.exists(pdf):
        print(f"pdflatex FAILED for {tex}:\n", r.stdout[-3000:])
        sys.exit(1)
    kb = os.path.getsize(pdf) // 1024
    print(f"PDF saved: {pdf}  ({kb} KB)", flush=True)
    msg = MIMEMultipart()
    msg['From'] = 'jchen9@caltech.edu'
    msg['To']   = 'jchen9@caltech.edu'
    msg['Subject'] = subj
    msg.attach(MIMEText(subj + '\n\n(corrected ka indexing and Xo conjugation)', 'plain'))
    with open(pdf, 'rb') as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(pdf))
    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf)}"'
    msg.attach(part)
    with smtplib.SMTP('mail.caltech.edu', 25) as server:
        server.sendmail('jchen9@caltech.edu', 'jchen9@caltech.edu', msg.as_string())
    print(f"Email sent via smtplib: {subj}", flush=True)

print("Done.")
