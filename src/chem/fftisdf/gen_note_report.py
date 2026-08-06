#!/usr/bin/env python3
"""
Generate LaTeX note + PDF: tensor decompositions of the ISDF Coulomb kernel W.
Six methods. Table: (a) fixed truncation eps=1e-2, (b) tuned eps for actual error<1e-2.
"""
import pickle, h5py, subprocess, os, sys
import numpy as np

os.chdir('/resnick/home/jchen9/fftisdf')
with open('data/SCF_diamond_4x4x4_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf = pickle.load(f)
with h5py.File('data/ISDFopt_diamond_4x4x4_gth-dzvp_c5_norm0.1.chk', 'r') as f:
    W = np.asarray(f['coul_kpt'])

Q, mu_dim, nu_dim = W.shape
N_orig = Q * mu_dim * nu_dim
W_norm = np.linalg.norm(W)
W_norms_exact = np.array([np.linalg.svd(W[q], compute_uv=False)[0] for q in range(Q)])
W_op = W_norms_exact.max()
print(f"W.shape={W.shape}  N_orig={N_orig:,}  W_op={W_op:.4f}", flush=True)

def rel_error(W_rec):
    return np.linalg.norm(W_rec - W) / W_norm

def specnorm(M):
    sv = np.linalg.svd(M, compute_uv=False)
    return sv[0] if len(sv) else 0.0

# Pre-compute mode-unfolding SVDs (reused for Tucker)
UQ_full,  sQ,  _ = np.linalg.svd(W.reshape(Q, -1),                       full_matrices=False)
Umu_full, smu, _ = np.linalg.svd(W.transpose(1,0,2).reshape(mu_dim, -1), full_matrices=False)
Unu_full, snu, _ = np.linalg.svd(W.transpose(2,0,1).reshape(nu_dim, -1), full_matrices=False)

def trank(s, eps): return max(1, int(np.sum(s / s[0] > eps)))

# ── Method 1: Per-Q eigendecomposition ────────────────────────────────────────
def method1(eps):
    W_rec = np.zeros_like(W)
    N_comp = 0
    bounds = []
    for q in range(Q):
        lam, U = np.linalg.eigh(W[q])
        idx = np.argsort(np.abs(lam))[::-1]
        lam = lam[idx]; U = U[:, idx]
        r = max(1, int(np.sum(np.abs(lam) / np.abs(lam[0]) > eps)))
        W_rec[q] = (U[:, :r] * lam[:r]) @ U[:, :r].conj().T
        bounds.append(float(np.abs(lam[0])))
        N_comp += mu_dim * r + r
    return rel_error(W_rec), N_comp, max(bounds) / W_op

# ── Method 2: Tucker / HOSVD ──────────────────────────────────────────────────
def method2(eps):
    rQ  = trank(sQ,  eps); rmu = trank(smu, eps); rnu = trank(snu, eps)
    UQ  = UQ_full[:, :rQ];  Umu = Umu_full[:, :rmu];  Unu = Unu_full[:, :rnu]
    G1  = (UQ.conj().T  @ W.reshape(Q,-1)).reshape(rQ,mu_dim,nu_dim)
    G2  = (Umu.conj().T @ G1.transpose(1,0,2).reshape(mu_dim,-1)).reshape(rmu,rQ,nu_dim).transpose(1,0,2)
    G   = (Unu.conj().T @ G2.transpose(2,0,1).reshape(nu_dim,-1)).reshape(rnu,rQ,rmu).transpose(1,2,0)
    R2  = (Umu @ G.transpose(1,0,2).reshape(rmu,-1)).reshape(mu_dim,rQ,rnu).transpose(1,0,2)
    R3  = (Unu @ R2.transpose(2,0,1).reshape(rnu,-1)).reshape(nu_dim,rQ,mu_dim).transpose(1,2,0)
    W_rec = (UQ @ R3.reshape(rQ,-1)).reshape(Q, mu_dim, nu_dim)
    N_comp = G.size + Q*rQ + mu_dim*rmu + nu_dim*rnu
    Gr = G.reshape(rQ,-1)
    bounds = [float(specnorm((UQ[q,:] @ Gr).reshape(rmu,rnu))) for q in range(Q)]
    return rel_error(W_rec), N_comp, max(bounds) / W_op

# ── Method 3: 3-site TT (base-4 Q, open mu/nu) ────────────────────────────────
def method3(eps):
    nk = 4
    M1 = W.reshape(nk,nk,nk,mu_dim,nu_dim).transpose(3,0,1,2,4).reshape(mu_dim*nk, nk*nk*nu_dim)
    UA, sA, VhA = np.linalg.svd(M1, full_matrices=False)
    r1 = trank(sA, eps)
    A = UA[:,:r1].reshape(mu_dim, nk, r1)
    M2 = (np.diag(sA[:r1]) @ VhA[:r1]).reshape(r1*nk, nk*nu_dim)
    UB, sB, VhB = np.linalg.svd(M2, full_matrices=False)
    r2 = trank(sB, eps)
    B = UB[:,:r2].reshape(r1, nk, r2)
    C = (np.diag(sB[:r2]) @ VhB[:r2]).reshape(r2, nk, nu_dim)
    W_rec = np.zeros_like(W)
    bounds = []
    for q in range(Q):
        q1,q2,q3 = q//16, (q//4)%4, q%4
        BQ = A[:,q1,:] @ B[:,q2,:]
        Cq = C[:,q3,:]
        W_rec[q] = BQ @ Cq
        bounds.append(specnorm(BQ) * specnorm(Cq))
    return rel_error(W_rec), A.size+B.size+C.size, max(bounds)/W_op

# ── Method 4: Binary QTT on Q (MSB-first, open mu/nu) ────────────────────────
def method4(eps):
    n_bits = 6
    arr = W.transpose(1,0,2).reshape([mu_dim]+[2]*n_bits+[nu_dim])
    T = arr.copy()
    cores = []
    for k in range(n_bits - 1):
        nl = T.shape[0]*T.shape[1]; nr = T.size//nl
        U, s, Vh = np.linalg.svd(T.reshape(nl,nr), full_matrices=False)
        r = trank(s, eps)
        cores.append(U[:,:r].reshape(T.shape[0],T.shape[1],r))
        T = (np.diag(s[:r]) @ Vh[:r]).reshape([r]+list(T.shape[2:]))
    cores.append(T)
    W_rec = np.zeros_like(W)
    for q in range(Q):
        bits = [(q >> (n_bits-1-k)) & 1 for k in range(n_bits)]
        M = cores[0][:,bits[0],:]
        for k in range(1, n_bits): M = M @ cores[k][:,bits[k],:]
        W_rec[q] = M
    err = rel_error(W_rec)
    N_comp = sum(c.size for c in cores)
    # Tight in canonical form: use exact spectral norms
    norm_inc = max(specnorm(W_rec[q]) for q in range(Q)) / W_op
    return err, N_comp, norm_inc

# ── Method 5: Binary MPO on (mu, nu), Q as open left boundary ─────────────────
def method5(eps):
    n_mb = 8; Qp = 256
    Wp = np.zeros((Q, Qp, Qp), dtype=W.dtype)
    Wp[:, :mu_dim, :nu_dim] = W
    perm = [0] + [j for k in range(n_mb) for j in (k+1, k+1+n_mb)]
    arr = Wp.reshape(Q, *([2]*n_mb), *([2]*n_mb)).transpose(perm).reshape(Q, *([4]*n_mb))
    T = arr.copy()
    cores = []
    for k in range(n_mb - 1):
        nl = T.shape[0]*T.shape[1]; nr = T.size//nl
        U, s, Vh = np.linalg.svd(T.reshape(nl,nr), full_matrices=False)
        r = trank(s, eps)
        cores.append(U[:,:r].reshape(T.shape[0],T.shape[1],r))
        T = (np.diag(s[:r]) @ Vh[:r]).reshape([r]+list(T.shape[2:]))
    cores.append(T)
    def mpo_mat(q):
        state = cores[0][q,:,:].reshape(2, 2, cores[0].shape[2])
        for k in range(1, n_mb):
            Ck = cores[k]
            if Ck.ndim == 3:
                Ck4 = Ck.reshape(Ck.shape[0],2,2,Ck.shape[2])
                ns = np.tensordot(state, Ck4, axes=([-1],[0]))
                s_ = ns.shape
                state = ns.transpose(0,2,1,3,4).reshape(s_[0]*2,s_[1]*2,s_[4])
            else:
                Ck4 = Ck.reshape(Ck.shape[0],2,2)
                ns = np.tensordot(state, Ck4, axes=([-1],[0]))
                s_ = ns.shape
                state = ns.transpose(0,2,1,3).reshape(s_[0]*2,s_[1]*2)
        return state
    W_rec = np.zeros_like(W)
    for q in range(Q):
        W_rec[q] = mpo_mat(q)[:mu_dim,:nu_dim]
    err = rel_error(W_rec)
    N_comp = sum(c.size for c in cores)
    norm_inc = max(specnorm(W_rec[q]) for q in range(Q)) / W_op
    return err, N_comp, norm_inc

# ── Method 6: Hierarchical digit-wise QTT (3-site then binary-split each site) ─
def method6(eps):
    nk = 4
    M1 = W.reshape(nk,nk,nk,mu_dim,nu_dim).transpose(3,0,1,2,4).reshape(mu_dim*nk, nk*nk*nu_dim)
    UA, sA, VhA = np.linalg.svd(M1, full_matrices=False)
    r1 = trank(sA, eps)
    A3 = UA[:,:r1].reshape(mu_dim, nk, r1)
    M2 = (np.diag(sA[:r1]) @ VhA[:r1]).reshape(r1*nk, nk*nu_dim)
    UB, sB, VhB = np.linalg.svd(M2, full_matrices=False)
    r2 = trank(sB, eps)
    B3 = UB[:,:r2].reshape(r1, nk, r2)
    C3 = (np.diag(sB[:r2]) @ VhB[:r2]).reshape(r2, nk, nu_dim)

    def split2(core3d, n_left0, n_left1, n_right, eps_):
        # core3d: (n_left0, 4, n_right) -> split 4->(2,2) at (n_left0,high)|(low,n_right)
        U, s, Vh = np.linalg.svd(core3d.reshape(n_left0*2, 2*n_right), full_matrices=False)
        r = trank(s, eps_)
        L = U[:,:r].reshape(n_left0, 2, r)
        R = (np.diag(s[:r]) @ Vh[:r]).reshape(r, 2, n_right)
        return L, R

    A1, A2 = split2(A3, mu_dim, 2, r1, eps)   # A1[mu,2,sA], A2[sA,2,r1]
    B1, B2 = split2(B3, r1,     2, r2, eps)   # B1[r1,2,sB], B2[sB,2,r2]
    C1, C2 = split2(C3, r2,     2, nu_dim, eps)  # C1[r2,2,sC], C2[sC,2,nu]
    cores6 = [A1, A2, B1, B2, C1, C2]

    W_rec = np.zeros_like(W)
    bounds = []
    for q in range(Q):
        q1,q2,q3 = q//16, (q//4)%4, q%4
        a1,b1 = q1//2, q1%2;  a2,b2 = q2//2, q2%2;  a3,b3 = q3//2, q3%2
        M = A1[:,a1,:] @ A2[:,b1,:] @ B1[:,a2,:] @ B2[:,b2,:] @ C1[:,a3,:]
        W_rec[q] = M @ C2[:,b3,:]
        bounds.append(specnorm(M) * specnorm(C2[:,b3,:]))
    err = rel_error(W_rec)
    N_comp = sum(c.size for c in cores6)
    return err, N_comp, max(bounds) / W_op

# ── Run all methods ────────────────────────────────────────────────────────────
method_fns   = [method1, method2, method3, method4, method5, method6]
method_names = [
    'Per-$Q$ eigendecomp.',
    'Tucker / HOSVD',
    '3-site TT (base-4 $Q$)',
    'Binary QTT on $Q$ (MSB-first)',
    r'Binary MPO on $(\mu,\nu)$',
    'Digit-wise hierarchical QTT',
]

EPS_FIXED = 1e-2
TARGET    = 1e-2

print(f"\n=== Fixed eps={EPS_FIXED} ===", flush=True)
rows_fixed = []
for i, (name, fn) in enumerate(zip(method_names, method_fns)):
    print(f"  M{i+1} ...", end=' ', flush=True)
    err, Nc, ni = fn(EPS_FIXED)
    rows_fixed.append((err, Nc, ni))
    print(f"err={err:.3e}  ratio={Nc/N_orig:.3f}  norm_inc={ni:.4f}", flush=True)

def bisect(fn, target=TARGET, lo=1e-6, hi=EPS_FIXED, n_iter=30):
    """Largest eps s.t. fn(eps)[0] < target."""
    if fn(hi)[0] < target: return hi
    if fn(lo)[0] >= target: return lo
    for _ in range(n_iter):
        mid = np.exp(0.5*(np.log(lo)+np.log(hi)))
        if fn(mid)[0] < target: lo = mid
        else:                    hi = mid
        if hi/lo < 1.002: break
    return lo

print(f"\n=== Bisecting for actual error < {TARGET} ===", flush=True)
rows_tuned = []
for i, (name, fn) in enumerate(zip(method_names, method_fns)):
    print(f"  M{i+1} bisecting ...", end=' ', flush=True)
    eps_opt = bisect(fn)
    err, Nc, ni = fn(eps_opt)
    rows_tuned.append((eps_opt, err, Nc, ni))
    print(f"eps={eps_opt:.2e}  err={err:.3e}  ratio={Nc/N_orig:.3f}  norm_inc={ni:.4f}", flush=True)

# ── Build LaTeX table rows ─────────────────────────────────────────────────────
def fmt_row(name, eps_str, err, Nc, ni):
    ratio = Nc / N_orig
    comp  = (1 - ratio) * 100 if ratio < 1 else 0.0
    comp_s = f'{comp:.1f}\\%' if ratio < 1 else r'\textbf{none}'
    return (f'  {name} & {eps_str} & ${err:.2e}$ & '
            f'${ratio:.3f}$ ({comp_s}) & ${ni:.4f}$ \\\\\n')

trows_A = ''
for (name, (err, Nc, ni)) in zip(method_names, rows_fixed):
    trows_A += fmt_row(name, '$10^{-2}$', err, Nc, ni)

trows_B = ''
for (name, (eps_opt, err, Nc, ni)) in zip(method_names, rows_tuned):
    trows_B += fmt_row(name, f'${eps_opt:.1e}$'.replace('e-0','\\!\\times\\!10^{-').replace('e-','\\!\\times\\!10^{-') + ('}' if 'times' in f'${eps_opt:.1e}$' else ''), err, Nc, ni)

# Simpler eps formatting
trows_B = ''
for (name, (eps_opt, err, Nc, ni)) in zip(method_names, rows_tuned):
    exp = int(np.floor(np.log10(eps_opt)))
    man = eps_opt / 10**exp
    if abs(man - 1.0) < 0.05:
        eps_s = f'$10^{{{exp}}}$'
    else:
        eps_s = f'${man:.1f}\\times10^{{{exp}}}$'
    trows_B += fmt_row(name, eps_s, err, Nc, ni)

# Bond dims for descriptions
def get_bonds4(eps):
    n_bits = 6
    arr = W.transpose(1,0,2).reshape([mu_dim]+[2]*n_bits+[nu_dim])
    T = arr.copy(); bonds = []
    for k in range(n_bits - 1):
        nl = T.shape[0]*T.shape[1]; nr = T.size//nl
        U, s, Vh = np.linalg.svd(T.reshape(nl,nr), full_matrices=False)
        r = trank(s, eps); bonds.append(r)
        T = (np.diag(s[:r]) @ Vh[:r]).reshape([r]+list(T.shape[2:]))
    return bonds

def get_bonds5(eps):
    n_mb = 8; Qp = 256
    Wp = np.zeros((Q,Qp,Qp), dtype=W.dtype); Wp[:,:mu_dim,:nu_dim] = W
    perm = [0]+[j for k in range(n_mb) for j in (k+1,k+1+n_mb)]
    arr = Wp.reshape(Q,*([2]*n_mb),*([2]*n_mb)).transpose(perm).reshape(Q,*([4]*n_mb))
    T = arr.copy(); bonds = []
    for k in range(n_mb-1):
        nl = T.shape[0]*T.shape[1]; nr = T.size//nl
        U, s, Vh = np.linalg.svd(T.reshape(nl,nr), full_matrices=False)
        r = trank(s, eps); bonds.append(r)
        T = (np.diag(s[:r]) @ Vh[:r]).reshape([r]+list(T.shape[2:]))
    return bonds

bonds4 = get_bonds4(EPS_FIXED)
bonds5 = get_bonds5(EPS_FIXED)
bond4_str = ', '.join(str(b) for b in bonds4)
bond5_str = ', '.join(str(b) for b in bonds5)

# ── LaTeX document ─────────────────────────────────────────────────────────────
import datetime
today = datetime.date.today().isoformat()

latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb,booktabs,geometry,microtype,hyperref,array}
\geometry{margin=2.2cm}
\hypersetup{colorlinks,linkcolor=blue}
\title{\textbf{Tensor Decompositions of the ISDF Coulomb Kernel $W_{Q\mu\nu}$}}
\author{}
\date{""" + today + r"""}
\begin{document}
\maketitle\thispagestyle{empty}

\section*{Setup}
$W\!\in\!\mathbb{C}^{N_Q\times N_\mu\times N_\nu}$,\ $N_Q=64$,\ $N_\mu=N_\nu=130$
(diamond $4\!\times\!4\!\times\!4$ $k$-mesh, ISDF auxiliary basis).
$W[Q]\!\equiv\!W[Q,\cdot,\cdot]$ is a Hermitian matrix for each $Q$ (input $\nu$, output $\mu$).

\paragraph{Norm.}
Each method writes $\hat{W}[Q]=M_1^{(Q)}\cdots M_k^{(Q)}$ and the \emph{norm bound}
is $\prod_k\|M_k^{(Q)}\|_2$.  The \emph{norm increase} is
$\max_Q(\text{bound}_Q)/\|W\|_{\mathrm{op}}$,
$\|W\|_{\mathrm{op}}=\max_Q\|W[Q]\|_2=""" + f"{W_op:.4f}" + r"""$.
$N_{\mathrm{orig}}=N_QN_\mu N_\nu=1{,}081{,}600$.

%──────────────────────────────────────────────────────────────────────────────
\section{Per-$Q$ Eigendecomposition}
\begin{equation}
  W[Q]\approx U^{(Q)}\Lambda^{(Q)}\bigl(U^{(Q)}\bigr)^\dagger,\quad
  U^{(Q)}\!\in\!\mathbb{C}^{N_\mu\times r_Q},\;
  \Lambda^{(Q)}=\operatorname{diag}(\lambda_1^{(Q)},\ldots,\lambda_{r_Q}^{(Q)}).
\end{equation}
Retain eigenvalues with $|\lambda_k|/|\lambda_1|>\varepsilon$.
Storage: $r_Q(N_\mu+1)$ per $Q$, summed.
Norm bound $=|\lambda_1^{(Q)}|=\|W[Q]\|_2$ (\emph{tight}: $U^{(Q)}$ is an isometry).

%──────────────────────────────────────────────────────────────────────────────
\section{Tucker / HOSVD}
\begin{equation}
  W[Q,\mu,\nu]\approx\sum_{ijk}\mathcal{G}_{ijk}\,U_Q(Q,i)\,U_\mu(\mu,j)\,U_\nu(\nu,k),
  \qquad W[Q]\approx U_\mu\mathcal{G}_Q U_\nu^\dagger.
\end{equation}
$U_Q,U_\mu,U_\nu$ from mode-unfolding SVDs; $\mathcal{G}\!\in\!\mathbb{C}^{r_Q\times r_\mu\times r_\nu}$.
Storage: $r_Qr_\mu r_\nu+N_Qr_Q+N_\mu r_\mu+N_\nu r_\nu$.
Norm bound $=\|\mathcal{G}_Q\|_2=\|W[Q]\|_2$ (\emph{tight}: $U_\mu,U_\nu$ are isometries).

%──────────────────────────────────────────────────────────────────────────────
\section{3-Site Tensor Train (Base-4 $Q$)}
Write $Q=16q_1+4q_2+q_3$, $q_i\!\in\!\{0,1,2,3\}$.
\begin{equation}
  W[Q,\mu,\nu]\approx
  \sum_{\alpha_1\alpha_2}A[\mu,q_1,\alpha_1]\,B[\alpha_1,q_2,\alpha_2]\,C[\alpha_2,q_3,\nu],
\end{equation}
$A\!\in\!\mathbb{C}^{N_\mu\times4\times r_1}$, $B\!\in\!\mathbb{C}^{r_1\times4\times r_2}$,
$C\!\in\!\mathbb{C}^{r_2\times4\times N_\nu}$. For fixed $Q$:
$\hat{W}[Q]=A_{q_1}B_{q_2}C_{q_3}$.
Left-first norm bound: $\|A_{q_1}B_{q_2}\|_2\|C_{q_3}\|_2$.

%──────────────────────────────────────────────────────────────────────────────
\section{Binary QTT on $Q$ (MSB-First, Open $\mu/\nu$ Boundaries)}
Write $Q=\sum_{k=0}^5 b_k\,2^{5-k}$ (MSB first).
\begin{equation}
  W[Q,\mu,\nu]\approx
  \sum_{\alpha_0\cdots\alpha_4}
  A_0[\mu,b_0,\alpha_0]\,A_1[\alpha_0,b_1,\alpha_1]\cdots A_5[\alpha_4,b_5,\nu],
\end{equation}
$A_0\!\in\!\mathbb{C}^{N_\mu\times2\times r_0}$, $A_k\!\in\!\mathbb{C}^{r_{k-1}\times2\times r_k}$,
$A_5\!\in\!\mathbb{C}^{r_4\times2\times N_\nu}$.
Storage: $2(N_\mu r_0+\sum_{k=1}^4 r_{k-1}r_k+r_4 N_\nu)$.
Bond dims at $\varepsilon=10^{-2}$: $[""" + bond4_str + r"""]$.
In left-canonical form, $\|W[Q]\|_2=\|A_0^{(b_0)}\|_2$ (\emph{tight}).

%──────────────────────────────────────────────────────────────────────────────
\section{Binary MPO on $(\mu,\nu)$, $Q$ as Open Left Boundary}
Pad to $N_\mu'=N_\nu'=256=2^8$; encode $\mu,\nu$ as 8-bit integers.
Let $d_k=2b_k^\mu+b_k^\nu\in\{0,1,2,3\}$ (interleaved bit-pairs, $k=0,\ldots,7$).
\begin{equation}
  W[Q,\mu,\nu]\approx
  \sum_{\alpha_1\cdots\alpha_7}
  M_0[Q,d_0,\alpha_1]\,M_1[\alpha_1,d_1,\alpha_2]\cdots M_7[\alpha_7,d_7],
\end{equation}
$M_0\!\in\!\mathbb{C}^{N_Q\times4\times r_1}$, $M_k\!\in\!\mathbb{C}^{r_{k-1}\times4\times r_k}$.
Storage: $4(N_Qr_1+\sum_{k=1}^6 r_{k-1}r_k+r_6)$.
Bond dims at $\varepsilon=10^{-2}$: $[""" + bond5_str + r"""]$.
In left-canonical form, $\|W[Q]\|_2=\|M_0[Q,\cdot,\cdot]\|_2$ (\emph{tight}).

%──────────────────────────────────────────────────────────────────────────────
\section{Digit-Wise Hierarchical Binary QTT}
\emph{Two-level decomposition.}
\textbf{Step 1}: Decompose as the 3-site base-4 TT (Section~3), yielding cores
$A\!\in\!\mathbb{C}^{N_\mu\times4\times r_1}$, $B\!\in\!\mathbb{C}^{r_1\times4\times r_2}$,
$C\!\in\!\mathbb{C}^{r_2\times4\times N_\nu}$.
\textbf{Step 2}: For each site, split the 4-valued index into two binary indices
($q_i=2a_i+b_i$) via an additional truncated SVD:
$A\to A_1[\mu,a_1,s_A]\cdot A_2[s_A,b_1,r_1]$, and similarly for $B,C$.
The result is a 6-site binary MPS:
\begin{equation}
  \hat{W}[Q]=A_1^{(a_1)}\,A_2^{(b_1)}\,B_1^{(a_2)}\,B_2^{(b_2)}\,C_1^{(a_3)}\,C_2^{(b_3)},
\end{equation}
with bonds $(s_A,r_1,s_B,r_2,s_C)$.
Unlike the direct 6-site QTT (Section~4), within-digit bonds $(s_A,s_B,s_C)$ are bounded
by $2\min(r_1,r_2)$, while cross-digit bonds $(r_1,r_2)$ are inherited from Step~1.

%──────────────────────────────────────────────────────────────────────────────
\section*{Summary Table}

\begin{table}[h!]
\centering\small
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Trunc.\ $\varepsilon$} & \textbf{Actual error}
  & \textbf{Ratio} (saved) & \textbf{Norm increase} \\
\midrule
\multicolumn{5}{l}{\textit{(A) Fixed truncation $\varepsilon = 10^{-2}$}} \\[2pt]
""" + trows_A + r"""\midrule
\multicolumn{5}{l}{\textit{(B) Tuned $\varepsilon$ for actual error $< 10^{-2}$}} \\[2pt]
""" + trows_B + r"""\bottomrule
\end{tabular}
\caption{All six representations at two truncation levels.
\emph{Ratio}$=N_{\mathrm{comp}}/N_{\mathrm{orig}}$; \emph{saved}$=(1-\mathrm{ratio})\times100\%$.
\emph{Norm increase}$=\max_Q(\text{bound}_Q)/\|W\|_{\mathrm{op}}$
(1.000 = tight; $>1$ = loose bound on approximation norm).}
\end{table}
\end{document}
"""

# ── Compile LaTeX ──────────────────────────────────────────────────────────────
tex_path = '/tmp/coulomb_decomp_note.tex'
pdf_path = '/tmp/coulomb_decomp_note.pdf'
with open(tex_path, 'w') as f:
    f.write(latex)

print("\nCompiling LaTeX ...", flush=True)
for _ in range(2):
    r = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-output-directory=/tmp', tex_path],
        capture_output=True, text=True)
if not os.path.exists(pdf_path):
    print("pdflatex FAILED:\n", r.stdout[-3000:])
    sys.exit(1)

kb = os.path.getsize(pdf_path) // 1024
print(f"PDF saved: {pdf_path}  ({kb} KB)", flush=True)

ret = subprocess.run(
    f'echo "W tensor decomposition note — 6 methods" | '
    f'mail -s "Tensor decomp note (W, 6 methods)" -a {pdf_path} jchen9@caltech.edu',
    shell=True)
print(f"Email sent (exit={ret.returncode})")
print("Done.")
