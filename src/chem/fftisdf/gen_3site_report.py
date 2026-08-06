"""
3-site QTT / rank analysis of W[Q, mu, nu].

Q = nk^3 is split into its 3 mesh dimensions: Q -> (q1, q2, q3) with qi in {0,..,nk-1}.
We investigate:
  A. Bipartition SVD spectra for all sequential cuts in the chain
       mu -- q1 -- q2 -- q3 -- nu
  B. 3-site open-boundary MPS (mu left, nu right) at multiple eps
  C. Per-(mu,nu): 3-site MPS of each W[:,mu,nu] reshaped to (nk,nk,nk)
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

# ── Load data ─────────────────────────────────────────────────────────────────
with open('data/SCF_diamond_4x4x4_gth-dzvp_ke40.0.pkl', 'rb') as f:
    mf = pickle.load(f)
with h5py.File('data/ISDFopt_diamond_4x4x4_gth-dzvp_c5_norm0.1.chk', 'r') as f:
    W = np.asarray(f['coul_kpt'])

Q, mu_dim, nu_dim = W.shape
nk   = round(Q ** (1/3))                         # 4
assert nk**3 == Q, f"Q={Q} is not a perfect cube"
print(f"W.shape={W.shape}, nk={nk}, Q={Q}={nk}^3", flush=True)

# reshape W to (q1, q2, q3, mu, nu)
W5 = W.reshape(nk, nk, nk, mu_dim, nu_dim)       # (4,4,4,130,130)

EPSILONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]

# ──────────────────────────────────────────────────────────────────────────────
# A. Bipartition SVD spectra for the chain  mu | q1 | q2 | q3 | nu
# ──────────────────────────────────────────────────────────────────────────────
# We transpose to (mu, q1, q2, q3, nu) and matricize at each cut.
print("A. Bipartition SVD analysis ...", flush=True)
W_ord = W5.transpose(3, 0, 1, 2, 4)              # (mu, q1, q2, q3, nu)
dims  = [mu_dim, nk, nk, nk, nu_dim]             # [130, 4, 4, 4, 130]
labels = ['mu', 'q1', 'q2', 'q3', 'nu']

# cuts: left indices | right indices
cuts = [
    ([0],          [1,2,3,4]),   # mu         | q1 q2 q3 nu
    ([0,1],        [2,3,4]),     # mu q1       | q2 q3 nu
    ([0,1,2],      [3,4]),       # mu q1 q2    | q3 nu
    ([0,1,2,3],    [4]),         # mu q1 q2 q3 | nu
]
svd_spectra = {}   # cut_name -> singular values
for left_idx, right_idx in cuts:
    left_dims  = [dims[i] for i in left_idx]
    right_dims = [dims[i] for i in right_idx]
    mat = W_ord.reshape(np.prod(left_dims), np.prod(right_dims))
    s   = np.linalg.svd(mat, compute_uv=False)
    name = '|'.join(labels[i] for i in left_idx) + ' | ' + '|'.join(labels[i] for i in right_idx)
    svd_spectra[name] = s
    ranks = {f'e{int(np.log10(e))}': int(np.sum(s/s[0] > e)) for e in EPSILONS}
    print(f"  {name}: max_rank={len(s)}, ranks={ranks}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# B. 3-site open-boundary MPS  A[mu,q1,r1] -- B[r1,q2,r2] -- C[r2,q3,nu]
# ──────────────────────────────────────────────────────────────────────────────
print("B. 3-site open-boundary MPS ...", flush=True)
arr = W_ord   # (mu, q1, q2, q3, nu)
expB = {}
for eps in EPSILONS:
    T = qtn.Tensor(arr, inds=['mu', 'q1', 'q2', 'q3', 'nu'])
    T_left, T = T.split(['mu', 'q1'], cutoff=eps, cutoff_mode='rel', bond_ind='r1')
    T_mid,  T = T.split(['r1', 'q2'], cutoff=eps, cutoff_mode='rel', bond_ind='r2')
    cores = [T_left, T_mid, T]
    bd    = [T_left.ind_size('r1'), T_mid.ind_size('r2')]
    # reconstruct
    res = T_left.data
    for c in [T_mid, T]:
        res = np.tensordot(res, c.data, axes=([-1], [0]))
    W_rec = res.reshape(mu_dim, nk, nk, nk, nu_dim).transpose(1,2,3,0,4).reshape(Q, mu_dim, nu_dim)
    err   = np.linalg.norm(W - W_rec) / np.linalg.norm(W)
    expB[eps] = (bd, err)
    print(f"  eps={eps:.0e}: bond_dims={bd}  err={err:.2e}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# C. Per-(mu,nu): 3-site MPS of W[:,mu,nu].reshape(nk,nk,nk)
# ──────────────────────────────────────────────────────────────────────────────
print("C. Per-(mu,nu) 3-site MPS ...", flush=True)
expC = {}
for eps in EPSILONS:
    all_bd = np.zeros((mu_dim, nu_dim, 2), dtype=int)
    for mu_i in range(mu_dim):
        for nu_i in range(nu_dim):
            v   = W[:, mu_i, nu_i].reshape(nk, nk, nk)  # (4,4,4)
            mps = qtn.MatrixProductState.from_dense(v.ravel(), dims=[nk]*3)
            mps.compress(cutoff=eps, cutoff_mode='rel')
            all_bd[mu_i, nu_i] = [mps.bond_size(i, i+1) for i in range(2)]
    expC[eps] = all_bd
    means = [f"{all_bd[:,:,b].mean():.2f}" for b in range(2)]
    maxs  = [all_bd[:,:,b].max() for b in range(2)]
    print(f"  eps={eps:.0e}: mean={means}  max={maxs}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────────────────────
print("Generating PDF ...", flush=True)
pdf_path = '/tmp/qtt_3site_report.pdf'
max_poss_B = [min(mu_dim*nk, nk*nk*nu_dim), min(mu_dim*nk*nk, nk*nu_dim)]  # [520, 520]
max_poss_C = [min(nk, nk*nk), min(nk*nk, nk)]                               # [4, 4]

def dark_header(t, ncols):
    for j in range(ncols):
        t[0, j].set_facecolor('#1a252f')
        t[0, j].set_text_props(color='white', fontweight='bold')

with PdfPages(pdf_path) as pdf:

    # ── Page 1: Title & overview ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.5, 0.97, "3-Site QTT Rank Report", ha='center', va='top',
            fontsize=26, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.91, "ISDF Coulomb Kernel W  —  Diamond 4×4×4  (Q = q1 × q2 × q3 = 4³)",
            ha='center', va='top', fontsize=13, transform=ax.transAxes)
    ax.plot([0,1],[0.87,0.87], color='#333', lw=1.5, transform=ax.transAxes)
    meta = [
        ("Date",        datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Host",        socket.gethostname()),
        ("W shape",     f"{W.shape}  →  reshaped to {W5.shape}"),
        ("Chain order", "mu  —  q1  —  q2  —  q3  —  nu"),
        ("eps tested",  "  ".join(f"1e{int(np.log10(e))}" for e in EPSILONS)),
    ]
    y = 0.83
    for k, v in meta:
        ax.text(0.07, y, f"{k}:", ha='left', va='top', fontsize=11,
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.30, y, v, ha='left', va='top', fontsize=11, transform=ax.transAxes)
        y -= 0.04
    ax.plot([0,1],[y-0.01,y-0.01], color='#aaa', lw=0.8, ls='--', transform=ax.transAxes)
    summary = (
        "Experiments\n\n"
        "A. Bipartition SVD spectra for all sequential cuts in the chain\n"
        "     mu | q1 q2 q3 nu\n"
        "     mu q1 | q2 q3 nu\n"
        "     mu q1 q2 | q3 nu\n"
        "     mu q1 q2 q3 | nu\n"
        "   Max ranks: 130, 520, 520, 130\n\n"
        "B. 3-site open-boundary MPS  A[mu,q1,r1] -- B[r1,q2,r2] -- C[r2,q3,nu]\n"
        "   Max bond dims: r1=520, r2=520\n\n"
        "C. Per-(mu,nu): 3-site MPS of each W[:,mu,nu].reshape(4,4,4)\n"
        "   Max bond dims: r1=4, r2=4  (all 16,900 pairs)"
    )
    ax.text(0.07, y - 0.04, summary, ha='left', va='top', fontsize=10,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='#f5f5f5', ec='#bbb'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 2: SVD spectra ───────────────────────────────────────────────────
    cut_names = list(svd_spectra.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 9))
    fig.suptitle("A. Bipartition SVD spectra  (normalised to largest singular value)",
                 fontsize=12, fontweight='bold')
    for ax, name in zip(axes.flat, cut_names):
        s = svd_spectra[name] / svd_spectra[name][0]
        ax.semilogy(np.arange(1, len(s)+1), s, 'o-', ms=4, lw=1.5)
        for eps in [1e-2, 1e-3]:
            r = int(np.sum(s > eps))
            ax.axhline(eps, color='gray', lw=0.7, ls='--')
            ax.text(len(s)*0.98, eps*1.5, f'eps=1e{int(np.log10(eps))}: r={r}',
                    ha='right', fontsize=7, color='gray')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Singular value index', fontsize=8)
        ax.set_ylabel('Normalised singular value', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 3: SVD rank table ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 6)); ax.axis('off')
    fig.suptitle("A. Numerical rank at each cut for each eps",
                 fontsize=12, fontweight='bold', y=0.97)
    col_labels = ['Cut', 'Max rank'] + [f'eps=1e{int(np.log10(e))}' for e in EPSILONS]
    rows = []
    for name, s in svd_spectra.items():
        s_norm = s / s[0]
        row = [name, str(len(s))]
        for eps in EPSILONS:
            row.append(str(int(np.sum(s_norm > eps))))
        rows.append(row)
    t = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 2.2)
    dark_header(t, len(col_labels))
    for i, (name, s) in enumerate(svd_spectra.items()):
        s_norm = s / s[0]
        for j, eps in enumerate(EPSILONS, start=2):
            r   = int(np.sum(s_norm > eps))
            mp  = len(s)
            frac = r / mp
            t[i+1, j].set_facecolor(plt.cm.RdYlGn(1.0 - frac))
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 4: Experiment B table + plot ─────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 9))
    fig.suptitle("B. 3-site Open-boundary MPS  A[mu,q1,r1]–B[r1,q2,r2]–C[r2,q3,nu]",
                 fontsize=12, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, top=0.93, bottom=0.06)

    ax0 = fig.add_subplot(gs[0]); ax0.axis('off')
    col_labelsB = ['eps', f'r1 (max={max_poss_B[0]})', f'r2 (max={max_poss_B[1]})', 'rel err']
    rowsB = []
    for eps in EPSILONS:
        bd, err = expB[eps]
        rowsB.append([f'{eps:.0e}', str(bd[0]), str(bd[1]), f'{err:.2e}'])
    tB = ax0.table(cellText=rowsB, colLabels=col_labelsB, loc='center', cellLoc='center')
    tB.auto_set_font_size(False); tB.set_fontsize(10); tB.scale(1, 2.2)
    dark_header(tB, len(col_labelsB))
    for i, eps in enumerate(EPSILONS):
        bd, _ = expB[eps]
        for j, (d, mp) in enumerate(zip(bd, max_poss_B), start=1):
            tB[i+1, j].set_facecolor(plt.cm.RdYlGn(1.0 - d/mp))

    ax1 = fig.add_subplot(gs[1])
    xlabels = [f'{e:.0e}' for e in EPSILONS]
    for b, mp in enumerate(max_poss_B):
        vals = [expB[eps][0][b] for eps in EPSILONS]
        ax1.plot(xlabels, vals, 'o-', label=f'r{b+1} (max={mp})')
        ax1.axhline(mp, ls='--', lw=0.6, color='gray')
    ax1.set_xlabel('eps'); ax1.set_ylabel('Bond dimension')
    ax1.set_title('Bond dims vs eps'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 5: Experiment C table + mean-bond plot ───────────────────────────
    fig = plt.figure(figsize=(8.5, 9))
    fig.suptitle("C. Per-(mu,nu) 3-site MPS of W[:,mu,nu].reshape(4,4,4)",
                 fontsize=12, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, top=0.93, bottom=0.06)

    ax0 = fig.add_subplot(gs[0]); ax0.axis('off')
    col_labelsC = ['eps', f'r1 (max={max_poss_C[0]})', f'r2 (max={max_poss_C[1]})',
                   'r1 mean', 'r2 mean', 'r1 % at max', 'r2 % at max']
    rowsC = []
    for eps in EPSILONS:
        all_bd = expC[eps]
        row = [f'{eps:.0e}']
        for b in range(2):
            row.append(str(all_bd[:,:,b].max()))
        for b in range(2):
            row.append(f'{all_bd[:,:,b].mean():.2f}')
        for b in range(2):
            mp  = max_poss_C[b]
            pct = 100*(all_bd[:,:,b] == mp).mean()
            row.append(f'{pct:.1f}%')
        rowsC.append(row)
    tC = ax0.table(cellText=rowsC, colLabels=col_labelsC, loc='center', cellLoc='center')
    tC.auto_set_font_size(False); tC.set_fontsize(9); tC.scale(1, 2.2)
    dark_header(tC, len(col_labelsC))

    ax1 = fig.add_subplot(gs[1])
    for b, mp in enumerate(max_poss_C):
        means = [expC[eps][:,:,b].mean() for eps in EPSILONS]
        ax1.plot(xlabels, means, 'o-', label=f'r{b+1} mean (max={mp})')
        ax1.axhline(mp, ls='--', lw=0.6, color='gray')
    ax1.set_xlabel('eps'); ax1.set_ylabel('Mean bond dim over (mu,nu)')
    ax1.set_title('Mean per-(mu,nu) bond dims vs eps')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 6: Heatmaps of per-(mu,nu) bond dims at eps=1e-2 ────────────────
    for eps in [1e-2, 1e-1]:
        all_bd = expC[eps]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
        fig.suptitle(f"C. Per-(mu,nu) bond-dim heatmaps   eps={eps:.0e}",
                     fontsize=12, fontweight='bold')
        for b, ax in enumerate(axes):
            mp = max_poss_C[b]
            im = ax.imshow(all_bd[:,:,b], vmin=1, vmax=mp, cmap='viridis', aspect='auto')
            ax.set_title(f'r{b+1}  (max={mp})', fontsize=10)
            ax.set_xlabel('nu', fontsize=9); ax.set_ylabel('mu', fontsize=9)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 7: Comparison 3-site vs 6-site (binary) QTT ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5))
    fig.suptitle("Comparison: 3-site (base-4) vs 6-site (binary) QTT",
                 fontsize=12, fontweight='bold', y=1.01)

    # open-boundary comparison
    ax = axes[0]
    # 6-site binary results from previous run (hard-coded from earlier output)
    bd6_by_eps = {
        1e-1: [41, 46, 67, 67, 84], 1e-2: [176, 247, 446, 470, 260],
        1e-3: [258, 483, 919, 520, 260], 1e-4: [260, 520, 1024, 520, 260],
        1e-6: [260, 520, 1040, 520, 260],
    }
    ax.plot(xlabels, [max(bd6_by_eps[e]) for e in EPSILONS], 's--',
            color='#c0392b', label='6-site binary (max bond)')
    ax.plot(xlabels, [max(expB[e][0]) for e in EPSILONS], 'o-',
            color='#2980b9', label='3-site base-4 (max bond)')
    ax.set_xlabel('eps'); ax.set_ylabel('Max bond dimension')
    ax.set_title('Open-boundary MPS\nmax bond dim vs eps')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # per-(mu,nu) comparison
    ax = axes[1]
    bd6_per = {  # max possible for 6-site binary: 2,4,8,8,4,2 → central=8
        e: 8 for e in EPSILONS   # from previous run all bonds were at max
    }
    ax.plot(xlabels, [expC[e][:,:,0].mean() for e in EPSILONS], 'o-',
            color='#27ae60', label='3-site r1 mean (max=4)')
    ax.plot(xlabels, [expC[e][:,:,1].mean() for e in EPSILONS], 's-',
            color='#8e44ad', label='3-site r2 mean (max=4)')
    ax.axhline(4, ls='--', lw=0.7, color='#27ae60', alpha=0.5)
    ax.axhline(8, ls='--', lw=0.7, color='#c0392b', alpha=0.5)
    ax.text(xlabels[-1], 4.1, '3-site max', ha='right', fontsize=7, color='#27ae60')
    ax.text(xlabels[-1], 8.2, '6-site max', ha='right', fontsize=7, color='#c0392b')
    ax.set_xlabel('eps'); ax.set_ylabel('Mean bond dim over (mu,nu)')
    ax.set_title('Per-(mu,nu) MPS\nmean bond dim vs eps')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 8: Interpretation ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    ax.text(0.5, 0.97, "Interpretation", ha='center', va='top',
            fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.plot([0,1],[0.93,0.93], color='#333', lw=1.5, transform=ax.transAxes)
    txt = (
        "BIPARTITION SVD (Exp A)\n"
        "  - Cut  mu | q1 q2 q3 nu  (max rank 130): reveals how many\n"
        "    independent mu-components couple to the full Q space.\n"
        "  - Cut  mu q1 | q2 q3 nu  (max rank 520): rank between the\n"
        "    first k-mesh dimension combined with mu vs the rest.\n"
        "  - Cut  mu q1 q2 | q3 nu  (max rank 520): symmetric.\n"
        "  - Cut  mu q1 q2 q3 | nu  (max rank 130): same as first by symmetry.\n"
        "  Fast singular value decay at a cut -> that cut can be compressed.\n\n"
        "3-SITE OPEN-BOUNDARY MPS (Exp B)\n"
        "  Chain: A[mu,q1,r1] -- B[r1,q2,r2] -- C[r2,q3,nu]\n"
        "  Max possible bond dims: r1=r2=520 (much smaller than 6-site binary: 1040).\n"
        "  If the 3-site bond dims are significantly below 520, the base-4\n"
        "  factorisation captures more structure than binary.\n\n"
        "PER-(mu,nu) 3-SITE MPS (Exp C)\n"
        "  Each W[:,mu,nu] is a 4x4x4 tensor with max bond dims r1=r2=4.\n"
        "  Compare to 6-site binary: max bond dims were all at ceiling (2,4,8,4,2).\n"
        "  The 3-site base-4 representation has ceiling 4 vs 8 for binary -- so\n"
        "  even at full rank the base-4 MPS is cheaper by 2x at the central bond.\n"
        "  If the actual bond dims drop below 4, additional savings are achieved.\n\n"
        "PHYSICAL INTERPRETATION\n"
        "  The k-mesh is a 4x4x4 Monkhorst-Pack grid. Splitting Q into (q1,q2,q3)\n"
        "  respects the crystal symmetry (each qi is a direction in reciprocal space).\n"
        "  Low bond dimensions at these cuts would mean the Coulomb interaction\n"
        "  factorises approximately across the three reciprocal lattice directions --\n"
        "  physically plausible if the crystal is nearly isotropic and the BZ is\n"
        "  approximately spherical. The 1/|q|^2 singularity is the dominant feature;\n"
        "  whether it factorises across the 3 directions depends on the metric."
    )
    ax.text(0.04, 0.90, txt, ha='left', va='top', fontsize=9.5,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', fc='#fafafa', ec='#ccc'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    d = pdf.infodict()
    d['Title']   = '3-site QTT Rank Report — ISDF W tensor'
    d['Author']  = 'jchen9@caltech.edu'

print(f"PDF saved: {pdf_path}  ({os.path.getsize(pdf_path)//1024} KB)", flush=True)

# ── Email ─────────────────────────────────────────────────────────────────────
body = (
    "Hi,\n\n3-site QTT rank analysis report attached (PDF).\n\n"
    "8-page PDF covering:\n"
    "  p1: Overview & metadata\n"
    "  p2: Bipartition SVD spectra (4 cuts, log-scale plots)\n"
    "  p3: Numerical rank table at each cut for each eps\n"
    "  p4: 3-site open-boundary MPS bond dims (table + plot)\n"
    "  p5: Per-(mu,nu) 3-site MPS stats (table + mean-bond plot)\n"
    "  p6-7: Per-(mu,nu) bond-dim heatmaps at eps=1e-2 and 1e-1\n"
    "  p8: Comparison 3-site base-4 vs 6-site binary QTT\n"
    "  p9: Interpretation\n\n"
    "Best,\nClaude\n"
)
res = subprocess.run(
    ['mail', '-s', '3-site QTT Rank Report — ISDF W diamond 4x4x4',
     '-a', pdf_path, 'jchen9@caltech.edu'],
    input=body, text=True, capture_output=True
)
if res.returncode != 0:
    os.system(f'uuencode {pdf_path} qtt_3site_report.pdf | '
              f'mail -s "3-site QTT Rank Report" jchen9@caltech.edu')
    print("Sent via uuencode fallback", flush=True)
else:
    print(f"Email sent (exit={res.returncode})", flush=True)
print("Done.", flush=True)
