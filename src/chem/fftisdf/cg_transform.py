"""Whole-space-group CG (symmetry-adapting) transform for a symmetry-adapted ISDF-THC chk.

Notation (matches symmetry.ipynb):
    i      irrep label index
    m_i    irrep dimension        (the "partner" index; size of the irrep)
    n_i    multiplicity           (how many copies of irrep i appear)

The whole space-group representation carried by an index of dimension  N = sum_i m_i * n_i
is reduced by a single unitary Q (the CG / symmetry-adapting transform):

    grid index (k, I)  , dim nkpts*nIP = 1088  ->  Q_grid  (1088 x 1088)
    AO   index (k, mu) , dim nkpts*nao = 208   ->  Q_AO    (208  x 208)

W has two grid indices, X has one grid index (rows) and one AO index (cols):

    Q_grid^dagger ( (+)_q W[q] ) Q_grid = (+)_i  ( 1_{m_i} (x) w_i ) ,  w_i in C^{n_i x n_i}
    Q_grid^dagger ( (+)_k X[k] ) Q_AO   = (+)_i  ( 1_{m_i} (x) x_i ) ,  x_i in C^{n_i^grid x n_i^AO}

i.e. W and X are the IDENTITY on the m_i (partner) index and act non-trivially only on the
multiplicity index.  This is Schur's lemma for the whole space group (the group is handled as a
whole -- point operations carry k-point translation phases -- with no translation-first /
little-group induction step).

Construction:
  1. average a random Hermitian operator over the group  -> invariant A (commutes with the rep);
  2. eigendecompose A: its degenerate eigenspaces are the individual irrep copies (dim m_i);
  3. merge eigenspaces with equal character vector into isotypic classes (this is what makes W come
     out block-diagonal: a generic invariant splits the n_i copies, W couples them);
  4. inside each class, align partners across the n_i copies with a second invariant B (for the grid)
     / with X^dagger (for the AO side) so that the reduced blocks are literally 1_{m_i} (x) w_i.

Run in jsun3's THC_general environment (needs pyscf space-group symmetry + torch):
    cd /central/groups/changroup/members/jsun3/xprize/THC_general
    python /path/to/cg_transform.py                     # diamond 2x2x2 c5
    python /path/to/cg_transform.py --save-transform    # also dump Q_grid, Q_AO
"""
import argparse
import os
import pickle

import h5py
import numpy as np
import torch

import libsymm

torch.manual_seed(0)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="/central/groups/changroup/members/jsun3/xprize/THC_general/data_diamond_gth-cc-pvdz_final")
    ap.add_argument("--klabel", default="2x2x2")
    ap.add_argument("--kmesh", type=int, nargs=3, default=(2, 2, 2))
    ap.add_argument("--cisdf", type=int, default=5)
    ap.add_argument("--tag", default="diamond_2x2x2_c5")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--save-transform", action="store_true")
    args = ap.parse_args()
    kmesh = tuple(args.kmesh)

    chk = os.path.join(args.data_dir, f"ISDFov_bareGDF_symm_{args.klabel}_c{args.cisdf}.chk")
    dft = os.path.join(args.data_dir, f"DFT_{args.klabel}_symm.pkl")
    with open(dft, "rb") as f:
        mf = pickle.load(f)
    with h5py.File(chk, "r") as f:
        X_ao = torch.tensor(f["inpv_kpt"][:])
        W = torch.tensor(f["coul_kpt"][:])
        mesh = np.asarray(f["mesh"], dtype=np.int64)
        ix_sel = np.asarray(f["ix_sel"], dtype=np.int64)

    cell = mf.cell
    kpts = cell.make_kpts(kmesh)
    cell_isdf = cell.copy(); cell_isdf.mesh = mesh
    coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)
    symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=torch.complex128)
    perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)
    kmap, U = symm.kmap, symm.U
    nk, nI, nao, nops = len(kpts), len(ix_sel), cell.nao_nr(), symm.nops
    phase_grid = phase.conj()
    tp = translation_phase(cell, kpts, kmesh)
    print(f"nk={nk} nI={nI} nao={nao} nops={nops}  grid dim={nk*nI}  AO dim={nk*nao}")

    def sym_mono(A):
        out = torch.zeros_like(A)
        for iop in range(perm.shape[0]):
            p, v = perm[iop], phase_grid[iop]
            Ag = v[:, :, None] * A * v.conj()[:, None, :]
            Ap = torch.zeros_like(Ag); Ap[:, p[:, None], p[None, :]] = Ag
            out.scatter_add_(0, kmap[:, iop].reshape(A.shape[0], 1, 1).expand_as(Ap), Ap)
        return out / perm.shape[0]

    def sym_uni(A):
        out = torch.zeros_like(A)
        for iop in range(U.shape[0]):
            Ag = torch.einsum("kpa,kab,kqb->kpq", U[iop], A, U[iop].conj())
            out.scatter_add_(0, kmap[:, iop].reshape(A.shape[0], 1, 1).expand_as(Ag), Ag)
        return out / U.shape[0]

    def rand_herm(n):
        A = torch.randn(nk, n, n, dtype=torch.float64) + 1j * torch.randn(nk, n, n, dtype=torch.float64)
        return (A + A.conj().transpose(-1, -2)) / 2

    # ---- copies (A eigenspaces) + isotypic classes, for grid and AO ----
    def decompose(kind):
        A = sym_mono(rand_herm(nI)) if kind == "grid" else sym_uni(rand_herm(nao))
        A = (A + A.conj().transpose(-1, -2)) / 2
        e, Qk = torch.linalg.eigh(A)
        es, Q = block_eigh_to_global(e, Qk)
        copies = degenerate_groups(es.numpy(), args.tol)              # each = one irrep copy (dim m_i)
        chars = (chars_monomial(kmap, perm, phase_grid, Q, copies, nk, nI, tp) if kind == "grid"
                 else chars_unitary(kmap, U, Q, copies, nk, nao, tp))
        classes = merge_by_character(copies, chars, args.tol * 10)     # each = isotypic component
        return Q, copies, chars, classes

    Qg0, cop_g, ch_g, cls_g = decompose("grid")
    Qa0, cop_a, ch_a, cls_a = decompose("ao")
    Bop = block_operator(sym_mono(rand_herm(nI)))                      # 2nd grid invariant for alignment
    Bop = (Bop + Bop.conj().T) / 2

    M = block_operator_from_q(W, nk, nI)                              # (+)_q W[q]
    Xop = block_operator_rect(X_ao, nk, nI, nao)                     # (+)_k X[k]

    # ---- grid: partner-consistent basis, a-major within each isotypic class ----
    Qg_cols, irrep_rows = [], []      # irrep_rows: (m_i, n_i, char, col_start, col_len)
    for cls in cls_g:
        mi = cop_g[cls[0]][1] - cop_g[cls[0]][0]
        ni = len(cls)
        ref = [Qg0[:, c] for c in range(cop_g[cls[0]][0], cop_g[cls[0]][1])]   # m_i ref partners
        aligned = [[None] * ni for _ in range(mi)]                             # [a][copy]
        for a in range(mi):
            va = Bop @ ref[a]
            for ci, gi in enumerate(cls):
                P = Qg0[:, cop_g[gi][0]:cop_g[gi][1]]
                v = P @ (P.conj().T @ va)
                aligned[a][ci] = v / torch.linalg.norm(v)
        start = len(Qg_cols)
        for a in range(mi):
            for ci in range(ni):
                Qg_cols.append(aligned[a][ci])
        irrep_rows.append((mi, ni, ch_g[cls[0]], start, mi * ni))
    Qg = torch.stack(Qg_cols, dim=1)

    # ---- AO: match to grid irreps by character; align AO partners to grid via X^dagger ----
    used = [False] * len(cls_a)
    Qa_cols, X_info = [], []          # X_info: (grid_irrep_index, m_i, n_grid, n_ao, grid_row_slice, ao_col_slice)
    for gidx, (mi, ni_g, cg, gstart, glen) in enumerate(irrep_rows):
        hit = None
        for aj, ca in enumerate(cls_a):
            if used[aj]:
                continue
            s = max(torch.linalg.norm(cg).item(), torch.linalg.norm(ch_a[cls_a[aj][0]]).item(), 1.0)
            if (torch.linalg.norm(cg - ch_a[cls_a[aj][0]]) / s).item() < 1e-6:
                hit = aj; break
        if hit is None:
            continue
        used[hit] = True
        cls = cls_a[hit]
        ni_a = len(cls)
        # grid reference partners for this irrep (first grid copy), a-major layout: col gstart + a*ni_g
        gref = [Qg[:, gstart + a * ni_g] for a in range(mi)]
        aligned = [[None] * ni_a for _ in range(mi)]
        for a in range(mi):
            va = Xop.conj().T @ gref[a]           # grid partner-a  ->  AO partner-a
            for ci, gi in enumerate(cls):
                P = Qa0[:, cop_a[gi][0]:cop_a[gi][1]]
                v = P @ (P.conj().T @ va)
                aligned[a][ci] = v / torch.linalg.norm(v)
        ao_start = len(Qa_cols)
        for a in range(mi):
            for ci in range(ni_a):
                Qa_cols.append(aligned[a][ci])
        X_info.append((gidx, mi, ni_g, ni_a, (gstart, gstart + glen),
                       (ao_start, ao_start + mi * ni_a)))
    Qa = torch.stack(Qa_cols, dim=1)

    # ---- transform + verify ----
    Wt = Qg.conj().T @ M @ Qg
    Xt = Qg.conj().T @ Xop @ Qa

    # block-diagonalization leakage
    rb = np.cumsum([0] + [glen for _, _, _, _, glen in irrep_rows])
    offW = torch.ones_like(Wt, dtype=torch.bool)
    for a in range(len(irrep_rows)):
        offW[rb[a]:rb[a + 1], rb[a]:rb[a + 1]] = False
    leakW = (Wt[offW].abs().max() / Wt.abs().max()).item()
    allowedX = torch.zeros_like(Xt, dtype=torch.bool)
    for (_, mi, ng, na, (r0, r1), (c0, c1)) in X_info:
        allowedX[r0:r1, c0:c1] = True
    leakX = (Xt[~allowedX].abs().max() / Xt.abs().max()).item()

    # identity-on-m_i:  1_{m_i} (x) w_i residual (partner-consistent, a-major)
    resW = kron_residual_W(Wt, irrep_rows)
    resX = kron_residual_X(Xt, X_info)

    print(f"W: {len(irrep_rows)} irrep blocks, off-block leak/|W| = {leakW:.1e}, "
          f"1_(m_i)(x)w_i residual = {resW:.1e}")
    print(f"X: {len(X_info)} coupled blocks, off-block leak/|X| = {leakX:.1e}, "
          f"1_(m_i)(x)x_i residual = {resX:.1e}")

    # ---- save block metadata (small) ----
    W_blocks = np.array([(mi, ni) for mi, ni, _c, _s, _l in irrep_rows], dtype=np.int64)   # (m_i, n_i)
    X_blocks = np.array([(mi, ng, na) for (_g, mi, ng, na, _r, _c) in X_info], dtype=np.int64)
    npz = os.path.join(OUT_DIR, f"cg_blocks_{args.tag}.npz")
    np.savez(npz, W_blocks=W_blocks, X_blocks=X_blocks,
             leakW=leakW, leakX=leakX, resW=resW, resX=resX,
             nk=nk, nI=nI, nao=nao, nops=nops)
    print("saved", npz)
    if args.save_transform:
        npz2 = os.path.join(OUT_DIR, f"cg_transform_{args.tag}.npz")
        np.savez(npz2, Q_grid=Qg.numpy(), Q_AO=Qa.numpy())
        print("saved", npz2, "(full CG change-of-basis matrices)")

    # coupled grid rows only, for the X figure
    coupled_rows = [r for (_g, mi, ng, na, (r0, r1), _c) in X_info for r in range(r0, r1)]
    make_block_figure(M.abs().numpy(), Wt.abs().numpy(), Xop.abs().numpy(),
                      Xt[coupled_rows, :].abs().numpy(), W_blocks, X_blocks, leakW, leakX)
    make_cg_matrix_figure(Qg.abs().numpy(), Qa.abs().numpy(), Wt.abs().numpy(),
                          W_blocks, nk, nI, nao)


# ----------------------------- helpers -----------------------------
def block_operator_from_q(W, nk, nI):
    M = torch.zeros(nk * nI, nk * nI, dtype=torch.complex128)
    for q in range(nk):
        M[q * nI:(q + 1) * nI, q * nI:(q + 1) * nI] = W[q]
    return M


def block_operator_rect(X_ao, nk, nI, nao):
    Xop = torch.zeros(nk * nI, nk * nao, dtype=torch.complex128)
    for k in range(nk):
        Xop[k * nI:(k + 1) * nI, k * nao:(k + 1) * nao] = X_ao[k]
    return Xop


def block_operator(A):
    nk, n, _ = A.shape
    M = torch.zeros(nk * n, nk * n, dtype=A.dtype)
    for k in range(nk):
        M[k * n:(k + 1) * n, k * n:(k + 1) * n] = A[k]
    return M


def block_eigh_to_global(e, Qk):
    nkpts, ninner = e.shape
    n = nkpts * ninner
    order = torch.argsort(e.reshape(n))
    Q = torch.zeros((n, n), dtype=Qk.dtype)
    for cn, co in enumerate(order):
        k, a = int(co // ninner), int(co % ninner)
        Q[k * ninner:(k + 1) * ninner, cn] = Qk[k, :, a]
    return e.reshape(n)[order], Q


def degenerate_groups(e, tol):
    g, i0 = [], 0
    while i0 < len(e):
        i1 = i0 + 1
        sc = max(abs(float(e[i0])), 1.0)
        while i1 < len(e) and abs(float(e[i1] - e[i0])) <= tol * sc:
            i1 += 1
        g.append((i0, i1)); i0 = i1
    return g


def translation_phase(cell, kpts, kmesh):
    ks = cell.get_scaled_kpts(kpts)
    ki = np.rint(ks * np.asarray(kmesh)[None]).astype(np.int64) % np.asarray(kmesh)[None]
    R = np.asarray(np.meshgrid(*[np.arange(n) for n in kmesh], indexing="ij")).reshape(3, -1).T
    kr = np.einsum("kd,td,d->tk", ki, R, 1.0 / np.asarray(kmesh))
    return torch.tensor(np.exp(2j * np.pi * kr))


def chars_monomial(kmap, perm, phase, Q, groups, nk, nI, tp):
    ch, ko = [], kmap.T.contiguous()
    for i0, i1 in groups:
        Qa = Q[:, i0:i1].reshape(nk, nI, i1 - i0); Qo = Qa[ko]
        idx = perm[:, None, :, None].expand(perm.shape[0], nk, nI, i1 - i0)
        Qr = torch.gather(Qo, 2, idx)
        Bin = torch.einsum("okia,oki,kia->ok", Qr.conj(), phase, Qa)
        B = torch.zeros_like(Bin); B.scatter_add_(1, ko, Bin)
        ch.append(torch.einsum("tk,ok->ot", tp, B).reshape(-1))
    return ch


def chars_unitary(kmap, U, Q, groups, nk, nao, tp):
    ch, ko = [], kmap.T.contiguous()
    for i0, i1 in groups:
        Qa = Q[:, i0:i1].reshape(nk, nao, i1 - i0); Qo = Qa[ko]
        Bin = torch.einsum("okpa,okpq,kqa->ok", Qo.conj(), U, Qa)
        B = torch.zeros_like(Bin); B.scatter_add_(1, ko, Bin)
        ch.append(torch.einsum("tk,ok->ot", tp, B).reshape(-1))
    return ch


def merge_by_character(groups, chars, tol):
    classes, used = [], [False] * len(groups)
    for i in range(len(groups)):
        if used[i]:
            continue
        used[i] = True; cls = [i]
        for j in range(i + 1, len(groups)):
            sc = max(torch.linalg.norm(chars[i]).item(), 1.0)
            if not used[j] and (torch.linalg.norm(chars[i] - chars[j]) / sc).item() < tol:
                used[j] = True; cls.append(j)
        classes.append(cls)
    return classes


def kron_residual_W(Wt, irrep_rows):
    """max_i || Wt_block_i - 1_{m_i} (x) w_i || / |Wt| with a-major column order."""
    res = 0.0
    for mi, ni, _c, s, _l in irrep_rows:
        blk = Wt[s:s + mi * ni, s:s + mi * ni].reshape(mi, ni, mi, ni)
        w = blk[0, :, 0, :]
        for a in range(mi):
            for ap in range(mi):
                t = blk[a, :, ap, :]
                d = (t - w).abs().max().item() if a == ap else t.abs().max().item()
                res = max(res, d)
    return res / (Wt.abs().max().item() + 1e-30)


def kron_residual_X(Xt, X_info):
    res = 0.0
    for (_g, mi, ng, na, (r0, r1), (c0, c1)) in X_info:
        blk = Xt[r0:r1, c0:c1].reshape(mi, ng, mi, na)
        x = blk[0, :, 0, :]
        for a in range(mi):
            for ap in range(mi):
                t = blk[a, :, ap, :]
                d = (t - x).abs().max().item() if a == ap else t.abs().max().item()
                res = max(res, d)
    return res / (Xt.abs().max().item() + 1e-30)


def _show(ax, Mabs, title, rlines=None, clines=None, minor_r=None, minor_c=None):
    vmax = np.percentile(Mabs[Mabs > 0], 99) if (Mabs > 0).any() else 1.0
    ax.imshow(Mabs, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
    for pos in ([] if minor_r is None else minor_r):
        ax.axhline(pos - 0.5, color="#bbbbbb", lw=0.3)
    for pos in ([] if minor_c is None else minor_c):
        ax.axvline(pos - 0.5, color="#bbbbbb", lw=0.3)
    for pos in ([] if rlines is None else rlines):
        ax.axhline(pos - 0.5, color="#d62728", lw=0.6)
    for pos in ([] if clines is None else clines):
        ax.axvline(pos - 0.5, color="#d62728", lw=0.6)
    ax.set_title(title, fontsize=10)


def make_block_figure(M_abs, Wt_abs, Xop_abs, Xt_abs, Wb, Xb, leakW, leakX):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    r_iso = np.cumsum([0] + list(Wb[:, 0] * Wb[:, 1]))           # isotypic block bounds
    r_tile = np.cumsum([0] + list(np.repeat(Wb[:, 1], Wb[:, 0])))  # 1_{m_i}(x)w_i tiles
    _show(ax[0, 0], M_abs, "|W| raw basis\n(block-diagonal in q only: 8×136)")
    _show(ax[0, 1], Wt_abs, "|W| in CG basis = ⊕_i 1_{m_i}⊗w_i\n(red: irrep i; grey: m_i identical w_i tiles)",
          rlines=r_iso[1:-1], clines=r_iso[1:-1], minor_r=r_tile, minor_c=r_tile)
    _show(ax[1, 0], Xop_abs, "|X| raw basis\n(block-diagonal in k: 8 blocks 136×26)")
    xr = np.cumsum([0] + list(Xb[:, 0] * Xb[:, 1]))
    xc = np.cumsum([0] + list(Xb[:, 0] * Xb[:, 2]))
    _show(ax[1, 1], Xt_abs, "|X| in CG basis = ⊕_i 1_{m_i}⊗x_i\n(grid-irrep ↔ AO-irrep)",
          rlines=xr[1:-1], clines=xc[1:-1])
    fig.suptitle(f"Whole space-group CG transform block-diagonalizes W and X  "
                 f"(diamond 2×2×2 c5; leak_W={leakW:.0e}, leak_X={leakX:.0e})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"symmetry_cg_blocks.{ext}"), dpi=140, bbox_inches="tight")
    print("saved symmetry_cg_blocks.{png,pdf}")


def make_cg_matrix_figure(Qg_abs, Qa_abs, Wt_abs, Wb, nk, nI, nao):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5.2))
    axg = fig.add_subplot(1, 3, 1); axa = fig.add_subplot(1, 3, 2); axz = fig.add_subplot(1, 3, 3)
    krow = [j * nI for j in range(1, nk)]
    r_iso = np.cumsum([0] + list(Wb[:, 0] * Wb[:, 1]))
    _show(axg, Qg_abs, "|Q_grid|  (1088×1088)\nrows: (k,I) [8 k-blocks]; cols: irrep-adapted",
          minor_r=krow, clines=r_iso[1:-1])
    axg.set_xlabel("irrep-adapted column"); axg.set_ylabel("grid index (k,I)")
    krowa = [j * nao for j in range(1, nk)]
    _show(axa, Qa_abs, "|Q_AO|  (208×208)\nrows: (k,µ) [8 k-blocks]; cols: irrep-adapted",
          minor_r=krowa)
    axa.set_xlabel("irrep-adapted column"); axa.set_ylabel("AO index (k,µ)")
    # zoom: the largest isotypic block of W, showing 1_{m_i}(x)w_i
    sizes = Wb[:, 0] * Wb[:, 1]
    b = int(np.argmax(sizes)); s0 = int(r_iso[b]); s1 = int(r_iso[b + 1])
    mi, ni = int(Wb[b, 0]), int(Wb[b, 1])
    tiles = [s0 + t * ni for t in range(mi + 1)]
    _show(axz, Wt_abs[s0:s1, s0:s1] if False else Wt_abs[s0:s1, s0:s1],
          f"|W| zoom: one irrep block (m_i={mi}, n_i={ni})\n= 1_{{{mi}}} ⊗ w_i  ({mi} identical {ni}×{ni} tiles)",
          rlines=[t - s0 for t in tiles[1:-1]], clines=[t - s0 for t in tiles[1:-1]])
    fig.suptitle("The CG (symmetry-adapting) transformation Q, and the 1_{m_i}⊗w_i block form", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"symmetry_cg_matrix.{ext}"), dpi=140, bbox_inches="tight")
    print("saved symmetry_cg_matrix.{png,pdf}")


if __name__ == "__main__":
    main()
