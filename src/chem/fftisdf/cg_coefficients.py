"""Concrete Clebsch-Gordan coefficients of the diamond (2x2x2) whole space group.

Two things are computed from the symmetry-adapted ISDF-THC chk:

  1. the gauge-invariant FUSION RULES  N^c_{ab}:  irrep_a (x) irrep_b = (+)_c N^c_{ab} irrep_c
     (needs only the characters -> `cg_fusion_diamond_2x2x2.txt`);

  2. an explicit CG coefficient matrix for one chosen multiplicity-free triple
     (default: the first 2-dim irrep squared, r5 (x) r5 = r1 (+) r5 (+) r18), obtained by
     extracting the irrep matrices D_lambda(g) from the partner-consistent CG basis and reducing
     the product representation D_a(g) (x) D_a(g).

Verified: |G| = 384, character orthonormality ~1e-15, D_lambda unitary + character-matching +
great-orthogonality ~1e-15, and V^dagger (D_a (x) D_a) V = (+)_c D_c to ~1e-14.

The CG numbers are only defined once a basis is fixed inside each irrep; here that basis comes from
a particular (arbitrary but self-consistent) numerical construction, and each CG column is then
phase-fixed so its first significant entry is real positive.  The gauge-INVARIANT content is the
fusion table and the |coefficient| pattern.

Run in jsun3's THC_general environment (pyscf space-group symmetry + torch):
    cd /central/groups/changroup/members/jsun3/xprize/THC_general
    python <this_folder>/cg_coefficients.py
"""
import os
import pickle

import h5py
import numpy as np
import torch

import libsymm

torch.manual_seed(0)
np.set_printoptions(precision=4, suppress=True, linewidth=140)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/central/groups/changroup/members/jsun3/xprize/THC_general/data_diamond_gth-cc-pvdz_final"
KM = (2, 2, 2)


def build():
    mf = pickle.load(open(os.path.join(DATA_DIR, "DFT_2x2x2_symm.pkl"), "rb"))
    with h5py.File(os.path.join(DATA_DIR, "ISDFov_bareGDF_symm_2x2x2_c5.chk"), "r") as f:
        mesh = np.asarray(f["mesh"], dtype=np.int64)
        ix_sel = np.asarray(f["ix_sel"], dtype=np.int64)
    cell = mf.cell
    kpts = cell.make_kpts(KM)
    cell_isdf = cell.copy(); cell_isdf.mesh = mesh
    coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)
    symm = libsymm.PBCSymmetry(cell_isdf, KM, kpts, dtype=torch.complex128)
    perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)
    return cell, kpts, symm, perm.numpy(), phase.conj().resolve_conj().numpy(), symm.kmap.numpy()


def main():
    cell, kpts, symm, perm, phase_grid, kmap = build()
    nk, nI, nops = len(kpts), perm.shape[1], symm.nops
    Ngrid = nk * nI
    ks = cell.get_scaled_kpts(kpts)
    ki = np.rint(ks * np.asarray(KM)[None]).astype(np.int64) % np.asarray(KM)[None]
    Rt = np.asarray(np.meshgrid(*[np.arange(n) for n in KM], indexing="ij")).reshape(3, -1).T
    tp = np.exp(2j * np.pi * np.einsum("kd,td,d->tk", ki, Rt, 1.0 / np.asarray(KM)))   # (nk_trans, nk)

    pg = torch.from_numpy(phase_grid); pr = torch.from_numpy(perm); km = torch.from_numpy(kmap); TP = torch.from_numpy(tp)

    def sym_mono(A):
        o = torch.zeros_like(A)
        for iop in range(nops):
            p, v = pr[iop], pg[iop]
            Ag = v[:, :, None] * A * v.conj()[:, None, :]
            Ap = torch.zeros_like(Ag); Ap[:, p[:, None], p[None, :]] = Ag
            o.scatter_add_(0, km[:, iop].reshape(A.shape[0], 1, 1).expand_as(Ap), Ap)
        return o / nops

    def rand_inv():
        A = torch.randn(nk, nI, nI, dtype=torch.float64) + 1j * torch.randn(nk, nI, nI, dtype=torch.float64)
        A = sym_mono((A + A.conj().transpose(-1, -2)) / 2)
        return (A + A.conj().transpose(-1, -2)) / 2

    def beg(e, Qk):
        ni = e.shape[1]; order = torch.argsort(e.reshape(-1))
        Q = torch.zeros((Ngrid, Ngrid), dtype=Qk.dtype)
        for cn, co in enumerate(order):
            k, a = int(co // ni), int(co % ni)
            Q[k * ni:(k + 1) * ni, cn] = Qk[k, :, a]
        return e.reshape(-1)[order], Q

    def dgroups(e, tol=1e-8):
        g, i0 = [], 0
        while i0 < len(e):
            i1 = i0 + 1; sc = max(abs(float(e[i0])), 1.0)
            while i1 < len(e) and abs(float(e[i1] - e[i0])) <= tol * sc:
                i1 += 1
            g.append((i0, i1)); i0 = i1
        return g

    def chars(Q, groups):
        ch, ko = [], km.T.contiguous()
        for i0, i1 in groups:
            Qa = Q[:, i0:i1].reshape(nk, nI, i1 - i0); Qo = Qa[ko]
            idx = pr[:, None, :, None].expand(nops, nk, nI, i1 - i0); Qr = torch.gather(Qo, 2, idx)
            Bin = torch.einsum("okia,oki,kia->ok", Qr.conj(), pg, Qa)
            B = torch.zeros_like(Bin); B.scatter_add_(1, ko, Bin)
            ch.append(torch.einsum("tk,ok->ot", TP, B).reshape(-1))
        return ch

    # ---- irreps + characters ----
    A = rand_inv()
    e, Qk = torch.linalg.eigh(A); es, Q = beg(e, Qk)
    groups = dgroups(es.numpy()); ch = chars(Q, groups)
    used, classes = [False] * len(groups), []
    for a in range(len(groups)):
        if used[a]:
            continue
        used[a] = True; cls = [a]
        for b in range(a + 1, len(groups)):
            if not used[b] and (torch.linalg.norm(ch[a] - ch[b]) / max(torch.linalg.norm(ch[a]).item(), 1.0)).item() < 1e-7:
                used[b] = True; cls.append(b)
        classes.append(cls)
    dim = [groups[c[0]][1] - groups[c[0]][0] for c in classes]
    mult = [len(c) for c in classes]
    chi = [ch[c[0]].numpy() for c in classes]
    nirr = len(classes)
    G = float(sum(abs(chi[[i for i in range(nirr) if dim[i] == 1][0]]) ** 2).real)
    ortho = max(abs((chi[i].conj() * chi[j]).sum() / G - (1 if i == j else 0)) for i in range(nirr) for j in range(nirr))

    Nf = np.rint(np.array([[[(chi[a] * chi[b] * chi[c].conj()).sum().real / G
                             for c in range(nirr)] for b in range(nirr)] for a in range(nirr)]))
    print(f"|G| = {G:.0f} ,  #irreps = {nirr} ,  char-orthonormality err = {ortho:.1e}")
    print("irrep dims  m_i:", dim)
    print("multipl.    n_i:", mult)

    # ---- write the full fusion table ----
    lines = ["# Fusion rules  r_a (x) r_b = (+)_c N^c_ab r_c   for the diamond 2x2x2 whole space group",
             f"# |G| = {G:.0f};  irrep i has (dim m_i, mult n_i) below",
             "# i : (m_i, n_i)"]
    for i in range(nirr):
        lines.append(f"#   r{i:<2d}: (m={dim[i]}, n={mult[i]})")
    lines.append("")
    for a in range(nirr):
        for b in range(a, nirr):
            terms = [(int(Nf[a, b, c]), c) for c in range(nirr) if Nf[a, b, c] > 0]
            rhs = " + ".join((f"{n}*" if n > 1 else "") + f"r{c}" for n, c in terms)
            lines.append(f"r{a}(d{dim[a]}) (x) r{b}(d{dim[b]}) = {rhs}")
    open(os.path.join(OUT_DIR, "cg_fusion_diamond_2x2x2.txt"), "w").write("\n".join(lines) + "\n")
    print("wrote cg_fusion_diamond_2x2x2.txt")

    # ---- partner-consistent basis + irrep matrices D_lambda(g) ----
    Bop_k = rand_inv()
    Bop = torch.zeros(Ngrid, Ngrid, dtype=torch.complex128)
    for k in range(nk):
        Bop[k * nI:(k + 1) * nI, k * nI:(k + 1) * nI] = Bop_k[k]

    def partner_Q(cls):
        mi, ni = groups[cls[0]][1] - groups[cls[0]][0], len(cls)
        ref = [Q[:, c] for c in range(groups[cls[0]][0], groups[cls[0]][1])]
        cols = []
        for a in range(mi):
            va = Bop @ ref[a]
            for gi in cls:
                P = Q[:, groups[gi][0]:groups[gi][1]]; v = P @ (P.conj().T @ va)
                cols.append(v / torch.linalg.norm(v))
        return torch.stack(cols, dim=1), mi, ni

    row_dest = np.empty((nops, Ngrid), dtype=np.int64)
    row_ph = np.empty((nops, Ngrid), dtype=np.complex128)
    for op in range(nops):
        for k in range(nk):
            for I in range(nI):
                s = k * nI + I
                row_dest[op, s] = kmap[k, op] * nI + perm[op, I]
                row_ph[op, s] = phase_grid[op, k, I]
    kof = np.repeat(np.arange(nk), nI)

    def rho_apply(op, t, V):
        out = torch.zeros_like(V)
        out[torch.from_numpy(row_dest[op])] = torch.from_numpy(row_ph[op])[:, None] * V
        return out * torch.from_numpy(tp[t, kof])[:, None]

    def extract_D(cls):
        Qb, mi, ni = partner_Q(cls)
        Dg = np.empty((nops * nk, mi, mi), dtype=np.complex128)
        for op in range(nops):
            for t in range(nk):
                M = (Qb.conj().T @ rho_apply(op, t, Qb)).numpy().reshape(mi, ni, mi, ni)
                Dg[op * nk + t] = M[:, 0, :, 0]
        return Dg, mi

    # choose the triple: first 2-dim irrep squared
    a = [i for i in range(nirr) if dim[i] == 2][0]
    tg = [c for c in range(nirr) if Nf[a, a, c] > 0]
    print(f"\nchosen triple:  r{a} (x) r{a} = " + " + ".join(f"r{c}(d{dim[c]})" for c in tg))

    Da, ma = extract_D(classes[a])
    uerr = max(np.abs(Da[g] @ Da[g].conj().T - np.eye(ma)).max() for g in range(nops * nk))
    cherr = np.abs(np.array([np.trace(Da[g]) for g in range(nops * nk)]) - chi[a]).max()
    go = np.einsum("gab,gcd->abcd", Da, Da.conj()) / G
    goerr = np.abs(go - np.einsum("ac,bd->abcd", np.eye(ma), np.eye(ma)) / ma).max()
    print(f"D_r{a}: unitary {uerr:.1e} | char-match {cherr:.1e} | great-orthogonality {goerr:.1e}")
    Dt = {c: extract_D(classes[c])[0] for c in tg}

    # ---- reduce the product rep -> CG coefficients ----
    R = np.einsum("gab,gcd->gacbd", Da, Da).reshape(nops * nk, ma * ma, ma * ma)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(ma * ma) + 1j * rng.standard_normal(ma * ma)
    cols, labels = [], []
    for c in tg:
        d3, D3 = dim[c], Dt[c]
        O = (d3 / G) * np.einsum("gab,gXY->abXY", D3.conj(), R)   # transfer ops [alpha,beta,X,Y]
        Psi = np.array([O[al, 0] @ u for al in range(d3)])        # partner vectors (mutually consistent)
        Psi = Psi / np.sqrt((np.abs(Psi) ** 2).sum() / d3)
        # ONE global phase for the whole irrep block (keep partner phases relative)
        j = int(np.argmax(np.abs(Psi[0])))
        Psi = Psi * np.exp(-1j * np.angle(Psi[0, j]))
        for al in range(d3):
            cols.append(Psi[al]); labels.append((c, al))
    V = np.array(cols).T
    uerrV = np.abs(V.conj().T @ V - np.eye(ma * ma)).max()

    def blockdiag(g):
        B = np.zeros((ma * ma, ma * ma), dtype=complex); o = 0
        for c in tg:
            d3 = dim[c]; B[o:o + d3, o:o + d3] = Dt[c][g]; o += d3
        return B
    rec = max(np.abs(V.conj().T @ R[g] @ V - blockdiag(g)).max() for g in range(nops * nk))
    print(f"CG matrix V: unitary {uerrV:.1e} | V^dag (D_r{a} x D_r{a}) V == (+) D_target: {rec:.1e}")

    print(f"\nExtracted 2-D irrep D_r{a}(g) at a few group elements g=(op,trans):")
    for g in (nk * 1 + 0, nk * 5 + 3, nk * 20 + 1):        # a few representative elements
        print(f"  g#{g}:  D = {np.round(Da[g],3).tolist()}")

    print(f"\nCG coefficient matrix  C[(alpha1,alpha2), (lambda3,alpha3)]")
    print("  rows (alpha1,alpha2) = (0,0),(0,1),(1,0),(1,1)")
    print("  cols (lambda3,alpha3) =", [(f"r{c}", al) for c, al in labels])
    print(np.round(V, 4))
    print("\n|CG| magnitudes (gauge-invariant pattern):")
    print(np.round(np.abs(V), 4))

    np.savez(os.path.join(OUT_DIR, "cg_coefficients_diamond_2x2x2.npz"),
             V=V, labels=np.array(labels), a=a, targets=np.array(tg),
             dim=np.array(dim), mult=np.array(mult), Da=Da, G=G)
    print("\nsaved cg_coefficients_diamond_2x2x2.npz  and  cg_fusion_diamond_2x2x2.txt")


if __name__ == "__main__":
    main()
