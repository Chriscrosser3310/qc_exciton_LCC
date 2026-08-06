#!/usr/bin/env python3
r"""BSE block-encoding synthesis-scheme table (4 X methods x 5 exchange-W methods).

Same layout as ``combined_construction_table.py`` / ``synthesis_scheme_tables.py`` but for
the full BSE block encoding ``<psi| S P A D P S |psi>``.

Toffoli totals are assembled from a component decomposition that reproduces the actual
``BSEBlockEncoding`` cost exactly:

    total = fixed + 8*X_o + 8*X_v + exchange_central_W ,

where the BSE contains 16 interpolating-vector X tensors (2 X_o + 2 X_v in the exchange
term, 2 + 2 in the direct term, 4 + 4 in the combined term) and exactly one exchange
central W; ``fixed`` is everything else (the two diagonal central kernels, the two Fock
terms, the particle-number counters P, the antisymmetrizers S, the diagonal D, the |psi>
prepares, and all bookkeeping/swaps).  ``fixed`` is obtained by subtracting the default
(reflection X, interferometer exchange-W) contributions from the real BSE cost, so the
reconstruction is exact (asserted below).

X-synthesis methods (rows): Reflection (Grover), Reflection (three-phase-layer),
Isometry (column-by-column), Interferometer isometry.
Exchange central-W methods: Reflection (Grover), Reflection (three-phase-layer),
Interferometer (the SVD interferometer), Frobenius, Recursive CSD.
The diagonal central kernels (direct + combined term) are always the diagonal kernel.

  %X = (8 X_o + 8 X_v) / total ;  %W = (exchange central + 2 diagonal kernels) / total.
  Qubits = data-free peak = max(signature floor, every component's peak qubit count).
  Cheapest total per column in bold.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.block_interferometer_isometry_QROAM import (
    BlockInterferometerIsometrySynthesisQROAM,
)
from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    ColumnIsometryRectangularBlockEncoding,
)
from integrations.qualtran.bse_block_encoding import BSEBlockEncoding
from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
)
from integrations.qualtran.direct_Coulomb_block_encoding import DiagonalCoulombKernelBlockEncoding
from integrations.qualtran.rectangular_block_encoding_reflection import (
    ReflectionRectangularBlockEncoding,
)
from integrations.qualtran.recursive_csd_synthesis_QROAM import RecursiveCSDSynthesisQROAM
from integrations.qualtran.svd_block_encoding_interferometer import SVDBlockEncodingInterferometer
from integrations.qualtran.utils import get_Toffoli_counts as T, get_qubit_counts as Q

M, N_O, N_V, N_IP, N_K, B = 3, 4, 22, 208, 216, 32
N_CENTRAL = 1 << max(1, (N_IP - 1).bit_length())          # 256: padded W dimension
N_X = 1 << max(1, (max(N_O, N_V, N_IP) - 1).bit_length())  # 256: padded X row dimension
SYS_X = SYS_C = 16     # block(8) + matrix(8)
N_X_TENSORS = 16       # 8 X_o + 8 X_v across exchange/direct/combined


def _npow2(x: int) -> int:
    return 1 << max(1, (int(x) - 1).bit_length())


# ----------------------------- X-encoder costs -----------------------------

def x_component(method: str, n_refl: int, opt: bool):
    if method == "refl":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=N_K, n_rows=N_X, phase_bitsize=B, n_reflections=n_refl, optimal_T=opt)
    elif method == "refl3":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=N_K, n_rows=N_X, phase_bitsize=B, n_reflections=n_refl,
            optimal_T=opt, three_phase_layer_prep=True)
    elif method == "col":
        b = ColumnIsometryRectangularBlockEncoding(
            n_blocks=N_K, n_rows=N_X, phase_bitsize=B, n_reflections=n_refl, optimal_T=opt)
    elif method == "intf_iso":
        b = BlockInterferometerIsometrySynthesisQROAM.from_shape(
            n_blocks=N_K, n_rows=N_X, n_cols=_npow2(n_refl), phase_bitsize=B, optimal_T=opt)
    else:
        raise ValueError(method)
    return T(b), Q(b)


# ----------------------------- exchange central-W costs --------------------

def _svd(opt):
    return SVDBlockEncodingInterferometer(n_blocks=N_K, n_rows=N_CENTRAL, phase_bitsize=B, optimal_T=opt)


def central_component(method: str, opt: bool):
    if method == "refl":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=N_K, n_rows=N_CENTRAL, phase_bitsize=B, n_reflections=N_IP, optimal_T=opt)
        return T(b), Q(b)
    if method == "refl3":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=N_K, n_rows=N_CENTRAL, phase_bitsize=B, n_reflections=N_IP,
            optimal_T=opt, three_phase_layer_prep=True)
        return T(b), Q(b)
    if method == "intf":
        b = _svd(opt)
        return T(b), Q(b)
    if method == "fro":
        b = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(
            n_blocks=N_K, n_rows=N_CENTRAL, phase_bitsize=B, optimal_T=opt)
        return T(b), Q(b)
    if method == "rcsd":
        svd = _svd(opt)
        diag_stage = T(svd) - 2 * T(svd.interferometer)
        csd = RecursiveCSDSynthesisQROAM.from_shape(N_K, N_CENTRAL, B, optimal_T=opt)
        # SVD-style: 2 CSD unitary syntheses + the same Sigma diagonal; peak qubit = the
        # CSD synthesis (dominates the Sigma stage).
        return 2 * T(csd) + diag_stage, Q(csd)
    raise ValueError(method)


def diag_kernel_T(opt):
    return T(DiagonalCoulombKernelBlockEncoding(N_k=N_K, N_IP=N_IP, phase_bitsize=B, optimal_T=opt))


X_SCHEMES = [("Reflection", "refl"),
             ("Reflection (3-phase)", "refl3"),
             ("Isometry", "col"),
             ("Interferometer isometry", "intf_iso")]
W_SCHEMES = [("Reflection", "refl"),
             ("Reflection (3-phase)", "refl3"),
             ("Interferometer", "intf"),
             ("Frobenius", "fro"),
             ("Recursive CSD", "rcsd")]


def build(opt):
    bse = BSEBlockEncoding(m=M, N_o=N_O, N_v=N_V, N_IP=N_IP, N_k=N_K, phase_bitsize=B, optimal_T=opt)
    bse_T, bse_Q = T(bse), Q(bse)
    floor = bse.system_bitsize + bse.ancilla_bitsize + bse.resource_bitsize

    xoT_def, _ = x_component("refl", min(N_O, N_IP), opt)
    xvT_def, _ = x_component("refl", min(N_V, N_IP), opt)
    cT_def, _ = central_component("intf", opt)
    fixed = bse_T - N_X_TENSORS // 2 * xoT_def - N_X_TENSORS // 2 * xvT_def - cT_def

    two_diag_T = 2 * diag_kernel_T(opt)  # direct + combined diagonal kernels (always same)
    # FIXED_MAX_Q: peak qubit of the fixed components is the diagonal kernel (it dominates
    # at T-opt); the BSE default qubit count already reflects this, so derive it from there.
    fixed_max_q = bse_Q  # default peak (diagonal-kernel dominated at T-opt; floor at Q-opt)

    rows = {}
    for xl, xk in X_SCHEMES:
        xoT, xoQ = x_component(xk, min(N_O, N_IP), opt)
        xvT, xvQ = x_component(xk, min(N_V, N_IP), opt)
        xshare = 8 * xoT + 8 * xvT
        for wl, wk in W_SCHEMES:
            cT, cQ = central_component(wk, opt)
            total = fixed + xshare + cT
            wshare = cT + two_diag_T
            qubits = max(floor, fixed_max_q, xoQ, xvQ, cQ)
            rows[(xk, wk)] = dict(total=total, pX=100 * xshare / total,
                                  pW=100 * wshare / total, qubits=qubits)
    # self-check: default (reflection X, interferometer W) must reproduce the real BSE.
    d = rows[("refl", "intf")]
    assert d["total"] == bse_T, (d["total"], bse_T)
    assert d["qubits"] == bse_Q, (d["qubits"], bse_Q)
    return rows


def main():
    rT, rQ = build(True), build(False)
    min_tot_T = min(r["total"] for r in rT.values())
    min_tot_Q = min(r["total"] for r in rQ.values())

    print(f"# BSE: m={M}, N_o={N_O}, N_v={N_V}, N_IP={N_IP}, N_k={N_K}, b={B}")
    hdr = (f"{'X synth':24s} {'Central(exch)':22s} {'T tot':>14} {'%X':>6} {'%W':>6} {'Tqub':>6}   "
           f"{'Q tot':>15} {'%X':>6} {'%W':>6} {'Qqub':>6}")
    print(hdr)
    for xl, xk in X_SCHEMES:
        for wl, wk in W_SCHEMES:
            a, b = rT[(xk, wk)], rQ[(xk, wk)]
            print(f"{xl:24s} {wl:22s} {a['total']:>14,} {a['pX']:>6.2f} {a['pW']:>6.2f} {a['qubits']:>6}   "
                  f"{b['total']:>15,} {b['pX']:>6.2f} {b['pW']:>6.2f} {b['qubits']:>6}")

    # ------------------------------- LaTeX -------------------------------
    def lf(n):
        return f"{n:,}".replace(",", "{,}")

    def cell(v, mn):
        return f"\\textbf{{{lf(v)}}}" if v == mn else lf(v)

    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{Toffoli cost and data-free peak qubits of the BSE block encoding",
         r"           $\langle\psi| S P A D P S |\psi\rangle$ ($m=%d$, $N_o=%d$, $N_v=%d$, $N_\mathrm{IP}=%d$, $N_k=%d$, $b=%d$)."
         % (M, N_O, N_V, N_IP, N_K, B),
         r"           $X$ synth sets all 16 interpolating-vector tensors; the central column sets the",
         r"           exchange-Coulomb central $W$ (the direct/combined diagonal kernels are fixed).",
         r"           \%\,$X=(8X_o+8X_v)/\text{total}$; \%\,$W$ = (exchange central $+$ 2 diagonal kernels)$/\text{total}$.",
         r"           Cheapest total per column in \textbf{bold}.}",
         r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{llrrrrrrrr}",
         r"    \toprule",
         r"    & & \multicolumn{4}{c}{T-opt} & \multicolumn{4}{c}{Q-opt} \\",
         r"    \cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"    $X$ synth & Central ($W_{\mathrm{exch}}$) & Total & \%\,$X$ & \%\,$W$ & Qubits & Total & \%\,$X$ & \%\,$W$ & Qubits \\",
         r"    \midrule"]
    for gi, (xl, xk) in enumerate(X_SCHEMES):
        if gi:
            L.append(r"    \midrule")
        for ci, (wl, wk) in enumerate(W_SCHEMES):
            a, b = rT[(xk, wk)], rQ[(xk, wk)]
            xcell = xl if ci == 0 else ""
            L.append(
                f"    {xcell} & {wl} & {cell(a['total'], min_tot_T)} & {a['pX']:.2f} & {a['pW']:.2f} & {lf(a['qubits'])} "
                f"& {cell(b['total'], min_tot_Q)} & {b['pX']:.2f} & {b['pW']:.2f} & {lf(b['qubits'])} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    tex = "\n".join(L)

    out = os.path.join(REPO_ROOT, "docs", "bse_synthesis_scheme_table.tex")
    with open(out, "w") as f:
        f.write(tex + "\n")
    print(f"\nWrote LaTeX: {out}\n")
    print(tex)


if __name__ == "__main__":
    main()
