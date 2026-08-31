#!/usr/bin/env python3
r"""Combined-construction Toffoli/qubit table (extended synthesis schemes).

Reproduces and extends the combined construction
    total = 6 X_o + 6 X_v + W,   W = 2 (diagonal) + (central)
at N_up=4, N_down=22, N_IP=208, N_k=216, b=32.

  %X = (6 X_o + 6 X_v) / total ;  %W = (2 diagonal + central) / total.
  qubits = 6 ceil(log2 N_IP) + max over the (sequentially applied) component ancilla
           footprints, where footprint(component) = qubits(component) - system_bits(component).

X-synthesis schemes (rows):
  * Reflection (Grover state prep)
  * Reflection (three-phase-layer state prep)        [added]
  * Isometry (column-by-column)
  * Interferometer isometry                          [added]
Central-W schemes (rows, the diagonal is UNCHANGED):
  * Reflection (Grover)
  * Reflection (three-phase-layer)                   [added]
  * Interferometer  (formerly "SVD")
  * Frobenius
  * Recursive CSD  (SVD-style: 2 CSD unitary syntheses + Sigma diagonal)   [added]

The interferometer-isometry X and recursive-CSD central are bare synthesis bloqs (no
BlockEncoding wrapper); their Toffoli is the synthesis cost (a wrapper adds only
negligible reflection-about-zero/signal overhead).  Recursive CSD encodes the contraction
W^q via the SVD pattern, swapping the interferometer unitary synthesis for CSD.
All counts from Qualtran's QECGatesCost / QubitCount walking each build_call_graph.
"""

from __future__ import annotations

import math
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
from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
)
from integrations.qualtran.direct_Coulomb_block_encoding import DirectCoulombBlockEncoding
from integrations.qualtran.rectangular_block_encoding_reflection import (
    ReflectionRectangularBlockEncoding,
)
from integrations.qualtran.recursive_csd_synthesis_QROAM import RecursiveCSDSynthesisQROAM
from integrations.qualtran.svd_block_encoding_interferometer import SVDBlockEncodingInterferometer
from integrations.qualtran.utils import get_Toffoli_counts as T, get_qubit_counts as Q

NK, NIP, NUP, NDN, B = 216, 208, 4, 22, 32
NROWS = 1 << max(1, (max(NUP, NDN, NIP) - 1).bit_length())  # 256
MATRIX_BITS = (NROWS - 1).bit_length()                       # 8
BLOCK_BITS = (NK - 1).bit_length()                           # 8
SYS_X = BLOCK_BITS + MATRIX_BITS                             # 16  (block + matrix)
SYS_C = BLOCK_BITS + MATRIX_BITS                             # 16
IP_BITS = (NIP - 1).bit_length()                             # 8
SYS_DIAG = BLOCK_BITS + 2 * IP_BITS                          # 24  (Q + I + J)
XO_REFL, XV_REFL = min(NUP, NIP), min(NDN, NIP)              # 4, 22


def _npow2(x: int) -> int:
    return 1 << max(1, (int(x) - 1).bit_length())


# (Toffoli, ancilla-footprint = qubits - system_bits) for each component, per policy.
def x_component(synth: str, n: int, opt: bool):
    if synth == "refl":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=n, optimal_T=opt)
        return T(b), Q(b) - SYS_X
    if synth == "refl3":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=n,
            optimal_T=opt, three_phase_layer_prep=True)
        return T(b), Q(b) - SYS_X
    if synth == "iso":
        b = ColumnIsometryRectangularBlockEncoding(
            n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=n, optimal_T=opt)
        return T(b), Q(b) - SYS_X
    if synth == "intf_iso":
        b = BlockInterferometerIsometrySynthesisQROAM.from_shape(
            n_blocks=NK, n_rows=NROWS, n_cols=_npow2(n), phase_bitsize=B, optimal_T=opt)
        return T(b), Q(b) - SYS_X
    raise ValueError(synth)


def _svd(opt):
    return SVDBlockEncodingInterferometer(n_blocks=NK, n_rows=NROWS, phase_bitsize=B, optimal_T=opt)


def central_component(scheme: str, opt: bool):
    if scheme == "refl":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=NIP, optimal_T=opt)
        return T(b), Q(b) - SYS_C
    if scheme == "refl3":
        b = ReflectionRectangularBlockEncoding(
            n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=NIP,
            optimal_T=opt, three_phase_layer_prep=True)
        return T(b), Q(b) - SYS_C
    if scheme == "intf":
        b = _svd(opt)
        return T(b), Q(b) - SYS_C
    if scheme == "fro":
        b = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(
            n_blocks=NK, n_rows=NROWS, phase_bitsize=B, optimal_T=opt)
        return T(b), Q(b) - SYS_C
    if scheme == "rcsd":
        svd = _svd(opt)
        diag_stage = T(svd) - 2 * T(svd.interferometer)
        csd = RecursiveCSDSynthesisQROAM.from_shape(NK, NROWS, B, optimal_T=opt)
        toff = 2 * T(csd) + diag_stage
        # SVD-style: sequential CSD(V) -> Sigma -> CSD(U). Peak = CSD synthesis (dominates
        # the Sigma stage, whose peak < SVD peak < CSD peak here).
        return toff, Q(csd) - SYS_C
    raise ValueError(scheme)


def diag_component(opt):
    b = DirectCoulombBlockEncoding(
        N_up=NUP, N_down=NDN, N_IP=NIP, N_k=NK, phase_bitsize=B, optimal_T=opt).C_diag
    return T(b), Q(b) - SYS_DIAG


X_SCHEMES = [("Reflection", "refl"),
             ("Reflection (3-phase)", "refl3"),
             ("Isometry", "iso"),
             ("Interferometer isometry", "intf_iso")]
C_SCHEMES = [("Reflection", "refl"),
             ("Reflection (3-phase)", "refl3"),
             ("Interferometer", "intf"),
             ("Frobenius", "fro"),
             ("Recursive CSD", "rcsd")]


def build(opt):
    dT, dfp = diag_component(opt)
    rows = {}
    for xl, xk in X_SCHEMES:
        xoT, xofp = x_component(xk, XO_REFL, opt)
        xvT, xvfp = x_component(xk, XV_REFL, opt)
        xfp = max(xofp, xvfp)
        for cl, ck in C_SCHEMES:
            cT, cfp = central_component(ck, opt)
            total = 6 * xoT + 6 * xvT + 2 * dT + cT
            xshare = 6 * xoT + 6 * xvT
            wshare = 2 * dT + cT
            qubits = 6 * IP_BITS + max(xfp, dfp, cfp)
            rows[(xk, ck)] = dict(total=total, pX=100 * xshare / total,
                                  pW=100 * wshare / total, qubits=qubits)
    return rows


def main():
    rT, rQ = build(True), build(False)
    min_tot_T = min(r["total"] for r in rT.values())
    min_tot_Q = min(r["total"] for r in rQ.values())

    # ---- readable ----
    print(f"# N_up={NUP} N_down={NDN} N_IP={NIP} N_k={NK} b={B}; "
          f"n_cols(intf-iso): X_o={_npow2(XO_REFL)}, X_v={_npow2(XV_REFL)}")
    hdr = f"{'X synth':24s} {'Central':22s} {'T tot':>13} {'%X':>6} {'%W':>6} {'Tqub':>6}   {'Q tot':>14} {'%X':>6} {'%W':>6} {'Qqub':>6}"
    print(hdr)
    for xl, xk in X_SCHEMES:
        for cl, ck in C_SCHEMES:
            a, b = rT[(xk, ck)], rQ[(xk, ck)]
            print(f"{xl:24s} {cl:22s} {a['total']:>13,} {a['pX']:>6.2f} {a['pW']:>6.2f} {a['qubits']:>6}   "
                  f"{b['total']:>14,} {b['pX']:>6.2f} {b['pW']:>6.2f} {b['qubits']:>6}")

    # ---- LaTeX ----
    def lf(n):
        return f"{n:,}".replace(",", "{,}")

    def cell_total(v, mn):
        return f"\\textbf{{{lf(v)}}}" if v == mn else lf(v)

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"  \centering")
    L.append(r"  \caption{Toffoli cost and total qubits of the combined construction")
    L.append(r"           $6X_o + 6X_v + W$, $W = 2\,(\text{diagonal}) + (\text{central})$")
    L.append(r"           ($N_\mathrm{up}=4$, $N_\mathrm{down}=22$, $N_\mathrm{IP}=208$, $N_k=216$, $b=32$).")
    L.append(r"           \%\,$X = (6X_o+6X_v)/\text{total}$; \%\,$W$ the central+diagonal share.")
    L.append(r"           Qubits $= 6\lceil\log_2 N_\mathrm{IP}\rceil + \max$ over the (sequentially applied)")
    L.append(r"           component ancilla footprints. Cheapest total per column in \textbf{bold}.}")
    L.append(r"  \setlength{\tabcolsep}{4pt}")
    L.append(r"  \begin{tabular}{llrrrrrrrr}")
    L.append(r"    \toprule")
    L.append(r"    & & \multicolumn{4}{c}{T-opt} & \multicolumn{4}{c}{Q-opt} \\")
    L.append(r"    \cmidrule(lr){3-6}\cmidrule(lr){7-10}")
    L.append(r"    $X$ synth & Central & Total & \%\,$X$ & \%\,$W$ & Qubits & Total & \%\,$X$ & \%\,$W$ & Qubits \\")
    L.append(r"    \midrule")
    for gi, (xl, xk) in enumerate(X_SCHEMES):
        if gi:
            L.append(r"    \midrule")
        for ci, (cl, ck) in enumerate(C_SCHEMES):
            a, b = rT[(xk, ck)], rQ[(xk, ck)]
            xcell = xl if ci == 0 else ""
            L.append(
                f"    {xcell} & {cl} & {cell_total(a['total'], min_tot_T)} & {a['pX']:.2f} & {a['pW']:.2f} & {lf(a['qubits'])} "
                f"& {cell_total(b['total'], min_tot_Q)} & {b['pX']:.2f} & {b['pW']:.2f} & {lf(b['qubits'])} \\\\")
    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}")
    L.append(r"\end{table}")
    tex = "\n".join(L)

    out = os.path.join(REPO_ROOT, "docs", "combined_construction_table.tex")
    with open(out, "w") as f:
        f.write(tex + "\n")
    print(f"\nWrote LaTeX: {out}\n")
    print(tex)


if __name__ == "__main__":
    main()
