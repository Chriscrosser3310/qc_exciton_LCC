#!/usr/bin/env python3
"""Toffoli-count tables for THC block-encoding synthesis schemes.

Fixed parameters: N_k = 256, N_IP = 26*8 = 208, N_up = 4, N_down = 22, b = 32.

Three tables (each with T-opt and Q-opt columns), all Toffoli counts:

  Table 1 (X encoders, X_o = X_up, X_v = X_down): synthesis of the rectangular
    interpolating-vector isometry under
      * Reflection (Grover-type state prep)        -- current production
      * Reflection (three-phase-layer state prep)
      * Interferometer isometry synthesis

  Table 2 (central W encoders): block encoding of sum_q |q><q| (x) W^q under
      * Interferometer   (the SVD-interferometer central; formerly "SVD")
      * Reflection (Grover-type state prep)
      * Reflection (three-phase-layer state prep)
      * Frobenius        (Clader block-matrix)
      * Recursive CSD    (SVD-style: 2 CSD unitary syntheses + Sigma diagonal)
      * Diagonal         (the DIRECT construction's central kernel; UNCHANGED ref.)

  Table 3 (full-system totals): 2 X_o + 2 X_v + central + momentum bookkeeping.
    Exchange rows use the production Reflection-Grover X; the Diagonal row is the
    full direct construction.

The interferometer-isometry X and recursive-CSD W are bare synthesis bloqs (no
BlockEncoding wrapper exists); their Toffoli is the synthesis cost (a BE wrapper would
add only negligible reflection-about-zero / signal overhead).  All counts come from
Qualtran's QECGatesCost walking each bloq's build_call_graph.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.block_interferometer_isometry_QROAM import (
    BlockInterferometerIsometrySynthesisQROAM,
)
from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
)
from integrations.qualtran.direct_Coulomb_block_encoding import DirectCoulombBlockEncoding
from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
from integrations.qualtran.rectangular_block_encoding_reflection import (
    ReflectionRectangularBlockEncoding,
)
from integrations.qualtran.recursive_csd_synthesis_QROAM import RecursiveCSDSynthesisQROAM
from integrations.qualtran.svd_block_encoding_interferometer import SVDBlockEncodingInterferometer
from integrations.qualtran.utils import get_Toffoli_counts as T

NK, NIP, NUP, NDN, B = 256, 26 * 8, 4, 22, 32
NROWS = 1 << max(1, (max(NUP, NDN, NIP) - 1).bit_length())  # 256


def _npow2(x: int) -> int:
    return 1 << max(1, (int(x) - 1).bit_length())


# ----------------------------- X-encoder costs -----------------------------

def x_costs(n_reflections: int, opt: bool):
    rg = ReflectionRectangularBlockEncoding(
        n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=n_reflections, optimal_T=opt
    )
    r3 = ReflectionRectangularBlockEncoding(
        n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=n_reflections,
        optimal_T=opt, three_phase_layer_prep=True,
    )
    ii = BlockInterferometerIsometrySynthesisQROAM.from_shape(
        n_blocks=NK, n_rows=NROWS, n_cols=_npow2(n_reflections), phase_bitsize=B, optimal_T=opt
    )
    return {"refl_grover": T(rg), "refl_3phase": T(r3), "intf_iso": T(ii)}


# ----------------------------- W-central costs -----------------------------

def _svd_central(opt: bool) -> SVDBlockEncodingInterferometer:
    return SVDBlockEncodingInterferometer(n_blocks=NK, n_rows=NROWS, phase_bitsize=B, optimal_T=opt)


def _diag_stage_toffoli(opt: bool) -> int:
    # Sigma (singular-value) rotation stage of the SVD interferometer central:
    #   T(svd) = 2*T(interferometer) + T(diag stage).  Shared by the recursive-CSD central.
    svd = _svd_central(opt)
    return T(svd) - 2 * T(svd.interferometer)


def w_costs(opt: bool):
    svd = _svd_central(opt)
    wrg = ReflectionRectangularBlockEncoding(
        n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=NIP, optimal_T=opt
    )
    wr3 = ReflectionRectangularBlockEncoding(
        n_blocks=NK, n_rows=NROWS, phase_bitsize=B, n_reflections=NIP,
        optimal_T=opt, three_phase_layer_prep=True,
    )
    fro = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(
        n_blocks=NK, n_rows=NROWS, phase_bitsize=B, optimal_T=opt
    )
    csd = RecursiveCSDSynthesisQROAM.from_shape(NK, NROWS, B, optimal_T=opt)
    csd_central = 2 * T(csd) + _diag_stage_toffoli(opt)
    # Diagonal kernel (direct construction's central) -- unchanged reference.
    diag = DirectCoulombBlockEncoding(
        N_up=NUP, N_down=NDN, N_IP=NIP, N_k=NK, phase_bitsize=B, optimal_T=opt
    ).C_diag
    return {
        "intf": T(svd),
        "refl_grover": T(wrg),
        "refl_3phase": T(wr3),
        "fro": T(fro),
        "recursive_csd": csd_central,
        "diag": T(diag),
    }


# ------------------------------ Full totals --------------------------------

def totals(opt: bool, w: dict):
    # X + momentum/LCU bookkeeping is identical across exchange centrals, so derive it
    # once: T(exchange_default) - T(its SVD central).  Then total = that + chosen central.
    exch = ExchangeCoulombBlockEncoding(
        N_up=NUP, N_down=NDN, N_IP=NIP, N_k=NK, phase_bitsize=B, optimal_T=opt
    )
    x_and_bookkeeping = T(exch) - w["intf"]
    direct = DirectCoulombBlockEncoding(
        N_up=NUP, N_down=NDN, N_IP=NIP, N_k=NK, phase_bitsize=B, optimal_T=opt
    )
    return {
        "intf": x_and_bookkeeping + w["intf"],
        "refl_grover": x_and_bookkeeping + w["refl_grover"],
        "refl_3phase": x_and_bookkeeping + w["refl_3phase"],
        "fro": x_and_bookkeeping + w["fro"],
        "recursive_csd": x_and_bookkeeping + w["recursive_csd"],
        "diag": T(direct),
    }, x_and_bookkeeping


def fmt(n: int) -> str:
    return f"{n:,}"


def latex_fmt(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main():
    XO, XV = min(NUP, NIP), min(NDN, NIP)  # 4, 22
    xT = {"o": x_costs(XO, True), "v": x_costs(XV, True)}
    xQ = {"o": x_costs(XO, False), "v": x_costs(XV, False)}
    wT, wQ = w_costs(True), w_costs(False)
    totT, xbkT = totals(True, wT)
    totQ, xbkQ = totals(False, wQ)

    # sanity: interferometer total must equal the directly-built exchange total
    print(f"# params: N_k={NK}, N_IP={NIP}, N_up={NUP}, N_down={NDN}, b={B}, n_rows={NROWS}")
    print(f"# X+bookkeeping (T-opt)={fmt(xbkT)}  (Q-opt)={fmt(xbkQ)}")
    print(f"# n_cols (intf-iso): X_o={_npow2(XO)}, X_v={_npow2(XV)}\n")

    xrows = [("Reflection (Grover)", "refl_grover"),
             ("Reflection (three-phase-layer)", "refl_3phase"),
             ("Interferometer isometry", "intf_iso")]
    wrows = [("Interferometer", "intf"),
             ("Reflection (Grover)", "refl_grover"),
             ("Reflection (three-phase-layer)", "refl_3phase"),
             ("Frobenius", "fro"),
             ("Recursive CSD", "recursive_csd"),
             ("Diagonal (direct central)", "diag")]
    trows = [("Exchange, Interferometer", "intf"),
             ("Exchange, Reflection (Grover)", "refl_grover"),
             ("Exchange, Reflection (three-phase-layer)", "refl_3phase"),
             ("Exchange, Frobenius", "fro"),
             ("Exchange, Recursive CSD", "recursive_csd"),
             ("Direct (Diagonal)", "diag")]

    print("=== TABLE 1: X encoders (Toffoli) ===")
    print(f"{'method':32s} {'Xo T-opt':>12} {'Xo Q-opt':>13} {'Xv T-opt':>13} {'Xv Q-opt':>14}")
    for lbl, k in xrows:
        print(f"{lbl:32s} {fmt(xT['o'][k]):>12} {fmt(xQ['o'][k]):>13} {fmt(xT['v'][k]):>13} {fmt(xQ['v'][k]):>14}")

    print("\n=== TABLE 2: central W encoders (Toffoli) ===")
    print(f"{'scheme':34s} {'T-opt':>14} {'Q-opt':>16}")
    for lbl, k in wrows:
        print(f"{lbl:34s} {fmt(wT[k]):>14} {fmt(wQ[k]):>16}")

    print("\n=== TABLE 3: full-system totals (Toffoli) ===")
    print(f"{'scheme':42s} {'T-opt':>14} {'Q-opt':>16}")
    for lbl, k in trows:
        print(f"{lbl:42s} {fmt(totT[k]):>14} {fmt(totQ[k]):>16}")

    # ---------------------------- LaTeX ----------------------------
    tex = []
    tex.append("% \\usepackage{booktabs}")
    tex.append(f"% Toffoli counts at N_k={NK}, N_IP={NIP}, N_up={NUP}, N_down={NDN}, b={B}.\n")

    tex.append("\\begin{table}[htbp]\n  \\centering")
    tex.append("  \\caption{Toffoli cost of the outer $X$-matrix isometry encoders "
               "($X_o$=up, $N_\\mathrm{up}=4$; $X_v$=down, $N_\\mathrm{down}=22$).}")
    tex.append("  \\begin{tabular}{lrrrr}")
    tex.append("    \\toprule")
    tex.append("    & \\multicolumn{2}{c}{$X_o$} & \\multicolumn{2}{c}{$X_v$} \\\\")
    tex.append("    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    tex.append("    Synthesis & T-opt & Q-opt & T-opt & Q-opt \\\\")
    tex.append("    \\midrule")
    for lbl, k in xrows:
        tex.append(f"    {lbl} & {latex_fmt(xT['o'][k])} & {latex_fmt(xQ['o'][k])} "
                   f"& {latex_fmt(xT['v'][k])} & {latex_fmt(xQ['v'][k])} \\\\")
    tex.append("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

    tex.append("\\begin{table}[htbp]\n  \\centering")
    tex.append("  \\caption{Toffoli cost of the central $W$ block encoder (single application).}")
    tex.append("  \\begin{tabular}{lrr}")
    tex.append("    \\toprule")
    tex.append("    Scheme & T-opt & Q-opt \\\\")
    tex.append("    \\midrule")
    for lbl, k in wrows:
        tex.append(f"    {lbl} & {latex_fmt(wT[k])} & {latex_fmt(wQ[k])} \\\\")
    tex.append("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

    tex.append("\\begin{table}[htbp]\n  \\centering")
    tex.append("  \\caption{Total Toffoli cost of the full block encoding "
               "($2X_o + 2X_v + W + \\text{momentum bookkeeping}$). Exchange rows use the "
               "production Reflection--Grover $X$.}")
    tex.append("  \\begin{tabular}{lrr}")
    tex.append("    \\toprule")
    tex.append("    Scheme & T-opt & Q-opt \\\\")
    tex.append("    \\midrule")
    for lbl, k in trows:
        tex.append(f"    {lbl} & {latex_fmt(totT[k])} & {latex_fmt(totQ[k])} \\\\")
    tex.append("    \\bottomrule\n  \\end{tabular}\n\\end{table}")

    tex_str = "\n".join(tex)
    out_tex = os.path.join(REPO_ROOT, "docs", "synthesis_scheme_tables.tex")
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write(tex_str + "\n")
    print(f"\nWrote LaTeX: {out_tex}")
    print("\n" + "=" * 60 + " LATEX " + "=" * 60)
    print(tex_str)


if __name__ == "__main__":
    main()
