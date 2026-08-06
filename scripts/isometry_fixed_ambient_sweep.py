#!/usr/bin/env python3
r"""Isometry-synthesis sweep at FIXED ambient dimension, varying the synthesized column count.

The ambient (row) dimension is fixed at ``M' = d*N = 256`` and the number of synthesized
columns ``N`` is swept over ``4, 8, 16, ..., 256`` (so ``d = M'/N = 64, 32, ..., 1``): the
isometry is always ``256 x N`` (synthesize the first ``N`` columns of a 256-dim unitary).
``b = 32``, single block.

Two methods, same notation/convention as the earlier d-sweep:

  * Interferometer isometry (``InterferometerIsometrySynthesisQROAM``): the Eq.-36 staircase
    of ``d - 1`` fixed ``d=2`` steps (Sec. III B of arXiv:2409.11748), each a ``2N``-dim step.
  * Reflection isometry (``BlockUnitaryReflectionQROAM``, ``n_blocks=1``): ``N`` Householder
    reflections (Sec. 4 of arXiv:1812.00954), one per synthesized column, each about a
    256-dim state prepared by QROAM rotations.

Reports both un-batched (``Lambda=1``) and Toffoli-optimal (``optimal_T=True``) regimes,
emits a PDF (Toffoli + qubits vs N, with d annotated; table), and emails it.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.interferometer_isometry_QROAM import (
    InterferometerIsometrySynthesisQROAM,
)
from integrations.qualtran.block_unitary_reflection_QROAM import BlockUnitaryReflectionQROAM
from integrations.qualtran.utils import get_qubit_counts, get_Toffoli_counts

M_AMBIENT = 256                    # fixed ambient (row) dimension M' = d*N
N_VALUES = [4, 8, 16, 32, 64, 128, 256]   # synthesized column count N (d = M'/N)
B = int(os.environ.get("QC_PHASE_BITSIZE", "32"))   # phase bitsize (override via env)
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"isometry_fixed_ambient_sweep_M{M_AMBIENT}_b{B}.pdf")


def _intf(N: int, opt: bool):
    return InterferometerIsometrySynthesisQROAM.from_shape(
        n_rows=M_AMBIENT, n_cols=N, phase_bitsize=B, optimal_T=opt
    )


def _refl(N: int, opt: bool):
    return BlockUnitaryReflectionQROAM.from_shape(
        1, M_AMBIENT, B, n_reflections=N, optimal_T=opt
    )


METHODS = [("Interferometer", _intf, "#1f77b4"), ("Reflection", _refl, "#d62728")]


def main():
    Ns = np.array(N_VALUES, dtype=float)
    ds = M_AMBIENT / Ns
    data = {}
    for label, fn, _ in METHODS:
        # Qubit-optimal (lambda=1) and Toffoli-optimal (optimal_T) end-points of the tradeoff.
        T1 = np.array([int(get_Toffoli_counts(fn(N, False))) for N in N_VALUES], dtype=float)
        Q1 = np.array([int(get_qubit_counts(fn(N, False))) for N in N_VALUES], dtype=float)
        Topt = np.array([int(get_Toffoli_counts(fn(N, True))) for N in N_VALUES], dtype=float)
        Qopt = np.array([int(get_qubit_counts(fn(N, True))) for N in N_VALUES], dtype=float)
        data[label] = dict(T1=T1, Q1=Q1, Topt=Topt, Qopt=Qopt)

    print(f"Isometry synthesis at fixed ambient M' = d*N = {M_AMBIENT}, b = {B}, single block")
    print("Tradeoff: qubit-optimal (Lambda=1) vs Toffoli-optimal (optimal_T) -- Toffoli DOWN, qubits UP.")
    for label, _, _ in METHODS:
        dd = data[label]
        print(f"\n  [{label}]")
        print(f"  {'N':>5} | {'d':>4} | {'T(Lam=1)':>11} {'q(Lam=1)':>9} | {'T(opt)':>11} {'q(opt)':>9} | "
              f"{'T saved':>8} {'q added':>8}")
        for i, N in enumerate(N_VALUES):
            tsav = 100 * (dd['T1'][i] - dd['Topt'][i]) / dd['T1'][i]
            qadd = int(dd['Qopt'][i] - dd['Q1'][i])
            print(f"  {N:>5} | {int(M_AMBIENT // N):>4} | {int(dd['T1'][i]):>11,} {int(dd['Q1'][i]):>9,} | "
                  f"{int(dd['Topt'][i]):>11,} {int(dd['Qopt'][i]):>9,} | {tsav:>7.1f}% {qadd:>+8,}")

    # method crossover (Toffoli-optimal)
    diff = data["Interferometer"]["Topt"] - data["Reflection"]["Topt"]
    cheaper = ["Interf" if x < 0 else "Reflect" for x in diff]
    print("\nToffoli-optimal cheaper method per N:", dict(zip(N_VALUES, cheaper)))

    # ------------------------------- PDF -------------------------------
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        # 1) Toffoli vs N (both methods x both regimes)
        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        for label, _, color in METHODS:
            dd = data[label]
            ax.plot(Ns, dd['T1'], marker="o", color=color, linestyle="-", linewidth=1.8,
                    label=f"{label}  (Λ=1)")
            ax.plot(Ns, dd['Topt'], marker="s", color=color, linestyle="--", linewidth=1.6,
                    label=f"{label}  (opt)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("N = synthesized columns   (ambient M' = d·N = 256;  d = 256/N)")
        ax.set_ylabel("Toffoli count")
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([f"{N}\n(d={M_AMBIENT // N})" for N in N_VALUES], fontsize=8)
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9)
        ax.set_title(f"Isometry synthesis at fixed ambient M' = {M_AMBIENT}: Toffoli vs N\n"
                     f"(interferometer vs reflection, b={B}, single block)")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 2) Toffoli<->qubit tradeoff: qubits vs N for BOTH regimes (opt uses more qubits).
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        for label, _, color in METHODS:
            dd = data[label]
            ax.plot(Ns, dd['Q1'], marker="o", color=color, linestyle="-", linewidth=1.8,
                    label=f"{label}  qubit-opt (Λ=1)")
            ax.plot(Ns, dd['Qopt'], marker="s", color=color, linestyle="--", linewidth=1.6,
                    label=f"{label}  Toffoli-opt")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N = synthesized columns   (ambient M' = 256;  d = 256/N)")
        ax.set_ylabel("peak logical qubits")
        ax.set_xticks(N_VALUES); ax.set_xticklabels([str(N) for N in N_VALUES])
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8.5)
        ax.set_title(f"Tradeoff at fixed ambient M' = {M_AMBIENT}, b = {B}: Toffoli-optimal spends qubits\n"
                     "(gap between dashed and solid = qubits added to lower Toffoli)")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 3) tradeoff table: T and qubits at both end-points, per method.
        fig, ax = plt.subplots(figsize=(12, 4.2)); ax.axis("off")
        cols = ["N", "d"]
        for lbl, _, _ in METHODS:
            cols += [f"{lbl[:5]} T(Λ=1)", f"{lbl[:5]} q(Λ=1)", f"{lbl[:5]} T(opt)", f"{lbl[:5]} q(opt)"]
        cells = []
        for i, N in enumerate(N_VALUES):
            row = [str(N), str(int(M_AMBIENT // N))]
            for lbl, _, _ in METHODS:
                dd = data[lbl]
                row += [f"{int(dd['T1'][i]):,}", f"{int(dd['Q1'][i]):,}",
                        f"{int(dd['Topt'][i]):,}", f"{int(dd['Qopt'][i]):,}"]
            cells.append(row)
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a"); cell.set_text_props(color="white", weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
        ax.set_title(f"Isometry synthesis Toffoli/qubit tradeoff at fixed ambient M' = d·N = {M_AMBIENT}  (b={B})\n"
                     "Λ=1 = qubit-optimal;  opt = Toffoli-optimal (QROAM batching)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        pdf.infodict()["Title"] = "Isometry Synthesis at Fixed Ambient Dimension: Toffoli/Qubit Tradeoff"
    print(f"\nWrote report: {OUT_PDF}")

    # ------------------------------- email -------------------------------
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = (f"Isometry synthesis at fixed ambient M'=d*N={M_AMBIENT}: Toffoli vs N "
                      f"(interferometer vs reflection, b={B})")
    body = [
        "Hi,", "",
        f"Isometry-synthesis sweep at FIXED ambient dimension M' = d*N = {M_AMBIENT}, varying the",
        f"synthesized column count N over {N_VALUES} (so d = M'/N = {[M_AMBIENT//N for N in N_VALUES]}).",
        f"The isometry is always {M_AMBIENT} x N.  b = {B}, single block.", "",
        "Shown as the Toffoli<->qubit tradeoff: Lambda=1 (qubit-optimal) vs optimal_T (Toffoli-optimal,",
        "QROAM batching).  optimal_T lowers Toffoli by spending more workspace qubits.", "",
    ]
    for label, _, _ in METHODS:
        dd = data[label]
        body.append(f"[{label}]")
        body.append(f"{'N':>5} | {'d':>4} | {'T(Λ=1)':>11} {'q(Λ=1)':>8} | {'T(opt)':>11} {'q(opt)':>8} | {'T saved':>7} {'q added':>7}")
        for i, N in enumerate(N_VALUES):
            tsav = 100 * (dd['T1'][i] - dd['Topt'][i]) / dd['T1'][i]
            qadd = int(dd['Qopt'][i] - dd['Q1'][i])
            body.append(f"{N:>5} | {int(M_AMBIENT//N):>4} | {int(dd['T1'][i]):>11,} {int(dd['Q1'][i]):>8,} | "
                        f"{int(dd['Topt'][i]):>11,} {int(dd['Qopt'][i]):>8,} | {tsav:>6.1f}% {qadd:>+7,}")
        body.append("")
    body += [
        "Notes: interferometer = Eq.-36 staircase of (d-1) fixed d=2 steps, each a 2N-dim step;",
        "reflection = N Householder reflections, each about a 256-dim QROAM-prepared state.",
        "At b=32 the interferometer's QROAM optimum Lambda* = 0.5*sqrt(N_un/b) ~ 1 for these dims, so its",
        "Toffoli<->qubit tradeoff is mild; the reflection (Lambda* = sqrt(N/b) ~ 3) trades more.",
        "N=256 (d=1) is the full 256x256 unitary for both.",
        "", "PDF (Toffoli vs N, qubit tradeoff vs N, tradeoff table) attached.", ""]
    msg.attach(MIMEText("\n".join(body), "plain"))
    with open(OUT_PDF, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(OUT_PDF))
    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(OUT_PDF)}"'
    msg.attach(part)
    try:
        with smtplib.SMTP("localhost", 25, timeout=10) as smtp:
            smtp.sendmail(msg["From"], [RECIPIENT], msg.as_string())
        print(f"Email sent via localhost:25 to {RECIPIENT}")
    except Exception as e:
        print(f"localhost:25 failed: {e}")
        proc = subprocess.run(["/usr/sbin/sendmail", "-t", "-oi"],
                              input=msg.as_string().encode(), capture_output=True, timeout=30)
        print(f"Email sent via sendmail to {RECIPIENT}" if proc.returncode == 0
              else f"sendmail failed: {proc.stderr.decode(errors='replace')[:200]}")


if __name__ == "__main__":
    main()
