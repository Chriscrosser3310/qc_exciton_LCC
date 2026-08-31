#!/usr/bin/env python3
r"""Joint isometry-synthesis scaling sweep over the reduction parameter ``d``: two methods.

Synthesize a ``(d*N) x N`` isometry -- the first ``M = N`` columns of a ``(d*N)``-dimensional
unitary -- with the column count fixed at ``N = 32`` and the row count ``d*N`` grown as
``d = rows/cols = 1, 2, ..., 1024`` (``d`` = Sec. III B reduction parameter of arXiv:2409.11748).
``b = 32``, single block.

Two synthesis methods, same notation/convention:

  * Interferometer isometry (``InterferometerIsometrySynthesisQROAM``): the Eq.-36
    staircase of Sec. III B of arXiv:2409.11748 -- ``d - 1`` fixed ``d=2`` steps (each a
    full ``V`` + multiplexed ``R_y`` + 2-block ``U``), iterated per "cost multiplied by d-1".
  * Reflection isometry (``BlockUnitaryReflectionQROAM``, ``n_blocks=1``): ``M`` Householder
    reflections (Sec. 4 of arXiv:1812.00954), one per synthesized column, each reflecting
    about a ``(d*N)``-dimensional Householder state prepared by QROAM rotations.

Both are ASYMPTOTICALLY LINEAR in ``d``: the interferometer's per-block phase tables and
the reflection's ``M`` growing state-preparations each contribute a cost linear in ``d``.
The marginal panel (``dT/dd``, ``Lambda=1``) converges to a constant for both, the clean
signature of linear scaling; the large-``d`` power-law fit slope -> 1.

Reports both the un-batched (``Lambda=1``, closest to the paper's operation count) and the
Toffoli-optimal (``optimal_T=True``, QROAM batching) regimes.  Emits a PDF and emails it.
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

N_COLS = 32                        # fixed synthesized column count (M = N)
D_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]   # d = rows / cols
FIT_TAIL = 64                      # large-d power-law fit uses d >= FIT_TAIL
B = 32                             # phase bitsize
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"isometry_d_sweep_joint_N{N_COLS}_b{B}.pdf")


def _intf(d: int, opt: bool):
    return InterferometerIsometrySynthesisQROAM.from_shape(
        n_rows=d * N_COLS, n_cols=N_COLS, phase_bitsize=B, optimal_T=opt
    )


def _refl(d: int, opt: bool):
    return BlockUnitaryReflectionQROAM.from_shape(
        1, d * N_COLS, B, n_reflections=N_COLS, optimal_T=opt
    )


METHODS = [("Interferometer", _intf, "#1f77b4"), ("Reflection", _refl, "#d62728")]


def _fit_tail(ds, ys, lo):
    m = ds >= lo
    p, loga = np.polyfit(np.log(ds[m]), np.log(ys[m]), 1)
    return float(np.exp(loga)), float(p)


def main():
    ds = np.array(D_VALUES, dtype=float)
    # data[label] = dict(T1=, Topt=, Q=)
    data = {}
    for label, fn, _ in METHODS:
        T1 = np.array([int(get_Toffoli_counts(fn(d, False))) for d in D_VALUES], dtype=float)
        Topt = np.array([int(get_Toffoli_counts(fn(d, True))) for d in D_VALUES], dtype=float)
        Q = np.array([int(get_qubit_counts(fn(d, True))) for d in D_VALUES], dtype=float)
        a1, p1 = _fit_tail(ds, T1, FIT_TAIL)
        ao, po = _fit_tail(ds, Topt, FIT_TAIL)
        marg = np.diff(T1) / np.diff(ds)
        data[label] = dict(T1=T1, Topt=Topt, Q=Q, a1=a1, p1=p1, ao=ao, po=po, marg=marg)

    print(f"Isometry synthesis (d*N x N), N (cols) = {N_COLS}, b = {B}, single block")
    hdr = f"{'d':>5} | {'rows':>6} |"
    for label, _, _ in METHODS:
        hdr += f" {label+' T(L=1)':>20} | {label+' T(opt)':>20} |"
    print(hdr)
    for i, d in enumerate(D_VALUES):
        row = f"{d:>5} | {d*N_COLS:>6} |"
        for label, _, _ in METHODS:
            row += f" {int(data[label]['T1'][i]):>20,} | {int(data[label]['Topt'][i]):>20,} |"
        print(row)
    for label, _, _ in METHODS:
        dd = data[label]
        print(f"\n{label}: large-d fit (d>={FIT_TAIL}):  Lambda=1: T~{dd['a1']:.3g}*d^{dd['p1']:.3f}"
              f"   opt: T~{dd['ao']:.3g}*d^{dd['po']:.3f}"
              f"   marginal(L=1)->{dd['marg'][-1]:.0f}/unit d")

    # ------------------------------- PDF -------------------------------
    marg_d = ds[1:]
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        # 1) log-log Toffoli, both methods x both regimes, with large-d linear fits
        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        for label, _, color in METHODS:
            dd = data[label]
            ax.plot(ds, dd['T1'], marker="o", color=color, linewidth=1.8, linestyle="-",
                    label=f"{label}  (Λ=1)")
            ax.plot(ds, dd['Topt'], marker="s", color=color, linewidth=1.6, linestyle="--",
                    label=f"{label}  (opt)")
        d_fit = np.array([FIT_TAIL, max(D_VALUES)], dtype=float)
        for label, _, color in METHODS:
            dd = data[label]
            ax.plot(d_fit, dd['a1'] * d_fit ** dd['p1'], color=color, linestyle=":", linewidth=1.0)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("d  =  rows / cols   (rows = d·32,  cols = 32)")
        ax.set_ylabel("Toffoli count")
        ax.set_xticks(D_VALUES); ax.set_xticklabels([str(d) for d in D_VALUES], rotation=45)
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9)
        ax.set_title(f"Isometry synthesis (d·N × N): interferometer vs reflection, Toffoli vs d\n"
                     f"(N={N_COLS}, b={B}, single block) -- both asymptotically LINEAR in d")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 2) marginal Toffoli per unit d (Lambda=1) -> constant proves linearity
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        for label, _, color in METHODS:
            dd = data[label]
            ax.plot(marg_d, dd['marg'], marker="o", color=color, linewidth=1.8,
                    label=f"{label}  (→ {dd['marg'][-1]:.0f}/unit d)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("d"); ax.set_ylabel("marginal Toffoli per unit d  (ΔT/Δd, Λ=1)")
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9.5)
        ax.set_title("Marginal Toffoli per unit d converges to a constant  ⇒  T linear in d")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 3) qubits
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        for label, _, color in METHODS:
            ax.plot(ds, data[label]['Q'], marker="s", color=color, linewidth=1.8, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("d"); ax.set_ylabel("peak logical qubits  (Toffoli-optimal)")
        ax.set_xticks(D_VALUES); ax.set_xticklabels([str(d) for d in D_VALUES], rotation=45)
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9.5)
        ax.set_title(f"Isometry synthesis: peak qubits vs d  (N={N_COLS}, b={B})")
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # 4) table
        fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off")
        cols = ["d", "rows"] + [f"{lbl} T(Λ=1)" for lbl, _, _ in METHODS] + \
               [f"{lbl} T(opt)" for lbl, _, _ in METHODS]
        cells = []
        for i, d in enumerate(D_VALUES):
            cells.append([str(d), str(d * N_COLS)]
                         + [f"{int(data[lbl]['T1'][i]):,}" for lbl, _, _ in METHODS]
                         + [f"{int(data[lbl]['Topt'][i]):,}" for lbl, _, _ in METHODS])
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.8); tbl.scale(1, 1.45)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a"); cell.set_text_props(color="white", weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
        fit_lines = "   ".join(
            f"{lbl}: Λ=1 d^{data[lbl]['p1']:.2f}, opt d^{data[lbl]['po']:.2f}"
            for lbl, _, _ in METHODS)
        ax.set_title(f"Isometry synthesis Toffoli: interferometer vs reflection  (N={N_COLS}, b={B})\n"
                     f"large-d fit (d≥{FIT_TAIL}):  {fit_lines}", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        pdf.infodict()["Title"] = "Isometry Synthesis Toffoli Scaling vs d: Interferometer vs Reflection"
    print(f"\nWrote report: {OUT_PDF}")

    # ------------------------------- email -------------------------------
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = (f"Isometry synthesis: interferometer vs reflection, Toffoli vs d "
                      f"(N={N_COLS}, b={B})")
    body = [
        "Hi,", "",
        f"Joint isometry-synthesis scaling sweep: synthesize the first M = N = {N_COLS} columns of a",
        f"(d·N)-dimensional unitary, growing d = rows/cols over {D_VALUES} (b = {B}, single block).",
        "Same convention as before; two methods compared.", "",
        "Both are LINEAR in d (asymptotic regime); marginal dT/dd (Λ=1) -> a constant:",
    ]
    for label, _, _ in METHODS:
        dd = data[label]
        body.append(f"  {label:>14}:  Λ=1  T ~ {dd['a1']:.3g}·d^{dd['p1']:.3f}"
                     f"  (-> {dd['marg'][-1]:.0f}/unit d);   opt  T ~ {dd['ao']:.3g}·d^{dd['po']:.3f}")
    body += ["",
             f"{'d':>5} | {'rows':>6} | " + " | ".join(
                 f"{lbl[:5]} Λ=1" .rjust(13) + f" | {lbl[:5]} opt".rjust(13) for lbl, _, _ in METHODS)]
    for i, d in enumerate(D_VALUES):
        row = f"{d:>5} | {d*N_COLS:>6} | " + " | ".join(
            f"{int(data[lbl]['T1'][i]):>11,} | {int(data[lbl]['Topt'][i]):>11,}" for lbl, _, _ in METHODS)
        body.append(row)
    body += ["",
             "Interferometer (Sec III B, arXiv:2409.11748): Eq.-36 staircase of d-1 fixed d=2 steps",
             "(V + mux-R_y + 2-block U), each a fixed-size 2M-dim synthesis -> cost linear in d.",
             "Reflection (Sec 4, arXiv:1812.00954): M=32 Householder reflections, each about a (d·N)-dim",
             "state prepared by QROAM rotations -- M fixed, state dimension grows linearly in d.",
             "", "PDF (Toffoli log-log + linear fits, marginal-cost panel, qubits, table) attached.", ""]
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
