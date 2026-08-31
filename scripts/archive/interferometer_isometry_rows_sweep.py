#!/usr/bin/env python3
r"""Isometry synthesis optimal-Toffoli sweep over the synthesized dimension: two methods.

Two ways to block-encode an ``N x M`` isometry (synthesize ``M`` columns of an
``N``-dimensional block unitary), with ambient ``N = 256`` fixed and ``M`` swept over
``1, 2, 4, ..., 256``, at optimal Toffoli (``optimal_T=True``):

  * Interferometer isometry (``BlockInterferometerIsometrySynthesisQROAM``): thin-row-CSD
    recursion; requires ``M >= 4`` and power-of-two, so ``M = 1, 2`` are n/a.
  * Direct column-by-column isometry (``ColumnIsometryRectangularBlockEncoding``): Iten
    column-by-column synthesis; allows arbitrary ``M`` (incl. 1, 2).

Reported for a single block (``N_k = 1``) and the session block count (``N_k = 216``).
Emits a PDF (table + log-log plot) and emails it.
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

from integrations.qualtran.block_interferometer_isometry_QROAM import (
    BlockInterferometerIsometrySynthesisQROAM,
)
from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    ColumnIsometryRectangularBlockEncoding,
)
from integrations.qualtran.utils import get_Toffoli_counts

N_ROWS = 256                       # fixed ambient dimension ("cols" = 256)
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]   # synthesized column count ("rows")
N_K_VALUES = [1, 216]
B = 32
MIN_M = 4                          # interferometer-isometry floor
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"isometry_synthesis_rows_sweep_N{N_ROWS}_b{B}.pdf")

# method key -> (label, callable(n_blocks, m) -> toffoli or None)
def _intf(n_blocks, m):
    if m < MIN_M:
        return None
    b = BlockInterferometerIsometrySynthesisQROAM.from_shape(
        n_blocks=n_blocks, n_rows=N_ROWS, n_cols=m, phase_bitsize=B, optimal_T=True)
    return int(get_Toffoli_counts(b))


def _col(n_blocks, m):
    b = ColumnIsometryRectangularBlockEncoding(
        n_blocks=n_blocks, n_rows=N_ROWS, n_reflections=m, phase_bitsize=B, optimal_T=True)
    return int(get_Toffoli_counts(b))


METHODS = [("Interferometer", _intf), ("Column", _col)]


SERIES = [(meth, lbl, nk) for (lbl, meth) in METHODS for nk in N_K_VALUES]
COL_HEADERS = [f"{lbl} N_k={nk}" for (lbl, _) in METHODS for nk in N_K_VALUES]


def main():
    # data[m][(label, nk)] = toffoli or None
    data = {m: {(lbl, nk): fn(nk, m) for (lbl, fn) in METHODS for nk in N_K_VALUES}
            for m in M_VALUES}

    def fmt(v):
        return "n/a (M<4)" if v is None else f"{v:,}"

    print(f"Isometry synthesis, ambient N (cols) = {N_ROWS}, b = {B}, optimal_T")
    print(f"{'M':>5} | " + " | ".join(f"{h:>22}" for h in COL_HEADERS))
    for m in M_VALUES:
        print(f"{m:>5} | " + " | ".join(
            f"{fmt(data[m][(lbl, nk)]):>22}" for (lbl, _) in METHODS for nk in N_K_VALUES))

    # ------------------------------- PDF -------------------------------
    styles = {("Interferometer", 1): ("#1f77b4", "o", "-"),
              ("Interferometer", 216): ("#d62728", "s", "-"),
              ("Column", 1): ("#1f77b4", "o", "--"),
              ("Column", 216): ("#d62728", "s", "--")}
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for (lbl, fn) in METHODS:
            for nk in N_K_VALUES:
                xs = [m for m in M_VALUES if data[m][(lbl, nk)] is not None]
                ys = [data[m][(lbl, nk)] for m in xs]
                color, marker, ls = styles[(lbl, nk)]
                ax.plot(xs, ys, marker=marker, color=color, linestyle=ls, linewidth=1.7,
                        label=f"{lbl}, N_k={nk}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("M = synthesized column count  (ambient N = 256)")
        ax.set_ylabel("Optimal Toffoli count")
        ax.set_xticks(M_VALUES); ax.set_xticklabels([str(m) for m in M_VALUES])
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9)
        ax.set_title(f"Isometry synthesis: interferometer vs column, Toffoli vs M (N={N_ROWS}, b={B})")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off")
        cols = ["M"] + COL_HEADERS
        cells = [[str(m)] + [fmt(data[m][(lbl, nk)]) for (lbl, _) in METHODS for nk in N_K_VALUES]
                 for m in M_VALUES]
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a"); cell.set_text_props(color="white", weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
        ax.set_title(f"Isometry synthesis optimal Toffoli: interferometer vs column "
                     f"(ambient N = {N_ROWS}, b = {B})", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        pdf.infodict()["Title"] = "Isometry Synthesis Toffoli Sweep: Interferometer vs Column"
    print(f"\nWrote report: {OUT_PDF}")

    # ------------------------------- email -------------------------------
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "Isometry synthesis: interferometer vs column, optimal Toffoli vs M (N=256)"
    body = [
        "Hi,", "",
        f"Isometry-synthesis optimal Toffoli sweep, ambient N (cols) = {N_ROWS}, b = {B}.",
        "M = number of synthesized columns; interferometer requires M >= 4 (else n/a),",
        "column-by-column allows any M.", "",
        f"{'M':>5} | " + " | ".join(f"{h}" for h in COL_HEADERS),
    ]
    for m in M_VALUES:
        body.append(f"{m:>5} | " + " | ".join(
            fmt(data[m][(lbl, nk)]) for (lbl, _) in METHODS for nk in N_K_VALUES))
    body += ["", "PDF (table + log-log plot) attached.", ""]
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
