#!/usr/bin/env python3
r"""Recursive-CSD optimal-Toffoli scaling for N x N unitaries.

Sweeps the synthesis of a full ``N x N`` unitary (single block, ``N_k = 1``) via
``RecursiveCSDSynthesisQROAM`` at optimal Toffoli (``optimal_T=True``), for
``N = 4, 8, 16, ..., 1024``.  Fits the large-N scaling as a power law ``T ~ a * N^p``
(and reports the ``N (log2 N)^2`` reference, the textbook CSD gate-count scaling).

Emits a PDF (log-log plot with fit + table) and emails it.
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

from integrations.qualtran.recursive_csd_synthesis_QROAM import RecursiveCSDSynthesisQROAM
from integrations.qualtran.utils import get_Toffoli_counts

N_VALUES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
B = 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"recursive_csd_scaling_b{B}.pdf")


def toffoli(N: int) -> int:
    b = RecursiveCSDSynthesisQROAM.from_shape(1, N, B, optimal_T=True)
    return int(get_Toffoli_counts(b))


def main():
    Ns = np.array(N_VALUES, dtype=float)
    T = np.array([toffoli(n) for n in N_VALUES], dtype=float)

    # power-law fit over the large-N tail (N >= 64) where the asymptotics dominate.
    tail = Ns >= 64
    p, loga = np.polyfit(np.log(Ns[tail]), np.log(T[tail]), 1)
    a = np.exp(loga)
    # textbook CSD reference scaling N (log2 N)^2, normalized at the largest N.
    ref = Ns * np.log2(Ns) ** 2
    ref = ref * (T[-1] / ref[-1])

    print(f"Recursive CSD optimal Toffoli (N_k=1, b={B}, optimal_T)")
    print(f"{'N':>6} | {'Toffoli':>14} | {'T/N':>12} | {'T/(N log2^2 N)':>16}")
    for n, t in zip(N_VALUES, T):
        print(f"{n:>6} | {int(t):>14,} | {t/n:>12,.1f} | {t/(n*np.log2(n)**2):>16.2f}")
    print(f"\nLarge-N power-law fit (N>=64):  T ~ {a:.3g} * N^{p:.3f}")

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        # log-log scaling plot
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.plot(Ns, T, marker="o", color="#1f77b4", linewidth=1.8, label="Recursive CSD (optimal T)")
        ax.plot(Ns, a * Ns ** p, color="#d62728", linestyle="--", linewidth=1.4,
                label=f"power-law fit  ~ N^{{{p:.2f}}}  (N>=64)")
        ax.plot(Ns, ref, color="#2ca02c", linestyle=":", linewidth=1.4,
                label="reference  N (log2 N)^2")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("N  (N x N unitary dimension)")
        ax.set_ylabel("Optimal Toffoli count")
        ax.set_xticks(N_VALUES); ax.set_xticklabels([str(n) for n in N_VALUES])
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=9)
        ax.set_title(f"Recursive CSD synthesis: Toffoli scaling vs N  (single unitary, b={B})")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # table
        fig, ax = plt.subplots(figsize=(8, 5)); ax.axis("off")
        cols = ["N", "Toffoli", "T / N", "T / (N log2^2 N)"]
        cells = [[str(n), f"{int(t):,}", f"{t/n:,.1f}", f"{t/(n*np.log2(n)**2):.2f}"]
                 for n, t in zip(N_VALUES, T)]
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a"); cell.set_text_props(color="white", weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
        ax.set_title(f"Recursive CSD optimal Toffoli vs N  (b={B}); fit T ~ N^{p:.2f}", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        pdf.infodict()["Title"] = "Recursive CSD Toffoli Scaling"
    print(f"\nWrote report: {OUT_PDF}")

    # email
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "Recursive CSD: optimal Toffoli scaling vs N (N x N unitary)"
    body = [
        "Hi,", "",
        f"Recursive-CSD optimal Toffoli scaling for N x N unitaries (single block, b={B}):", "",
        f"{'N':>6} | {'Toffoli':>14}",
    ]
    for n, t in zip(N_VALUES, T):
        body.append(f"{n:>6} | {int(t):>14,}")
    body += ["", f"Large-N power-law fit (N>=64):  T ~ {a:.3g} * N^{p:.3f}",
             "(reference textbook CSD scaling: N (log2 N)^2.)", "",
             "PDF (log-log plot with fit + table) attached.", ""]
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
