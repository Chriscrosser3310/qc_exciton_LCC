#!/usr/bin/env python3
"""Central-tensor (W) block-encoding Toffoli comparison: four schemes.

Isolates the *central* block encoder used inside the THC constructions -- the
encoding of the Coulomb kernel W alone (not the outer X matrices) -- and compares the
four schemes head to head:

  1. Diagonal kernel  -- DiagonalCoulombKernelBlockEncoding: sum_{Q,I,J} W^Q_{IJ}
                         |Q,I,J><Q,I,J| via QROAM -> Ry -> QROAM^dag (the direct BE's
                         central piece).
  2. SVD interferometer-- SVDBlockEncodingInterferometer of sum_Q |Q><Q| (x) W^q
                         ("Unitary Synthesis", exchange-US central piece).
  3. Frobenius        -- BlockDiagonalClassicalMatrixBlockEncoding of sum_Q |Q><Q| (x) W^q
                         (Clader U_A = U_L^dag U_R, exchange-Frobenius central piece).
  4. Reflection       -- ReflectionRectangularBlockEncoding of sum_Q |Q><Q| (x) W^q
                         (Householder isometry, n_reflections = N_IP, exchange-Refl piece).

Each is the central bloq extracted from the corresponding full block encoding
(``DirectCoulombBlockEncoding.C_diag`` / ``ExchangeCoulombBlockEncoding.C_inner``) so the
sizing and optimal_T propagation match the four-way full-system report exactly.

Parameters:
  N_IP (central bond) = 256, N_k = k^3 for k = 1..6, phase_bitsize = 32.
  (N_up / N_down are irrelevant to the central tensor and are fixed only to instantiate
  the parent BE; they do not affect the central encoder.)

Each scheme is reported under two QROAM tuning policies (T-opt: closed-form Toffoli
optimum; Q-opt: no blocking -> minimal qubits).  Emits a PDF and emails it.  All counts
come from Qualtran's QECGatesCost / QubitCount walking each bloq's build_call_graph.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from dataclasses import dataclass
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

from integrations.qualtran.direct_Coulomb_block_encoding import DirectCoulombBlockEncoding
from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

N_UP, N_DOWN, N_IP = 4, 22, 256
K_VALUES = [1, 2, 3, 4, 5, 6]
N_K_VALUES = [k ** 3 for k in K_VALUES]
PHASE_BITSIZE = 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"central_tensor_report_NIP{N_IP}_b{PHASE_BITSIZE}.pdf",
)


def _central(variant: str, N_k: int, optimal_T: bool):
    """Return the central W block encoder for one scheme."""
    common = dict(N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k,
                  phase_bitsize=PHASE_BITSIZE, optimal_T=optimal_T)
    if variant == "diag":
        return DirectCoulombBlockEncoding(**common).C_diag
    if variant == "us":
        return ExchangeCoulombBlockEncoding(**common).C_inner
    if variant == "fro":
        return ExchangeCoulombBlockEncoding(use_fro_BE=True, **common).C_inner
    if variant == "refl":
        return ExchangeCoulombBlockEncoding(central_via_reflection=True, **common).C_inner
    raise ValueError(variant)


# (variant, policy) -> (label, color, marker, linestyle)
STYLES = {
    ("diag", "topt"): ("Diagonal kernel, T-opt",     "#1f77b4", "o", "-"),
    ("diag", "qopt"): ("Diagonal kernel, Q-opt",     "#1f77b4", "o", "--"),
    ("us", "topt"):   ("SVD interferometer, T-opt",  "#d62728", "s", "-"),
    ("us", "qopt"):   ("SVD interferometer, Q-opt",  "#d62728", "s", "--"),
    ("fro", "topt"):  ("Frobenius, T-opt",           "#2ca02c", "^", "-"),
    ("fro", "qopt"):  ("Frobenius, Q-opt",           "#2ca02c", "^", "--"),
    ("refl", "topt"): ("Reflection, T-opt",          "#9467bd", "D", "-"),
    ("refl", "qopt"): ("Reflection, Q-opt",          "#9467bd", "D", "--"),
}
VARIANTS = ("diag", "us", "fro", "refl")


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int


def evaluate(variant: str, N_k: int, optimal_T: bool) -> Rec:
    b = _central(variant, N_k, optimal_T)
    return Rec(int(get_Toffoli_counts(b)), int(get_qubit_counts(b)))


def main():
    print("=" * 78)
    print(f"Central-tensor (W) BE comparison: N_IP={N_IP}, b={PHASE_BITSIZE}")
    print(f"  N_k = k^3 for k in {K_VALUES}")
    print("=" * 78)

    rows = []  # (k, N_k, dict[(variant, policy)] -> Rec)
    for k, N_k in zip(K_VALUES, N_K_VALUES):
        recs = {}
        for variant in VARIANTS:
            recs[(variant, "topt")] = evaluate(variant, N_k, True)
            recs[(variant, "qopt")] = evaluate(variant, N_k, False)
        rows.append((k, N_k, recs))
        print(f"k={k} (N_k={N_k}):")
        for key in STYLES:
            r = recs[key]
            print(f"    {STYLES[key][0]:28s}  T={r.toffoli:>14,}  Q={r.qubits}")

    Karr = np.asarray(N_K_VALUES, dtype=float)

    def plot(metric, ylabel):
        fig, ax = plt.subplots(figsize=(9.8, 6.0))
        for key, (lbl, color, marker, ls) in STYLES.items():
            data = np.array([getattr(r[2][key], metric) for r in rows], dtype=float)
            ax.plot(Karr, data, marker=marker, linestyle=ls, color=color,
                    linewidth=1.7, markersize=6, label=lbl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N_k = k^3")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_K_VALUES)
        ax.set_xticklabels([f"{n}\nk={k}" for n, k in zip(N_K_VALUES, K_VALUES)])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        ax.set_title(
            f"{ylabel}: central W block encoder, four schemes  "
            f"(N_IP={N_IP}, b={PHASE_BITSIZE})"
        )
        fig.tight_layout()
        return fig

    def _metric_table(ax, metric, fmt, title):
        ax.axis("off")
        cols = ["central scheme"] + [f"N_k={n}" for n in N_K_VALUES]
        cells = []
        for key in STYLES:
            label = STYLES[key][0]
            vals = [fmt(getattr(r[2][key], metric)) for r in rows]
            cells.append([label] + vals)
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.55)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
            if c == 0:
                cell.set_text_props(ha="left")
        ax.set_title(title, fontsize=12)

    def toffoli_table_page():
        fig, ax = plt.subplots(figsize=(12, 4.5))
        _metric_table(ax, "toffoli", lambda v: f"{v:,}", "Toffoli count: central W block encoder, four schemes")
        return fig

    def qubit_table_page():
        fig, ax = plt.subplots(figsize=(12, 4.5))
        _metric_table(ax, "qubits", lambda v: f"{v:,}", "Peak logical qubits: central W block encoder, four schemes")
        return fig

    def summary_page():
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.axis("off")
        lines = [
            "Central-tensor (W) block-encoding comparison: four schemes",
            "",
            f"Parameters:  N_IP (central bond) = {N_IP}",
            f"             N_k = k^3 for k in {K_VALUES} (so N_k in {N_K_VALUES})",
            f"             phase_bitsize b = {PHASE_BITSIZE}",
            "",
            "This isolates the CENTRAL block encoder of the Coulomb kernel W only -- the",
            "outer X-matrix block encodings are excluded.  Each bloq is the central piece",
            "extracted from the corresponding full THC block encoding (C_diag / C_inner),",
            "so sizing and optimal_T propagation match the four-way full-system report.",
            "",
            "  1. Diagonal kernel: sum_{Q,I,J} W^Q_{IJ} |Q,I,J><Q,I,J| via QROAM -> Ry ->",
            "     QROAM^dag (the direct BE's central piece; W treated as diagonal in I,J).",
            "  2. SVD interferometer: sum_Q |Q><Q| (x) W^q via the block-unitary",
            "     interferometer (exchange Unitary-Synthesis central piece).",
            "  3. Frobenius: sum_Q |Q><Q| (x) W^q via the Clader block-matrix scheme",
            "     (BlockDiagonalClassicalMatrixBlockEncoding, U_A = U_L^dag U_R).",
            "  4. Reflection: sum_Q |Q><Q| (x) W^q via the Householder reflection isometry",
            "     (ReflectionRectangularBlockEncoding, n_reflections = N_IP).",
            "",
            "Tuning policies (per curve):",
            "  T-opt: every QROAM log_block_sizes at the closed-form Toffoli optimum.",
            "  Q-opt: no QROAM blocking (log_block_sizes = 0) -> minimal qubits.",
            "",
            "Note: schemes 1-3 (diagonal / Frobenius) trade many qubits for low T at T-opt;",
            "  their Q-opt qubit counts grow only logarithmically in N_k.",
            "",
            "All Toffoli / qubit counts come from Qualtran's QECGatesCost / QubitCount.",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9.0)
        fig.tight_layout()
        return fig

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        for make in (
            summary_page,
            lambda: plot("toffoli", "Toffoli count"),
            lambda: plot("qubits", "Peak logical qubits"),
            toffoli_table_page,
            qubit_table_page,
        ):
            fig = make()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "Central Tensor (W) Block Encoding: Four Schemes"
    print(f"\nWrote report: {OUT_PDF}")

    # email
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "Central-tensor (W) block-encoding comparison: four schemes"
    body_lines = [
        "Hi,",
        "",
        "Attached: central-tensor (Coulomb kernel W) block-encoding comparison for",
        f"  N_IP (central bond) = {N_IP}, N_k = k^3 for k in {K_VALUES}, phase_bitsize = {PHASE_BITSIZE}",
        "",
        "Four central schemes: diagonal kernel (direct), SVD interferometer (exchange US),",
        "Frobenius block-matrix (use_fro_BE), and reflection isometry (central_via_reflection).",
        "Outer X-matrix encodings are excluded -- this is the central W encoder only.",
        "Counts from Qualtran's QECGatesCost / QubitCount walking build_call_graph.",
        "",
    ]
    for (k, N_k, recs) in rows:
        body_lines.append(f"  k={k} (N_k={N_k:3d}):")
        for key in STYLES:
            r = recs[key]
            body_lines.append(f"      {STYLES[key][0]:28s} T={r.toffoli:>13,}  Q={r.qubits}")
    msg.attach(MIMEText("\n".join(body_lines), "plain"))
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
        if proc.returncode == 0:
            print(f"Email sent via sendmail to {RECIPIENT}")
        else:
            print(f"sendmail failed: {proc.stderr.decode(errors='replace')[:200]}")


if __name__ == "__main__":
    main()
