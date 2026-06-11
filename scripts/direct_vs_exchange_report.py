#!/usr/bin/env python3
"""Four-way THC block-encoding Toffoli-count comparison.

Compares the Toffoli count of four block encodings of the periodic THC two-electron
integral tensor (arXiv:2601.16379 Eq. 9):

  1. direct       -- DirectCoulombBlockEncoding (diagonal central Coulomb kernel).
  2. exchange-US  -- ExchangeCoulombBlockEncoding with the central W^q encoded by the SVD
                     interferometer ("Unitary Synthesis"), the default.
  3. exchange-Fro -- ExchangeCoulombBlockEncoding with the central W^q encoded by the
                     Clader-Frobenius block-matrix scheme (``use_fro_BE=True``;
                     BlockDiagonalClassicalMatrixBlockEncoding, U_A = U_L^dag U_R).
  4. exchange-Refl-- ExchangeCoulombBlockEncoding with the central W^q encoded by the
                     Householder reflection isometry (``central_via_reflection=True``;
                     ReflectionRectangularBlockEncoding with n_reflections = N_IP).

Parameters:
  N_up = 4, N_down = 22, N_IP (central bond) = 256, N_k = k^3 for k = 1..6
  (so N_k in {1, 8, 27, 64, 125, 216}), phase_bitsize = 32.

Each variant is reported under two QROAM tuning policies:
  * T-opt: every QROAM ``log_block_sizes`` at the closed-form Toffoli optimum
    (``optimal_T=True``).
  * Q-opt: no QROAM blocking (``log_block_sizes = 0``) -> minimal qubits.

Emits a PDF (Toffoli plot + qubit plot + table) and emails it.  All counts come from
Qualtran's QECGatesCost / QubitCount walking each bloq's build_call_graph.
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
    REPO_ROOT, "docs",
    f"thc_be_four_way_Nup{N_UP}_Ndown{N_DOWN}_NIP{N_IP}_b{PHASE_BITSIZE}.pdf",
)

# variant key -> (constructor(N_k, optimal_T) -> bloq, pretty label).
VARIANTS = {
    "direct": lambda N_k, opt: DirectCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE, optimal_T=opt
    ),
    "exch_us": lambda N_k, opt: ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE,
        optimal_T=opt, use_fro_BE=False,
    ),
    "exch_fro": lambda N_k, opt: ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE,
        optimal_T=opt, use_fro_BE=True,
    ),
    "exch_refl": lambda N_k, opt: ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE,
        optimal_T=opt, central_via_reflection=True,
    ),
}

# (variant, policy) -> (label, color, marker, linestyle)
STYLES = {
    ("direct", "topt"):    ("Direct, T-opt",             "#1f77b4", "o", "-"),
    ("direct", "qopt"):    ("Direct, Q-opt",             "#1f77b4", "o", "--"),
    ("exch_us", "topt"):   ("Exchange US, T-opt",        "#d62728", "s", "-"),
    ("exch_us", "qopt"):   ("Exchange US, Q-opt",        "#d62728", "s", "--"),
    ("exch_fro", "topt"):  ("Exchange Frobenius, T-opt", "#2ca02c", "^", "-"),
    ("exch_fro", "qopt"):  ("Exchange Frobenius, Q-opt", "#2ca02c", "^", "--"),
    ("exch_refl", "topt"): ("Exchange Reflection, T-opt", "#9467bd", "D", "-"),
    ("exch_refl", "qopt"): ("Exchange Reflection, Q-opt", "#9467bd", "D", "--"),
}


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int


def evaluate(variant: str, N_k: int, optimal_T: bool) -> Rec:
    b = VARIANTS[variant](N_k, optimal_T)
    return Rec(int(get_Toffoli_counts(b)), int(get_qubit_counts(b)))


def main():
    print("=" * 78)
    print(f"Four-way THC BE: N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP}")
    print(f"  N_k = k^3 for k in {K_VALUES}; phase_bitsize={PHASE_BITSIZE}")
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
            print(f"    {STYLES[key][0]:24s}  T={r.toffoli:>14,}  Q={r.qubits}")

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
            f"{ylabel}: THC block encoding, four constructions  "
            f"(N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP}, b={PHASE_BITSIZE})"
        )
        fig.tight_layout()
        return fig

    def _metric_table(ax, metric, fmt, title):
        # Transposed layout: one row per (variant, policy) series, one column per N_k.
        ax.axis("off")
        cols = ["construction"] + [f"N_k={n}" for n in N_K_VALUES]
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
        _metric_table(ax, "toffoli", lambda v: f"{v:,}", "Toffoli count: THC block encoding, four constructions")
        return fig

    def qubit_table_page():
        fig, ax = plt.subplots(figsize=(12, 4.5))
        _metric_table(ax, "qubits", lambda v: f"{v:,}", "Peak logical qubits: THC block encoding, four constructions")
        return fig

    def summary_page():
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.axis("off")
        lines = [
            "THC block-encoding Toffoli comparison: four constructions",
            "",
            f"Parameters:  N_up = {N_UP},  N_down = {N_DOWN},  N_IP (central bond) = {N_IP}",
            f"             N_k = k^3 for k in {K_VALUES} (so N_k in {N_K_VALUES})",
            f"             phase_bitsize b = {PHASE_BITSIZE}",
            "",
            "All four encode the periodic THC tensor (arXiv:2601.16379 Eq. 9):",
            "",
            "  1. Direct: orientation flipped -- inputs (mu,k_mu),(lam,k_lam), outputs the",
            "     other two.  |Q> uniform LCU, two X reflections, two momentum modular adds",
            "     k += Q, central DIAGONAL kernel sum W^Q_{IJ}|QIJ><QIJ| (QROAM->Ry->QROAM^dag),",
            "     then the two X^dagger.",
            "",
            "  Exchange variants share bra-pair X reflections; they differ in the central W^q:",
            "  2. Exchange US (Unitary Synthesis): central W^q via the SVD block-unitary",
            "     interferometer (default central encoder).",
            "  3. Exchange Frobenius (use_fro_BE=True): central W^q via the Clader-Frobenius",
            "     block-matrix scheme (BlockDiagonalClassicalMatrixBlockEncoding, U_A=U_L^dag U_R).",
            "  4. Exchange Reflection (central_via_reflection=True): central W^q via the",
            "     Householder reflection isometry (ReflectionRectangularBlockEncoding, N_IP refl).",
            "",
            "Tuning policies (per curve):",
            "  T-opt: every QROAM log_block_sizes at the closed-form Toffoli optimum.",
            "  Q-opt: no QROAM blocking (log_block_sizes = 0) -> minimal qubits.",
            "",
            "All Toffoli / qubit counts come from Qualtran's QECGatesCost / QubitCount",
            "  walking each bloq's build_call_graph (no analytic formulas).",
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
        info["Title"] = "THC Block Encoding: Direct vs Exchange (US / Frobenius / Reflection)"
    print(f"\nWrote report: {OUT_PDF}")

    # email
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "THC block encoding: direct vs exchange (US / Frobenius / Reflection) Toffoli comparison"
    body_lines = [
        "Hi,",
        "",
        "Attached: four-way THC block-encoding Toffoli/qubit comparison for",
        f"  N_up = {N_UP}, N_down = {N_DOWN}, N_IP (central bond) = {N_IP}",
        f"  N_k = k^3 for k in {K_VALUES}",
        f"  phase_bitsize = {PHASE_BITSIZE}",
        "",
        "Four constructions: direct; exchange with central W via Unitary Synthesis (SVD),",
        "via Frobenius block-matrix (use_fro_BE=True), and via reflection isometry",
        "(central_via_reflection=True).",
        "Counts from Qualtran's QECGatesCost / QubitCount walking build_call_graph.",
        "",
    ]
    for (k, N_k, recs) in rows:
        body_lines.append(f"  k={k} (N_k={N_k:3d}):")
        for key in STYLES:
            r = recs[key]
            body_lines.append(
                f"      {STYLES[key][0]:26s} T={r.toffoli:>13,}  Q={r.qubits}"
            )
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
