#!/usr/bin/env python3
"""X-matrix (interpolating-vector) block-encoding Toffoli comparison: up vs down.

Isolates the *outer* block encoders of the THC constructions -- the encoders of the
interpolating-vector matrices X (not the central Coulomb kernel W) -- for the up and
down channels.  Both the direct and exchange constructions use the same outer encoders,
so this is shared across all four full-system schemes.

  * X_up   -- ReflectionRectangularBlockEncoding of sum_k |k><k| (x) X^{up,k}
              (N_up x N_IP isometry; n_reflections = min(N_up, N_IP)).
  * X_down -- ReflectionRectangularBlockEncoding of sum_k |k><k| (x) X^{down,k}
              (N_down x N_IP isometry; n_reflections = min(N_down, N_IP)).

Each is the outer bloq extracted from the full block encoding
(``ExchangeCoulombBlockEncoding.B_up`` / ``.B_down``; identical to the direct BE's
``B_mu`` / ``B_lam``), so sizing and optimal_T propagation match the full-system reports.

Parameters:
  N_up = 4, N_down = 22, N_IP (central bond) = 256, N_k = k^3 for k = 1..6,
  phase_bitsize = 32.

Each channel is reported under two QROAM tuning policies (T-opt: closed-form Toffoli
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

from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

N_UP, N_DOWN, N_IP = 4, 22, 256
K_VALUES = [1, 2, 3, 4, 5, 6]
N_K_VALUES = [k ** 3 for k in K_VALUES]
PHASE_BITSIZE = 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"x_matrix_report_Nup{N_UP}_Ndown{N_DOWN}_NIP{N_IP}_b{PHASE_BITSIZE}.pdf",
)


def _x(side: str, N_k: int, optimal_T: bool):
    """Return the outer X-matrix block encoder for one channel."""
    e = ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k,
        phase_bitsize=PHASE_BITSIZE, optimal_T=optimal_T,
    )
    return e.B_up if side == "up" else e.B_down


# (side, policy) -> (label, color, marker, linestyle)
STYLES = {
    ("up", "topt"):   (f"X_up (N_up={N_UP}), T-opt",      "#1f77b4", "o", "-"),
    ("up", "qopt"):   (f"X_up (N_up={N_UP}), Q-opt",      "#1f77b4", "o", "--"),
    ("down", "topt"): (f"X_down (N_down={N_DOWN}), T-opt", "#d62728", "s", "-"),
    ("down", "qopt"): (f"X_down (N_down={N_DOWN}), Q-opt", "#d62728", "s", "--"),
}
SIDES = ("up", "down")


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int


def evaluate(side: str, N_k: int, optimal_T: bool) -> Rec:
    b = _x(side, N_k, optimal_T)
    return Rec(int(get_Toffoli_counts(b)), int(get_qubit_counts(b)))


def main():
    print("=" * 78)
    print(f"X-matrix (outer) BE comparison: N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP}, b={PHASE_BITSIZE}")
    print(f"  N_k = k^3 for k in {K_VALUES}")
    print("=" * 78)

    rows = []  # (k, N_k, dict[(side, policy)] -> Rec)
    for k, N_k in zip(K_VALUES, N_K_VALUES):
        recs = {}
        for side in SIDES:
            recs[(side, "topt")] = evaluate(side, N_k, True)
            recs[(side, "qopt")] = evaluate(side, N_k, False)
        rows.append((k, N_k, recs))
        print(f"k={k} (N_k={N_k}):")
        for key in STYLES:
            r = recs[key]
            print(f"    {STYLES[key][0]:28s}  T={r.toffoli:>12,}  Q={r.qubits}")

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
        ax.legend(fontsize=9, ncol=2)
        ax.set_title(
            f"{ylabel}: outer X-matrix block encoders (up vs down)  "
            f"(N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP}, b={PHASE_BITSIZE})"
        )
        fig.tight_layout()
        return fig

    def _metric_table(ax, metric, fmt, title):
        ax.axis("off")
        cols = ["X encoder"] + [f"N_k={n}" for n in N_K_VALUES]
        cells = []
        for key in STYLES:
            label = STYLES[key][0]
            vals = [fmt(getattr(r[2][key], metric)) for r in rows]
            cells.append([label] + vals)
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
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
        fig, ax = plt.subplots(figsize=(12, 3.2))
        _metric_table(ax, "toffoli", lambda v: f"{v:,}", "Toffoli count: outer X-matrix block encoders")
        return fig

    def qubit_table_page():
        fig, ax = plt.subplots(figsize=(12, 3.2))
        _metric_table(ax, "qubits", lambda v: f"{v:,}", "Peak logical qubits: outer X-matrix block encoders")
        return fig

    def summary_page():
        fig, ax = plt.subplots(figsize=(10, 7.0))
        ax.axis("off")
        lines = [
            "Outer X-matrix block-encoding comparison: up vs down",
            "",
            f"Parameters:  N_up = {N_UP},  N_down = {N_DOWN},  N_IP (central bond) = {N_IP}",
            f"             N_k = k^3 for k in {K_VALUES} (so N_k in {N_K_VALUES})",
            f"             phase_bitsize b = {PHASE_BITSIZE}",
            "",
            "This isolates the OUTER block encoders of the interpolating-vector matrices X",
            "only -- the central Coulomb kernel W is excluded.  Both direct and exchange",
            "use the same outer encoders, so these costs are shared across all four schemes.",
            "Each is extracted from the full BE (ExchangeCoulombBlockEncoding.B_up / .B_down,",
            "identical to the direct BE's B_mu / B_lam).",
            "",
            "  X_up:   sum_k |k><k| (x) X^{up,k}   (N_up x N_IP isometry via",
            f"          ReflectionRectangularBlockEncoding, n_reflections = min(N_up, N_IP) = {min(N_UP, N_IP)}).",
            "  X_down: sum_k |k><k| (x) X^{down,k} (N_down x N_IP isometry,",
            f"          n_reflections = min(N_down, N_IP) = {min(N_DOWN, N_IP)}).",
            "",
            "Both isometries are padded to n_rows = next_pow2(max(N, N_IP)) = "
            f"{1 << max(1, (max(N_DOWN, N_IP) - 1).bit_length())}, so the cost",
            "scales with the reflection count (= short dimension), not the padded size --",
            "hence X_down (22 reflections) costs ~5.5x X_up (4 reflections).",
            "",
            "Tuning policies (per curve):",
            "  T-opt: every QROAM log_block_sizes at the closed-form Toffoli optimum.",
            "  Q-opt: no QROAM blocking (log_block_sizes = 0) -> minimal qubits",
            "         (qubit count then grows only logarithmically in N_k).",
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
        info["Title"] = "Outer X-Matrix Block Encoding: Up vs Down"
    print(f"\nWrote report: {OUT_PDF}")

    # email
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "Outer X-matrix block-encoding comparison: up vs down"
    body_lines = [
        "Hi,",
        "",
        "Attached: outer X-matrix (interpolating-vector) block-encoding comparison for",
        f"  N_up = {N_UP}, N_down = {N_DOWN}, N_IP (central bond) = {N_IP}, "
        f"N_k = k^3 for k in {K_VALUES}, phase_bitsize = {PHASE_BITSIZE}",
        "",
        "Up vs down channels (ReflectionRectangularBlockEncoding); the central W kernel is",
        "excluded.  These outer encoders are shared by both direct and exchange.",
        "Counts from Qualtran's QECGatesCost / QubitCount walking build_call_graph.",
        "",
    ]
    for (k, N_k, recs) in rows:
        body_lines.append(f"  k={k} (N_k={N_k:3d}):")
        for key in STYLES:
            r = recs[key]
            body_lines.append(f"      {STYLES[key][0]:28s} T={r.toffoli:>12,}  Q={r.qubits}")
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
