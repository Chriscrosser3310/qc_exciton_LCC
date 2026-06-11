#!/usr/bin/env python3
"""Resource comparison: column-by-column isometry synthesis vs reflection vs interferometer.

Compares three QROAM-rotation synthesis methods for an ``m -> n`` isometry (the first ``K`` columns
of an ``N x N`` unitary, ``N = 2^n``), all at their Toffoli-optimal QROAM block sizes:

  * column-by-column  -- ``BlockIsometryColumnSynthesisQROAM`` (Iten 1501.06911 + Berry 2409.11748 Eq.24)
  * reflection (LKS)  -- ``BlockUnitaryReflectionQROAM`` (arXiv:1812.00954 Sec. 4 Householder)
  * interferometer    -- ``BlockUnitaryInterferometerSynthesisQROAM`` (Berry 2409.11748 Sec. III A,
                          full-unitary only -- it always synthesizes all N columns)

For each method we plot Toffoli (n_ccz-equivalent) and peak-qubit counts vs ``N`` for the full unitary
(``K = N``) and the half-column isometry (``K = N/2``), with power-law fits.  Writes a multi-page PDF to
``docs/`` and, when ``SEND_EMAIL=1``, emails it.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    BlockIsometryColumnSynthesisQROAM,
)
from integrations.qualtran.block_unitary_interferometer_QROAM import (
    BlockUnitaryInterferometerSynthesisQROAM,
)
from integrations.qualtran.block_unitary_reflection_QROAM import BlockUnitaryReflectionQROAM
from integrations.qualtran.utils import get_qubit_counts, get_Toffoli_counts

N_QUBITS = [3, 4, 5, 6, 7, 8]
N_ROWS = [1 << n for n in N_QUBITS]
PHASE_BITSIZE = 32
N_BLOCKS = 1
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"isometry_column_synthesis_vs_reflection_vs_interferometer_b{PHASE_BITSIZE}.pdf"
)


@dataclass(frozen=True)
class Record:
    n: int
    N: int
    K: int
    toffoli: int
    qubits: int


def _record(bloq, n: int, N: int, K: int) -> Optional[Record]:
    try:
        return Record(n, N, K, int(get_Toffoli_counts(bloq)), int(get_qubit_counts(bloq)))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    skipped n={n} K={K}: {type(exc).__name__}: {exc}")
        return None


def colcol(n: int, N: int, K: int) -> Optional[Record]:
    bloq = BlockIsometryColumnSynthesisQROAM.from_shape(
        N_BLOCKS, N, PHASE_BITSIZE, n_cols=K, optimal_T=True
    )
    return _record(bloq, n, N, K)


def reflection(n: int, N: int, K: int) -> Optional[Record]:
    bloq = BlockUnitaryReflectionQROAM.from_shape(
        n_blocks=N_BLOCKS, n_rows=N, phase_bitsize=PHASE_BITSIZE, n_reflections=K, optimal_T=True
    )
    return _record(bloq, n, N, K)


def interferometer(n: int, N: int) -> Optional[Record]:
    bloq = BlockUnitaryInterferometerSynthesisQROAM.from_shape(
        N_BLOCKS, N, PHASE_BITSIZE, optimal_T=True
    )
    return _record(bloq, n, N, N)


SERIES_BUILDERS = {
    "colcol_full": lambda n, N: colcol(n, N, N),
    "colcol_half": lambda n, N: colcol(n, N, max(1, N // 2)),
    "reflection_full": lambda n, N: reflection(n, N, N),
    "reflection_half": lambda n, N: reflection(n, N, max(1, N // 2)),
    "interferometer_full": lambda n, N: interferometer(n, N),
}
LABELS = {
    "colcol_full": "Column-by-column, K=N",
    "colcol_half": "Column-by-column, K=N/2 (isometry)",
    "reflection_full": "Reflection (LKS), K=N",
    "reflection_half": "Reflection (LKS), K=N/2 (isometry)",
    "interferometer_full": "Interferometer, full unitary",
}
COLORS = {
    "colcol_full": "#1f77b4",
    "colcol_half": "#17becf",
    "reflection_full": "#d62728",
    "reflection_half": "#ff7f0e",
    "interferometer_full": "#2ca02c",
}
MARKERS = {
    "colcol_full": "o",
    "colcol_half": "v",
    "reflection_full": "^",
    "reflection_half": "s",
    "interferometer_full": "D",
}


def fit_power_law(x: np.ndarray, y: List[int]) -> tuple:
    alpha, log_c = np.polyfit(np.log(x), np.log(np.asarray(y, dtype=float)), 1)
    return float(alpha), float(np.exp(log_c))


print("=" * 78)
print("Isometry synthesis comparison (column-by-column vs reflection vs interferometer)")
print(f"  n = {N_QUBITS}  (N = 2^n), phase_bitsize b = {PHASE_BITSIZE}, n_blocks = {N_BLOCKS}")
print("=" * 78)

records: Dict[str, List[Record]] = {name: [] for name in SERIES_BUILDERS}
for n, N in zip(N_QUBITS, N_ROWS):
    print(f"n={n} N={N}")
    for name, build in SERIES_BUILDERS.items():
        rec = build(n, N)
        if rec is not None:
            records[name].append(rec)
            print(f"  {LABELS[name]:38s} T={rec.toffoli:>12,}  Q={rec.qubits}")

Nvec = np.asarray(N_ROWS, dtype=float)
fits = {
    name: {
        "toffoli": fit_power_law(Nvec, [r.toffoli for r in recs]),
        "qubits": fit_power_law(Nvec, [r.qubits for r in recs]),
    }
    for name, recs in records.items()
    if recs
}

print("\nPower-law fits y = c * N^alpha")
for name, fit in fits.items():
    ta, tc = fit["toffoli"]
    qa, qc = fit["qubits"]
    print(f"  {LABELS[name]:38s} Toffoli alpha={ta:.3f} c={tc:.2e}; qubits alpha={qa:.3f} c={qc:.2e}")


def plot_metric(metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for name, recs in records.items():
        if not recs:
            continue
        xs = [r.N for r in recs]
        ys = [getattr(r, metric) for r in recs]
        alpha, c = fits[name][metric]
        ax.plot(xs, ys, marker=MARKERS[name], color=COLORS[name], linewidth=1.7, label=LABELS[name])
        xfit = np.linspace(min(xs), max(xs), 200)
        ax.plot(xfit, c * xfit ** alpha, ":", color=COLORS[name], linewidth=1.0, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N = 2^n (matrix dimension)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(N_ROWS)
    ax.set_xticklabels([str(N) for N in N_ROWS])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(f"{ylabel} vs N for isometry synthesis methods (b={PHASE_BITSIZE})")
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.axis("off")
    cols = ["n", "N"] + [LABELS[name] + " T" for name in records]
    rows = []
    for i, n in enumerate(N_QUBITS):
        row = [n, N_ROWS[i]]
        for name in records:
            recs = records[name]
            row.append(f"{recs[i].toffoli:,}" if i < len(recs) else "-")
        rows.append(row)
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#25364a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2:
            cell.set_facecolor("#f3f6fa")
    ax.set_title("Toffoli (n_ccz) counts by method (Toffoli-optimal QROAM)", fontsize=12)
    return fig


def summary_page():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    lines = [
        "Column-by-column isometry synthesis (Iten 1501.06911) with Berry Eq. 24 QROAM layers",
        "",
        f"n = {N_QUBITS}  (N = 2^n),  phase precision b = {PHASE_BITSIZE},  n_blocks = {N_BLOCKS}.",
        "All curves use the Toffoli-optimal QROAM block size (optimal_T=True).",
        "Toffoli = get_Toffoli_counts (n_ccz-equivalent); qubits = QubitCount peak logical.",
        "",
        "Methods:",
        "  - Column-by-column: K column operations, each a staircase of merged multiplexed",
        "      single-qubit gates (Delta_s . C^u_{n-1-s}(U^u_s)) realized via Eq. 24",
        "      (R(phi) H R(theta) H + QROAM phases) plus subleading multi-controlled gates.",
        "      Leading cost ~ K * 2^n / Lambda; synthesizes only the K needed columns.",
        "  - Reflection (LKS): K Householder reflections (arXiv:1812.00954 Sec. 4).",
        "  - Interferometer: full-unitary rectangular mesh (Berry Sec. III A); always N columns.",
        "",
        "Block-aware: BlockIsometryColumnSynthesisQROAM carries an optional read-only block",
        "  register (sum_j |j><j| (x) V_j); shown here with n_blocks=1 (single isometry).",
        "",
        "Fitted power laws y = c * N^alpha:",
    ]
    for name in records:
        if name not in fits:
            continue
        ta, tc = fits[name]["toffoli"]
        qa, qc = fits[name]["qubits"]
        lines.append(f"  {LABELS[name]}: T alpha={ta:.3f} c={tc:.2e}; Q alpha={qa:.3f} c={qc:.2e}")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.0)
    fig.tight_layout()
    return fig


os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with pdf_backend.PdfPages(OUT_PDF) as pdf:
    for make in (
        summary_page,
        lambda: plot_metric("toffoli", "Toffoli count (n_ccz)"),
        lambda: plot_metric("qubits", "Peak logical qubits"),
        table_page,
    ):
        fig = make()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    info = pdf.infodict()
    info["Title"] = "Isometry column-by-column QROAM resource comparison"
    info["Author"] = "qc_exciton_LCC"

print(f"\nWrote report: {OUT_PDF}")


def send_email(recipient: str, pdf_path: str) -> bool:
    fit_lines = []
    for name in records:
        if name not in fits:
            continue
        ta, tc = fits[name]["toffoli"]
        qa, qc = fits[name]["qubits"]
        fit_lines.append(f"{LABELS[name]}: Toffoli alpha={ta:.3f}, c={tc:.2e}; qubits alpha={qa:.3f}, c={qc:.2e}")
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached is the isometry-synthesis resource comparison: the new column-by-column
        synthesizer (Iten 1501.06911 column-by-column with Berry 2409.11748 Eq. 24 QROAM
        layers) against the LKS Householder reflection synthesis and the full-unitary
        interferometer synthesis.

        Parameters:
          n = {N_QUBITS}  (N = 2^n)
          phase_bitsize b = {PHASE_BITSIZE}, n_blocks = {N_BLOCKS}
          all curves at Toffoli-optimal QROAM block size

        Fits y = c * N^alpha:
        """
    )
    body += "\n".join(f"          {line}" for line in fit_lines) + "\n"

    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = recipient
    msg["Subject"] = "Isometry column-by-column QROAM resource report"
    msg.attach(MIMEText(body, "plain"))
    with open(pdf_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(pdf_path)}"'
    msg.attach(part)
    try:
        with smtplib.SMTP("localhost", 25, timeout=10) as smtp:
            smtp.sendmail(msg["From"], [recipient], msg.as_string())
        print(f"Email sent via localhost:25 to {recipient}")
        return True
    except Exception as exc:
        print(f"localhost:25 email failed: {exc}")
    try:
        proc = subprocess.run(
            ["/usr/sbin/sendmail", "-t", "-oi"], input=msg.as_string().encode(),
            capture_output=True, timeout=30,
        )
        if proc.returncode == 0:
            print(f"Email sent via /usr/sbin/sendmail to {recipient}")
            return True
        print(f"sendmail failed with {proc.returncode}: {proc.stderr.decode(errors='replace')[:300]}")
    except Exception as exc:
        print(f"sendmail email failed: {exc}")
    return False


if os.environ.get("SEND_EMAIL") == "1":
    send_email(RECIPIENT, OUT_PDF)
else:
    print("Set SEND_EMAIL=1 to email the report.")
