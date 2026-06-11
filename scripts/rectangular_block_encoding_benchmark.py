#!/usr/bin/env python3
"""Benchmark RectangularBlockEncodingReflection across non-power-of-2 (M, N).

For each (M, N) we instantiate the bloq, sweep its QROAM log_block_sizes, and pick
Toffoli-optimal and qubit-optimal configurations.  All costs come from Qualtran's
get_cost_value walking build_call_graph / build_composite_bloq.
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
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.rectangular_block_encoding_reflection import (
    RectangularBlockEncodingReflection,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

# (M, N) pairs; intentionally non-power-of-2 dimensions.
SHAPES: list[Tuple[int, int]] = [
    (13, 19),
    (20, 20),
    (26, 26),
    (13, 37),    # M < N (transposed)
    (50, 20),    # M > N (standard)
    (40, 60),    # M < N
    (60, 40),    # M > N
    (100, 28),   # M > N, large
    (28, 100),   # M < N, large
]
K = 8
PHASE_BITSIZE = 32
SWEEP_LBS = list(range(0, 12))
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"rectangular_block_encoding_K{K}_b{PHASE_BITSIZE}.pdf"
)
RECIPIENT = "jchen9@caltech.edu"


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    lbs_fwd: int
    lbs_adj: int


def evaluate(K: int, m: int, n: int, lbs_fwd: int, lbs_adj: int) -> Optional[Rec]:
    try:
        b = RectangularBlockEncodingReflection(
            n_blocks=K, m_rows=m, n_cols=n, phase_bitsize=PHASE_BITSIZE,
            amp_log_block_sizes=[lbs_fwd],
            amp_adjoint_log_block_sizes=[lbs_adj],
            phase_log_block_sizes=[lbs_fwd],
            phase_adjoint_log_block_sizes=[lbs_adj],
        )
        return Rec(int(get_Toffoli_counts(b)), int(get_qubit_counts(b)), lbs_fwd, lbs_adj)
    except Exception:
        return None


def optimize(K: int, m: int, n: int) -> Tuple[Rec, Rec]:
    recs = []
    for lf in SWEEP_LBS:
        for la in SWEEP_LBS:
            r = evaluate(K, m, n, lf, la)
            if r is not None:
                recs.append(r)
    if not recs:
        raise RuntimeError(f"no valid config for M={m}, N={n}")
    return min(recs, key=lambda r: (r.toffoli, r.qubits)), min(recs, key=lambda r: (r.qubits, r.toffoli))


print("=" * 78)
print(f"RectangularBlockEncodingReflection benchmark  K={K} phase_bitsize={PHASE_BITSIZE}")
print("=" * 78)

records: list[Tuple[int, int, Rec, Rec, int, int, bool]] = []
for m, n in SHAPES:
    t_opt, q_opt = optimize(K, m, n)
    b_ref = RectangularBlockEncodingReflection(
        n_blocks=K, m_rows=m, n_cols=n, phase_bitsize=PHASE_BITSIZE
    )
    D = b_ref.matrix_dim
    n_refl = b_ref.n_reflections
    transposed = b_ref.transposed
    records.append((m, n, t_opt, q_opt, D, n_refl, transposed))
    method = "transposed (rows)" if transposed else "standard (cols)"
    print(f"M={m:3d} N={n:3d} D={D:3d} n_reflections={n_refl:3d} [{method}]")
    print(f"  T-opt: T={t_opt.toffoli:,} Q={t_opt.qubits} lbs=(fwd={t_opt.lbs_fwd}, adj={t_opt.lbs_adj})")
    print(f"  Q-opt: T={q_opt.toffoli:,} Q={q_opt.qubits} lbs=(fwd={q_opt.lbs_fwd}, adj={q_opt.lbs_adj})")


def plot_metric(metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    xs = np.arange(len(records))
    t_data = [getattr(r[2], metric) for r in records]
    q_data = [getattr(r[3], metric) for r in records]
    w = 0.38
    ax.bar(xs - w / 2, t_data, w, label="Toffoli-opt", color="#1f77b4")
    ax.bar(xs + w / 2, q_data, w, label="qubit-opt", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"M={r[0]}\nN={r[1]}\nD={r[4]}\n{'T' if r[6] else 'S'}{r[5]}" for r in records],
        fontsize=8,
    )
    ax.set_xlabel("(M, N) with padded dim D and (T)ransposed/(S)tandard, n_reflections")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{ylabel}: rectangular block encoding  (K={K}, b={PHASE_BITSIZE})")
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.axis("off")
    cols = ["M", "N", "D", "n_refl", "method",
            "T-opt T", "T-opt Q", "Q-opt T", "Q-opt Q",
            "T-opt lbs", "Q-opt lbs"]
    rows = []
    for (m, n, t, q, D, n_refl, transposed) in records:
        method = "transposed" if transposed else "standard"
        rows.append([
            m, n, D, n_refl, method,
            f"{t.toffoli:,}", t.qubits,
            f"{q.toffoli:,}", q.qubits,
            f"({t.lbs_fwd}, {t.lbs_adj})",
            f"({q.lbs_fwd}, {q.lbs_adj})",
        ])
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#25364a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2:
            cell.set_facecolor("#f3f6fa")
    ax.set_title(f"Resource counts (K={K}, b={PHASE_BITSIZE})", fontsize=12)
    return fig


def summary_page():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    lines = [
        "Rectangular block encoding via state-reflection synthesis",
        "",
        f"K = {K} blocks; phase_bitsize b = {PHASE_BITSIZE}.",
        "",
        "Each A_k is an M x N matrix with possibly non-power-of-2 M, N.",
        "The padded unitary dimension is D = smallest power of 2 >= M + N.",
        "  - If M <= N: 'transposed' synthesis builds the first M rows of a D x D",
        "    unitary using M state reflections (each in a D-dim Hilbert space).",
        "  - If M > N : 'standard' synthesis builds the first N columns of a",
        "    D x D unitary using N state reflections (each in a D-dim Hilbert space).",
        "",
        "Underlying bloq:",
        "  BlockUnitaryReflectionQROAM(n_blocks=K, n_rows=D, n_reflections=min(M, N)).",
        "  Reflections are loaded by QROAM with the rows (transposed) or columns",
        "  (standard) of A_k, padded with zeros up to D.  No comparator is needed in",
        "  the circuit; the non-power-of-2 cutoff is enforced purely by the data.",
        "",
        "build_call_graph and build_composite_bloq are both implemented and yield",
        "the same Toffoli and (peak) qubit counts via Qualtran's get_cost_value.",
        "",
        "Each row's log_block_sizes (forward + adjoint) is swept over",
        f"{SWEEP_LBS[0]}..{SWEEP_LBS[-1]} and minimized for (T, Q) lex (Toffoli-opt)",
        "and (Q, T) lex (qubit-opt).",
    ]
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
    fig.tight_layout()
    return fig


os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with pdf_backend.PdfPages(OUT_PDF) as pdf:
    for make in (
        summary_page,
        lambda: plot_metric("toffoli", "Toffoli count"),
        lambda: plot_metric("qubits", "Peak logical qubits"),
        table_page,
    ):
        fig = make()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    info = pdf.infodict()
    info["Title"] = "Rectangular block encoding (Toffoli / qubit benchmark)"
    info["Author"] = "qc_exciton_LCC"
print(f"\nWrote report: {OUT_PDF}")


def send_email(recipient: str, pdf_path: str) -> bool:
    subject = "Rectangular block-encoding (non-power-of-2 M, N) resource report"
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached: resource benchmark for the new RectangularBlockEncodingReflection
        bloq.  It block-encodes K = {K} matrices A_k of (possibly non-power-of-2)
        sizes M x N using the state-reflection construction in a padded D x D
        Hilbert space with D = smallest power of 2 >= M + N.

          - M <= N : 'transposed' (build M rows via M reflections).
          - M  > N : 'standard'   (build N cols via N reflections).

        Both Toffoli-optimal and qubit-optimal log_block_size choices are reported.
        Cost via Qualtran's get_cost_value on the bloq; build_call_graph and
        build_composite_bloq agree.

        Shapes:
        """
    )
    for (m, n, t, q, D, n_refl, transposed) in records:
        method = "transposed" if transposed else "standard"
        body += (
            f"          M={m:3d} N={n:3d}  D={D:3d}  n_refl={n_refl:3d}  ({method})  "
            f"T-opt T={t.toffoli:,} Q={t.qubits} ; Q-opt T={q.toffoli:,} Q={q.qubits}\n"
        )

    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = recipient
    msg["Subject"] = subject
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
            ["/usr/sbin/sendmail", "-t", "-oi"],
            input=msg.as_string().encode(),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0:
            print(f"Email sent via /usr/sbin/sendmail to {recipient}")
            return True
        print(f"sendmail failed with {proc.returncode}: {proc.stderr.decode(errors='replace')[:300]}")
    except Exception as exc:
        print(f"sendmail email failed: {exc}")
    return False


EMAIL_SENT = send_email(RECIPIENT, OUT_PDF)
if not EMAIL_SENT:
    print("Email was not sent; report is saved locally.")
