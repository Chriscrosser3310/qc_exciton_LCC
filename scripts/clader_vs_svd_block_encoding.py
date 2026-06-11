#!/usr/bin/env python3
"""Block-encode a dense N x N matrix (N = 2^n, n = 4..10): Clader-Frobenius vs SVD.

Compares two block-encoding constructions for a single dense matrix, each in a
Toffoli(T)-count-optimal and a qubit-count-optimal configuration:

  (1) Clader-Frobenius  (arXiv:2206.03505, minimal-T-count version)
        U_A = U_R^dag U_L, alpha = ||A||_F.  Implemented by
        ClassicalMatrixBlockEncoding: a plain state prep of the row-norm state |phi>
        plus an n-qubit swap (U_L), and an adjoint controlled row state prep (U_R^dag),
        both backed by select-swap / QROAM rotations.
          - T-opt  : analytic per-layer lambda* QROAM batching (optimal_T=True).
          - Q-opt  : un-batched lambda = 1 QROAM, minimal ancilla (optimal_T=False).

  (2) SVD interferometer  (alpha = 1)
        A = U Sigma V; U, V synthesized by the block-unitary interferometer and the
        diagonal Sigma by QROAMClean + ctrl-AddIntoPhaseGrad + QROAMCleanAdjoint.
        Implemented by SVDBlockEncodingInterferometer (n_blocks = 1).  Both metrics are
        optimized over a uniform QROAM batching parameter lambda in {0..n}.

Costs come from utils.get_Toffoli_counts / get_qubit_counts (Qualtran QECGatesCost /
QubitCount).  Output: a PDF report (table + log-scale plots + notes), emailed to the
recipient below.

IMPORTANT subnormalization caveat (shown in the report): the two methods do NOT have the
same alpha.  Clader-Frobenius has alpha = ||A||_F (~ N for a dense O(1)-entry matrix),
while the SVD interferometer has alpha = 1.  Downstream algorithms (QSVT/QPE) need ~alpha
queries, so a fair "effective" comparison multiplies the Clader per-query cost by ||A||_F.
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
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    ClassicalMatrixBlockEncoding,
)
from integrations.qualtran.svd_block_encoding_interferometer import (
    SVDBlockEncodingInterferometer,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

N_VALUES = list(range(4, 11))  # n = 4..10  ->  N = 16..1024
PHASE_BITSIZE = 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"clader_vs_svd_blockencoding_n4to10_b{PHASE_BITSIZE}.pdf"
)


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    detail: str


# ------------------------------- Clader-Frobenius --------------------------------


def clader_costs(n: int) -> Tuple[Rec, Rec]:
    """(T-opt, Q-opt) for ClassicalMatrixBlockEncoding of one 2^n x 2^n matrix."""
    N = 1 << n
    bt = ClassicalMatrixBlockEncoding.from_bitsize(N, PHASE_BITSIZE, optimal_T=True)
    bq = ClassicalMatrixBlockEncoding.from_bitsize(N, PHASE_BITSIZE, optimal_T=False)
    t_opt = Rec(int(get_Toffoli_counts(bt)), int(get_qubit_counts(bt)), "per-layer lambda*")
    q_opt = Rec(int(get_Toffoli_counts(bq)), int(get_qubit_counts(bq)), "lambda=1")
    return t_opt, q_opt


# ------------------------------- SVD interferometer ------------------------------


def _svd_bloq(N: int, lam: int) -> SVDBlockEncodingInterferometer:
    t = (lam,)  # n_blocks = 1 -> 1-D QROAM data on every stage
    return SVDBlockEncodingInterferometer(
        n_blocks=1,
        n_rows=N,
        phase_bitsize=PHASE_BITSIZE,
        interferometer_log_block_sizes=t,
        interferometer_final_log_block_sizes=t,
        interferometer_final_adjoint_log_block_sizes=t,
        diag_log_block_sizes=t,
        diag_adjoint_log_block_sizes=t,
        optimal_T=False,
    )


def svd_costs(n: int) -> Tuple[Rec, Rec]:
    """(T-opt, Q-opt) for SVDBlockEncodingInterferometer, optimized over lambda in {0..n}."""
    N = 1 << n
    recs: List[Tuple[int, int, int]] = []  # (lambda, T, Q)
    for lam in range(0, n + 1):
        try:
            bl = _svd_bloq(N, lam)
            recs.append((lam, int(get_Toffoli_counts(bl)), int(get_qubit_counts(bl))))
        except Exception:  # noqa: BLE001 - skip invalid batching choices
            continue
    if not recs:
        raise RuntimeError(f"no valid SVD configuration for n={n}")
    t_lam, t_T, t_Q = min(recs, key=lambda r: (r[1], r[2]))
    q_lam, q_T, q_Q = min(recs, key=lambda r: (r[2], r[1]))
    return Rec(t_T, t_Q, f"lambda={t_lam}"), Rec(q_T, q_Q, f"lambda={q_lam}")


# ------------------------------------ Fits ---------------------------------------


def fit_exp(ns: List[int], ys: List[int]) -> Tuple[float, float]:
    """Fit y = c * 2^(a*n); return (a, c)."""
    a, logc = np.polyfit(ns, np.log2(ys), 1)
    return float(a), float(2.0**logc)


# --------------------------------- Data assembly ---------------------------------

SERIES = {
    "clader_topt": ("Clader-Frobenius, Toffoli-opt", "#1f77b4", "o", "-"),
    "clader_qopt": ("Clader-Frobenius, qubit-opt", "#1f77b4", "o", "--"),
    "svd_topt": ("SVD interferometer, Toffoli-opt", "#d62728", "^", "-"),
    "svd_qopt": ("SVD interferometer, qubit-opt", "#d62728", "^", "--"),
}

DATA = {k: {"toffoli": [], "qubits": [], "detail": []} for k in SERIES}

for n in N_VALUES:
    c_t, c_q = clader_costs(n)
    s_t, s_q = svd_costs(n)
    for key, rec in (
        ("clader_topt", c_t),
        ("clader_qopt", c_q),
        ("svd_topt", s_t),
        ("svd_qopt", s_q),
    ):
        DATA[key]["toffoli"].append(rec.toffoli)
        DATA[key]["qubits"].append(rec.qubits)
        DATA[key]["detail"].append(rec.detail)

fits = {}
for key in SERIES:
    fits[key + "_toffoli"] = fit_exp(N_VALUES, DATA[key]["toffoli"])
    fits[key + "_qubits"] = fit_exp(N_VALUES, DATA[key]["qubits"])


# ------------------------------------ Plots --------------------------------------


def plot_metric(metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    n_dims = [1 << n for n in N_VALUES]  # N = 2^n on the x-axis
    for key, (lbl, color, marker, ls) in SERIES.items():
        ys = DATA[key][metric]
        ax.plot(n_dims, ys, marker=marker, color=color, linestyle=ls, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("N  (matrix dimension, N = 2^n)")
    ax.set_ylabel(ylabel + "  (log2 scale)")
    ax.set_title(f"{ylabel} vs N   (phase_bitsize = {PHASE_BITSIZE})")
    ax.set_xticks(n_dims)
    ax.set_xticklabels([str(N) for N in n_dims])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.axis("off")
    col_labels = ["n", "N"]
    for lbl, _, _, _ in SERIES.values():
        col_labels.append(lbl.replace(", ", "\n"))
    rows = []
    for i, n in enumerate(N_VALUES):
        row = [str(n), str(1 << n)]
        for key in SERIES:
            row.append(f"T={DATA[key]['toffoli'][i]:,}\nQ={DATA[key]['qubits'][i]:,}")
        rows.append(row)
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 2.1)
    ax.set_title(
        "Toffoli (T) and qubit (Q) counts: dense N x N block-encoding\n"
        f"phase_bitsize = {PHASE_BITSIZE};  alpha(Clader) = ||A||_F,  alpha(SVD) = 1",
        fontsize=11,
    )
    return fig


def notes_page():
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.text(
        0.06,
        0.95,
        "Dense matrix block-encoding: Clader-Frobenius vs SVD interferometer",
        fontsize=14,
        weight="bold",
        va="top",
    )
    body = textwrap.dedent(
        f"""\
        Task: block-encode a single dense N x N matrix of classical data, N = 2^n,
        n = {N_VALUES[0]}..{N_VALUES[-1]}, phase_bitsize = {PHASE_BITSIZE}.  Two methods, each
        in a Toffoli(T)-count-optimal and a qubit-count-optimal configuration.

        (1) CLADER-FROBENIUS  (arXiv:2206.03505, minimal-T-count construction)
            U_A = U_R^dag U_L,  subnormalization alpha = ||A||_F.
              U_L  = plain state prep of |phi> = sum_j (||A_j||/||A||_F) |j>  (+ register swap)
              U_R  = controlled state prep of  |psi_j> = sum_k (A_jk/||A_j||) |k>
            Both are select-swap / QROAM-rotation state preparations (Low-Kliuchnikov-
            Schaeffer phase-gradient rotations).
              T-opt : per-layer analytic lambda* QROAM batching (fewest T, more ancilla).
              Q-opt : un-batched lambda = 1 QROAM (fewest qubits, more T).

        (2) SVD INTERFEROMETER  (subnormalization alpha = 1)
            A = U Sigma V; U, V via the block-unitary interferometer (n_blocks = 1), the
            diagonal Sigma via QROAMClean + ctrl-AddIntoPhaseGrad + QROAMCleanAdjoint.
            Each metric is optimized over a uniform QROAM batching lambda in {{0..n}}.

        SCALING FITS  y = c * 2^(a*n)  (a ~ slope per doubling of n):
        """
    )
    for key, (lbl, _, _, _) in SERIES.items():
        at, ct = fits[key + "_toffoli"]
        aq, cq = fits[key + "_qubits"]
        body += f"            {lbl:34s} T: a={at:.2f} c={ct:.2e}   Q: a={aq:.2f} c={cq:.2e}\n"
    body += textwrap.dedent(
        """
        KEY CAVEAT - subnormalization is NOT equal:
          * Clader-Frobenius gives alpha = ||A||_F, which is ~ N for a dense matrix with
            O(1) entries.  SVD interferometer gives the spectral-norm-optimal alpha = 1.
          * Downstream algorithms (QSVT, QPE, qubitization) require ~alpha block-encoding
            queries.  So the EFFECTIVE Clader cost for such an algorithm is roughly its
            per-query Toffoli count times ||A||_F (~N), whereas the SVD cost is per-query.
          * Read the raw T/Q here as the cost of ONE block-encoding query.  The SVD method
            pays more per query but needs far fewer queries; Clader is cheap per query but
            its large alpha must be amortized.  The right choice is application-dependent.

        Both constructions: error ~ O(2^-phase_bitsize) from angle discretization; counts
        from Qualtran QECGatesCost / QubitCount (utils.get_Toffoli_counts / get_qubit_counts).
        """
    )
    fig.text(0.06, 0.89, body, fontsize=9, family="monospace", va="top")
    return fig


# ------------------------------------ Render -------------------------------------

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with pdf_backend.PdfPages(OUT_PDF) as pdf:
    for fig_fn in (
        notes_page,
        table_page,
        lambda: plot_metric("toffoli", "Toffoli count"),
        lambda: plot_metric("qubits", "Qubit count"),
    ):
        fig = fig_fn()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
print(f"Wrote {OUT_PDF}")


# ------------------------------------- Email -------------------------------------


def send_email(recipient: str, pdf_path: str) -> bool:
    subject = "Dense block-encoding resource report: Clader-Frobenius vs SVD"
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached is the resource comparison for block-encoding a single dense N x N
        matrix of classical data (N = 2^n, n = {N_VALUES[0]}..{N_VALUES[-1]},
        phase_bitsize = {PHASE_BITSIZE}), for two methods, each in a Toffoli-optimal and
        a qubit-optimal configuration:

          (1) Clader-Frobenius  (arXiv:2206.03505, minimal-T-count): U_A = U_R^dag U_L,
              alpha = ||A||_F.  T-opt = per-layer lambda* QROAM; Q-opt = lambda=1 QROAM.
          (2) SVD interferometer: A = U Sigma V via block-unitary interferometer + QROAM
              diagonal, alpha = 1; each metric optimized over QROAM lambda in {{0..n}}.

        Selected counts (Toffoli / qubits):
        """
    )
    for i, n in enumerate(N_VALUES):
        if n in (4, 7, 10):
            body += f"          n={n:>2} N={1<<n:>5}: "
            body += "  ".join(
                f"{lbl.split(',')[1].strip()[:5]}:{DATA[key]['toffoli'][i]:,}/{DATA[key]['qubits'][i]:,}"
                for key, (lbl, *_ ) in SERIES.items()
            )
            body += "\n"
    body += textwrap.dedent(
        """
        Scaling fits y = c * 2^(a*n):
        """
    )
    for key, (lbl, *_ ) in SERIES.items():
        at, ct = fits[key + "_toffoli"]
        aq, cq = fits[key + "_qubits"]
        body += f"          {lbl:34s} T a={at:.2f}; Q a={aq:.2f}\n"
    body += textwrap.dedent(
        """
        NOTE: alpha differs (Clader ||A||_F ~ N vs SVD 1).  These are per-query costs;
        algorithms need ~alpha queries, so amortize the Clader cost by ||A||_F.  See the
        report's notes page for details.
        """
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
    except Exception as exc:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        print(f"sendmail email failed: {exc}")
    return False


if __name__ == "__main__":
    # Email is opt-in (SEND_EMAIL=1) so the default run only writes the local PDF.
    if os.environ.get("SEND_EMAIL") == "1":
        if not send_email(RECIPIENT, OUT_PDF):
            print("Email was not sent; report is saved locally.")
    else:
        print(f"Report saved locally at {OUT_PDF} (set SEND_EMAIL=1 to email it).")
