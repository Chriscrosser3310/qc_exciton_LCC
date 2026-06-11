#!/usr/bin/env python3
"""Block-DIAGONAL block-encoding of A_block = sum_k |k><k| (x) A_k (K blocks, each N x N).

Compares two constructions for the block-diagonal operator, each in a Toffoli(T)-count-
optimal and a qubit-count-optimal configuration, for N = 2^n, n = 4..10 at fixed K:

  (1) Clader-Frobenius BLOCK  (arXiv:2206.03505 generalization; this repo)
        Encodes sum_k |k><k| (x) A_k / F_max,  F_max = max_k ||A_k||_F.
        BlockDiagonalClassicalMatrixBlockEncoding: a k-controlled flag rotation
        |0>_f -> (F_k/F_max)|0> + sqrt(1-F_k^2/F_max^2)|1>, then the k-controlled
        Frobenius construction U_R^dag U_L (row/column QROAM-rotation state preps).
          - T-opt : Toffoli-minimal QROAM select-swap batch (optimal_T=True).
          - Q-opt : un-batched lambda = 1 QROAM (optimal_T=False).

  (2) SVD interferometer BLOCK  (alpha = 1)
        Encodes sum_k |k><k| (x) A_k with A_k = U_k Sigma_k V_k; U_k, V_k via the
        block-unitary interferometer and Sigma_k via QROAMClean + ctrl-AddIntoPhaseGrad
        + QROAMCleanAdjoint.  SVDBlockEncodingInterferometer(n_blocks=K).  Both metrics
        optimized over a QROAM batching parameter lambda (blocking the pair dimension).

Costs from utils.get_Toffoli_counts / get_qubit_counts.  Output: a PDF (table + log-log
plots vs N + notes), emailed to the recipient when SEND_EMAIL=1.

Subnormalization caveat (in the report): Clader-block has alpha = F_max (~N for dense
O(1) blocks, independent of K); SVD-block has alpha = 1.  Downstream algorithms need
~alpha queries, so amortize the Clader per-query cost by F_max.
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
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding,
)
from integrations.qualtran.svd_block_encoding_interferometer import (
    SVDBlockEncodingInterferometer,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

K = 8
N_VALUES = list(range(4, 11))  # n = 4..10  ->  N = 16..1024
PHASE_BITSIZE = 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"clader_block_vs_svd_block_K{K}_n4to10_b{PHASE_BITSIZE}.pdf"
)


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    detail: str


# --------------------------- Clader-Frobenius (block) ----------------------------


def clader_block_costs(n: int) -> Tuple[Rec, Rec]:
    N = 1 << n
    bt = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(K, N, PHASE_BITSIZE, optimal_T=True)
    bq = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(K, N, PHASE_BITSIZE, optimal_T=False)
    return (
        Rec(int(get_Toffoli_counts(bt)), int(get_qubit_counts(bt)), "T-min QROAM batch"),
        Rec(int(get_Toffoli_counts(bq)), int(get_qubit_counts(bq)), "lambda=1"),
    )


# ----------------------------- SVD interferometer (block) ------------------------


def _svd_block_bloq(N: int, lam: int) -> SVDBlockEncodingInterferometer:
    # K > 1 -> 2-D QROAM (block, row); batch the block/pair dimension with lambda.
    t = [0, lam] if K > 1 else [lam]
    return SVDBlockEncodingInterferometer(
        n_blocks=K,
        n_rows=N,
        phase_bitsize=PHASE_BITSIZE,
        interferometer_log_block_sizes=t,
        interferometer_final_log_block_sizes=t,
        interferometer_final_adjoint_log_block_sizes=t,
        diag_log_block_sizes=t,
        diag_adjoint_log_block_sizes=t,
        optimal_T=False,
    )


def svd_block_costs(n: int) -> Tuple[Rec, Rec]:
    N = 1 << n
    recs: List[Tuple[int, int, int]] = []
    for lam in range(0, n + 2):
        try:
            bl = _svd_block_bloq(N, lam)
            recs.append((lam, int(get_Toffoli_counts(bl)), int(get_qubit_counts(bl))))
        except Exception:  # noqa: BLE001
            continue
    if not recs:
        raise RuntimeError(f"no valid SVD-block configuration for n={n}")
    t_lam, t_T, t_Q = min(recs, key=lambda r: (r[1], r[2]))
    q_lam, q_T, q_Q = min(recs, key=lambda r: (r[2], r[1]))
    return Rec(t_T, t_Q, f"lambda={t_lam}"), Rec(q_T, q_Q, f"lambda={q_lam}")


# ------------------------------------ Fits ---------------------------------------


def fit_exp(ns: List[int], ys: List[int]) -> Tuple[float, float]:
    a, logc = np.polyfit(ns, np.log2(ys), 1)
    return float(a), float(2.0**logc)


# --------------------------------- Data assembly ---------------------------------

SERIES = {
    "clader_topt": ("Clader-Frobenius block, Toffoli-opt", "#1f77b4", "o", "-"),
    "clader_qopt": ("Clader-Frobenius block, qubit-opt", "#1f77b4", "o", "--"),
    "svd_topt": ("SVD interferometer block, Toffoli-opt", "#d62728", "^", "-"),
    "svd_qopt": ("SVD interferometer block, qubit-opt", "#d62728", "^", "--"),
}

DATA = {k: {"toffoli": [], "qubits": [], "detail": []} for k in SERIES}

for n in N_VALUES:
    c_t, c_q = clader_block_costs(n)
    s_t, s_q = svd_block_costs(n)
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
    ax.set_xlabel("N  (block dimension, N = 2^n)")
    ax.set_ylabel(ylabel + "  (log2 scale)")
    ax.set_title(f"{ylabel} vs N   (K = {K} blocks, phase_bitsize = {PHASE_BITSIZE})")
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
        col_labels.append(lbl.replace(" block, ", "\n"))
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
        f"Toffoli (T) / qubit (Q) counts: block-diagonal sum_k |k><k| (x) A_k, K = {K}\n"
        f"phase_bitsize = {PHASE_BITSIZE};  alpha(Clader) = F_max,  alpha(SVD) = 1",
        fontsize=11,
    )
    return fig


def notes_page():
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.text(
        0.06, 0.95,
        "Block-diagonal block-encoding: Clader-Frobenius vs SVD interferometer",
        fontsize=14, weight="bold", va="top",
    )
    body = textwrap.dedent(
        f"""\
        Task: block-encode the block-diagonal operator A_block = sum_k |k><k| (x) A_k with
        K = {K} blocks, each A_k of size N x N, N = 2^n, n = {N_VALUES[0]}..{N_VALUES[-1]},
        phase_bitsize = {PHASE_BITSIZE}.  Two methods, each Toffoli(T)-optimal and qubit-optimal.

        (1) CLADER-FROBENIUS BLOCK  (arXiv:2206.03505 generalization)
            Encodes sum_k |k><k| (x) A_k / F_max,  F_max = max_k ||A_k||_F.  A read-only k
            register addresses every QROAM:
              - flag : k-controlled rotation |0>_f -> (F_k/F_max)|0> + sqrt(1-F_k^2/F_max^2)|1>
              - U_L  : k-controlled prep of |phi_k> = sum_j ||(A_k)_{{j,.}}||/F_k |j> (+ swap)
              - U_R  : (k,j)-controlled prep of |psi_{{k,j}}> = sum_l (A_k)_{{j,l}}/||(A_k)_{{j,.}}|| |l>
            Post-selecting f=0 and the prep ancilla on |0> yields (A_k)_{{j,l}}/F_max.
              T-opt : Toffoli-minimal QROAM select-swap batch (more ancilla).
              Q-opt : un-batched lambda = 1 QROAM (fewest qubits, more T).

        (2) SVD INTERFEROMETER BLOCK  (alpha = 1)
            A_k = U_k Sigma_k V_k; U_k, V_k via block-unitary interferometer, Sigma_k via
            QROAMClean + ctrl-AddIntoPhaseGrad + QROAMCleanAdjoint.  Each metric optimized
            over a QROAM batching lambda (blocking the K-block / pair dimension).

        SCALING FITS  y = c * 2^(a*n)  (slope a per doubling of N; K fixed):
        """
    )
    for key, (lbl, _, _, _) in SERIES.items():
        at, ct = fits[key + "_toffoli"]
        aq, cq = fits[key + "_qubits"]
        body += f"            {lbl:38s} T: a={at:.2f}   Q: a={aq:.2f}\n"
    body += textwrap.dedent(
        """
        KEY CAVEAT - subnormalization is NOT equal:
          * Clader-block: alpha = F_max = max_k ||A_k||_F  (~N for dense O(1) blocks,
            independent of K).  SVD-block: spectral-norm-optimal alpha = 1.
          * Downstream algorithms (QSVT, QPE, qubitization) need ~alpha block-encoding
            queries; the EFFECTIVE Clader cost is per-query Toffoli times F_max (~N).
          * Read the raw T/Q as the cost of ONE block-encoding query.  The SVD method
            pays more per query but needs far fewer queries; Clader is cheap per query
            but its large alpha must be amortized.  Choice is application-dependent.

        Both: error ~ O(2^-phase_bitsize); counts from Qualtran QECGatesCost / QubitCount.
        """
    )
    fig.text(0.06, 0.90, body, fontsize=8.5, family="monospace", va="top")
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
    subject = f"Block-diagonal block-encoding report (K={K}): Clader-Frobenius vs SVD"
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached is the resource comparison for block-encoding the block-diagonal operator
        A_block = sum_k |k><k| (x) A_k with K = {K} blocks of size N x N
        (N = 2^n, n = {N_VALUES[0]}..{N_VALUES[-1]}, phase_bitsize = {PHASE_BITSIZE}),
        in Toffoli-optimal and qubit-optimal configurations:

          (1) Clader-Frobenius block (alpha = F_max = max_k ||A_k||_F): k-controlled flag
              rotation injecting F_k/F_max, then the k-controlled Frobenius construction.
          (2) SVD interferometer block (alpha = 1): A_k = U_k Sigma_k V_k via the
              block-unitary interferometer + QROAM diagonal.

        Selected counts (Toffoli / qubits):
        """
    )
    for i, n in enumerate(N_VALUES):
        if n in (4, 7, 10):
            body += f"          n={n:>2} N={1<<n:>5}: "
            body += "  ".join(
                f"{key.split('_')[0][:3]}-{key.split('_')[1][0]}:{DATA[key]['toffoli'][i]:,}/{DATA[key]['qubits'][i]:,}"
                for key in SERIES
            )
            body += "\n"
    body += "\n        Scaling fits y = c * 2^(a*n):\n"
    for key, (lbl, *_ ) in SERIES.items():
        at, _ = fits[key + "_toffoli"]
        aq, _ = fits[key + "_qubits"]
        body += f"          {lbl:38s} T a={at:.2f}; Q a={aq:.2f}\n"
    body += textwrap.dedent(
        """
        NOTE: alpha differs (Clader F_max ~ N vs SVD 1).  These are per-query costs;
        algorithms need ~alpha queries, so amortize the Clader cost by F_max.  See the
        report's notes page.
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
    if os.environ.get("SEND_EMAIL") == "1":
        if not send_email(RECIPIENT, OUT_PDF):
            print("Email was not sent; report is saved locally.")
    else:
        print(f"Report saved locally at {OUT_PDF} (set SEND_EMAIL=1 to email it).")
