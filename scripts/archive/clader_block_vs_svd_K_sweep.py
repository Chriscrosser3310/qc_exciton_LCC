#!/usr/bin/env python3
"""Block-diagonal block-encoding of sum_k |k><k| (x) A_k at FIXED N = 256, sweeping K.

Compares, as a function of the number of blocks K (1 .. 216), the two constructions for
A_block = sum_k |k><k| (x) A_k with each A_k of size N x N (N = 256 = 2^8), in a
Toffoli(T)-count-optimal and a qubit-count-optimal configuration:

  (1) Clader-Frobenius BLOCK  (arXiv:2206.03505 generalization; this repo)
        Encodes sum_k |k><k| (x) A_k / F_max,  F_max = max_k ||A_k||_F.
        BlockDiagonalClassicalMatrixBlockEncoding(K, N).
          - T-opt : Toffoli-minimal QROAM select-swap batch (optimal_T=True).
          - Q-opt : un-batched lambda = 1 QROAM (optimal_T=False).

  (2) SVD interferometer BLOCK  (alpha = 1)
        A_k = U_k Sigma_k V_k via the block-unitary interferometer + QROAM diagonal.
        SVDBlockEncodingInterferometer(n_blocks=K, n_rows=N); both metrics optimized
        over a QROAM batching lambda (blocking the K-block dimension).

K range 1..216 = 6^3 mirrors the N_k = 1^3..6^3 momentum-grid sizes used elsewhere in
this repo.  Costs from utils.get_Toffoli_counts / get_qubit_counts.  Output: a PDF
(table + log-log plots vs K + notes), emailed when SEND_EMAIL=1.

Caveat (in the report): Clader-block alpha = F_max (~N, independent of K); SVD-block
alpha = 1.  These are per-query costs; algorithms need ~alpha queries.
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

N = 256
N_BITS = 8  # log2(N)
PHASE_BITSIZE = 32
# K from 1 to 216 = 6^3: the perfect cubes 1^3..6^3 plus intervening powers of two.
K_VALUES = [1, 2, 4, 8, 16, 27, 32, 64, 125, 128, 216]
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"clader_block_vs_svd_N{N}_Ksweep_b{PHASE_BITSIZE}.pdf"
)


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    detail: str


def clader_block_costs(K: int) -> Tuple[Rec, Rec]:
    bt = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(K, N, PHASE_BITSIZE, optimal_T=True)
    bq = BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(K, N, PHASE_BITSIZE, optimal_T=False)
    return (
        Rec(int(get_Toffoli_counts(bt)), int(get_qubit_counts(bt)), "T-min QROAM batch"),
        Rec(int(get_Toffoli_counts(bq)), int(get_qubit_counts(bq)), "lambda=1"),
    )


def _svd_block_bloq(K: int, lam: int) -> SVDBlockEncodingInterferometer:
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


def svd_block_costs(K: int) -> Tuple[Rec, Rec]:
    block_bits = max(1, (K - 1).bit_length())
    recs: List[Tuple[int, int, int]] = []
    for lam in range(0, block_bits + 2):
        try:
            bl = _svd_block_bloq(K, lam)
            recs.append((lam, int(get_Toffoli_counts(bl)), int(get_qubit_counts(bl))))
        except Exception:  # noqa: BLE001
            continue
    if not recs:
        raise RuntimeError(f"no valid SVD-block configuration for K={K}")
    t_lam, t_T, t_Q = min(recs, key=lambda r: (r[1], r[2]))
    q_lam, q_T, q_Q = min(recs, key=lambda r: (r[2], r[1]))
    return Rec(t_T, t_Q, f"lambda={t_lam}"), Rec(q_T, q_Q, f"lambda={q_lam}")


def fit_pow(ks: List[int], ys: List[int]) -> Tuple[float, float]:
    """Fit y = c * K^a in log-log; return (a, c)."""
    a, logc = np.polyfit(np.log2(ks), np.log2(ys), 1)
    return float(a), float(2.0**logc)


SERIES = {
    "clader_topt": ("Clader-Frobenius block, Toffoli-opt", "#1f77b4", "o", "-"),
    "clader_qopt": ("Clader-Frobenius block, qubit-opt", "#1f77b4", "o", "--"),
    "svd_topt": ("SVD interferometer block, Toffoli-opt", "#d62728", "^", "-"),
    "svd_qopt": ("SVD interferometer block, qubit-opt", "#d62728", "^", "--"),
}

DATA = {k: {"toffoli": [], "qubits": [], "detail": []} for k in SERIES}

for K in K_VALUES:
    c_t, c_q = clader_block_costs(K)
    s_t, s_q = svd_block_costs(K)
    for key, rec in (
        ("clader_topt", c_t), ("clader_qopt", c_q),
        ("svd_topt", s_t), ("svd_qopt", s_q),
    ):
        DATA[key]["toffoli"].append(rec.toffoli)
        DATA[key]["qubits"].append(rec.qubits)
        DATA[key]["detail"].append(rec.detail)

fits = {}
for key in SERIES:
    fits[key + "_toffoli"] = fit_pow(K_VALUES, DATA[key]["toffoli"])
    fits[key + "_qubits"] = fit_pow(K_VALUES, DATA[key]["qubits"])


def plot_metric(metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for key, (lbl, color, marker, ls) in SERIES.items():
        ax.plot(K_VALUES, DATA[key][metric], marker=marker, color=color, linestyle=ls, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("K  (number of blocks)")
    ax.set_ylabel(ylabel + "  (log2 scale)")
    ax.set_title(f"{ylabel} vs K   (N = {N}, phase_bitsize = {PHASE_BITSIZE})")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES], rotation=45, fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.axis("off")
    col_labels = ["K"]
    for lbl, _, _, _ in SERIES.values():
        col_labels.append(lbl.replace(" block, ", "\n"))
    rows = []
    for i, K in enumerate(K_VALUES):
        row = [str(K)]
        for key in SERIES:
            row.append(f"T={DATA[key]['toffoli'][i]:,}\nQ={DATA[key]['qubits'][i]:,}")
        rows.append(row)
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 2.0)
    ax.set_title(
        f"Toffoli (T) / qubit (Q) counts vs K: sum_k |k><k| (x) A_k, N = {N}\n"
        f"phase_bitsize = {PHASE_BITSIZE};  alpha(Clader) = F_max,  alpha(SVD) = 1",
        fontsize=11,
    )
    return fig


def notes_page():
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.text(
        0.06, 0.95,
        f"Block-diagonal block-encoding vs K  (fixed N = {N}): Clader-Frobenius vs SVD",
        fontsize=13, weight="bold", va="top",
    )
    body = textwrap.dedent(
        f"""\
        Task: block-encode A_block = sum_k |k><k| (x) A_k, each A_k of size N x N with
        N = {N} = 2^{N_BITS}, sweeping the number of blocks K in {K_VALUES}
        (1 .. 216 = 6^3), phase_bitsize = {PHASE_BITSIZE}.  Two methods, each Toffoli-optimal
        and qubit-optimal.

        (1) CLADER-FROBENIUS BLOCK (alpha = F_max = max_k ||A_k||_F): a read-only k register
            addresses every QROAM; a k-controlled flag rotation injects F_k/F_max, then the
            k-controlled Frobenius construction U_R^dag U_L prepares the row/column states.
              T-opt : Toffoli-minimal QROAM select-swap batch (more ancilla).
              Q-opt : un-batched lambda = 1 QROAM (fewest qubits, more T).

        (2) SVD INTERFEROMETER BLOCK (alpha = 1): A_k = U_k Sigma_k V_k via the block-unitary
            interferometer + QROAM diagonal; each metric optimized over a QROAM batching
            lambda (blocking the K-block dimension).

        SCALING FITS  y = c * K^a  (slope a per doubling of K; N fixed):
        """
    )
    for key, (lbl, _, _, _) in SERIES.items():
        at, _ = fits[key + "_toffoli"]
        aq, _ = fits[key + "_qubits"]
        body += f"            {lbl:38s} T: a={at:.2f}   Q: a={aq:.2f}\n"
    body += textwrap.dedent(
        """
        Reading the K-dependence (the fits above bear this out):
          * Toffoli-OPT: T ~ sqrt(K)  (a ~ 0.5 for both methods).  The select-swap QROAM
            amortizes the K x N x N data load across blocks -- its optimum batch
            lambda* ~ sqrt(K N / b) gives cost ~ sqrt(K N b), so doubling K only raises T
            by ~sqrt(2).  This block-amortization is the whole point of select-swap.
          * Qubit-OPT (lambda = 1): T ~ K  (a ~ 1.0; ~0.9 for SVD).  With no batching every
            block's data is loaded serially, so Toffoli is linear in K -- but qubit count
            stays ~flat in K (a ~ 0.02-0.03): only log2(K) address qubits are added.
          * Clader T-opt qubit count grows with K (a ~ 0.44): its select-swap batch widens
            with the K x N x N data.  The k register itself adds only log2(K) qubits.

        KEY CAVEAT - subnormalization is NOT equal: Clader-block alpha = F_max (~N,
        independent of K) vs SVD-block alpha = 1.  These are per-query costs; downstream
        algorithms need ~alpha queries, so amortize the Clader cost by F_max.

        Counts from Qualtran QECGatesCost / QubitCount; error ~ O(2^-phase_bitsize).
        """
    )
    fig.text(0.06, 0.90, body, fontsize=8.5, family="monospace", va="top")
    return fig


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


def send_email(recipient: str, pdf_path: str) -> bool:
    subject = f"Block-diagonal block-encoding report (N={N}, K sweep): Clader vs SVD"
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached is the resource comparison for block-encoding A_block = sum_k |k><k| (x) A_k
        at fixed N = {N}, sweeping K in {K_VALUES} (1 .. 216 = 6^3),
        phase_bitsize = {PHASE_BITSIZE}, in Toffoli-optimal and qubit-optimal configurations:

          (1) Clader-Frobenius block (alpha = F_max): k-controlled flag rotation + the
              k-controlled Frobenius construction.
          (2) SVD interferometer block (alpha = 1): A_k = U_k Sigma_k V_k.

        Selected counts (Toffoli / qubits):
        """
    )
    for i, K in enumerate(K_VALUES):
        if K in (1, 8, 64, 216):
            body += f"          K={K:>3}: "
            body += "  ".join(
                f"{key.split('_')[0][:3]}-{key.split('_')[1][0]}:{DATA[key]['toffoli'][i]:,}/{DATA[key]['qubits'][i]:,}"
                for key in SERIES
            )
            body += "\n"
    body += "\n        Scaling fits y = c * K^a:\n"
    for key, (lbl, *_ ) in SERIES.items():
        at, _ = fits[key + "_toffoli"]
        aq, _ = fits[key + "_qubits"]
        body += f"          {lbl:38s} T a={at:.2f}; Q a={aq:.2f}\n"
    body += textwrap.dedent(
        """
        NOTE: alpha differs (Clader F_max ~ N vs SVD 1); per-query costs, amortize by F_max.
        See the report's notes page.
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
            input=msg.as_string().encode(), capture_output=True, timeout=30,
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
