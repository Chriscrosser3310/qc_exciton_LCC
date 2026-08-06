#!/usr/bin/env python3
"""Optimal T-count scaling of recursive-CSD N x N unitary synthesis (Tan, arXiv:2509.25702).

For ``N = 4, 8, ..., 1024`` this reports the optimal T-count of synthesizing an arbitrary ``N x N``
unitary with the recursive cosine-sine-decomposition algorithm of

    Xinyu Tan, "Unitary synthesis with fewer T gates", arXiv:2509.25702,

which factors ``U(2^n)`` into ``2^n - 1`` multi-controlled single-qubit unitaries, groups consecutive
runs of ``2^k - 1`` of them into ``2^{n-k}`` multi-controlled ``k``-qubit unitaries, and synthesizes
each via a generalized Gosset-Kothari-Wu diagonal synthesis at T-count ``2^{(n+k)/2} sqrt(L) + 4^k L``
(Thm 4.3).  Summed over the ``2^{n-k}`` blocks and minimized over the block size ``k`` (asymptotic
optimum ``k ~ n/3``) this gives the paper's

    T-count  =  O( 2^{4n/3} L^{2/3} )  ~  N^{4/3}  up to log factors        (Thm 1.1).

These T-counts are read off the ACTUAL constructed circuit ``PaperRecursiveCSDUnitarySynthesis``
(``2^{n-k}`` real ``MultiControlledKQubitUnitaryQROAM`` blocks: a QROAM lookup over the ``n-k`` controls
plus phase-gradient reconstruction on the ``k`` targets), counted by Qualtran's ``QECGatesCost`` -- not
from the analytic formula.  ``k`` is chosen to minimize the real gate count at each ``n``.

(For contrast, the explicit un-grouped circuit ``RecursiveCSDSynthesisQROAM`` -- the Sec. 3.1 naive
strategy that expands every k-qubit block into ``2^k - 1`` single-qubit multiplexed gates -- scales as
``~ N^2``; that was the earlier report's figure.)

Writes a multi-page PDF to ``docs/`` and, when ``SEND_EMAIL=1``, emails it.
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
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.recursive_csd_synthesis_QROAM import (
    PaperRecursiveCSDUnitarySynthesis,
    optimal_constructed_paper_unitary,
)

N_QUBITS = list(range(2, 11))            # N = 4, 8, ..., 1024  (requested range)
N_ROWS = [1 << n for n in N_QUBITS]
N_QUBITS_ASYMPTOTIC = list(range(2, 17))  # constructed as far as feasible (N up to 65536)
PHASE_BITSIZE = 32                       # word length L = n + b folded into phase_bitsize
BLOCK_CASES = [1, 216]                    # single unitary, and block-diagonal with N_k = 216 = 6^3
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"recursive_csd_unitary_tcount_b{PHASE_BITSIZE}.pdf")
COLORS = {1: "#1f77b4", 216: "#d62728"}
LABELS = {1: "single unitary ($N_k=1$)", 216: "block-diagonal ($N_k=216$)"}


@dataclass(frozen=True)
class Record:
    n: int
    N: int
    k: int
    n_blocks: int
    t_count: float


def measure(n: int, n_blocks: int) -> Record:
    k, T = optimal_constructed_paper_unitary(n, PHASE_BITSIZE, n_blocks=n_blocks)
    return Record(n, 1 << n, k, n_blocks, float(T))


def fit_power_law(x, y) -> tuple:
    alpha, log_c = np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)
    return float(alpha), float(np.exp(log_c))


print("=" * 78)
print("Recursive-CSD unitary synthesis (arXiv:2509.25702): CONSTRUCTED optimal T-count vs N")
print(f"  N = {N_ROWS},  b = {PHASE_BITSIZE},  N_k in {BLOCK_CASES}  (counted via QECGatesCost)")
print("=" * 78)

# records[n_blocks] = list of Record over N_QUBITS;  fits[n_blocks] = (alpha, c)
records = {nb: [measure(n, nb) for n in N_QUBITS] for nb in BLOCK_CASES}
fits = {nb: fit_power_law(N_ROWS, [r.t_count for r in records[nb]]) for nb in BLOCK_CASES}
for nb in BLOCK_CASES:
    print(f"\nN_k = {nb}:")
    for r in records[nb]:
        print(f"  N={r.N:>5}  k*={r.k}  blocks(2^(n-k))={1 << (r.n - r.k):>5}  T={r.t_count:>16,.0f}")
    print(f"  fit:  T ~ {fits[nb][1]:.3e} * N^{fits[nb][0]:.3f}")

# Asymptotic series (constructed as far as feasible) for the exponent-convergence panel.
asym = {nb: [measure(n, nb) for n in N_QUBITS_ASYMPTOTIC] for nb in BLOCK_CASES}
asym_local = {
    nb: np.diff(np.log([r.t_count for r in asym[nb]])) / np.diff(np.log([r.N for r in asym[nb]]))
    for nb in BLOCK_CASES
}
asym_midn = [(N_QUBITS_ASYMPTOTIC[i] + N_QUBITS_ASYMPTOTIC[i + 1]) / 2 for i in range(len(N_QUBITS_ASYMPTOTIC) - 1)]


def scaling_page():
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    xf = np.linspace(min(N_ROWS), max(N_ROWS), 200)
    for nb in BLOCK_CASES:
        T = np.asarray([r.t_count for r in records[nb]], float)
        a, cc = fits[nb]
        ax.plot(N_ROWS, T, "o-", color=COLORS[nb], lw=1.8, ms=7,
                label=f"{LABELS[nb]}: fit $N^{{{a:.3f}}}$")
        ax.plot(xf, cc * xf ** a, ":", color=COLORS[nb], lw=1.1)
    # N^{4/3}(log)^{2/3} guide (Thm 1.1) and N^2 reference, anchored at the single-unitary first point.
    T1 = records[1][0].t_count
    g43 = (xf ** (4 / 3)) * (np.log2(xf) ** (2 / 3))
    ax.plot(xf, T1 * g43 / g43[0], "--", color="#2ca02c", lw=1.1, label="$N^{4/3}(\\log N)^{2/3}$ (Thm 1.1)")
    ax.plot(xf, T1 * (xf / N_ROWS[0]) ** 2, "--", color="#888888", lw=1.0, label="$N^2$ (un-grouped)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N = 2^n  (matrix dimension)")
    ax.set_ylabel("optimal T-count (constructed, QECGatesCost)")
    ax.set_xticks(N_ROWS)
    ax.set_xticklabels([str(N) for N in N_ROWS])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5)
    ax.set_title(f"Optimal T-count vs N: single vs block-diagonal ($N_k=216$), b={PHASE_BITSIZE}")
    fig.tight_layout()
    return fig


def exponent_page():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    for nb in BLOCK_CASES:
        ax1.plot(asym_midn, asym_local[nb], "s-", color=COLORS[nb], lw=1.6, ms=4, label=LABELS[nb])
        ax2.plot(N_QUBITS_ASYMPTOTIC, [r.k for r in asym[nb]], "o-", color=COLORS[nb], lw=1.6, ms=4,
                 label=f"$k^*$, {LABELS[nb]}")
    ax1.axhline(4 / 3, color="#2ca02c", ls="--", lw=1.2, label="4/3 (asymptotic)")
    ax1.axhline(3 / 2, color="#888888", ls=":", lw=1.0, label="3/2 (previous best)")
    ax1.set_ylabel("local scaling exponent")
    ax1.set_ylim(1.0, 1.6)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title("Scaling exponent -> 4/3 for both cases (block label is a constant sqrt(N_k) factor)")
    ax2.plot(N_QUBITS_ASYMPTOTIC, [n / 3 for n in N_QUBITS_ASYMPTOTIC], "--", color="#2ca02c", lw=1.2,
             label="$n/3$ (asymptotic optimum)")
    ax2.set_xlabel("n = log2(N)")
    ax2.set_ylabel("optimal $k$")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    cols = ["n", "N", "k* (Nk=1)", "T (Nk=1)", "k* (Nk=216)", "T (Nk=216)", "ratio"]
    rows = []
    for i, n in enumerate(N_QUBITS):
        r1, r2 = records[1][i], records[216][i]
        rows.append([n, r1.N, r1.k, f"{r1.t_count:,.0f}", r2.k, f"{r2.t_count:,.0f}",
                     f"{r2.t_count / r1.t_count:.1f}"])
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for (rr, cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#25364a")
            cell.set_text_props(color="white", weight="bold")
        elif rr % 2:
            cell.set_facecolor("#f3f6fa")
    ax.set_title(f"Constructed optimal T-count: single vs N_k=216 block-diagonal (b={PHASE_BITSIZE})",
                 fontsize=12)
    return fig


def summary_page():
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.axis("off")
    lines = [
        "Recursive-CSD N x N unitary synthesis -- CONSTRUCTED optimal T-count vs N",
        "  Xinyu Tan, 'Unitary synthesis with fewer T gates', arXiv:2509.25702.",
        "",
        "Algorithm: recursive cosine-sine decomposition factors U(2^n) into 2^n - 1 multi-controlled",
        "  single-qubit unitaries (Thm 3.2/3.5).  Consecutive runs of 2^k - 1 of them that share the",
        "  same k-qubit target are GROUPED into a single (n-k)-controlled k-qubit unitary; there are",
        "  2^{n-k} such blocks.  Each is synthesized by a generalized Gosset-Kothari-Wu diagonal",
        "  synthesis at T-count  2^{(n+k)/2} sqrt(L) + 4^k L   (Thm 4.3).  Summing over 2^{n-k}",
        "  blocks and minimizing over k (asymptotic optimum k ~ n/3):",
        "",
        "      T = O( 2^{4n/3} L^{2/3} ) ~ N^{4/3}  up to logs        (Thm 1.1).",
        "",
        "These T-counts are READ OFF THE ACTUAL CIRCUIT, not the formula: each point is the",
        "Qualtran QECGatesCost of PaperRecursiveCSDUnitarySynthesis = 2^{n-k} constructed",
        "MultiControlledKQubitUnitaryQROAM blocks (a QROAM select-swap lookup over the n-k controls",
        "loading 4^k phase words + 4^k phase-gradient additions on the k targets).  k is chosen to",
        f"minimize the real gate count at each n; phase word length b = {PHASE_BITSIZE}.",
        "",
        "Block-diagonal U = sum_a |a><a| (x) U_a (here N_k = 216 = 6^3): the block label a is appended",
        "to every QROAM lookup address (table (N_k, 2^{n-k})), so only the lookup term grows -- by",
        "~sqrt(N_k) under the select-swap optimum -- while the 4^k reconstruction is unchanged.  The",
        "optimal k therefore shifts UP for the block case, and the N-scaling exponent is unchanged",
        "(sqrt(N_k) is a constant prefactor): both cases approach 4/3.",
        "",
        f"Power-law fits over N = 4..1024:",
        f"   single (N_k=1):     T ~ {fits[1][1]:.3e} * N^{fits[1][0]:.3f}",
        f"   block (N_k=216):    T ~ {fits[216][1]:.3e} * N^{fits[216][0]:.3f}",
        "The fitted exponent is below 4/3 in this finite range (small n keeps k small); the constructed",
        "local exponent climbs toward 4/3 as n grows (asymptotic page) with k* -> n/3.  Both are far",
        "below the N^2 cost of the explicit un-grouped circuit (RecursiveCSDSynthesisQROAM).",
    ]
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.8)
    fig.tight_layout()
    return fig


os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with pdf_backend.PdfPages(OUT_PDF) as pdf:
    for make in (summary_page, scaling_page, exponent_page, table_page):
        fig = make()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    info = pdf.infodict()
    info["Title"] = "Recursive-CSD unitary synthesis: optimal T-count scaling (N^4/3)"
    info["Author"] = "qc_exciton_LCC"

print(f"\nWrote report: {OUT_PDF}")


def send_email(recipient: str, pdf_path: str) -> bool:
    body = textwrap.dedent(
        f"""\
        Hi,

        Corrected optimal T-count scaling report for recursive-CSD synthesis of an arbitrary
        N x N unitary, following Xinyu Tan, "Unitary synthesis with fewer T gates"
        (arXiv:2509.25702).

        The algorithm groups the 2^n - 1 multi-controlled single-qubit unitaries from the
        recursive cosine-sine decomposition into 2^{{n-k}} multi-controlled k-qubit unitaries,
        each synthesized by a generalized Gosset-Kothari-Wu diagonal synthesis at T-count
        2^{{(n+k)/2}} sqrt(L) + 4^k L (Thm 4.3).  Minimizing over the block size k (asymptotic
        optimum k ~ n/3) gives  T = O(2^{{4n/3}} L^{{2/3}}) ~ N^{{4/3}} up to logs  (Thm 1.1).

        IMPORTANT: these T-counts are read off the ACTUAL constructed circuit
        (PaperRecursiveCSDUnitarySynthesis = 2^{{n-k}} real QROAM-lookup + phase-gradient blocks),
        counted by Qualtran's QECGatesCost -- not from the analytic formula.

        Also included: the block-diagonal case U = sum_a |a><a| (x) U_a with N_k = 216 = 6^3.
        The block label a is appended to every QROAM lookup address, so the lookup term grows by
        ~sqrt(N_k) (the reconstruction is unchanged and the optimal block size k shifts up); the
        N-scaling exponent is unchanged -- both single and 216-block cases approach 4/3.

        Parameters: N = {N_ROWS}, phase word length b = {PHASE_BITSIZE}.
        Power-law fits over N=4..1024:
          single   (N_k=1):   T ~ {fits[1][1]:.3e} * N^{fits[1][0]:.3f}
          block (N_k=216):    T ~ {fits[216][1]:.3e} * N^{fits[216][0]:.3f}

        Note: the fitted exponent is below 4/3 in this finite range (small n keeps k small); the
        constructed asymptotic page shows the local exponent climbing toward 4/3 with k* -> n/3.
        The earlier report's N^2 figure was the explicit un-grouped circuit (Sec. 3.1 naive
        strategy), not the grouped N^{{4/3}} route.

        Optimal T-counts (constructed):
        """
    )
    body += "\n".join(
        f"          N={records[1][i].N:>5}: "
        f"single k*={records[1][i].k} T={records[1][i].t_count:,.0f}  |  "
        f"N_k=216 k*={records[216][i].k} T={records[216][i].t_count:,.0f}"
        for i in range(len(N_QUBITS))
    ) + "\n"

    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = recipient
    msg["Subject"] = "Recursive-CSD unitary synthesis: optimal T-count scaling (N^4/3) [corrected]"
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
