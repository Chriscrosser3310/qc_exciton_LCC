#!/usr/bin/env python3
"""Generate block-unitary synthesis resource plots and email the report."""

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
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np
from qualtran.symbolics import Shaped

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.block_unitary_interferometer_QROAM import (
    estimate_interferometer_resources,
    optimal_interferometer_log_block_sizes,
)
from integrations.qualtran.block_unitary_synthesis_QROAM import BlockUnitarySynthesisQROAM
from integrations.qualtran.unitary_synthesis_QROAM import UnitarySynthesisQROAM
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

K_VALUES = list(range(1, 7))
N_BLOCKS_VALUES = [k**3 for k in K_VALUES]
N_SITES = 26
MULTIPLIER = 8
N_ROWS = 1 << (MULTIPLIER * N_SITES - 1).bit_length()
N_REFLECTIONS = N_ROWS
PHASE_BITSIZE = 32
SWEEP_LOG_BLOCK_SIZES = list(range(0, 11))
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"block_unitary_interferometer_resource_report_b{PHASE_BITSIZE}.pdf")
RECIPIENT = "jchen9@caltech.edu"


@dataclass(frozen=True)
class Record:
    toffoli: int
    qubits: int
    layer_lbs: int
    final_lbs: int


def compute_standard_resources(n_blocks: int, lbs: int) -> Optional[Record]:
    """Existing Householder/QROAM block-unitary resource estimate."""

    try:
        bloq = BlockUnitarySynthesisQROAM.from_shape(
            n_blocks=n_blocks,
            n_rows=N_ROWS,
            phase_bitsize=PHASE_BITSIZE,
            n_reflections=1,
            log_block_sizes=[lbs, lbs],
        )
        t_per_reflection = int(get_Toffoli_counts(bloq))
        if n_blocks == 1:
            fallback = UnitarySynthesisQROAM(
                unitary=Shaped((N_ROWS, 1)),
                phase_bitsize=PHASE_BITSIZE,
                log_block_sizes=[lbs],
            )
            q = int(get_qubit_counts(fallback))
        else:
            q = int(get_qubit_counts(bloq))
        return Record(N_REFLECTIONS * t_per_reflection, q, lbs, lbs)
    except Exception as exc:
        print(f"    standard lbs={lbs} skipped: {type(exc).__name__}: {exc}")
        return None


def optimize_standard(n_blocks: int) -> tuple[Record, Record]:
    records = [r for lbs in SWEEP_LOG_BLOCK_SIZES if (r := compute_standard_resources(n_blocks, lbs))]
    if not records:
        raise RuntimeError(f"no valid standard records for n_blocks={n_blocks}")
    return min(records, key=lambda r: (r.toffoli, r.qubits)), min(records, key=lambda r: (r.qubits, r.toffoli))


def compute_interferometer_resources(n_blocks: int, layer_lbs: int, final_lbs: int) -> Record:
    est = estimate_interferometer_resources(
        n_blocks=n_blocks,
        n_rows=N_ROWS,
        phase_bitsize=PHASE_BITSIZE,
        layer_log_block_size=layer_lbs,
        final_log_block_size=final_lbs,
    )
    return Record(est.toffoli, est.qubits, est.layer_log_block_size, est.final_log_block_size)


def optimize_interferometer(n_blocks: int) -> tuple[Record, Record]:
    records = [
        compute_interferometer_resources(n_blocks, layer_lbs, final_lbs)
        for layer_lbs in SWEEP_LOG_BLOCK_SIZES
        for final_lbs in SWEEP_LOG_BLOCK_SIZES
    ]
    return min(records, key=lambda r: (r.toffoli, r.qubits)), min(records, key=lambda r: (r.qubits, r.toffoli))


def fit_power_law(x: np.ndarray, y: list[int]) -> tuple[float, float]:
    alpha, log_c = np.polyfit(np.log(x), np.log(np.asarray(y, dtype=float)), 1)
    return float(alpha), float(np.exp(log_c))


def series(records: dict[str, list[Record]], metric: str) -> dict[str, list[int]]:
    return {name: [getattr(r, metric) for r in recs] for name, recs in records.items()}


print("=" * 78)
print("Block-unitary synthesis comparison")
print(f"  N = {N_ROWS} (smallest power of two >= 8*26 = {MULTIPLIER * N_SITES})")
print(f"  blocks M = k^3 for k={K_VALUES}")
print(f"  phase_bitsize = {PHASE_BITSIZE}")
print("=" * 78)

records: dict[str, list[Record]] = {
    "interferometer_topt": [],
    "interferometer_qopt": [],
    "standard_topt": [],
    "standard_qopt": [],
}

for k, n_blocks in zip(K_VALUES, N_BLOCKS_VALUES):
    print(f"k={k}, M={n_blocks}")
    it, iq = optimize_interferometer(n_blocks)
    st, sq = optimize_standard(n_blocks)
    records["interferometer_topt"].append(it)
    records["interferometer_qopt"].append(iq)
    records["standard_topt"].append(st)
    records["standard_qopt"].append(sq)
    approx_lbs = optimal_interferometer_log_block_sizes(n_blocks, N_ROWS, PHASE_BITSIZE)
    print(
        f"  interferometer T-opt T={it.toffoli:,} Q={it.qubits} "
        f"lbs=({it.layer_lbs},{it.final_lbs}); Q-opt T={iq.toffoli:,} Q={iq.qubits} "
        f"lbs=({iq.layer_lbs},{iq.final_lbs}); continuous-opt approx={approx_lbs}"
    )
    print(
        f"  standard       T-opt T={st.toffoli:,} Q={st.qubits} lbs={st.layer_lbs}; "
        f"Q-opt T={sq.toffoli:,} Q={sq.qubits} lbs={sq.layer_lbs}"
    )

M = np.asarray(N_BLOCKS_VALUES, dtype=float)
fits = {
    name: {
        "toffoli": fit_power_law(M, [r.toffoli for r in recs]),
        "qubits": fit_power_law(M, [r.qubits for r in recs]),
    }
    for name, recs in records.items()
}

print("\nPower-law fits y = c * M^alpha")
for name, fit in fits.items():
    ta, tc = fit["toffoli"]
    qa, qc = fit["qubits"]
    print(f"  {name:22s} Toffoli: alpha={ta:.3f}, c={tc:.3e}; Qubits: alpha={qa:.3f}, c={qc:.3e}")


LABELS = {
    "interferometer_topt": "Interferometer, Toffoli-opt",
    "interferometer_qopt": "Interferometer, qubit-opt",
    "standard_topt": "Standard Householder, Toffoli-opt",
    "standard_qopt": "Standard Householder, qubit-opt",
}
COLORS = {
    "interferometer_topt": "#1f77b4",
    "interferometer_qopt": "#2ca02c",
    "standard_topt": "#d62728",
    "standard_qopt": "#9467bd",
}
MARKERS = {
    "interferometer_topt": "o",
    "interferometer_qopt": "s",
    "standard_topt": "^",
    "standard_qopt": "D",
}


def plot_metric(metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    xfit = np.linspace(M.min(), M.max(), 300)
    for name, recs in records.items():
        data = [getattr(r, metric) for r in recs]
        alpha, c = fits[name][metric]
        ax.plot(M, data, marker=MARKERS[name], color=COLORS[name], linewidth=1.7, label=LABELS[name])
        ax.plot(xfit, c * xfit**alpha, ":", color=COLORS[name], linewidth=1.1, alpha=0.75)
        ax.text(M[-1] * 1.03, data[-1], f"a={alpha:.2f}\nc={c:.1e}", color=COLORS[name], fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M = k^3 blocks")
    ax.set_ylabel(ylabel)
    ax.set_xticks(M)
    ax.set_xticklabels([f"{int(m)}\nk={k}" for m, k in zip(M, K_VALUES)])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(f"{ylabel} for block unitary synthesis (N={N_ROWS}, b={PHASE_BITSIZE})")
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    cols = [
        "k",
        "M",
        "Int T-opt T",
        "Int T-opt Q",
        "Int Q-opt T",
        "Int Q-opt Q",
        "Std T-opt T",
        "Std T-opt Q",
        "Std Q-opt T",
        "Std Q-opt Q",
    ]
    rows = []
    for i, k in enumerate(K_VALUES):
        rows.append(
            [
                k,
                N_BLOCKS_VALUES[i],
                f"{records['interferometer_topt'][i].toffoli:,}",
                records["interferometer_topt"][i].qubits,
                f"{records['interferometer_qopt'][i].toffoli:,}",
                records["interferometer_qopt"][i].qubits,
                f"{records['standard_topt'][i].toffoli:,}",
                records["standard_topt"][i].qubits,
                f"{records['standard_qopt'][i].toffoli:,}",
                records["standard_qopt"][i].qubits,
            ]
        )
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#25364a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2:
            cell.set_facecolor("#f3f6fa")
    ax.set_title("Resource counts and optimization modes", fontsize=12)
    return fig


def summary_page():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    lines = [
        "Block-aware interferometer synthesis with QROAM-loaded phases",
        "",
        f"Dimension: N = {N_ROWS}, the smallest power of two >= 8*26 = {MULTIPLIER * N_SITES}.",
        f"Blocks: M = k^3 for k = {K_VALUES}. Phase precision b = {PHASE_BITSIZE}.",
        "",
        "Interferometer model from the note:",
        "  Each of N beamsplitter layers loads (alpha_l(x,j), beta_l(x,j)) by QROAM.",
        "  The QROAM address is (block x, pair j); the target shift acts only on the target register.",
        "  A final QROAM table addressed by (x,y) applies the block-dependent diagonal phase.",
        "",
        "Leading Toffoli estimate:",
        "  T_layer = ceil(M*N/(2*Lambda)) + 2*Lambda*b - 5",
        "  T_final = ceil(M*N/Lambda_f) + Lambda_f*b + ceil(M*N/Lambda_f) + Lambda_f - 6",
        "  T_total = N*T_layer + (log2(N)-2)*(N-1) + T_final",
        "",
        "Fitted power laws y = c * M^alpha:",
    ]
    for name in records:
        ta, tc = fits[name]["toffoli"]
        qa, qc = fits[name]["qubits"]
        lines.append(f"  {LABELS[name]}: T alpha={ta:.3f}, c={tc:.3e}; Q alpha={qa:.3f}, c={qc:.3e}")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
    fig.tight_layout()
    return fig


os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
with pdf_backend.PdfPages(OUT_PDF) as pdf:
    for make in (summary_page, lambda: plot_metric("toffoli", "Toffoli count"), lambda: plot_metric("qubits", "Peak logical qubits"), table_page):
        fig = make()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    info = pdf.infodict()
    info["Title"] = "Block-Unitary Interferometer QROAM Resource Comparison"
    info["Author"] = "qc_exciton_LCC"

print(f"\nWrote report: {OUT_PDF}")


def send_email(recipient: str, pdf_path: str) -> bool:
    subject = "Block-unitary interferometer QROAM resource report"
    fit_lines = []
    for name in records:
        ta, tc = fits[name]["toffoli"]
        qa, qc = fits[name]["qubits"]
        fit_lines.append(f"{LABELS[name]}: Toffoli alpha={ta:.3f}, c={tc:.3e}; qubits alpha={qa:.3f}, c={qc:.3e}")
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached is the block-unitary resource report comparing the new
        block-aware interferometer/QROAM synthesis against the existing
        Householder/QROAM block-unitary synthesis.

        Parameters:
          M = k^3 for k = {K_VALUES}
          N = {N_ROWS} (smallest power of two >= 8*26 = {MULTIPLIER * N_SITES})
          phase_bitsize = {PHASE_BITSIZE}

        Fits y = c * M^alpha:
        """
    )
    body += "\n".join(f"          {line}" for line in fit_lines)
    body += "\n"

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
