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
    BlockUnitaryInterferometerSynthesisQROAM,
    optimal_interferometer_log_block_sizes,
)
from integrations.qualtran.model_resource_counts import optimize_block_unitary_interferometer
from integrations.qualtran.block_unitary_synthesis_QROAM import BlockUnitarySynthesisQROAM
from integrations.qualtran.unitary_synthesis_QROAM import UnitarySynthesisQROAM
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

K_VALUES = list(range(1, 7))
N_BLOCKS_VALUES = [k**3 for k in K_VALUES]
N_SITES = 26
MULTIPLIER = 8
N_COLS = 1 << (MULTIPLIER * N_SITES - 1).bit_length()  # round up to power of 2
N_ROWS = N_COLS
N_REFLECTIONS = N_COLS
PHASE_BITSIZE = 32
QPE_SUBNORMALIZATION_FACTOR = 10**5
SWEEP_LOG_BLOCK_SIZES = list(range(0, 11))
OUT_PDF = os.path.join(
    REPO_ROOT,
    "docs",
    f"block_unitary_interferometer_resource_report_rows{N_ROWS}_cols{N_COLS}_bloq_vs_theory_b{PHASE_BITSIZE}.pdf",
)
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
    toffoli_opt = records[0]
    for record in records[1:]:
        if record.toffoli > toffoli_opt.toffoli:
            break
        if (record.toffoli, record.qubits) < (toffoli_opt.toffoli, toffoli_opt.qubits):
            toffoli_opt = record
    return toffoli_opt, min(records, key=lambda r: (r.qubits, r.toffoli))


def _lbs_for(lbs: int, n_blocks: int) -> list:
    """Map scalar lbs to the QROAM log_block_sizes list that gives total lambda = 2^lbs.

    For n_blocks > 1 the QROAM is 2-D (block, pair/system); we block only in the
    second (pair/system) dimension to match the analytic formula's single-lambda model.
    For n_blocks == 1 there is only one selection dimension, so a 1-element list suffices.
    """
    return [lbs] if n_blocks == 1 else [0, lbs]


def compute_interferometer_resources(
    n_blocks: int,
    lambda1_lbs: int,
    lambda2_lbs: int,
) -> Record:
    bloq = BlockUnitaryInterferometerSynthesisQROAM(
        n_blocks=n_blocks,
        n_rows=N_ROWS,
        phase_bitsize=PHASE_BITSIZE,
        n_layers=N_COLS,
        log_block_sizes=_lbs_for(lambda1_lbs, n_blocks),
        final_log_block_sizes=_lbs_for(lambda1_lbs, n_blocks),
        final_adjoint_log_block_sizes=_lbs_for(lambda2_lbs, n_blocks),
    )
    return Record(int(get_Toffoli_counts(bloq)), int(get_qubit_counts(bloq)), lambda1_lbs, lambda2_lbs)


def optimize_interferometer(n_blocks: int) -> tuple[Record, Record]:
    records = [
        compute_interferometer_resources(n_blocks, lambda1_lbs, lambda2_lbs)
        for lambda1_lbs in SWEEP_LOG_BLOCK_SIZES
        for lambda2_lbs in SWEEP_LOG_BLOCK_SIZES
    ]
    return min(records, key=lambda r: (r.toffoli, r.qubits)), min(records, key=lambda r: (r.qubits, r.toffoli))


def optimize_theory(n_blocks: int) -> tuple[Record, Record]:
    t_rec = optimize_block_unitary_interferometer(
        n_blocks, N_ROWS, PHASE_BITSIZE, objective="toffoli"
    )
    q_rec = optimize_block_unitary_interferometer(
        n_blocks, N_ROWS, PHASE_BITSIZE, objective="qubits"
    )
    return (
        Record(t_rec.toffoli, t_rec.qubits, t_rec.log_lambda_1, t_rec.log_lambda_2),
        Record(q_rec.toffoli, q_rec.qubits, q_rec.log_lambda_1, q_rec.log_lambda_2),
    )


def fit_power_law(x: np.ndarray, y: list[int]) -> tuple[float, float]:
    alpha, log_c = np.polyfit(np.log(x), np.log(np.asarray(y, dtype=float)), 1)
    return float(alpha), float(np.exp(log_c))


def series(records: dict[str, list[Record]], metric: str) -> dict[str, list[int]]:
    return {name: [getattr(r, metric) for r in recs] for name, recs in records.items()}


print("=" * 78)
print("Block-unitary synthesis comparison")
print(f"  rows = cols = {N_ROWS} (smallest power of two >= 8*26 = {MULTIPLIER*N_SITES})")
print(f"  cols/reflections/layers = {N_COLS} (rounded up from 8*26={MULTIPLIER*N_SITES})")
print(f"  blocks M = k^3 for k={K_VALUES}")
print(f"  phase_bitsize = {PHASE_BITSIZE}")
print("=" * 78)

records: dict[str, list[Record]] = {
    "interferometer_topt": [],
    "interferometer_qopt": [],
    "theory_topt": [],
    "theory_qopt": [],
    "standard_topt": [],
    "standard_qopt": [],
}

for k, n_blocks in zip(K_VALUES, N_BLOCKS_VALUES):
    print(f"k={k}, M={n_blocks}")
    it, iq = optimize_interferometer(n_blocks)
    tt, tq = optimize_theory(n_blocks)
    st, sq = optimize_standard(n_blocks)
    records["interferometer_topt"].append(it)
    records["interferometer_qopt"].append(iq)
    records["theory_topt"].append(tt)
    records["theory_qopt"].append(tq)
    records["standard_topt"].append(st)
    records["standard_qopt"].append(sq)
    print(
        f"  interferometer T-opt T={it.toffoli:,} Q={it.qubits} "
        f"lbs=(lambda1={it.layer_lbs}, lambda2={it.final_lbs}); "
        f"Q-opt T={iq.toffoli:,} Q={iq.qubits} "
        f"lbs=(lambda1={iq.layer_lbs}, lambda2={iq.final_lbs})"
    )
    print(
        f"  theoretical    T-opt T={tt.toffoli:,} Q={tt.qubits} "
        f"lbs=(lambda1={tt.layer_lbs}, lambda2={tt.final_lbs}); "
        f"Q-opt T={tq.toffoli:,} Q={tq.qubits} "
        f"lbs=(lambda1={tq.layer_lbs}, lambda2={tq.final_lbs})"
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
    "interferometer_topt": "Interferometer bloq, Toffoli-opt",
    "interferometer_qopt": "Interferometer bloq, qubit-opt",
    "theory_topt": "Theory formula, Toffoli-opt",
    "theory_qopt": "Theory formula, qubit-opt",
    "standard_topt": "Standard Householder, Toffoli-opt",
    "standard_qopt": "Standard Householder, qubit-opt",
}
COLORS = {
    "interferometer_topt": "#1f77b4",
    "interferometer_qopt": "#2ca02c",
    "theory_topt": "#17becf",
    "theory_qopt": "#bcbd22",
    "standard_topt": "#d62728",
    "standard_qopt": "#9467bd",
}
MARKERS = {
    "interferometer_topt": "o",
    "interferometer_qopt": "s",
    "theory_topt": "P",
    "theory_qopt": "X",
    "standard_topt": "^",
    "standard_qopt": "D",
}


def plot_metric(metric: str, ylabel: str, *, scale: int = 1, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    xfit = np.linspace(M.min(), M.max(), 300)
    for name, recs in records.items():
        data = [scale * getattr(r, metric) for r in recs]
        alpha, c = fits[name][metric]
        c *= scale
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
    ax.set_title(
        f"{ylabel} for block unitary synthesis "
        f"(rows={N_ROWS}, cols={N_COLS}, b={PHASE_BITSIZE}){title_suffix}"
    )
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.axis("off")
    cols = [
        "k",
        "M",
        "Int T-opt T",
        "Int T-opt Q",
        "Int Q-opt T",
        "Int Q-opt Q",
        "Theory T-opt T",
        "Theory T-opt Q",
        "Theory Q-opt T",
        "Theory Q-opt Q",
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
                f"{records['theory_topt'][i].toffoli:,}",
                records["theory_topt"][i].qubits,
                f"{records['theory_qopt'][i].toffoli:,}",
                records["theory_qopt"][i].qubits,
                f"{records['standard_topt'][i].toffoli:,}",
                records["standard_topt"][i].qubits,
                f"{records['standard_qopt'][i].toffoli:,}",
                records["standard_qopt"][i].qubits,
            ]
        )
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.2)
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
        f"Rows = Cols = {N_ROWS}, the smallest power of two >= 8*26 = {MULTIPLIER*N_SITES}.",
        f"Columns/reflections/layers: {N_COLS} (rounded up from 8*26={MULTIPLIER*N_SITES} to power of two).",
        f"Blocks: M = k^3 for k = {K_VALUES}. Phase precision b = {PHASE_BITSIZE}.",
        "",
        "Interferometer bloq curves use get_Toffoli_counts / get_qubit_counts on",
        "  BlockUnitaryInterferometerSynthesisQROAM (the actual Qualtran circuit).",
        "  Key differences from theory formula:",
        "  - Controlled AddIntoPhaseGrad (~2x vs uncontrolled) in each beamsplitter layer",
        "  - Shifts modeled by N//2 pairs of AddK (cyclic increment) instead of analytic formula",
        "  - QROAM cost from Qualtran's internal model (may differ by small constants)",
        "",
        "Theory formula curves use the N x N model from model_resource_counts.py:",
        "  T_layer = ceil(M*N/(2*L1)) + 2*b*L1 - 5 (uncontrolled AddIntoPhaseGrad approx)",
        "  T_final = ceil(M*N/L1) + b*L1 + ceil(M*N/L2) + L2 - 6",
        "  T_total = C*T_layer + (n-2)*(N-1) + T_final  [n=log2(N), C=layers]",
        "",
        f"Additional scaled plots multiply all counts by {QPE_SUBNORMALIZATION_FACTOR:.0e},",
        "  termed multiplied by QPE iteration and subnormalization. Qubits are unscaled.",
        "",
        "Optimization convention:",
        "  Lambda_1 is used for layer QROAM and final forward/phase shifts.",
        "  Lambda_2 is used for final-layer QROAM uncomputation.",
        "  The standard Householder baseline keeps its previous scalar early-stop sweep.",
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
    for make in (
        summary_page,
        lambda: plot_metric("toffoli", "Toffoli count"),
        lambda: plot_metric("qubits", "Peak logical qubits"),
        lambda: plot_metric(
            "toffoli",
            "Toffoli count multiplied by QPE iteration and subnormalization",
            scale=QPE_SUBNORMALIZATION_FACTOR,
            title_suffix="; multiplied by QPE iteration and subnormalization",
        ),
        table_page,
    ):
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
        Householder/QROAM block-unitary synthesis and the theoretical formula
        values from model_resource_counts.py.

        Parameters:
          M = k^3 for k = {K_VALUES}
          rows = {N_ROWS} (smallest power of two >= 8*26 = {N_COLS})
          cols/reflections/layers = {N_COLS} (strictly 8*26)
          phase_bitsize = {PHASE_BITSIZE}
          lambda_1 is used for layer QROAM and final forward/phase shifts
          lambda_2 is used for final-layer QROAM uncomputation
          theory curves use the N x N theoretical model
          Toffoli scaled plots multiply counts by {QPE_SUBNORMALIZATION_FACTOR:.0e};
          qubits are unscaled

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
