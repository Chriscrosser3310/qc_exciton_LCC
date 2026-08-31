#!/usr/bin/env python3
"""Exchange-Coulomb block-encoding resource report (PLANS.md item 2).

Parameters:
  N_up = 4, N_down = 22, N_IP = 26*8 = 208, N_k = k^3 for k = 1..6.

Reports Toffoli-optimal and qubit-optimal counts for both the controlled and
uncontrolled :class:`ExchangeCoulombBlockEncoding`, sweeping the QROAM
log_block_sizes inside each reflection rectangular block encoding.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from integrations.qualtran.exchange_Coulomb_block_encoding import (
    ExchangeCoulombBlockEncoding,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

N_UP, N_DOWN, N_IP = 4, 22, 26 * 8
K_VALUES = [1, 2, 3, 4, 5, 6]
N_K_VALUES = [k ** 3 for k in K_VALUES]
PHASE_BITSIZE = 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(REPO_ROOT, "docs", f"exchange_coulomb_report_Nup{N_UP}_Ndown{N_DOWN}_NIP{N_IP}_b{PHASE_BITSIZE}.pdf")


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    lbs: tuple


def _distribute(total: int, caps):
    """Greedy fill from the last (largest) dim first, capped by floor(log2 dim)."""
    out = [0] * len(caps)
    rem = total
    for i in range(len(caps) - 1, -1, -1):
        take = min(rem, caps[i])
        out[i] = take
        rem -= take
    return tuple(out)


def _opt_lbs(caps, b: int, *, adjoint: bool) -> tuple:
    """Closed-form optimal lambda for QROAM cost; distribute across dims.

    Forward QROAM:  T(L) = ceil(M/L) + b*(L-1)   ->   L* = sqrt(M/b),  T_min ~ 2*sqrt(M*b)
    Adjoint QROAM:  T(L) = ceil(M/L) + (L-1)     ->   L* = sqrt(M),     T_min ~ 2*sqrt(M)

    Returns the per-dim log_block_sizes that distribute the optimum log2(L*) across
    dimensions, capped by floor(log2(dim_size)).
    """
    log_M = sum(caps)
    log_b = int(np.log2(max(1, b)))
    raw = (log_M - log_b) // 2 if not adjoint else log_M // 2
    lbs_total = max(0, min(raw, log_M))
    return _distribute(lbs_total, caps)


def _caps_for_outer(N_k: int, n_rows: int):
    """Outer reflection-isometry: per-reflection QROAM shape is (K, n_rows)."""
    n_rows_bits = int(np.log2(n_rows))
    if N_k > 1:
        return (int(np.log2(N_k)), n_rows_bits)
    return (n_rows_bits,)


def _caps_for_inner_intf(N_k: int, n_rows: int):
    """SVD interferometer phase-layer QROAM shape is (K, n_rows/2)."""
    pair_bits = int(np.log2(max(1, n_rows // 2)))
    if N_k > 1:
        return (int(np.log2(N_k)), pair_bits)
    return (pair_bits,)


def _caps_for_inner_final(N_k: int, n_rows: int):
    """SVD interferometer final phase QROAM shape is (K, n_rows)."""
    return _caps_for_outer(N_k, n_rows)


def _caps_for_inner_diag(N_k: int, n_rows: int):
    """SVD diagonal Sigma_k QROAM shape is (K, n_rows)."""
    return _caps_for_outer(N_k, n_rows)


def _n_rows_up(N_up, N_IP):
    return 1 << max(1, (max(N_up, N_IP) - 1).bit_length())


def _n_rows_down(N_down, N_IP):
    return 1 << max(1, (max(N_down, N_IP) - 1).bit_length())


def _n_rows_inner(N_IP):
    return 1 << max(1, (N_IP - 1).bit_length())


def eval_uncontrolled(N_k, outer_fwd, outer_adj, inner_fwd, inner_adj) -> Rec:
    n_up = _n_rows_up(N_UP, N_IP)
    n_down = _n_rows_down(N_DOWN, N_IP)
    n_inner = _n_rows_inner(N_IP)
    o_caps = _caps_for_outer(N_k, max(n_up, n_down))
    i_caps = _caps_for_outer(N_k, n_inner)
    outer_lbs_fwd = tuple(_distribute(outer_fwd, o_caps))
    outer_lbs_adj = tuple(_distribute(outer_adj, o_caps))
    inner_lbs_fwd = tuple(_distribute(inner_fwd, i_caps))
    inner_lbs_adj = tuple(_distribute(inner_adj, i_caps))
    b = ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE,
        outer_amp_log_block_sizes=outer_lbs_fwd,
        outer_amp_adjoint_log_block_sizes=outer_lbs_adj,
        outer_phase_log_block_sizes=outer_lbs_fwd,
        outer_phase_adjoint_log_block_sizes=outer_lbs_adj,
        inner_intf_log_block_sizes=inner_lbs_fwd,
        inner_intf_final_log_block_sizes=inner_lbs_fwd,
        inner_intf_final_adjoint_log_block_sizes=inner_lbs_adj,
        inner_diag_log_block_sizes=inner_lbs_fwd,
        inner_diag_adjoint_log_block_sizes=inner_lbs_adj,
    )
    return Rec(int(get_Toffoli_counts(b)), int(get_qubit_counts(b)),
               (outer_fwd, outer_adj, inner_fwd, inner_adj))


def eval_controlled(N_k, outer_fwd, outer_adj, inner_fwd, inner_adj) -> Rec:
    n_up = _n_rows_up(N_UP, N_IP)
    n_down = _n_rows_down(N_DOWN, N_IP)
    n_inner = _n_rows_inner(N_IP)
    o_caps = _caps_for_outer(N_k, max(n_up, n_down))
    i_caps = _caps_for_outer(N_k, n_inner)
    outer_lbs_fwd = tuple(_distribute(outer_fwd, o_caps))
    outer_lbs_adj = tuple(_distribute(outer_adj, o_caps))
    inner_lbs_fwd = tuple(_distribute(inner_fwd, i_caps))
    inner_lbs_adj = tuple(_distribute(inner_adj, i_caps))
    inner = ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE,
        outer_amp_log_block_sizes=outer_lbs_fwd,
        outer_amp_adjoint_log_block_sizes=outer_lbs_adj,
        outer_phase_log_block_sizes=outer_lbs_fwd,
        outer_phase_adjoint_log_block_sizes=outer_lbs_adj,
        inner_intf_log_block_sizes=inner_lbs_fwd,
        inner_intf_final_log_block_sizes=inner_lbs_fwd,
        inner_intf_final_adjoint_log_block_sizes=inner_lbs_adj,
        inner_diag_log_block_sizes=inner_lbs_fwd,
        inner_diag_adjoint_log_block_sizes=inner_lbs_adj,
    )
    cb = inner.controlled()
    return Rec(int(get_Toffoli_counts(cb)), int(get_qubit_counts(cb)),
               (outer_fwd, outer_adj, inner_fwd, inner_adj))


def _build_bloq(N_k, controlled,
                outer_fwd, outer_adj,
                intf_layer, intf_final_fwd, intf_final_adj,
                diag_fwd, diag_adj):
    inner = ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_k, phase_bitsize=PHASE_BITSIZE,
        outer_amp_log_block_sizes=outer_fwd,
        outer_amp_adjoint_log_block_sizes=outer_adj,
        outer_phase_log_block_sizes=outer_fwd,
        outer_phase_adjoint_log_block_sizes=outer_adj,
        inner_intf_log_block_sizes=intf_layer,
        inner_intf_final_log_block_sizes=intf_final_fwd,
        inner_intf_final_adjoint_log_block_sizes=intf_final_adj,
        inner_diag_log_block_sizes=diag_fwd,
        inner_diag_adjoint_log_block_sizes=diag_adj,
    )
    bloq = inner.controlled() if controlled else inner
    return Rec(int(get_Toffoli_counts(bloq)), int(get_qubit_counts(bloq)),
               (outer_fwd, outer_adj, intf_layer, intf_final_fwd, intf_final_adj, diag_fwd, diag_adj))


def optimize(N_k, controlled: bool):
    """Closed-form T-opt and Q-opt: no sweep, just analytic optimum lambdas.

    T-opt: forward QROAM lambda ~ sqrt(M*b); adjoint QROAM lambda ~ sqrt(M).
    Q-opt: every lbs = 0 (no QROAM blocking -> minimal qubits).
    """
    n_up = _n_rows_up(N_UP, N_IP)
    n_inner = _n_rows_inner(N_IP)
    o_caps = _caps_for_outer(N_k, max(n_up, _n_rows_down(N_DOWN, N_IP)))
    intf_layer_caps = _caps_for_inner_intf(N_k, n_inner)
    intf_final_caps = _caps_for_inner_final(N_k, n_inner)
    diag_caps = _caps_for_inner_diag(N_k, n_inner)

    t_opt = _build_bloq(
        N_k, controlled,
        outer_fwd=_opt_lbs(o_caps, PHASE_BITSIZE, adjoint=False),
        outer_adj=_opt_lbs(o_caps, PHASE_BITSIZE, adjoint=True),
        intf_layer=_opt_lbs(intf_layer_caps, PHASE_BITSIZE, adjoint=False),
        intf_final_fwd=_opt_lbs(intf_final_caps, PHASE_BITSIZE, adjoint=False),
        intf_final_adj=_opt_lbs(intf_final_caps, PHASE_BITSIZE, adjoint=True),
        diag_fwd=_opt_lbs(diag_caps, PHASE_BITSIZE, adjoint=False),
        diag_adj=_opt_lbs(diag_caps, PHASE_BITSIZE, adjoint=True),
    )
    zo, zl, zf, zd = (tuple([0] * len(o_caps)), tuple([0] * len(intf_layer_caps)),
                      tuple([0] * len(intf_final_caps)), tuple([0] * len(diag_caps)))
    q_opt = _build_bloq(N_k, controlled, zo, zo, zl, zf, zf, zd, zd)
    return t_opt, q_opt


def main():
    print("=" * 78)
    print(f"Exchange-Coulomb resource sweep: N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP}")
    print(f"  N_k = k^3 for k in {K_VALUES}; phase_bitsize={PHASE_BITSIZE}")
    print("=" * 78)

    rows = []
    for k, N_k in zip(K_VALUES, N_K_VALUES):
        print(f"k={k}, N_k={N_k}")
        uT, uQ = optimize(N_k, controlled=False)
        cT, cQ = optimize(N_k, controlled=True)
        rows.append((k, N_k, uT, uQ, cT, cQ))
        print(f"  uncontrolled  T-opt: T={uT.toffoli:,} Q={uT.qubits} ; Q-opt: T={uQ.toffoli:,} Q={uQ.qubits}")
        print(f"  controlled    T-opt: T={cT.toffoli:,} Q={cT.qubits} ; Q-opt: T={cQ.toffoli:,} Q={cQ.qubits}")

    Karr = np.asarray(N_K_VALUES, dtype=float)
    series_T = {
        "u_topt": np.array([r[2].toffoli for r in rows], dtype=float),
        "u_qopt": np.array([r[3].toffoli for r in rows], dtype=float),
        "c_topt": np.array([r[4].toffoli for r in rows], dtype=float),
        "c_qopt": np.array([r[5].toffoli for r in rows], dtype=float),
    }
    series_Q = {
        "u_topt": np.array([r[2].qubits for r in rows], dtype=float),
        "u_qopt": np.array([r[3].qubits for r in rows], dtype=float),
        "c_topt": np.array([r[4].qubits for r in rows], dtype=float),
        "c_qopt": np.array([r[5].qubits for r in rows], dtype=float),
    }
    STYLES = {
        "u_topt": ("Uncontrolled, T-opt", "#1f77b4", "o"),
        "u_qopt": ("Uncontrolled, Q-opt", "#2ca02c", "s"),
        "c_topt": ("Controlled, T-opt",   "#d62728", "^"),
        "c_qopt": ("Controlled, Q-opt",   "#9467bd", "D"),
    }

    def plot(metric, ylabel):
        arr = series_T if metric == "toffoli" else series_Q
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for key, (lbl, color, marker) in STYLES.items():
            data = arr[key]
            ax.plot(Karr, data, marker=marker, linestyle="-", color=color, linewidth=1.7, label=lbl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N_k = k^3")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_K_VALUES)
        ax.set_xticklabels([f"{n}\nk={k}" for n, k in zip(N_K_VALUES, K_VALUES)])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_title(f"{ylabel}: exchange-Coulomb BE  (N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP}, b={PHASE_BITSIZE})")
        fig.tight_layout()
        return fig

    def table_page():
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis("off")
        cols = ["k", "N_k",
                "Uncontrolled T-opt T", "Uncontrolled T-opt Q",
                "Uncontrolled Q-opt T", "Uncontrolled Q-opt Q",
                "Controlled T-opt T", "Controlled T-opt Q",
                "Controlled Q-opt T", "Controlled Q-opt Q"]
        cells = []
        for (k, N_k, uT, uQ, cT, cQ) in rows:
            cells.append([
                k, N_k,
                f"{uT.toffoli:,}", uT.qubits, f"{uQ.toffoli:,}", uQ.qubits,
                f"{cT.toffoli:,}", cT.qubits, f"{cQ.toffoli:,}", cQ.qubits,
            ])
        tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
        ax.set_title("Resource counts: exchange-Coulomb BE", fontsize=12)
        return fig

    def summary_page():
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        lines = [
            "Exchange-Coulomb block-encoding resource sweep",
            "",
            f"Parameters:  N_up = {N_UP},  N_down = {N_DOWN},  N_IP = {N_IP}",
            f"             N_k = k^3 for k in {K_VALUES} (so N_k in {N_K_VALUES})",
            f"             phase_bitsize b = {PHASE_BITSIZE}",
            "",
            "Construction (data-free):",
            "  B = (B_up (x) B_down) . ModSub . PrepareUniformSuperposition_adj",
            "  Each B_up, B_down is a ReflectionRectangularBlockEncoding",
            "  Full = B . C . B^dagger  where C is another ReflectionRectangularBlockEncoding",
            "        on the inner matrix register (sum_Q |Q><Q| (x) A'_Q).",
            "",
            "Controlled version exploits the B-sandwich: only C is promoted to a controlled",
            "  block encoding; outer B / B^dagger pair self-cancels at ext_ctrl = 0.",
            "",
            "All Toffoli and qubit counts come from Qualtran's QECGatesCost / QubitCount",
            "  walking the bloq's build_call_graph (no analytic formulas).",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
        fig.tight_layout()
        return fig

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        for make in (
            summary_page,
            lambda: plot("toffoli", "Toffoli count"),
            lambda: plot("qubits", "Peak logical qubits"),
            table_page,
        ):
            fig = make()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "Exchange-Coulomb Block Encoding Resource Report"
    print(f"\nWrote report: {OUT_PDF}")

    # email
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "Exchange-Coulomb block encoding resource report"
    body_lines = [
        f"Hi,",
        "",
        f"Attached: exchange-Coulomb block-encoding cost report for",
        f"  N_up = {N_UP}, N_down = {N_DOWN}, N_IP = {N_IP}",
        f"  N_k = k^3 for k in {K_VALUES}",
        f"  phase_bitsize = {PHASE_BITSIZE}",
        "",
        "Costs come from Qualtran's QECGatesCost walking the bloq's build_call_graph.",
        "",
    ]
    for (k, N_k, uT, uQ, cT, cQ) in rows:
        body_lines.append(
            f"  k={k} (N_k={N_k:3d}):  uncontrolled T-opt T={uT.toffoli:>12,} Q={uT.qubits}; "
            f"Q-opt T={uQ.toffoli:>12,} Q={uQ.qubits}"
        )
        body_lines.append(
            f"                  controlled    T-opt T={cT.toffoli:>12,} Q={cT.qubits}; "
            f"Q-opt T={cQ.toffoli:>12,} Q={cQ.qubits}"
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
