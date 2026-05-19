#!/usr/bin/env python3
"""Block-encode K matrices A_k of size 2^n x 2^n: SVD vs direct dilation.

Approach 1 (SVD):
  A_k = U_k Sigma_k V_k
  Build sum_k |k><k| (x) U_k via BlockUnitaryInterferometerSynthesisQROAM.
  Implement sum_k |k><k| (x) Sigma_k via QROAMClean(load theta_{k,i})
    + controlled AddIntoPhaseGrad on a 1-qubit ancilla + QROAMCleanAdjoint.
  Build sum_k |k><k| (x) V_k via another BlockUnitaryInterferometerSynthesisQROAM.

Approach 2 (Dilation):
  Complete each A_k to a unitary W_k of size 2^(n+1) x 2^(n+1) (Stinespring).
  Synthesize sum_k |k><k| (x) W_k via BlockUnitaryInterferometerSynthesisQROAM
    on a system register one qubit wider.
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
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

from integrations.qualtran.block_unitary_interferometer_QROAM import (
    BlockUnitaryInterferometerSynthesisQROAM,
)
from integrations.qualtran.svd_block_encoding_interferometer import (
    SVDBlockEncodingInterferometer,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

K = 8
N_VALUES = [2, 3, 4, 5, 6, 7]
PHASE_BITSIZE = 32
SWEEP_LBS = list(range(0, 12))
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"svd_vs_dilation_blockencoding_K{K}_b{PHASE_BITSIZE}.pdf"
)
RECIPIENT = "jchen9@caltech.edu"


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    lbs: tuple


def _lbs_for_2d(lbs: int, n_blocks: int):
    """Match analytic single-lambda model. With >1 block, block only the pair dim."""
    return [lbs] if n_blocks == 1 else [0, lbs]


def _cap(lbs: int, max_lbs: int) -> int:
    return max(0, min(int(lbs), int(max_lbs)))


def interferometer_cost(n_blocks: int, n_rows: int, lbs: int):
    bloq = BlockUnitaryInterferometerSynthesisQROAM(
        n_blocks=n_blocks,
        n_rows=n_rows,
        phase_bitsize=PHASE_BITSIZE,
        n_layers=n_rows,
        log_block_sizes=_lbs_for_2d(lbs, n_blocks),
        final_log_block_sizes=_lbs_for_2d(lbs, n_blocks),
        final_adjoint_log_block_sizes=_lbs_for_2d(lbs, n_blocks),
    )
    return int(get_Toffoli_counts(bloq)), int(get_qubit_counts(bloq))


def optimize_interferometer(n_blocks: int, n_rows: int) -> tuple[Rec, Rec]:
    recs: list[Rec] = []
    for lbs in SWEEP_LBS:
        try:
            t, q = interferometer_cost(n_blocks, n_rows, lbs)
        except Exception:
            continue
        recs.append(Rec(t, q, (lbs,)))
    if not recs:
        raise RuntimeError(f"no valid interferometer config for K={n_blocks}, N={n_rows}")
    topt = min(recs, key=lambda r: (r.toffoli, r.qubits))
    qopt = min(recs, key=lambda r: (r.qubits, r.toffoli))
    return topt, qopt


def diagonal_cost(n_blocks: int, n_rows: int, lbs_fwd: int, lbs_adj: int):
    """QROAM(load theta_{k,i}) + ctrl-AddIntoPhaseGrad + QROAMCleanAdjoint."""
    if n_blocks > 1:
        data_shape = (n_blocks, n_rows)
        max_lbs1 = int(np.log2(n_rows))
        fwd_lbs = (0, _cap(lbs_fwd, max_lbs1))
        adj_lbs = (0, _cap(lbs_adj, max_lbs1))
    else:
        data_shape = (n_rows,)
        max_lbs0 = int(np.log2(n_rows))
        fwd_lbs = (_cap(lbs_fwd, max_lbs0),)
        adj_lbs = (_cap(lbs_adj, max_lbs0),)

    qroam = QROAMClean.build_from_bitsize(
        data_shape,
        target_bitsizes=(PHASE_BITSIZE,),
        log_block_sizes=fwd_lbs,
    )
    qroam_adj = QROAMCleanAdjoint.build_from_bitsize(
        data_shape,
        target_bitsizes=(PHASE_BITSIZE,),
        log_block_sizes=adj_lbs,
        target_shapes=(tuple(1 << b for b in adj_lbs),),
    )
    ctrl_add = AddIntoPhaseGrad(PHASE_BITSIZE, PHASE_BITSIZE).controlled()

    t = (
        int(get_Toffoli_counts(qroam))
        + int(get_Toffoli_counts(ctrl_add))
        + int(get_Toffoli_counts(qroam_adj))
    )
    # Peak qubits during diagonal stage (excluding BE ancilla, added later in _combine_svd):
    # selection (block+system) + QROAM target+junk are in qroam.qubits;
    # phase_gradient (b qubits) is also allocated in this stage, but is NOT in
    # qroam's signature, so add it explicitly. AddIntoPhaseGrad sub-bloq is small.
    q = max(int(get_qubit_counts(qroam)), int(get_qubit_counts(qroam_adj))) + PHASE_BITSIZE
    return t, q


def optimize_diagonal(n_blocks: int, n_rows: int) -> tuple[Rec, Rec]:
    recs: list[Rec] = []
    for lf in SWEEP_LBS:
        for la in SWEEP_LBS:
            try:
                t, q = diagonal_cost(n_blocks, n_rows, lf, la)
            except Exception:
                continue
            recs.append(Rec(t, q, (lf, la)))
    if not recs:
        raise RuntimeError(f"no diagonal config K={n_blocks}, N={n_rows}")
    topt = min(recs, key=lambda r: (r.toffoli, r.qubits))
    qopt = min(recs, key=lambda r: (r.qubits, r.toffoli))
    return topt, qopt


def _make_svd_bloq(n_blocks: int, n_rows: int, intf_lbs, diag_fwd_lbs, diag_adj_lbs):
    return SVDBlockEncodingInterferometer(
        n_blocks=n_blocks,
        n_rows=n_rows,
        phase_bitsize=PHASE_BITSIZE,
        n_layers=n_rows,
        interferometer_log_block_sizes=_lbs_for_2d(intf_lbs, n_blocks),
        interferometer_final_log_block_sizes=_lbs_for_2d(intf_lbs, n_blocks),
        interferometer_final_adjoint_log_block_sizes=_lbs_for_2d(intf_lbs, n_blocks),
        diag_log_block_sizes=_lbs_for_2d(diag_fwd_lbs, n_blocks),
        diag_adjoint_log_block_sizes=_lbs_for_2d(diag_adj_lbs, n_blocks),
    )


def _svd_eval(n_blocks: int, n_rows: int, intf_lbs: int, dfwd: int, dadj: int) -> Rec:
    bloq = _make_svd_bloq(n_blocks, n_rows, intf_lbs, dfwd, dadj)
    return Rec(int(get_Toffoli_counts(bloq)), int(get_qubit_counts(bloq)),
               ("intf", (intf_lbs,), "diag_fwd", (dfwd,), "diag_adj", (dadj,)))


def _qroam_T_Q(qroam_bloq):
    return int(get_Toffoli_counts(qroam_bloq)), int(get_qubit_counts(qroam_bloq))


def svd_total(n_blocks: int, n: int) -> tuple[Rec, Rec]:
    """Optimize SVDBlockEncodingInterferometer over (intf_lbs, diag_fwd_lbs, diag_adj_lbs).

    The bloq's build_call_graph emits exactly:
        2 x interferometer(intf_lbs) + QROAMClean(dfwd) + ctrl-AddPhaseGrad
        + QROAMCleanAdjoint(dadj)
    so T is separable (sum) and Q is component-max (Qualtran's QubitCount takes max
    over a sequential call graph).  We minimize each sub-bloq independently, then
    instantiate the SVD bloq and read T/Q via get_Toffoli_counts / get_qubit_counts.
    """
    n_rows = 1 << n

    # Per-component (T, Q) over the lbs sweep.
    intf_recs = [(lbs, *interferometer_cost(n_blocks, n_rows, lbs)) for lbs in SWEEP_LBS]
    fwd_recs = []
    adj_recs = []
    for lbs in SWEEP_LBS:
        bloq_template = _make_svd_bloq(n_blocks, n_rows, 0, lbs, 0)
        try:
            fwd_recs.append((lbs, *_qroam_T_Q(bloq_template.diag_qroam)))
        except Exception:
            pass
        bloq_template = _make_svd_bloq(n_blocks, n_rows, 0, 0, lbs)
        try:
            adj_recs.append((lbs, *_qroam_T_Q(bloq_template.diag_qroam_adjoint)))
        except Exception:
            pass

    def pick(records, key):
        return min(records, key=key)

    intf_T = pick(intf_recs, key=lambda r: (r[1], r[2]))
    fwd_T = pick(fwd_recs, key=lambda r: (r[1], r[2]))
    adj_T = pick(adj_recs, key=lambda r: (r[1], r[2]))
    intf_Q = pick(intf_recs, key=lambda r: (r[2], r[1]))
    fwd_Q = pick(fwd_recs, key=lambda r: (r[2], r[1]))
    adj_Q = pick(adj_recs, key=lambda r: (r[2], r[1]))

    t_opt = _svd_eval(n_blocks, n_rows, intf_T[0], fwd_T[0], adj_T[0])
    q_opt = _svd_eval(n_blocks, n_rows, intf_Q[0], fwd_Q[0], adj_Q[0])
    return t_opt, q_opt


def dilation_total(n_blocks: int, n: int) -> tuple[Rec, Rec]:
    topt, qopt = optimize_interferometer(n_blocks, 1 << (n + 1))
    return Rec(topt.toffoli, topt.qubits, ("intf", topt.lbs)), Rec(qopt.toffoli, qopt.qubits, ("intf", qopt.lbs))


print("=" * 78)
print("SVD vs direct-dilation block-encoding for K blocks of 2^n x 2^n matrices")
print(f"  K = {K}, n in {N_VALUES}, phase_bitsize = {PHASE_BITSIZE}")
print("=" * 78)

svd_topt: list[Rec] = []
svd_qopt: list[Rec] = []
dil_topt: list[Rec] = []
dil_qopt: list[Rec] = []
for n in N_VALUES:
    print(f"n={n} (matrix dim={1 << n})")
    st, sq = svd_total(K, n)
    dt, dq = dilation_total(K, n)
    svd_topt.append(st); svd_qopt.append(sq)
    dil_topt.append(dt); dil_qopt.append(dq)
    print(f"  SVD  T-opt: T={st.toffoli:,} Q={st.qubits}; Q-opt: T={sq.toffoli:,} Q={sq.qubits}")
    print(f"  Dil  T-opt: T={dt.toffoli:,} Q={dt.qubits}; Q-opt: T={dq.toffoli:,} Q={dq.qubits}")

n_arr = np.asarray(N_VALUES, dtype=float)
series = {
    "svd_topt": svd_topt, "svd_qopt": svd_qopt,
    "dil_topt": dil_topt, "dil_qopt": dil_qopt,
}
arrs_t = {k: np.asarray([r.toffoli for r in v], dtype=float) for k, v in series.items()}
arrs_q = {k: np.asarray([r.qubits for r in v], dtype=float) for k, v in series.items()}


def fit_exp_in_n(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = c * 2^(alpha * n). Returns (alpha, c)."""
    log2y = np.log2(y)
    alpha, log2c = np.polyfit(x, log2y, 1)
    return float(alpha), float(2.0 ** log2c)


fits = {}
for k in series:
    fits[k + "_toffoli"] = fit_exp_in_n(n_arr, arrs_t[k])
    fits[k + "_qubits"] = fit_exp_in_n(n_arr, arrs_q[k])

print("\nFits y = c * 2^(alpha * n):")
for name, (a, c) in fits.items():
    print(f"  {name:18s} alpha={a:.3f}, c={c:.3e}")


STYLES = {
    "svd_topt": ("SVD, Toffoli-opt", "#1f77b4", "o"),
    "svd_qopt": ("SVD, qubit-opt",   "#2ca02c", "s"),
    "dil_topt": ("Dilation, Toffoli-opt", "#d62728", "^"),
    "dil_qopt": ("Dilation, qubit-opt",   "#9467bd", "D"),
}


def plot_metric(metric: str, ylabel: str):
    arrs = arrs_t if metric == "toffoli" else arrs_q
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    xfit = np.linspace(n_arr.min(), n_arr.max(), 300)
    for key, (lbl, color, marker) in STYLES.items():
        data = arrs[key]
        a, c = fits[key + "_" + metric]
        ax.plot(n_arr, data, marker=marker, linestyle="-", color=color, linewidth=1.7, label=lbl)
        ax.plot(xfit, c * 2.0 ** (a * xfit), ":", color=color, alpha=0.6)
        ax.text(n_arr[-1] * 1.005, data[-1], f"α={a:.2f}\nc={c:.1e}", color=color, fontsize=7, va="center")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("n  (system size = 2^n)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(N_VALUES)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(f"{ylabel}: SVD vs direct dilation  (K={K} blocks, b={PHASE_BITSIZE})")
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    cols = ["n", "dim",
            "SVD T-opt T", "SVD T-opt Q", "SVD Q-opt T", "SVD Q-opt Q",
            "Dil T-opt T", "Dil T-opt Q", "Dil Q-opt T", "Dil Q-opt Q"]
    rows = []
    for i, n in enumerate(N_VALUES):
        st, sq = svd_topt[i], svd_qopt[i]
        dt, dq = dil_topt[i], dil_qopt[i]
        rows.append([
            n, 1 << n,
            f"{st.toffoli:,}", st.qubits, f"{sq.toffoli:,}", sq.qubits,
            f"{dt.toffoli:,}", dt.qubits, f"{dq.toffoli:,}", dq.qubits,
        ])
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
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
        "Block-encoding K matrices A_k of size 2^n x 2^n",
        "",
        f"K = {K} blocks; n sweep = {N_VALUES}; phase_bitsize b = {PHASE_BITSIZE}.",
        "",
        "Approach 1 (SVD-based):",
        "  A_k = U_k Sigma_k V_k.  Block-encoded as",
        "    [ sum_k |k><k| (x) U_k ]  ·  [diag(sigma) + ancilla rotation]  ·  [ sum_k |k><k| (x) V_k ].",
        "  - Interferometer cost for U_k and V_k:",
        "      BlockUnitaryInterferometerSynthesisQROAM(n_blocks=K, n_rows=2^n, n_layers=2^n).",
        "  - Diagonal cost:  QROAMClean(data_shape=(K, 2^n), b)  + ctrl-AddIntoPhaseGrad(b,b)  + QROAMCleanAdjoint.",
        "  - Toffoli total  = 2 * interferometer + diagonal.",
        "  - Qubit total    ~ max(interferometer, diagonal) + 1 (block-encoding ancilla).",
        "",
        "Approach 2 (direct dilation):",
        "  Each A_k is completed (Stinespring) to a unitary W_k of size 2^(n+1) x 2^(n+1).",
        "  Synthesize sum_k |k><k| (x) W_k via one interferometer on (n+1) system qubits:",
        "    BlockUnitaryInterferometerSynthesisQROAM(n_blocks=K, n_rows=2^(n+1), n_layers=2^(n+1)).",
        "",
        "All resource counts come from Qualtran's get_cost_value(QECGatesCost()), via",
        "  utils.get_Toffoli_counts and utils.get_qubit_counts.  Each cost is optimized over",
        f"  log_block_size in {SWEEP_LBS[0]}..{SWEEP_LBS[-1]}.",
        "",
        "Two optimization modes per approach are shown: Toffoli-optimal and qubit-optimal.",
        "",
        "Fits y = c * 2^(alpha * n):",
    ]
    for key, (lbl, _, _) in STYLES.items():
        at, ct = fits[key + "_toffoli"]
        aq, cq = fits[key + "_qubits"]
        lines.append(f"  {lbl:24s} T: alpha={at:.3f}, c={ct:.3e}; Q: alpha={aq:.3f}, c={cq:.3e}")
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
    info["Title"] = "SVD vs Dilation Block-Encoding Resource Comparison"
    info["Author"] = "qc_exciton_LCC"
print(f"\nWrote report: {OUT_PDF}")


def send_email(recipient: str, pdf_path: str) -> bool:
    subject = "SVD vs dilation block-encoding resource report"
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached is the resource comparison report for block-encoding K = {K}
        matrices A_k of size 2^n x 2^n, comparing:

          (1) SVD-based: A_k = U_k Sigma_k V_k, with U_k and V_k synthesized via the
              block-unitary interferometer, and the diagonal Sigma_k implemented as
              QROAMClean + controlled AddIntoPhaseGrad + QROAMCleanAdjoint on a
              single ancilla qubit.

          (2) Direct dilation: each A_k is Stinespring-dilated to a unitary W_k of
              size 2^(n+1) x 2^(n+1) and synthesized in one shot by the block-unitary
              interferometer on (n+1) system qubits.

        Parameters:
          K = {K}, n in {N_VALUES}, phase_bitsize = {PHASE_BITSIZE}.

        Fits y = c * 2^(alpha * n) (Toffoli-opt and qubit-opt curves both shown):
        """
    )
    for key, (lbl, _, _) in STYLES.items():
        at, ct = fits[key + "_toffoli"]
        aq, cq = fits[key + "_qubits"]
        body += f"          {lbl:24s} T alpha={at:.3f} c={ct:.3e}; Q alpha={aq:.3f} c={cq:.3e}\n"
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
