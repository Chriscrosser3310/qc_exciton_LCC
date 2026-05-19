#!/usr/bin/env python3
"""SVD vs direct dilation for fixed matrix size N=256, varying block count K=k^3.

Same two approaches as svd_vs_dilation_block_encoding.py, but here the system
size is fixed at 2^n = 256 and K = k^3 sweeps from 2^3=8 to 6^3=216.
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

from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

from integrations.qualtran.block_unitary_interferometer_QROAM import (
    BlockUnitaryInterferometerSynthesisQROAM,
)
from integrations.qualtran.svd_block_encoding_interferometer import (
    SVDBlockEncodingInterferometer,
)
from integrations.qualtran.utils import get_Toffoli_counts, get_qubit_counts

N_FIXED = 8                          # matrix dim 2^8 = 256
K_K_VALUES = [2, 3, 4, 5, 6]         # k in k^3
K_VALUES = [k ** 3 for k in K_K_VALUES]
PHASE_BITSIZE = 32
SWEEP_LBS = list(range(0, 12))
OUT_PDF = os.path.join(
    REPO_ROOT,
    "docs",
    f"svd_vs_dilation_blockencoding_N{1 << N_FIXED}_Ksweep_b{PHASE_BITSIZE}.pdf",
)
RECIPIENT = "jchen9@caltech.edu"


@dataclass(frozen=True)
class Rec:
    toffoli: int
    qubits: int
    lbs: tuple


def _lbs_for_2d(lbs: int, n_blocks: int):
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
        raise RuntimeError(f"no interferometer config K={n_blocks} N={n_rows}")
    return min(recs, key=lambda r: (r.toffoli, r.qubits)), min(recs, key=lambda r: (r.qubits, r.toffoli))


def diagonal_cost(n_blocks: int, n_rows: int, lbs_fwd: int, lbs_adj: int):
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
        data_shape, target_bitsizes=(PHASE_BITSIZE,), log_block_sizes=fwd_lbs
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
    # Peak qubits during diagonal stage (excluding BE ancilla, added later):
    # qroam.qubits has selection + target+junk but NOT the phase_gradient register;
    # add PHASE_BITSIZE for the phase_gradient that is allocated throughout.
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
        raise RuntimeError(f"no diagonal config K={n_blocks} N={n_rows}")
    return min(recs, key=lambda r: (r.toffoli, r.qubits)), min(recs, key=lambda r: (r.qubits, r.toffoli))


def _make_svd_bloq(K: int, n_rows: int, intf_lbs, dfwd, dadj):
    return SVDBlockEncodingInterferometer(
        n_blocks=K,
        n_rows=n_rows,
        phase_bitsize=PHASE_BITSIZE,
        n_layers=n_rows,
        interferometer_log_block_sizes=_lbs_for_2d(intf_lbs, K),
        interferometer_final_log_block_sizes=_lbs_for_2d(intf_lbs, K),
        interferometer_final_adjoint_log_block_sizes=_lbs_for_2d(intf_lbs, K),
        diag_log_block_sizes=_lbs_for_2d(dfwd, K),
        diag_adjoint_log_block_sizes=_lbs_for_2d(dadj, K),
    )


def _svd_eval(K: int, n_rows: int, intf_lbs: int, dfwd: int, dadj: int) -> Rec:
    bloq = _make_svd_bloq(K, n_rows, intf_lbs, dfwd, dadj)
    return Rec(int(get_Toffoli_counts(bloq)), int(get_qubit_counts(bloq)),
               ("intf", (intf_lbs,), "diag_fwd", (dfwd,), "diag_adj", (dadj,)))


def _qroam_T_Q(qroam_bloq):
    return int(get_Toffoli_counts(qroam_bloq)), int(get_qubit_counts(qroam_bloq))


def svd_total(K: int, n: int) -> tuple[Rec, Rec]:
    """Optimize SVDBlockEncodingInterferometer, separable T (sum) and Q (max)."""
    n_rows = 1 << n
    intf_recs = [(lbs, *interferometer_cost(K, n_rows, lbs)) for lbs in SWEEP_LBS]
    fwd_recs = []
    adj_recs = []
    for lbs in SWEEP_LBS:
        try:
            fwd_recs.append((lbs, *_qroam_T_Q(_make_svd_bloq(K, n_rows, 0, lbs, 0).diag_qroam)))
        except Exception:
            pass
        try:
            adj_recs.append((lbs, *_qroam_T_Q(_make_svd_bloq(K, n_rows, 0, 0, lbs).diag_qroam_adjoint)))
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

    return (_svd_eval(K, n_rows, intf_T[0], fwd_T[0], adj_T[0]),
            _svd_eval(K, n_rows, intf_Q[0], fwd_Q[0], adj_Q[0]))


def dilation_total(K: int, n: int) -> tuple[Rec, Rec]:
    it, iq = optimize_interferometer(K, 1 << (n + 1))
    return Rec(it.toffoli, it.qubits, ("intf", it.lbs)), Rec(iq.toffoli, iq.qubits, ("intf", iq.lbs))


print("=" * 78)
print(f"SVD vs direct dilation, fixed N = {1 << N_FIXED}, K sweep = {K_VALUES}")
print("=" * 78)

svd_topt: list[Rec] = []
svd_qopt: list[Rec] = []
dil_topt: list[Rec] = []
dil_qopt: list[Rec] = []
for K in K_VALUES:
    print(f"K={K}")
    st, sq = svd_total(K, N_FIXED)
    dt, dq = dilation_total(K, N_FIXED)
    svd_topt.append(st); svd_qopt.append(sq)
    dil_topt.append(dt); dil_qopt.append(dq)
    print(f"  SVD  T-opt: T={st.toffoli:,} Q={st.qubits}; Q-opt: T={sq.toffoli:,} Q={sq.qubits}")
    print(f"  Dil  T-opt: T={dt.toffoli:,} Q={dt.qubits}; Q-opt: T={dq.toffoli:,} Q={dq.qubits}")

K_arr = np.asarray(K_VALUES, dtype=float)
series = {
    "svd_topt": svd_topt, "svd_qopt": svd_qopt,
    "dil_topt": dil_topt, "dil_qopt": dil_qopt,
}
arrs_t = {k: np.asarray([r.toffoli for r in v], dtype=float) for k, v in series.items()}
arrs_q = {k: np.asarray([r.qubits for r in v], dtype=float) for k, v in series.items()}


def fit_power(x: np.ndarray, y: np.ndarray):
    a, logc = np.polyfit(np.log(x), np.log(y), 1)
    return float(a), float(np.exp(logc))


fits = {}
for k in series:
    fits[k + "_toffoli"] = fit_power(K_arr, arrs_t[k])
    fits[k + "_qubits"] = fit_power(K_arr, arrs_q[k])
print("\nFits y = c * K^alpha:")
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
    xfit = np.linspace(K_arr.min(), K_arr.max(), 300)
    for key, (lbl, color, marker) in STYLES.items():
        data = arrs[key]
        a, c = fits[key + "_" + metric]
        ax.plot(K_arr, data, marker=marker, linestyle="-", color=color, linewidth=1.7, label=lbl)
        ax.plot(xfit, c * xfit ** a, ":", color=color, alpha=0.6)
        ax.text(K_arr[-1] * 1.04, data[-1], f"α={a:.2f}\nc={c:.1e}", color=color, fontsize=7, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K = number of blocks (= k^3)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([f"{K}\nk={k}" for K, k in zip(K_VALUES, K_K_VALUES)])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(f"{ylabel}: SVD vs direct dilation (N={1 << N_FIXED}, b={PHASE_BITSIZE})")
    fig.tight_layout()
    return fig


def table_page():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    cols = ["k", "K=k^3",
            "SVD T-opt T", "SVD T-opt Q", "SVD Q-opt T", "SVD Q-opt Q",
            "Dil T-opt T", "Dil T-opt Q", "Dil Q-opt T", "Dil Q-opt Q"]
    rows = []
    for i, (k, K) in enumerate(zip(K_K_VALUES, K_VALUES)):
        st, sq = svd_topt[i], svd_qopt[i]
        dt, dq = dil_topt[i], dil_qopt[i]
        rows.append([
            k, K,
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
    ax.set_title(f"Resource counts (fixed N={1 << N_FIXED}, b={PHASE_BITSIZE})", fontsize=12)
    return fig


def summary_page():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    lines = [
        f"Block-encoding K matrices A_k of fixed size {1 << N_FIXED} x {1 << N_FIXED} = 2^{N_FIXED} x 2^{N_FIXED}",
        "",
        f"K = k^3 for k in {K_K_VALUES} (so K in {K_VALUES}); phase_bitsize b = {PHASE_BITSIZE}.",
        "",
        "Approach 1 (SVD): A_k = U_k Sigma_k V_k.",
        "  Cost = 2 * BlockUnitaryInterferometerSynthesisQROAM(K, 2^n)",
        "         + QROAMClean((K, 2^n), b) + ctrl-AddIntoPhaseGrad + QROAMCleanAdjoint.",
        "",
        "Approach 2 (direct dilation): A_k completed to W_k of size 2^(n+1) x 2^(n+1).",
        "  Cost = BlockUnitaryInterferometerSynthesisQROAM(K, 2^(n+1)).",
        "",
        "Each cost is optimized over its log_block_size sweep.",
        "",
        "Two optimization modes per approach: Toffoli-optimal and qubit-optimal.",
        "",
        "Fits y = c * K^alpha:",
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
    info["Title"] = "SVD vs Dilation, fixed N, K sweep"
    info["Author"] = "qc_exciton_LCC"
print(f"\nWrote report: {OUT_PDF}")


def send_email(recipient: str, pdf_path: str) -> bool:
    subject = f"SVD vs dilation block-encoding, fixed N={1 << N_FIXED}, K sweep"
    body = textwrap.dedent(
        f"""\
        Hi,

        Attached: resource comparison for block-encoding K matrices A_k of fixed
        size {1 << N_FIXED} x {1 << N_FIXED}, with K = k^3 swept over k in {K_K_VALUES}.

        Approaches:
          (1) SVD-based: 2 x BlockUnitaryInterferometerSynthesisQROAM(K, {1 << N_FIXED})
              + QROAM diagonal layer ({1 << N_FIXED} * K entries of b bits).
          (2) Direct dilation: BlockUnitaryInterferometerSynthesisQROAM(K, {1 << (N_FIXED + 1)}).

        Fits y = c * K^alpha (Toffoli-opt and qubit-opt curves both included):
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
