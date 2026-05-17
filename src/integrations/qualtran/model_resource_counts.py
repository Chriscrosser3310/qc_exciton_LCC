"""Theoretical resource-count models for Qualtran-adjacent algorithms.

The first model is a block-unitary interferometer synthesis estimate, based on
Sec. III.A of arXiv:2409.11748 with block-indexed QROM tables.
"""

from __future__ import annotations

import argparse
import math
import os
import smtplib
import subprocess
import textwrap
from dataclasses import dataclass
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Iterable, Sequence


QPE_SUBNORMALIZATION_FACTOR = 10**5


@dataclass(frozen=True)
class ResourceCount:
    """One Toffoli/qubit estimate with the selected QROM tradeoff parameters."""

    toffoli: int
    qubits: int
    lambda_1: int
    lambda_2: int
    log_lambda_1: int
    log_lambda_2: int


@dataclass(frozen=True)
class PreviousReportPoint:
    """Reference values from the previous generated report, kept read-only here."""

    k: int
    blocks: int
    interferometer_topt_toffoli: int
    interferometer_topt_qubits: int
    interferometer_qopt_toffoli: int
    interferometer_qopt_qubits: int
    standard_topt_toffoli: int
    standard_topt_qubits: int
    standard_qopt_toffoli: int
    standard_qopt_qubits: int


PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208: tuple[PreviousReportPoint, ...] = (
    PreviousReportPoint(1, 1, 54_408, 168, 67_574, 104, 363_376, 83, 363_376, 83),
    PreviousReportPoint(2, 8, 161_836, 299, 442_102, 107, 1_004_848, 183, 1_946_672, 89),
    PreviousReportPoint(3, 27, 289_748, 557, 1_458_678, 109, 2_178_384, 187, 6_094_192, 93),
    PreviousReportPoint(4, 64, 431_876, 1_070, 3_438_326, 110, 2_851_888, 571, 14_067_248, 95),
    PreviousReportPoint(5, 125, 635_860, 1_071, 6_702_070, 111, 3_905_200, 573, 27_229_072, 97),
    PreviousReportPoint(6, 216, 796_388, 2_096, 11_570_934, 112, 5_183_984, 575, 46_671_664, 99),
)


def ceil_log2(x: int) -> int:
    """Return ceil(log2(x)) for a positive integer."""

    if x <= 0:
        raise ValueError("x must be positive")
    return (x - 1).bit_length()


def assert_power_of_two(x: int, name: str) -> None:
    """Raise if ``x`` is not a positive power of two."""

    if x <= 0 or x != 1 << (x.bit_length() - 1):
        raise ValueError(f"{name} must be a positive power of two, got {x}")


def block_unitary_interferometer_toffoli(
    num_blocks: int,
    block_dim: int,
    bitsize: int,
    lambda_1: int,
    lambda_2: int,
) -> int:
    r"""Return the theoretical block-unitary interferometer Toffoli count.

    For ``K`` blocks, each ``N x N`` with ``N = 2^n``, this implements

    ``N(ceil(NK/(2 lambda_1)) + 2 lambda_1 b - 5)``
    ``+ (n-2)(N-1)``
    ``+ ceil(NK/lambda_1) + lambda_1 b``
    ``+ ceil(NK/lambda_2) + lambda_2 - 6``.

    The expression follows Sec. III.A of arXiv:2409.11748: the repeated
    interferometer layers use one QROM table per paired phase layer, and the
    final diagonal phase layer uses a QROM load plus an erasure term. The
    block modification replaces each table length ``N`` by ``N*K`` while
    preserving the common target-register interferometer skeleton.
    """

    assert_power_of_two(block_dim, "block_dim")
    if min(num_blocks, bitsize, lambda_1, lambda_2) <= 0:
        raise ValueError("num_blocks, bitsize, lambda_1, and lambda_2 must be positive")
    n = int(math.log2(block_dim))
    table_len = block_dim * num_blocks
    layer_cost = math.ceil(table_len / (2 * lambda_1)) + 2 * lambda_1 * bitsize - 5
    shift_cost = max(0, n - 2) * (block_dim - 1)
    final_cost = (
        math.ceil(table_len / lambda_1)
        + lambda_1 * bitsize
        + math.ceil(table_len / lambda_2)
        + lambda_2
        - 6
    )
    return block_dim * layer_cost + shift_cost + final_cost


def block_unitary_interferometer_qubits(
    num_blocks: int,
    block_dim: int,
    bitsize: int,
    lambda_1: int,
    lambda_2: int,
) -> int:
    """Return a peak logical qubit estimate for the theoretical model.

    This counts the block register, target register, phase-gradient register,
    and the larger of the QROM workspaces in the requested formula. The largest
    layer table output stores two ``b``-bit angles, so its workspace is modeled
    as ``2*b*lambda_1``. The final erasure workspace is modeled as ``lambda_2``.
    """

    assert_power_of_two(block_dim, "block_dim")
    if min(num_blocks, bitsize, lambda_1, lambda_2) <= 0:
        raise ValueError("num_blocks, bitsize, lambda_1, and lambda_2 must be positive")
    base = ceil_log2(num_blocks) + int(math.log2(block_dim)) + bitsize
    workspace = max(2 * bitsize * lambda_1, bitsize * lambda_1, lambda_2)
    return base + workspace


def block_unitary_interferometer_count(
    num_blocks: int,
    block_dim: int,
    bitsize: int,
    lambda_1: int,
    lambda_2: int,
) -> ResourceCount:
    """Return both Toffoli and qubit counts for the theoretical model."""

    if lambda_1 != 1 << (lambda_1.bit_length() - 1):
        raise ValueError("lambda_1 must be a power of two")
    if lambda_2 != 1 << (lambda_2.bit_length() - 1):
        raise ValueError("lambda_2 must be a power of two")
    return ResourceCount(
        toffoli=block_unitary_interferometer_toffoli(
            num_blocks, block_dim, bitsize, lambda_1, lambda_2
        ),
        qubits=block_unitary_interferometer_qubits(
            num_blocks, block_dim, bitsize, lambda_1, lambda_2
        ),
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        log_lambda_1=int(math.log2(lambda_1)),
        log_lambda_2=int(math.log2(lambda_2)),
    )


def optimize_block_unitary_interferometer(
    num_blocks: int,
    block_dim: int,
    bitsize: int,
    *,
    max_log_lambda: int | None = None,
    objective: str = "toffoli",
) -> ResourceCount:
    """Optimize ``lambda_1`` and ``lambda_2`` over powers of two."""

    if objective not in {"toffoli", "qubits"}:
        raise ValueError("objective must be 'toffoli' or 'qubits'")
    if max_log_lambda is None:
        max_log_lambda = max(1, ceil_log2(block_dim * num_blocks))
    candidates = [
        block_unitary_interferometer_count(num_blocks, block_dim, bitsize, 2**l1, 2**l2)
        for l1 in range(max_log_lambda + 1)
        for l2 in range(max_log_lambda + 1)
    ]
    if objective == "toffoli":
        return min(candidates, key=lambda c: (c.toffoli, c.qubits))
    return min(candidates, key=lambda c: (c.qubits, c.toffoli))


def power_law_fit(xs: Sequence[int], ys: Sequence[int]) -> tuple[float, float]:
    """Fit ``y = c*x^alpha`` in log-log space."""

    import numpy as np

    alpha, log_c = np.polyfit(np.log(np.asarray(xs, dtype=float)), np.log(ys), 1)
    return float(alpha), float(np.exp(log_c))


def _plot_report(
    *,
    block_dim: int,
    bitsize: int,
    k_values: Sequence[int],
    out_pdf: str,
) -> tuple[list[ResourceCount], list[ResourceCount]]:
    """Generate a PDF report and return theoretical T/Q optimized records."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.backends.backend_pdf as pdf_backend
    import matplotlib.pyplot as plt
    import numpy as np

    blocks = [k**3 for k in k_values]
    t_opt = [
        optimize_block_unitary_interferometer(k**3, block_dim, bitsize, objective="toffoli")
        for k in k_values
    ]
    q_opt = [
        optimize_block_unitary_interferometer(k**3, block_dim, bitsize, objective="qubits")
        for k in k_values
    ]

    fits = {
        "theory_topt_t": power_law_fit(blocks, [r.toffoli for r in t_opt]),
        "theory_qopt_t": power_law_fit(blocks, [r.toffoli for r in q_opt]),
        "theory_topt_q": power_law_fit(blocks, [r.qubits for r in t_opt]),
        "theory_qopt_q": power_law_fit(blocks, [r.qubits for r in q_opt]),
    }

    prev_blocks = [p.blocks for p in PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208]
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    def setup_log_axis(ax, ylabel: str, title: str):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("K = k^3 blocks")
        ax.set_ylabel(ylabel)
        ax.set_xticks(blocks)
        ax.set_xticklabels([f"{b}\nk={k}" for b, k in zip(blocks, k_values)])
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title)

    def plot_count(metric: str, scale: int = 1):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        if metric == "toffoli":
            y_t = [scale * r.toffoli for r in t_opt]
            y_q = [scale * r.toffoli for r in q_opt]
            prev_t = [
                scale * p.interferometer_topt_toffoli
                for p in PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208
            ]
            prev_q = [
                scale * p.interferometer_qopt_toffoli
                for p in PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208
            ]
            ylabel = "Toffoli count" if scale == 1 else "Toffoli count x 1e5"
        else:
            y_t = [r.qubits for r in t_opt]
            y_q = [r.qubits for r in q_opt]
            prev_t = [
                p.interferometer_topt_qubits
                for p in PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208
            ]
            prev_q = [
                p.interferometer_qopt_qubits
                for p in PREVIOUS_MATCHED_UNCOMPUTE_REPORT_ROWS256_COLS208
            ]
            ylabel = "Peak logical qubits"
        ax.plot(blocks, y_t, "o-", label="theory equation, Toffoli-opt")
        ax.plot(blocks, y_q, "s-", label="theory equation, qubit-opt")
        ax.plot(prev_blocks, prev_t, "^--", label="previous report, interferometer Toffoli-opt")
        ax.plot(prev_blocks, prev_q, "D--", label="previous report, interferometer qubit-opt")
        suffix = " multiplied by QPE iteration and subnormalization" if scale != 1 else ""
        setup_log_axis(
            ax,
            ylabel,
            f"{ylabel} comparison (N={block_dim}, b={bitsize}){suffix}",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def plot_lambdas():
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.plot(blocks, [r.lambda_1 for r in t_opt], "o-", label="lambda_1, Toffoli-opt")
        ax.plot(blocks, [r.lambda_2 for r in t_opt], "s-", label="lambda_2, Toffoli-opt")
        ax.plot(blocks, [r.lambda_1 for r in q_opt], "^--", label="lambda_1, qubit-opt")
        ax.plot(blocks, [r.lambda_2 for r in q_opt], "D--", label="lambda_2, qubit-opt")
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_xlabel("K = k^3 blocks")
        ax.set_ylabel("lambda")
        ax.set_xticks(blocks)
        ax.set_xticklabels([f"{b}\nk={k}" for b, k in zip(blocks, k_values)])
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title("Optimizing QROM tradeoff parameters")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def table_page():
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax.axis("off")
        rows = []
        for k, blocks_i, t_rec, q_rec in zip(k_values, blocks, t_opt, q_opt):
            rows.append(
                [
                    k,
                    blocks_i,
                    f"{t_rec.toffoli:,}",
                    t_rec.qubits,
                    t_rec.lambda_1,
                    t_rec.lambda_2,
                    f"{q_rec.toffoli:,}",
                    q_rec.qubits,
                    q_rec.lambda_1,
                    q_rec.lambda_2,
                ]
            )
        tbl = ax.table(
            cellText=rows,
            colLabels=[
                "k",
                "K",
                "T-opt T",
                "T-opt Q",
                "T-opt lambda1",
                "T-opt lambda2",
                "Q-opt T",
                "Q-opt Q",
                "Q-opt lambda1",
                "Q-opt lambda2",
            ],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.55)
        ax.set_title("Theoretical block-unitary interferometer choices")
        return fig

    def summary_page():
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.axis("off")
        lines = [
            "Theoretical model: block-unitary interferometer synthesis",
            "",
            "Reference: arXiv:2409.11748 Sec. III.A, with QROM table length N*K.",
            "Verified expression:",
            "  T = N(ceil(NK/(2 lambda1)) + 2 lambda1 b - 5)",
            "      + (log2(N)-2)(N-1)",
            "      + ceil(NK/lambda1) + lambda1 b",
            "      + ceil(NK/lambda2) + lambda2 - 6.",
            "",
            f"Parameters plotted: N={block_dim}, b={bitsize}, K=k^3 for k={list(k_values)}.",
            "Qubit model: ceil(log2 K) + log2 N + b + max(2 b lambda1, lambda2).",
            "",
            "Power-law fits y = c*K^alpha:",
        ]
        for label, (alpha, coeff) in fits.items():
            lines.append(f"  {label}: alpha={alpha:.3f}, c={coeff:.3e}")
        lines.append("")
        lines.append("The comparison curves marked previous report are read-only constants from")
        lines.append("the earlier rows=256, cols=208 matched-uncompute interferometer report.")
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.2)
        fig.tight_layout()
        return fig

    with pdf_backend.PdfPages(out_pdf) as pdf:
        for make in (
            summary_page,
            plot_lambdas,
            lambda: plot_count("toffoli"),
            lambda: plot_count("qubits"),
            lambda: plot_count("toffoli", scale=QPE_SUBNORMALIZATION_FACTOR),
            table_page,
        ):
            fig = make()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "Theoretical Block-Unitary Interferometer Resource Counts"
        info["Author"] = "qc_exciton_LCC"

    return t_opt, q_opt


def email_report(recipient: str, report_path: str, *, note_path: str | None = None) -> bool:
    """Send ``report_path`` using the local cluster mail path."""

    subject = "Theoretical block-unitary interferometer resource counts"
    body = textwrap.dedent(
        """\
        Hi,

        Attached is the theoretical model-resource-count report for block-unitary
        interferometer synthesis. It includes lambda_1/lambda_2 choices that
        optimize Toffoli or qubits, compares against the previous generated
        report curves, and includes Toffoli-only 1e5 scaled plots.

        """
    )
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    for path in [p for p in (report_path, note_path) if p]:
        with open(path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(path))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
        msg.attach(part)

    try:
        with smtplib.SMTP("localhost", 25, timeout=10) as smtp:
            smtp.sendmail(msg["From"], [recipient], msg.as_string())
        print(f"Email sent via localhost:25 to {recipient}")
        return True
    except Exception as exc:
        print(f"localhost:25 email failed: {exc}")

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
    return False


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--bitsize", type=int, default=32)
    parser.add_argument("--recipient", default="jchen9@caltech.edu")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.getcwd(), "docs", "model_resource_counts_block_unitary_interferometer.pdf"
        ),
    )
    parser.add_argument("--email", action="store_true")
    parser.add_argument(
        "--note",
        default=os.path.join(
            os.getcwd(),
            "notes",
            "block_unitary_interferometer_synthesis_self_contained_with_givens.pdf",
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    t_opt, q_opt = _plot_report(
        block_dim=args.block_dim,
        bitsize=args.bitsize,
        k_values=range(1, 7),
        out_pdf=args.out,
    )
    print(f"Wrote report: {args.out}")
    for k, t_rec, q_rec in zip(range(1, 7), t_opt, q_opt):
        print(
            f"k={k}, K={k**3}: "
            f"T-opt T={t_rec.toffoli:,}, Q={t_rec.qubits}, "
            f"lambda1={t_rec.lambda_1}, lambda2={t_rec.lambda_2}; "
            f"Q-opt T={q_rec.toffoli:,}, Q={q_rec.qubits}, "
            f"lambda1={q_rec.lambda_1}, lambda2={q_rec.lambda_2}"
        )
    if args.email:
        note_path = args.note if os.path.exists(args.note) else None
        return 0 if email_report(args.recipient, args.out, note_path=note_path) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
