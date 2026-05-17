#!/usr/bin/env python3
"""Generate a compact autonomous-cycle PDF report with rendered equations.

This is intentionally lightweight: matplotlib's built-in mathtext renderer
handles LaTeX-style equations, so no external TeX installation is required.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from collections.abc import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "docs", "cycle_reports")
PAGE_W, PAGE_H = 8.5, 11.0
DEFAULT_EQUATIONS = (
    r"$T(M, N, K, b) = K \left[2(\log_2 N + 1)b + I_1(M, N)\right]$",
    r"$Q(M, N, b) = \lceil \log_2 M \rceil + 1 + \log_2 N"
    r" + b + W(M, N, b)$",
    r"$T(M)/M \to 0 \quad"
    r" \mathrm{for\ sublinear\ QROAMClean\ amortization}$",
)


def _new_page(title: str):
    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_H))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=15, weight="bold")
    return fig, ax


def _write_wrapped(
    ax, y: float, text: str, *, width: int = 94, fontsize: float = 10.5
) -> float:
    for line in textwrap.wrap(text, width=width) or [""]:
        ax.text(0.06, y, line, ha="left", va="top", fontsize=fontsize)
        y -= 0.030
    return y


def _write_bullets(ax, y: float, title: str, items: Sequence[str]) -> float:
    ax.text(0.05, y, title, ha="left", va="top", fontsize=12, weight="bold")
    y -= 0.045
    for item in items:
        ax.text(0.07, y, "-", ha="left", va="top", fontsize=10.5)
        y = _write_wrapped(ax, y, item, width=88, fontsize=10.5)
        y -= 0.012
    return y


def _cover_page(args: argparse.Namespace):
    fig, ax = _new_page(f"Cycle {args.cycle} Autonomous Report")
    y = 0.90
    y = _write_wrapped(ax, y, f"Date: {args.date}")
    y = _write_wrapped(ax, y, f"Repository: {args.repository}")
    y -= 0.020
    y = _write_bullets(ax, y, "Task selected", [args.task])
    y = _write_bullets(ax, y, "Major changes", args.change)
    y = _write_bullets(ax, y, "Files changed", args.file)
    return fig


def _math_page(args: argparse.Namespace):
    fig, ax = _new_page("Rendered Equations")
    y = 0.89
    ax.text(
        0.05,
        y,
        "This page is a rendering check for the inbox requirement that "
        "equations appear as LaTeX-style math.",
        ha="left",
        va="top",
        fontsize=10.5,
    )
    equations = args.equation or DEFAULT_EQUATIONS
    y = 0.77
    for eq in equations:
        ax.text(0.09, y, eq, ha="left", va="top", fontsize=15)
        y -= 0.13
    y -= 0.02
    y = _write_bullets(ax, y, "Goals already achieved", args.achieved)
    return fig


def _verification_page(args: argparse.Namespace):
    fig, ax = _new_page("Verification")
    y = 0.89
    y = _write_bullets(ax, y, "Tests and checks", args.check)
    y = _write_bullets(ax, y, "Achieved goal", [args.goal])
    y = _write_bullets(ax, y, "Next recommended task", [args.next_task])
    return fig


def write_report(args: argparse.Namespace) -> str:
    out = args.output
    if out is None:
        out = os.path.join(DEFAULT_OUT_DIR, f"cycle{args.cycle}_report.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    with pdf_backend.PdfPages(out) as pdf:
        for make_page in (_cover_page, _math_page, _verification_page):
            fig = make_page(args)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = f"qc_exciton_LCC cycle {args.cycle} report"
        info["Author"] = "qc_exciton_LCC autonomous agent"
        info["Subject"] = "Autonomous coding cycle report"
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", required=True, type=int)
    parser.add_argument("--date", required=True)
    parser.add_argument("--repository", default="qc_exciton_LCC")
    parser.add_argument("--task", required=True)
    parser.add_argument("--change", action="append", required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--check", action="append", required=True)
    parser.add_argument("--achieved", action="append", required=True)
    parser.add_argument(
        "--equation",
        action="append",
        help=(
            "LaTeX-style mathtext equation to render on the report's math page. "
            "May be passed multiple times; defaults to synthesis resource formulas."
        ),
    )
    parser.add_argument("--goal", required=True)
    parser.add_argument("--next-task", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = write_report(args)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
