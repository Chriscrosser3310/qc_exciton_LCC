#!/usr/bin/env python3
"""Generate the Cycle 16 PDF report (math rendered via matplotlib mathtext).

Per the cycle-16 inbox instruction "send me a pdf report each time, with
math equations rendered as latex". Mathtext (no external LaTeX) is used
so the report regenerates in any environment with matplotlib.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PDF = os.path.join(REPO_ROOT, "docs", "cycle_reports", "cycle16_report.pdf")
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

PAGE_W, PAGE_H = 8.5, 11.0  # US letter, portrait, in inches


def _begin_page(title: str):
    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_H))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=15, weight="bold")
    return fig, ax


def _line(ax, y, text, *, fontsize=10.5, family=None, math=True):
    ax.text(0.05, y, text, ha="left", va="top", fontsize=fontsize, family=family)


def cover_page():
    today = _dt.date.today().isoformat()
    fig, ax = _begin_page("Cycle 16 — Autonomous Agent Report")
    y = 0.90
    for line in [
        f"Date: {today}",
        "Branch: agent-auto",
        "Repository: qc_exciton_LCC",
        "",
        "Cycle goal:",
        "  Add a Bloq-driven helper that regenerates the two tabulated dicts the",
        "  analytic synthesis estimator depends on, so re-tabulation under a QROAMClean",
        "  upstream change becomes a one-command operation.",
        "",
        "Selected deliverable:",
        r"  - scripts/regenerate_synthesis_tables.py — extracts $I_1$ and $W$ from the bloq",
        "  - tests/test_regenerate_synthesis_tables.py — corner-sample regression + roundtrip",
        "",
        "Inbox compliance:",
        "  User instruction (2026-05-16): 'send me a pdf report each time, with math",
        "  equations rendered as latex'. This PDF satisfies that — all math below is",
        "  rendered via matplotlib mathtext (LaTeX-style) so the report regenerates in",
        "  any environment with matplotlib (no external LaTeX install required).",
    ]:
        _line(ax, y, line)
        y -= 0.030
    return fig


def math_page():
    fig, ax = _begin_page("Closed-form decomposition (recap)")
    blocks = [
        (0.91, "Toffoli decomposition pinned by tests/test_block_unitary_synthesis_scaling.py:"),
        (0.86, r"$T(M, N, K, b) \, = \, K \cdot ( \, 2 (\log_2 N + 1) \, b \; + \; I_1(M, N) \, )$"),
        (0.78, r"where $M$ = n_blocks, $N$ = n_rows, $K$ = n_reflections, $b$ = phase_bitsize."),
        (0.73, r"$I_1(M, N)$ is the per-reflection $b=0$ intercept tabulated as"),
        (0.68, r"$\mathrm{SYNTHESIS\_PER\_REFLECTION\_INTERCEPT}$ in model_resource_counts.py."),
        (0.61, "Qubit decomposition pinned by tests/test_block_unitary_synthesis_qubit_count.py:"),
        (0.56, r"$Q(M, N, b) \, = \, [\, \lceil \log_2 M \rceil + 1 + \log_2 N + b \,]_{\mathrm{sig}} \; + \; W(M, N, b)$"),
        (0.48, r"where $W$ is the transient QROAMClean workspace tabulated as"),
        (0.43, r"$\mathrm{SYNTHESIS\_WORKSPACE\_QUBITS}$."),
        (0.36, "Sub-linear amortization (cycle 15):"),
        (0.31, r"$T(M)/M \, \to \, 0, \qquad T(M) \sim c \, M^{\alpha}, \quad \alpha < 1.$"),
        (0.23, r"Empirically (N=256, b=32, K=256): $\alpha_T \approx 0.275$, $\alpha_Q \approx 0.024$."),
        (0.18, "This cycle does NOT change those identities; it adds a one-command"),
        (0.14, "re-tabulation helper that produces both dicts directly from the Bloq."),
    ]
    for y, text in blocks:
        ax.text(0.05, y, text, ha="left", va="top", fontsize=11.5)
    return fig


def helper_page():
    fig, ax = _begin_page("Helper: scripts/regenerate_synthesis_tables.py")
    blocks = [
        (0.92, "Grid:"),
        (0.88, r"$M \in \{1, 2, 4, 8, 16, 32, 64\}, \quad N \in \{4, 8, 16, 32, 64, 128, 256\}$"),
        (0.83, r"$b \in \{2, 4, 8, 16, 32\}$  (workspace only)"),
        (0.76, "Intercept extraction:"),
        (0.72, r"$I_1(M, N) \, = \, T_{\mathrm{bloq}}(M, N, K{=}1, b{=}b_{\mathrm{ref}}) \, - \, 2(\log_2 N + 1) \, b_{\mathrm{ref}}$"),
        (0.65, r"with $b_{\mathrm{ref}} = 4$.  Uses $\mathrm{QECGatesCost}$ on $\mathrm{BlockUnitarySynthesisQROAM.from\_shape}$."),
        (0.58, "Workspace extraction:"),
        (0.54, r"$W(M, N, b) \, = \, \mathrm{QubitCount}(\mathrm{bloq}) \, - \, |\mathrm{signature}|$"),
        (0.47, "Modes:"),
        (0.43, "  --check  (default)  diff regenerated dicts against the shipped tables"),
        (0.39, "  --print              emit Python source ready to paste back into the module"),
        (0.35, "  --no-workspace       skip the slower 245-point workspace sweep"),
        (0.28, "Verification run (this cycle):"),
        (0.24, "  SYNTHESIS_PER_REFLECTION_INTERCEPT: OK (49 entries match)"),
        (0.20, "  SYNTHESIS_WORKSPACE_QUBITS: OK (245 entries match)"),
    ]
    for y, text in blocks:
        ax.text(0.05, y, text, ha="left", va="top", fontsize=11.5)
    return fig


def tests_page():
    fig, ax = _begin_page("Tests & checks")
    rows = [
        ("tests/test_regenerate_synthesis_tables.py", "4/4 passed (new)"),
        ("tests/test_block_unitary_synthesis_b_intercept.py", "6/6 passed (unaffected)"),
        ("tests/test_model_resource_counts.py", "31/31 passed (unaffected)"),
        ("scripts/regenerate_synthesis_tables.py --check (full sweep)", "245 + 49 entries match"),
    ]
    ax.text(0.05, 0.92, "Targeted test runs this cycle:", fontsize=12, weight="bold")
    y = 0.86
    for name, status in rows:
        ax.text(0.07, y, name, fontsize=10, family="monospace")
        ax.text(0.92, y, status, fontsize=10, ha="right")
        y -= 0.045

    ax.text(0.05, 0.55, "Achieved goal:", fontsize=12, weight="bold")
    achieved = [
        "GOALS.md 'Improve constant factors in quantum algorithms' /",
        "'Any potential improvements on the final Toffoli complexity / qubit counts / scaling':",
        "the two tabulated dicts that back the analytic synthesis estimator are now",
        "regenerable from the Bloq with a single command, closing the cycle-12+ recommendation",
        "that re-tabulation under a QROAMClean upstream change should not require a manual edit.",
    ]
    y = 0.49
    for line in achieved:
        ax.text(0.05, y, line, fontsize=10.5)
        y -= 0.030

    ax.text(0.05, 0.30, "Next recommended task:", fontsize=12, weight="bold")
    nxt = [
        r"Derive $I_1(M, N)$ in closed form from QROAMClean's optimizer. The empirical",
        r"asymptotic $I_1(M, N) \sim c \sqrt{M N}$ (consistent with the cycle-15",
        r"$\alpha_T \approx 0.275$ at $N=256$) and the $n_{\mathrm{blocks}} / n_{\mathrm{rows}}$",
        "monotonicity invariants are the anchors a candidate closed form must satisfy.",
        "The script added this cycle provides a one-command regression check against",
        "the bloq for any candidate closed form.",
    ]
    y = 0.25
    for line in nxt:
        ax.text(0.05, y, line, fontsize=10.5)
        y -= 0.030
    return fig


def main() -> int:
    pages = [cover_page, math_page, helper_page, tests_page]
    with pdf_backend.PdfPages(OUT_PDF) as pdf:
        for make in pages:
            fig = make()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "qc_exciton_LCC cycle 16 report"
        info["Author"] = "qc_exciton_LCC autonomous agent"
        info["Subject"] = "Autonomous cycle 16 deliverables"
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
