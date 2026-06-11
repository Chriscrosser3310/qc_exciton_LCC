#!/usr/bin/env python3
"""Per-component exact T-count breakdown of ExchangeCoulombBlockEncoding, emailed.

Parameters: N_up=4, N_down=22, N_IP=26*8=208, N_k=216, phase_bitsize=32, optimal_T.
B_up / B_down use short-direction (transposed) reflection synthesis, so
n_reflections = min(N_up, N_IP) = 4 and min(N_down, N_IP) = 22 respectively.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qc-exciton")

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from qualtran.resource_counting import get_cost_value, QECGatesCost
from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding

N_UP, N_DOWN, N_IP, N_K, B = 4, 22, 26 * 8, 216, 32
RECIPIENT = "jchen9@caltech.edu"
OUT_PDF = os.path.join(
    REPO_ROOT, "docs", f"exchange_tcount_breakdown_Nup{N_UP}_Ndown{N_DOWN}_NIP{N_IP}_Nk{N_K}_b{B}.pdf"
)


def costs(bloq):
    gc = get_cost_value(bloq, QECGatesCost())
    tc = gc.to_legacy_t_complexity()
    t = int(tc.t_incl_rotations(eps=1e-11))
    ccz = int(gc.total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"])
    return t, ccz


def make_pdf(be, rows, total_t, total_ccz, whole_t, whole_ccz, out_pdf):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with pdf_backend.PdfPages(out_pdf) as pdf:
        # Page 1: table
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.axis("off")
        col_labels = ["component", "n_refl", "mult", "T (each)", "CCZ (each)", "T (total)"]
        cell_text = []
        for name, nr, mult, t, ccz in rows:
            cell_text.append([name, nr, str(mult), f"{t:,}", f"{ccz:,}", f"{mult*t:,}"])
        cell_text.append(["TOTAL", "", "", "", f"{total_ccz:,} CCZ", f"{total_t:,}"])
        tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#25364a")
                cell.set_text_props(color="white", weight="bold")
            elif r == len(cell_text):  # TOTAL row
                cell.set_facecolor("#dde6f0")
                cell.set_text_props(weight="bold")
            elif r % 2:
                cell.set_facecolor("#f3f6fa")
        ax.set_title(
            "Exchange-Coulomb block encoding: per-component T-count\n"
            f"N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP} (=26*8), N_k={N_K}, b={B}, optimal_T",
            fontsize=12,
        )
        fig.text(0.5, 0.04,
                 f"Whole-bloq cross-check: T={whole_t:,}, CCZ={whole_ccz:,}  "
                 f"({'matches' if whole_t == total_t else 'MISMATCH'} component sum)",
                 ha="center", fontsize=9, style="italic")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: stacked / bar contribution of T (total) per component
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        names = [r[0].split()[0] for r in rows]
        tt = [r[2] * r[3] for r in rows]  # mult * T(each)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][: len(rows)]
        bars = ax.bar(names, tt, color=colors)
        ax.set_ylabel("T (total, incl. rotations)")
        ax.set_yscale("log")
        ax.set_title(f"T-count contribution by component (total = {total_t:,} T)")
        for b_, v in zip(bars, tt):
            pct = 100.0 * v / total_t
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:,}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        info = pdf.infodict()
        info["Title"] = "Exchange-Coulomb per-component T-count breakdown"
        info["Author"] = "qc_exciton_LCC"


def main():
    be = ExchangeCoulombBlockEncoding(
        N_up=N_UP, N_down=N_DOWN, N_IP=N_IP, N_k=N_K, phase_bitsize=B, optimal_T=True
    )
    components = [
        ("B_up   (rect. reflection BE)", be.B_up, 2, be.B_up.n_reflections_effective),
        ("B_down (rect. reflection BE)", be.B_down, 2, be.B_down.n_reflections_effective),
        ("C_inner (SVD interferometer BE)", be.C_inner, 1, None),
        ("mod_sub (Subtract on k)", be.mod_sub, 2, None),
        ("uniform_prep (Prepare uniform)", be.uniform_prep, 2, None),
    ]

    rows = []  # (name, n_refl_str, mult, t_each, ccz_each)
    total_t = 0
    total_ccz = 0
    for name, bloq, mult, n_refl in components:
        t, ccz = costs(bloq)
        total_t += mult * t
        total_ccz += mult * ccz
        rows.append((name, str(n_refl) if n_refl is not None else "-", mult, t, ccz))

    whole_t, whole_ccz = costs(be)

    lines = []
    lines.append("Exchange-Coulomb block encoding: exact T-count per component")
    lines.append("=" * 72)
    lines.append(f"Parameters: N_up={N_UP}, N_down={N_DOWN}, N_IP={N_IP} (=26*8), "
                 f"N_k={N_K}, phase_bitsize={B}, optimal_T=True")
    lines.append(f"Padded dims: n_rows_up={be.n_rows_up}, n_rows_down={be.n_rows_down}, "
                 f"n_rows_inner={be.n_rows_inner}, k_bitsize={be.k_bitsize}")
    lines.append("Full encoding = B . C . B^dagger  (each outer piece counted x2).")
    lines.append("B_up / B_down use short-direction (transposed) reflection synthesis:")
    lines.append("  n_reflections = min(N_up, N_IP) = 4  and  min(N_down, N_IP) = 22.")
    lines.append("")
    header = f"{'component':<34}{'n_refl':>7}{'mult':>6}{'T (each)':>16}{'CCZ (each)':>14}{'T (total)':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, nr, mult, t, ccz in rows:
        lines.append(f"{name:<34}{nr:>7}{mult:>6}{t:>16,}{ccz:>14,}{mult*t:>16,}")
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':<34}{'':>7}{'':>6}{'':>16}{'':>14}{total_t:>16,}")
    lines.append(f"{'TOTAL (CCZ / Toffoli-equiv)':<34}{'':>7}{'':>6}{'':>16}{'':>14}{total_ccz:>16,}")
    lines.append("")
    lines.append(f"Whole-bloq cross-check:  T={whole_t:,}   CCZ={whole_ccz:,}")
    lines.append(f"  matches component sum: T {'OK' if whole_t == total_t else 'MISMATCH'}, "
                 f"CCZ {'OK' if whole_ccz == total_ccz else 'MISMATCH'}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - T counts include rotation synthesis (Ross-Selinger, eps=1e-11).")
    lines.append("  - CCZ = Toffoli-equivalent count (n_ccz, ts_per_rotation=0).")
    lines.append("  - C_inner (the middle SVD block encoding) is the largest component.")

    body = "\n".join(lines)
    print(body)

    make_pdf(be, rows, total_t, total_ccz, whole_t, whole_ccz, OUT_PDF)
    print(f"\nWrote PDF: {OUT_PDF}")

    send_email(body, OUT_PDF)


def send_email(body, pdf_path):
    msg = MIMEMultipart()
    msg["From"] = "noreply@localhost"
    msg["To"] = RECIPIENT
    msg["Subject"] = "Exchange-Coulomb block encoding: per-component T-count breakdown"
    msg.attach(MIMEText(body, "plain"))
    with open(pdf_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(pdf_path)}"'
    msg.attach(part)

    try:
        with smtplib.SMTP("localhost", 25, timeout=10) as smtp:
            smtp.sendmail(msg["From"], [RECIPIENT], msg.as_string())
        print(f"Email sent via localhost:25 to {RECIPIENT}")
        return
    except Exception as exc:
        print(f"localhost:25 email failed: {exc}")
    try:
        proc = subprocess.run(
            ["/usr/sbin/sendmail", "-t", "-oi"],
            input=msg.as_string().encode(), capture_output=True, timeout=30,
        )
        if proc.returncode == 0:
            print(f"Email sent via /usr/sbin/sendmail to {RECIPIENT}")
        else:
            print(f"sendmail failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:300]}")
    except Exception as exc:
        print(f"sendmail email failed: {exc}")


if __name__ == "__main__":
    main()
