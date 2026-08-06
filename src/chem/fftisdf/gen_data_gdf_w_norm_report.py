#!/usr/bin/env python3
import argparse
import csv
import os
import re
import smtplib
import subprocess
import sys
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_DATA_DIR = Path("/resnick/groups/changroup/members/jsun3/xprize/fit_THC/data_GDF")
DEFAULT_EMAIL = "jchen9@caltech.edu"


def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(part) if part.isdigit() else part for part in parts]


def w_norms(chk_path):
    with h5py.File(chk_path, "r") as h5:
        if "coul_kpt" not in h5:
            raise KeyError("missing dataset coul_kpt")
        w = h5["coul_kpt"]
        if len(w.shape) != 3:
            raise ValueError(f"expected coul_kpt rank 3, got shape {w.shape}")

        nq = w.shape[0]
        max_abs = 0.0
        max_2norm = 0.0
        max_frob = 0.0

        for q in range(nq):
            wq = np.asarray(w[q])
            max_abs = max(max_abs, float(np.abs(wq).max()))
            max_2norm = max(max_2norm, float(np.linalg.svdvals(wq).max()))
            max_frob = max(max_frob, float(np.linalg.norm(wq)))

        return {
            "shape": tuple(int(x) for x in w.shape),
            "W_max": max_abs,
            "W_2norm": max_2norm,
            "W_frob": max_frob,
            "W_frob_over_2norm": max_frob / max_2norm if max_2norm != 0.0 else np.nan,
        }


def collect_rows(data_dir):
    paths = sorted(data_dir.glob("ISDF*.chk"), key=natural_key)
    rows = []
    for i, path in enumerate(paths, 1):
        print(f"[{i:4d}/{len(paths):4d}] {path.name}", flush=True)
        row = {"file": path.name, "path": str(path)}
        try:
            row.update(w_norms(path))
            row["status"] = "ok"
            print(
                "    "
                f"shape={row['shape']} "
                f"W_max={row['W_max']:.1f} "
                f"W_2norm={row['W_2norm']:.1f} "
                f"W_frob={row['W_frob']:.1f} "
                f"F/2={row['W_frob_over_2norm']:.1f}",
                flush=True,
            )
        except Exception as err:
            row.update({
                "shape": "",
                "W_max": np.nan,
                "W_2norm": np.nan,
                "W_frob": np.nan,
                "W_frob_over_2norm": np.nan,
            })
            row["status"] = f"error: {err}"
            print(f"    ERROR: {err}", flush=True)
        rows.append(row)
    return rows


def write_csv(rows, csv_path):
    fields = ["file", "shape", "W_max", "W_2norm", "W_frob", "W_frob_over_2norm", "status", "path"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            for field in ["W_max", "W_2norm", "W_frob", "W_frob_over_2norm"]:
                out[field] = fmt_value(out[field])
            writer.writerow(out)


def fmt_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.1f}"
    return str(value)


def add_table_page(pdf, rows, title, start, per_page=24):
    page_rows = rows[start : start + per_page]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=18)

    col_labels = [
        "file",
        "coul_kpt shape",
        "max |W[q]|",
        "max ||W[q]||2",
        "max ||W[q]||F",
        "F/2",
        "status",
    ]
    table_rows = [
        [
            row["file"],
            str(row.get("shape", "")),
            fmt_value(row.get("W_max", "")),
            fmt_value(row.get("W_2norm", "")),
            fmt_value(row.get("W_frob", "")),
            fmt_value(row.get("W_frob_over_2norm", "")),
            row.get("status", ""),
        ]
        for row in page_rows
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.39, 0.13, 0.10, 0.11, 0.11, 0.06, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.0)
    table.scale(1.0, 1.25)

    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9eef7")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f6f6")

    page_no = start // per_page + 1
    page_count = (len(rows) + per_page - 1) // per_page
    fig.text(0.5, 0.02, f"Page {page_no} / {page_count}", ha="center", fontsize=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_summary_page(pdf, rows, data_dir):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    title = "ISDF data_GDF W Norm Report"
    ax.set_title(title, fontsize=16, pad=18)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = [
        f"Data directory: {data_dir}",
        f"Generated: {generated}",
        f"Files scanned: {len(rows)}",
        f"Successful files: {len(ok_rows)}",
        f"Failed files: {len(rows) - len(ok_rows)}",
        "",
        "Definitions:",
        "W_max   = max_q max_ij |W[q, i, j]|",
        "W_2norm = max_q ||W[q]||_2",
        "W_frob  = max_q ||W[q]||_F",
        "F/2     = W_frob / W_2norm",
    ]
    ax.text(0.05, 0.88, "\n".join(text), va="top", ha="left", fontsize=11, family="monospace")

    if ok_rows:
        top_2 = sorted(ok_rows, key=lambda row: row["W_2norm"], reverse=True)[:10]
        top_f = sorted(ok_rows, key=lambda row: row["W_frob"], reverse=True)[:10]
        summary = ["Top 10 by W_2norm:"]
        summary += [f"{row['W_2norm']:.1f}  {row['file']}" for row in top_2]
        summary += ["", "Top 10 by W_frob:"]
        summary += [f"{row['W_frob']:.1f}  {row['file']}" for row in top_f]
        ax.text(0.05, 0.50, "\n".join(summary), va="top", ha="left", fontsize=8, family="monospace")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf(rows, pdf_path, data_dir):
    with PdfPages(pdf_path) as pdf:
        add_summary_page(pdf, rows, data_dir)
        for start in range(0, len(rows), 24):
            add_table_page(pdf, rows, "ISDF W Norms: max over q/k-points", start)


def send_email_smtp(pdf_path, csv_path, recipient):
    msg = MIMEMultipart()
    msg["From"] = recipient
    msg["To"] = recipient
    msg["Subject"] = "ISDF data_GDF W norm report"
    msg.attach(
        MIMEText(
            "Attached are the PDF and CSV reports for max-over-q W norms "
            "for ISDF*.chk files in fit_THC/data_GDF.\n",
            "plain",
        )
    )

    for path in [pdf_path, csv_path]:
        with open(path, "rb") as f:
            part = MIMEApplication(f.read(), Name=Path(path).name)
        part["Content-Disposition"] = f'attachment; filename="{Path(path).name}"'
        msg.attach(part)

    with smtplib.SMTP("mail.caltech.edu", 25, timeout=30) as server:
        server.sendmail(recipient, recipient, msg.as_string())


def send_email_mail(pdf_path, csv_path, recipient):
    body = (
        "Attached are the PDF and CSV reports for max-over-q W norms "
        "for ISDF*.chk files in fit_THC/data_GDF.\n"
    )
    cmd = [
        "mail",
        "-s",
        "ISDF data_GDF W norm report",
        "-a",
        str(pdf_path),
        "-a",
        str(csv_path),
        recipient,
    ]
    return subprocess.run(cmd, input=body, text=True, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pdf", type=Path, default=Path("isdf_data_gdf_w_norm_report.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("isdf_data_gdf_w_norm_report.csv"))
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()

    rows = collect_rows(args.data_dir)
    write_csv(rows, args.csv)
    write_pdf(rows, args.pdf, args.data_dir)

    print(f"CSV written: {args.csv}", flush=True)
    print(f"PDF written: {args.pdf}", flush=True)

    if args.no_email:
        return 0

    try:
        send_email_smtp(args.pdf, args.csv, args.email)
        print(f"Email sent via SMTP to {args.email}", flush=True)
    except Exception as smtp_err:
        print(f"SMTP email failed: {smtp_err}", flush=True)
        res = send_email_mail(args.pdf, args.csv, args.email)
        print(f"mail exit={res.returncode}", flush=True)
        if res.returncode != 0:
            return res.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
