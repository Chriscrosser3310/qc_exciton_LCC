"""Smoke tests for the reusable autonomous-cycle PDF report helper."""

from __future__ import annotations

import os
import sys
import tempfile

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import autonomous_cycle_report as acr  # noqa: E402


def test_cycle_report_writes_nonempty_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cycle_report.pdf")
        rc = acr.main(
            [
                "--cycle",
                "999",
                "--date",
                "2026-05-16",
                "--task",
                "Smoke-test the reusable report generator.",
                "--change",
                "Generated a PDF with mathtext-rendered equations.",
                "--file",
                "docs/cycle_reports/cycle999_report.pdf",
                "--check",
                "PDF file exists and is non-empty.",
                "--achieved",
                "Improve constant-factor reporting infrastructure.",
                "--equation",
                r"$\alpha_T < 1$",
                "--equation",
                r"$I_1(M,N) = T(M,N,1,0)$",
                "--goal",
                "Reusable report generation keeps per-cycle PDF delivery reviewable.",
                "--next-task",
                "Use the helper for future autonomous cycles.",
                "--output",
                out,
            ]
        )
        assert rc == 0
        assert os.path.getsize(out) > 1000


if __name__ == "__main__":
    failed = 0
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
