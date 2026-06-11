"""Smoke test for ``scripts/regenerate_synthesis_tables.py``.

The full regeneration sweep takes a few minutes (it instantiates 245
QROAMClean-backed bloqs and runs ``QubitCount`` on each), so this test
samples a small corner of the grid and verifies:

  * ``extract_intercept(n_blocks, n_rows)`` agrees with the shipped
    ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` lookup;
  * ``extract_workspace(n_blocks, n_rows, bitsize)`` agrees with the
    shipped ``SYNTHESIS_WORKSPACE_QUBITS`` lookup;
  * the formatters round-trip back to Python source that imports cleanly
    and reproduces the input dict.

Together with
``tests/test_block_unitary_reflection_b_intercept.py::test_reference_table_matches_module_table``
(which independently re-derives the intercept table from the bloq) the
script's extract functions are guarded against silent drift.
"""

from __future__ import annotations

import ast
import os
import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for envs without pytest

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__}, got nothing")
            return issubclass(exc_type, self.exc)

    class _PytestShim:
        @staticmethod
        def raises(exc):
            return _Raises(exc)

        @staticmethod
        def importorskip(name):
            return __import__(name)

    pytest = _PytestShim()  # type: ignore[assignment]
    sys.modules["pytest"] = pytest  # type: ignore[assignment]


qualtran = pytest.importorskip("qualtran")
_ = qualtran

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import regenerate_synthesis_tables as rst  # noqa: E402

from integrations.qualtran.model_resource_counts import (  # noqa: E402
    SYNTHESIS_PER_REFLECTION_INTERCEPT,
    SYNTHESIS_WORKSPACE_QUBITS,
)


SAMPLE_INTERCEPT_KEYS = [
    (1, 4),
    (1, 16),
    (4, 8),
    (16, 32),
]

SAMPLE_WORKSPACE_KEYS = [
    (1, 4, 4),
    (4, 16, 8),
    (16, 32, 4),
]


def test_extract_intercept_matches_shipped():
    for (n_blocks, n_rows) in SAMPLE_INTERCEPT_KEYS:
        got = rst.extract_intercept(n_blocks, n_rows)
        expected = SYNTHESIS_PER_REFLECTION_INTERCEPT[(n_blocks, n_rows)]
        assert got == expected, (n_blocks, n_rows, got, expected)


def test_extract_workspace_matches_shipped():
    for (n_blocks, n_rows, bitsize) in SAMPLE_WORKSPACE_KEYS:
        got = rst.extract_workspace(n_blocks, n_rows, bitsize)
        expected = SYNTHESIS_WORKSPACE_QUBITS[(n_blocks, n_rows, bitsize)]
        assert got == expected, (n_blocks, n_rows, bitsize, got, expected)


def test_intercept_formatter_roundtrip():
    """``_format_intercept_table`` emits a parseable ``{...}`` literal."""
    sample = {
        (1, 4): SYNTHESIS_PER_REFLECTION_INTERCEPT[(1, 4)],
        (1, 8): SYNTHESIS_PER_REFLECTION_INTERCEPT[(1, 8)],
        (2, 4): SYNTHESIS_PER_REFLECTION_INTERCEPT[(2, 4)],
        (2, 8): SYNTHESIS_PER_REFLECTION_INTERCEPT[(2, 8)],
    }
    src = rst._format_intercept_table(sample, (1, 2), (4, 8))
    body = src.split("=", 1)[1].strip()
    parsed = ast.literal_eval(body)
    assert parsed == sample


def test_workspace_formatter_roundtrip():
    sample = {
        (1, 4, 4): SYNTHESIS_WORKSPACE_QUBITS[(1, 4, 4)],
        (1, 8, 4): SYNTHESIS_WORKSPACE_QUBITS[(1, 8, 4)],
        (2, 4, 4): SYNTHESIS_WORKSPACE_QUBITS[(2, 4, 4)],
        (2, 8, 4): SYNTHESIS_WORKSPACE_QUBITS[(2, 8, 4)],
    }
    src = rst._format_workspace_table(sample, (1, 2), (4, 8), (4,))
    body = src.split("=", 1)[1].strip()
    parsed = ast.literal_eval(body)
    assert parsed == sample


def test_power_law_fit_recovers_quadratic():
    alpha, coeff = rst._power_law_fit((1, 2, 4, 8), (3, 12, 48, 192))
    assert abs(alpha - 2.0) < 1e-12
    assert abs(coeff - 3.0) < 1e-12


def test_scaling_summary_reports_canonical_fits():
    summary = rst._format_scaling_summary(
        SYNTHESIS_PER_REFLECTION_INTERCEPT,
        SYNTHESIS_WORKSPACE_QUBITS,
    )
    assert "I_1(n_blocks, N=256)" in summary
    assert "I_1(n_blocks=64, N)" in summary
    assert "W(n_blocks, N=256, b=32)" in summary
    assert "W(n_blocks=64, N, b=32)" in summary
    assert "n_blocks^" in summary
    assert "N^" in summary


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
