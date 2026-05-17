#!/usr/bin/env python3
"""Regenerate the synthesis intercept and workspace tables from the Bloq.

The closed-form Toffoli decomposition for ``BlockUnitarySynthesisQROAM`` is

    T(n_blocks, N, K, b) = K * (2 * (log2(N) + 1) * b + I_1(n_blocks, N))

where ``I_1`` depends only on ``(n_blocks, N)``. The peak qubit count
satisfies

    Q(n_blocks, N, b) = signature_qubits(n_blocks, N, b) + W(n_blocks, N, b)

where ``W`` is the QROAMClean workspace contribution. ``I_1`` and ``W``
are both produced by Qualtran's QROAMClean optimizer, which is non-trivial
to reproduce in closed form; the current
``src/integrations/qualtran/model_resource_counts.py`` module ships them
as tabulated dicts (``SYNTHESIS_PER_REFLECTION_INTERCEPT`` and
``SYNTHESIS_WORKSPACE_QUBITS``).

This script regenerates both dicts directly from the Bloq so that under
a QROAMClean upstream change re-tabulation is a one-command operation
rather than a manual edit. By default it ``--check``s the existing
dicts against the freshly extracted values; pass ``--print`` to emit
Python source ready to paste back into ``model_resource_counts.py``.

Both tables match the ones already shipped with the module (verified by
``tests/test_block_unitary_synthesis_b_intercept.py::test_reference_table_matches_module_table``
and the cycle-11 round-trip test for workspaces); this script just
captures the procedure in one runnable place.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Sequence

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

from integrations.qualtran.block_unitary_synthesis_QROAM import (
    BlockUnitarySynthesisQROAM,
)
from integrations.qualtran.model_resource_counts import (
    SYNTHESIS_PER_REFLECTION_INTERCEPT,
    SYNTHESIS_WORKSPACE_QUBITS,
)


N_BLOCKS_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
N_ROWS_GRID: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
BITSIZE_GRID: tuple[int, ...] = (2, 4, 8, 16, 32)
B_REF_FOR_INTERCEPT = 4
SUMMARY_N_ROWS = 256
SUMMARY_BITSIZE = 32


def _build(n_blocks: int, n_rows: int, bitsize: int) -> BlockUnitarySynthesisQROAM:
    return BlockUnitarySynthesisQROAM.from_shape(
        n_blocks=n_blocks, n_rows=n_rows, phase_bitsize=bitsize, n_reflections=1
    )


def extract_intercept(n_blocks: int, n_rows: int) -> int:
    """Per-reflection ``b=0`` intercept ``I_1(n_blocks, n_rows)``.

    Slope is ``2*(log2(N)+1)`` by the K-linearity / b-affineness identities
    pinned in ``tests/test_block_unitary_synthesis_scaling.py``. Subtract
    the slope contribution at ``b = B_REF_FOR_INTERCEPT`` to recover the
    intercept.
    """

    bloq = _build(n_blocks, n_rows, B_REF_FOR_INTERCEPT)
    t = int(get_cost_value(bloq, QECGatesCost()).toffoli)
    slope = 2 * (int(math.log2(n_rows)) + 1)
    return t - slope * B_REF_FOR_INTERCEPT


def extract_workspace(n_blocks: int, n_rows: int, bitsize: int) -> int:
    """QROAMClean workspace ``Q(n_blocks, N, b) - signature_qubits``."""

    bloq = _build(n_blocks, n_rows, bitsize)
    q = int(get_cost_value(bloq, QubitCount()))
    return q - bloq.signature.n_qubits()


def regenerate_intercept_table(
    n_blocks_grid: Sequence[int] = N_BLOCKS_GRID,
    n_rows_grid: Sequence[int] = N_ROWS_GRID,
) -> dict[tuple[int, int], int]:
    return {
        (nb, nr): extract_intercept(nb, nr)
        for nb in n_blocks_grid
        for nr in n_rows_grid
    }


def regenerate_workspace_table(
    n_blocks_grid: Sequence[int] = N_BLOCKS_GRID,
    n_rows_grid: Sequence[int] = N_ROWS_GRID,
    bitsize_grid: Sequence[int] = BITSIZE_GRID,
) -> dict[tuple[int, int, int], int]:
    return {
        (nb, nr, b): extract_workspace(nb, nr, b)
        for b in bitsize_grid
        for nb in n_blocks_grid
        for nr in n_rows_grid
    }


def _format_intercept_table(
    table: dict[tuple[int, int], int],
    n_blocks_grid: Sequence[int],
    n_rows_grid: Sequence[int],
) -> str:
    lines = ["SYNTHESIS_PER_REFLECTION_INTERCEPT: dict[tuple[int, int], int] = {"]
    for nb in n_blocks_grid:
        entries = ", ".join(f"({nb}, {nr}): {table[(nb, nr)]}" for nr in n_rows_grid)
        lines.append(f"    {entries},")
    lines.append("}")
    return "\n".join(lines)


def _format_workspace_table(
    table: dict[tuple[int, int, int], int],
    n_blocks_grid: Sequence[int],
    n_rows_grid: Sequence[int],
    bitsize_grid: Sequence[int],
) -> str:
    lines = ["SYNTHESIS_WORKSPACE_QUBITS: dict[tuple[int, int, int], int] = {"]
    for b in bitsize_grid:
        lines.append(f"    # bitsize = {b}")
        for nb in n_blocks_grid:
            entries = ", ".join(
                f"({nb}, {nr}, {b}): {table[(nb, nr, b)]}" for nr in n_rows_grid
            )
            lines.append(f"    {entries},")
    lines.append("}")
    return "\n".join(lines)


def _check(
    name: str,
    expected: dict,
    actual: dict,
) -> int:
    if expected == actual:
        print(f"  {name}: OK ({len(actual)} entries match)")
        return 0
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    differ = {k: (expected[k], actual[k]) for k in expected.keys() & actual.keys() if expected[k] != actual[k]}
    print(f"  {name}: MISMATCH")
    if missing:
        print(f"    missing in extracted: {sorted(missing)[:10]}{' …' if len(missing) > 10 else ''}")
    if extra:
        print(f"    extra in extracted:   {sorted(extra)[:10]}{' …' if len(extra) > 10 else ''}")
    if differ:
        for k, (e, a) in list(differ.items())[:10]:
            print(f"    {k}: shipped={e}, extracted={a}")
        if len(differ) > 10:
            print(f"    … and {len(differ) - 10} more differing entries")
    return 1


def _power_law_fit(xs: Sequence[int], ys: Sequence[int]) -> tuple[float, float]:
    """Return ``(alpha, coeff)`` for ``ys ~= coeff * xs**alpha``."""

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) < 2:
        raise ValueError("at least two points are required")
    if any(x <= 0 for x in xs) or any(y <= 0 for y in ys):
        raise ValueError("power-law fit requires positive x and y values")

    log_xs = [math.log(x) for x in xs]
    log_ys = [math.log(y) for y in ys]
    mean_x = sum(log_xs) / len(log_xs)
    mean_y = sum(log_ys) / len(log_ys)
    denom = sum((x - mean_x) ** 2 for x in log_xs)
    if denom == 0:
        raise ValueError("x values must not all be equal")
    alpha = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_xs, log_ys)) / denom
    coeff = math.exp(mean_y - alpha * mean_x)
    return alpha, coeff


def _format_scaling_summary(
    intercept: dict[tuple[int, int], int],
    workspace: dict[tuple[int, int, int], int] | None,
    *,
    n_rows: int = SUMMARY_N_ROWS,
    bitsize: int = SUMMARY_BITSIZE,
    n_blocks_grid: Sequence[int] = N_BLOCKS_GRID,
) -> str:
    """Summarize canonical-slice power-law fits for closed-form work."""

    n_blocks = [nb for nb in n_blocks_grid if (nb, n_rows) in intercept]
    intercept_values = [intercept[(nb, n_rows)] for nb in n_blocks]
    alpha_i, coeff_i = _power_law_fit(n_blocks, intercept_values)
    lines = [
        "Scaling summary for canonical report slice:",
        f"  I_1(n_blocks, N={n_rows}) ≈ {coeff_i:.6g} * n_blocks^{alpha_i:.6f}",
    ]
    if workspace is not None:
        workspace_blocks = [
            nb for nb in n_blocks_grid if (nb, n_rows, bitsize) in workspace
        ]
        workspace_values = [
            workspace[(nb, n_rows, bitsize)] for nb in workspace_blocks
        ]
        alpha_w, coeff_w = _power_law_fit(workspace_blocks, workspace_values)
        lines.append(
            f"  W(n_blocks, N={n_rows}, b={bitsize}) ≈ "
            f"{coeff_w:.6g} * n_blocks^{alpha_w:.6f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print",
        dest="emit",
        action="store_true",
        help="Print regenerated dicts as Python source on stdout (for paste-back).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Diff regenerated dicts against the shipped tables and exit non-zero on drift.",
    )
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Skip the (slower) workspace table; only regenerate the intercept table.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print canonical-slice power-law fits for I_1 and W. This is a "
            "diagnostic aid for deriving closed-form table expressions."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.emit and not args.check and not args.summary:
        args.check = True

    print(
        f"Extracting intercept on grid n_blocks∈{N_BLOCKS_GRID} × n_rows∈{N_ROWS_GRID} "
        f"(b_ref={B_REF_FOR_INTERCEPT}) …"
    )
    intercept = regenerate_intercept_table()
    workspace: dict[tuple[int, int, int], int] | None = None
    if not args.no_workspace:
        print(
            f"Extracting workspace on grid × bitsize∈{BITSIZE_GRID} "
            f"({len(N_BLOCKS_GRID)*len(N_ROWS_GRID)*len(BITSIZE_GRID)} points) …"
        )
        workspace = regenerate_workspace_table()

    rc = 0
    if args.check:
        print("Checking against shipped tables:")
        rc |= _check("SYNTHESIS_PER_REFLECTION_INTERCEPT", SYNTHESIS_PER_REFLECTION_INTERCEPT, intercept)
        if workspace is not None:
            rc |= _check("SYNTHESIS_WORKSPACE_QUBITS", SYNTHESIS_WORKSPACE_QUBITS, workspace)

    if args.emit:
        print()
        print(_format_intercept_table(intercept, N_BLOCKS_GRID, N_ROWS_GRID))
        if workspace is not None:
            print()
            print(_format_workspace_table(workspace, N_BLOCKS_GRID, N_ROWS_GRID, BITSIZE_GRID))

    if args.summary:
        print()
        print(_format_scaling_summary(intercept, workspace))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
