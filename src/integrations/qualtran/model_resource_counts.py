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
class SynthesisResourceCount:
    """Analytic resource estimate for ``BlockUnitaryReflectionQROAM``.

    ``signature_qubits`` is the persistent ``block + reflection_ancilla +
    system + phase_gradient`` register width. ``workspace_qubits`` is the
    transient QROAMClean workspace (tabulated; see
    ``SYNTHESIS_WORKSPACE_QUBITS``). ``total_qubits = signature_qubits +
    workspace_qubits`` matches the bloq's ``QubitCount`` exactly across the
    tabulated grid.
    """

    toffoli: int
    signature_qubits: int
    workspace_qubits: int
    total_qubits: int
    n_blocks: int
    n_rows: int
    bitsize: int
    n_reflections: int


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


# Per-reflection b=0 intercept I_1(n_blocks, N) for
# BlockUnitaryReflectionQROAM, pinned by
# ``tests/test_block_unitary_reflection_b_intercept.py``. Combined with the
# K-linear, b-affine, slope-= 2*(log2(N)+1) identities from
# ``tests/test_block_unitary_reflection_scaling.py`` it gives the closed-form
# decomposition
#
#     T(n_blocks, N, K, b) = K * (2 * (log2(N) + 1) * b + I_1(n_blocks, N)).
#
# The table is the analytic-side anchor; any QROAMClean upstream change that
# moves these values will surface both here and in the b-intercept test.
SYNTHESIS_PER_REFLECTION_INTERCEPT: dict[tuple[int, int], int] = {
    (1, 4): -6, (1, 8): -2, (1, 16): 6, (1, 32): 22, (1, 64): 46, (1, 128): 86, (1, 256): 142,
    (2, 4): 4, (2, 8): 16, (2, 16): 32, (2, 32): 64, (2, 64): 104, (2, 128): 176, (2, 256): 264,
    (4, 4): 12, (4, 8): 28, (4, 16): 52, (4, 32): 92, (4, 64): 148, (4, 128): 236, (4, 256): 356,
    (8, 4): 32, (8, 8): 64, (8, 16): 104, (8, 32): 176, (8, 64): 264, (8, 128): 416, (8, 256): 600,
    (16, 4): 48, (16, 8): 88, (16, 16): 144, (16, 32): 232, (16, 64): 352, (16, 128): 536, (16, 256): 784,
    (32, 4): 88, (32, 8): 160, (32, 16): 248, (32, 32): 400, (32, 64): 584, (32, 128): 896, (32, 256): 1272,
    (64, 4): 120, (64, 8): 208, (64, 16): 328, (64, 32): 512, (64, 64): 760, (64, 128): 1136, (64, 256): 1640,
}


def block_unitary_synthesis_toffoli(
    n_blocks: int,
    n_rows: int,
    bitsize: int,
    n_reflections: int,
) -> int:
    r"""Return the analytic Toffoli count for ``BlockUnitaryReflectionQROAM``.

    Implements the decomposition

        ``T = K * (2 * (log2(N) + 1) * b + I_1(n_blocks, N))``

    pinned by the ``tests/test_block_unitary_reflection_*`` family. The
    per-reflection intercept ``I_1`` is looked up from
    ``SYNTHESIS_PER_REFLECTION_INTERCEPT``; an unknown
    ``(n_blocks, n_rows)`` raises ``KeyError`` rather than silently
    extrapolating.
    """

    assert_power_of_two(n_rows, "n_rows")
    assert_power_of_two(n_blocks, "n_blocks")
    if min(bitsize, n_reflections) <= 0:
        raise ValueError("bitsize and n_reflections must be positive")
    if n_reflections > n_rows:
        raise ValueError("n_reflections must be <= n_rows")
    try:
        intercept = SYNTHESIS_PER_REFLECTION_INTERCEPT[(n_blocks, n_rows)]
    except KeyError as exc:
        raise KeyError(
            f"no tabulated synthesis intercept for n_blocks={n_blocks}, "
            f"n_rows={n_rows}; add an entry to "
            "SYNTHESIS_PER_REFLECTION_INTERCEPT after pinning it in "
            "tests/test_block_unitary_reflection_b_intercept.py"
        ) from exc
    slope = 2 * (int(math.log2(n_rows)) + 1)
    return n_reflections * (slope * bitsize + intercept)


# Transient QROAMClean workspace contribution
# (``QubitCount(BlockUnitaryReflectionQROAM.from_shape(...)) - signature.n_qubits()``)
# tabulated over the same 49-point ``(n_blocks, n_rows)`` grid as
# ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` plus ``bitsize`` in {2, 4, 8, 16, 32}.
# Values were extracted at ``n_reflections=1`` and are independent of
# ``n_reflections`` (each reflection reuses the same QROAMClean call graph;
# the workspace is the peak across reflections, not the sum).
SYNTHESIS_WORKSPACE_QUBITS: dict[tuple[int, int, int], int] = {
    # bitsize = 2
    (1, 4, 2): 6, (1, 8, 2): 8, (1, 16, 2): 9, (1, 32, 2): 13, (1, 64, 2): 14, (1, 128, 2): 22, (1, 256, 2): 23,
    (2, 4, 2): 7, (2, 8, 2): 9, (2, 16, 2): 10, (2, 32, 2): 14, (2, 64, 2): 15, (2, 128, 2): 23, (2, 256, 2): 24,
    (4, 4, 2): 8, (4, 8, 2): 10, (4, 16, 2): 11, (4, 32, 2): 15, (4, 64, 2): 16, (4, 128, 2): 24, (4, 256, 2): 25,
    (8, 4, 2): 10, (8, 8, 2): 14, (8, 16, 2): 15, (8, 32, 2): 23, (8, 64, 2): 24, (8, 128, 2): 40, (8, 256, 2): 41,
    (16, 4, 2): 11, (16, 8, 2): 15, (16, 16, 2): 16, (16, 32, 2): 24, (16, 64, 2): 25, (16, 128, 2): 41, (16, 256, 2): 42,
    (32, 4, 2): 15, (32, 8, 2): 23, (32, 16, 2): 24, (32, 32, 2): 40, (32, 64, 2): 41, (32, 128, 2): 73, (32, 256, 2): 74,
    (64, 4, 2): 16, (64, 8, 2): 24, (64, 16, 2): 25, (64, 32, 2): 41, (64, 64, 2): 42, (64, 128, 2): 74, (64, 256, 2): 75,
    # bitsize = 4
    (1, 4, 4): 8, (1, 8, 4): 9, (1, 16, 4): 13, (1, 32, 4): 14, (1, 64, 4): 22, (1, 128, 4): 23, (1, 256, 4): 39,
    (2, 4, 4): 9, (2, 8, 4): 10, (2, 16, 4): 14, (2, 32, 4): 15, (2, 64, 4): 23, (2, 128, 4): 24, (2, 256, 4): 40,
    (4, 4, 4): 10, (4, 8, 4): 11, (4, 16, 4): 15, (4, 32, 4): 16, (4, 64, 4): 24, (4, 128, 4): 25, (4, 256, 4): 41,
    (8, 4, 4): 11, (8, 8, 4): 12, (8, 16, 4): 16, (8, 32, 4): 17, (8, 64, 4): 25, (8, 128, 4): 26, (8, 256, 4): 42,
    (16, 4, 4): 15, (16, 8, 4): 16, (16, 16, 4): 24, (16, 32, 4): 25, (16, 64, 4): 41, (16, 128, 4): 42, (16, 256, 4): 74,
    (32, 4, 4): 16, (32, 8, 4): 17, (32, 16, 4): 25, (32, 32, 4): 26, (32, 64, 4): 42, (32, 128, 4): 43, (32, 256, 4): 75,
    (64, 4, 4): 24, (64, 8, 4): 25, (64, 16, 4): 41, (64, 32, 4): 42, (64, 64, 4): 74, (64, 128, 4): 75, (64, 256, 4): 139,
    # bitsize = 8
    (1, 4, 8): 12, (1, 8, 8): 13, (1, 16, 8): 14, (1, 32, 8): 22, (1, 64, 8): 23, (1, 128, 8): 39, (1, 256, 8): 40,
    (2, 4, 8): 13, (2, 8, 8): 14, (2, 16, 8): 15, (2, 32, 8): 23, (2, 64, 8): 24, (2, 128, 8): 40, (2, 256, 8): 41,
    (4, 4, 8): 14, (4, 8, 8): 15, (4, 16, 8): 16, (4, 32, 8): 24, (4, 64, 8): 25, (4, 128, 8): 41, (4, 256, 8): 42,
    (8, 4, 8): 15, (8, 8, 8): 16, (8, 16, 8): 17, (8, 32, 8): 25, (8, 64, 8): 26, (8, 128, 8): 42, (8, 256, 8): 43,
    (16, 4, 8): 16, (16, 8, 8): 17, (16, 16, 8): 18, (16, 32, 8): 26, (16, 64, 8): 27, (16, 128, 8): 43, (16, 256, 8): 44,
    (32, 4, 8): 24, (32, 8, 8): 25, (32, 16, 8): 26, (32, 32, 8): 42, (32, 64, 8): 43, (32, 128, 8): 75, (32, 256, 8): 76,
    (64, 4, 8): 25, (64, 8, 8): 26, (64, 16, 8): 27, (64, 32, 8): 43, (64, 64, 8): 44, (64, 128, 8): 76, (64, 256, 8): 77,
    # bitsize = 16
    (1, 4, 16): 20, (1, 8, 16): 21, (1, 16, 16): 22, (1, 32, 16): 23, (1, 64, 16): 39, (1, 128, 16): 40, (1, 256, 16): 72,
    (2, 4, 16): 21, (2, 8, 16): 22, (2, 16, 16): 23, (2, 32, 16): 24, (2, 64, 16): 40, (2, 128, 16): 41, (2, 256, 16): 73,
    (4, 4, 16): 22, (4, 8, 16): 23, (4, 16, 16): 24, (4, 32, 16): 25, (4, 64, 16): 41, (4, 128, 16): 42, (4, 256, 16): 74,
    (8, 4, 16): 23, (8, 8, 16): 24, (8, 16, 16): 25, (8, 32, 16): 26, (8, 64, 16): 42, (8, 128, 16): 43, (8, 256, 16): 75,
    (16, 4, 16): 24, (16, 8, 16): 25, (16, 16, 16): 26, (16, 32, 16): 27, (16, 64, 16): 43, (16, 128, 16): 44, (16, 256, 16): 76,
    (32, 4, 16): 25, (32, 8, 16): 26, (32, 16, 16): 27, (32, 32, 16): 28, (32, 64, 16): 44, (32, 128, 16): 45, (32, 256, 16): 77,
    (64, 4, 16): 41, (64, 8, 16): 42, (64, 16, 16): 43, (64, 32, 16): 44, (64, 64, 16): 76, (64, 128, 16): 77, (64, 256, 16): 141,
    # bitsize = 32
    (1, 4, 32): 36, (1, 8, 32): 37, (1, 16, 32): 38, (1, 32, 32): 39, (1, 64, 32): 40, (1, 128, 32): 72, (1, 256, 32): 73,
    (2, 4, 32): 37, (2, 8, 32): 38, (2, 16, 32): 39, (2, 32, 32): 40, (2, 64, 32): 41, (2, 128, 32): 73, (2, 256, 32): 74,
    (4, 4, 32): 38, (4, 8, 32): 39, (4, 16, 32): 40, (4, 32, 32): 41, (4, 64, 32): 42, (4, 128, 32): 74, (4, 256, 32): 75,
    (8, 4, 32): 39, (8, 8, 32): 40, (8, 16, 32): 41, (8, 32, 32): 42, (8, 64, 32): 43, (8, 128, 32): 75, (8, 256, 32): 76,
    (16, 4, 32): 40, (16, 8, 32): 41, (16, 16, 32): 42, (16, 32, 32): 43, (16, 64, 32): 44, (16, 128, 32): 76, (16, 256, 32): 77,
    (32, 4, 32): 41, (32, 8, 32): 42, (32, 16, 32): 43, (32, 32, 32): 44, (32, 64, 32): 45, (32, 128, 32): 77, (32, 256, 32): 78,
    (64, 4, 32): 42, (64, 8, 32): 43, (64, 16, 32): 44, (64, 32, 32): 45, (64, 64, 32): 46, (64, 128, 32): 78, (64, 256, 32): 79,
}


def block_unitary_synthesis_workspace_qubits(
    n_blocks: int,
    n_rows: int,
    bitsize: int,
) -> int:
    """Return the tabulated QROAMClean workspace qubits for ``BlockUnitaryReflectionQROAM``.

    Looked up from ``SYNTHESIS_WORKSPACE_QUBITS``. An untabulated
    ``(n_blocks, n_rows, bitsize)`` raises ``KeyError`` rather than
    silently extrapolating. The workspace is the peak transient
    contribution; ``signature_qubits + workspace_qubits`` matches the
    bloq's ``QubitCount`` exactly over the grid.
    """

    assert_power_of_two(n_rows, "n_rows")
    assert_power_of_two(n_blocks, "n_blocks")
    if bitsize <= 0:
        raise ValueError("bitsize must be positive")
    try:
        return SYNTHESIS_WORKSPACE_QUBITS[(n_blocks, n_rows, bitsize)]
    except KeyError as exc:
        raise KeyError(
            f"no tabulated synthesis workspace for n_blocks={n_blocks}, "
            f"n_rows={n_rows}, bitsize={bitsize}; add an entry to "
            "SYNTHESIS_WORKSPACE_QUBITS after extracting it from "
            "QubitCount on BlockUnitaryReflectionQROAM.from_shape(...)"
        ) from exc


def block_unitary_synthesis_signature_qubits(
    n_blocks: int,
    n_rows: int,
    bitsize: int,
) -> int:
    r"""Return the persistent (signature) qubit count for ``BlockUnitaryReflectionQROAM``.

    The bloq's external signature consists of four registers:
    ``block`` (``ceil_log2(n_blocks)`` qubits), ``reflection_ancilla`` (1 qubit),
    ``system`` (``log2(n_rows)`` qubits), and ``phase_gradient`` (``bitsize`` qubits).
    This is a lower bound on the peak qubit count; transient QROAMClean workspace
    is not modeled here and depends on the chosen ``log_block_sizes``.
    """

    assert_power_of_two(n_rows, "n_rows")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if bitsize <= 0:
        raise ValueError("bitsize must be positive")
    return ceil_log2(n_blocks) + 1 + int(math.log2(n_rows)) + bitsize


def block_unitary_synthesis_count(
    n_blocks: int,
    n_rows: int,
    bitsize: int,
    n_reflections: int,
) -> SynthesisResourceCount:
    """Return the analytic Toffoli and qubit estimate for the synthesis bloq.

    Composes ``block_unitary_synthesis_toffoli``,
    ``block_unitary_synthesis_signature_qubits``, and
    ``block_unitary_synthesis_workspace_qubits`` into a single record.
    ``total_qubits`` equals ``signature_qubits + workspace_qubits`` and
    matches the bloq's ``QubitCount`` exactly across the tabulated grid.
    """

    sig = block_unitary_synthesis_signature_qubits(n_blocks, n_rows, bitsize)
    workspace = block_unitary_synthesis_workspace_qubits(n_blocks, n_rows, bitsize)
    return SynthesisResourceCount(
        toffoli=block_unitary_synthesis_toffoli(n_blocks, n_rows, bitsize, n_reflections),
        signature_qubits=sig,
        workspace_qubits=workspace,
        total_qubits=sig + workspace,
        n_blocks=n_blocks,
        n_rows=n_rows,
        bitsize=bitsize,
        n_reflections=n_reflections,
    )


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


SYNTHESIS_PANEL_N_BLOCKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def _synthesis_panel_records(
    n_rows: int,
    bitsize: int,
    n_reflections: int,
    n_blocks_seq: Sequence[int] = SYNTHESIS_PANEL_N_BLOCKS,
) -> list[SynthesisResourceCount]:
    """Evaluate ``block_unitary_synthesis_count`` over a power-of-two ``n_blocks`` sweep.

    Only points present in the tabulated intercept and workspace grids are
    valid; the default sequence covers the full ``SYNTHESIS_PER_REFLECTION_INTERCEPT``
    grid for ``n_blocks``.
    """

    return [
        block_unitary_synthesis_count(nb, n_rows, bitsize, n_reflections)
        for nb in n_blocks_seq
    ]


def _plot_report(
    *,
    block_dim: int,
    bitsize: int,
    k_values: Sequence[int],
    out_pdf: str,
    synthesis_n_blocks: Sequence[int] = SYNTHESIS_PANEL_N_BLOCKS,
    synthesis_n_reflections: int | None = None,
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

    synth_n_reflections = synthesis_n_reflections if synthesis_n_reflections is not None else block_dim
    synth_records = _synthesis_panel_records(
        n_rows=block_dim,
        bitsize=bitsize,
        n_reflections=synth_n_reflections,
        n_blocks_seq=synthesis_n_blocks,
    )

    fits = {
        "theory_topt_t": power_law_fit(blocks, [r.toffoli for r in t_opt]),
        "theory_qopt_t": power_law_fit(blocks, [r.toffoli for r in q_opt]),
        "theory_topt_q": power_law_fit(blocks, [r.qubits for r in t_opt]),
        "theory_qopt_q": power_law_fit(blocks, [r.qubits for r in q_opt]),
    }
    # Sub-linear (alpha < 1) scaling of synthesis Toffoli in n_blocks is the
    # observable signature of QROAMClean amortization; pinned as a regression
    # guard by ``test_synthesis_panel_power_law_subllinear``.
    synth_fits = {
        "synth_t": power_law_fit(
            list(synthesis_n_blocks), [r.toffoli for r in synth_records]
        ),
        "synth_q": power_law_fit(
            list(synthesis_n_blocks), [r.total_qubits for r in synth_records]
        ),
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

    def plot_synthesis(metric: str):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        if metric == "toffoli":
            ys = [r.toffoli for r in synth_records]
            ylabel = "Synthesis Toffoli count"
            alpha, coeff = synth_fits["synth_t"]
        else:
            ys = [r.total_qubits for r in synth_records]
            ylabel = "Synthesis peak logical qubits"
            alpha, coeff = synth_fits["synth_q"]
        ax.plot(list(synthesis_n_blocks), ys, "o-",
                label=f"BlockUnitaryReflectionQROAM (K={synth_n_reflections})")
        x_arr = np.asarray(synthesis_n_blocks, dtype=float)
        ax.plot(
            x_arr,
            coeff * x_arr**alpha,
            "--",
            alpha=0.7,
            label=f"fit: c*n_blocks^{alpha:.3f}",
        )
        ax.set_xscale("log", base=2)
        if metric == "toffoli":
            ax.set_yscale("log")
        ax.set_xlabel("n_blocks (power of two)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(synthesis_n_blocks))
        ax.set_xticklabels([str(nb) for nb in synthesis_n_blocks])
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(
            f"{ylabel} (analytic; N={block_dim}, b={bitsize}, "
            f"K={synth_n_reflections})"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def synthesis_table_page():
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.axis("off")
        rows = [
            [
                r.n_blocks,
                f"{r.toffoli:,}",
                r.signature_qubits,
                r.workspace_qubits,
                r.total_qubits,
            ]
            for r in synth_records
        ]
        tbl = ax.table(
            cellText=rows,
            colLabels=[
                "n_blocks",
                "Toffoli",
                "signature qubits",
                "workspace qubits",
                "total qubits",
            ],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)
        ax.set_title(
            f"BlockUnitaryReflectionQROAM analytic counts "
            f"(N={block_dim}, b={bitsize}, K={synth_n_reflections})"
        )
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
        lines.append(
            f"Synthesis panel: BlockUnitaryReflectionQROAM analytic Toffoli + total qubits"
        )
        lines.append(
            f"  swept over n_blocks={list(synthesis_n_blocks)}, "
            f"N={block_dim}, b={bitsize}, K={synth_n_reflections}."
        )
        lines.append(
            "  total_qubits = signature + tabulated QROAMClean workspace; matches"
        )
        lines.append(
            "  the bloq's QubitCount exactly over the tabulated grid."
        )
        lines.append("  Synthesis power-law fits y = c*n_blocks^alpha (alpha<1 ⇒ QROAM amortization):")
        for label, (alpha_s, coeff_s) in synth_fits.items():
            lines.append(f"    {label}: alpha={alpha_s:.3f}, c={coeff_s:.3e}")
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
            lambda: plot_synthesis("toffoli"),
            lambda: plot_synthesis("qubits"),
            synthesis_table_page,
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
