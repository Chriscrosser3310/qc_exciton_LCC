#!/usr/bin/env python3
r"""Primitive-by-primitive: this package's Qualtran bloqs vs the analytic model.

Compares Toffoli counts of the bloqs used by the Sec.-2 BSE walk operator against
``XPRIZE_paper/script`` (the manuscript's analytic cost model) at matched parameters.

A **differential harness**: it executes both sides and diffs.  It draws no conclusion
about which side is right -- a mismatch is a finding, not a verdict.

All comparisons use **complex** data (Bloch orbitals at a general k-point are complex).
Bloqs are evaluated at the PHYSICAL dimension (no power-of-two padding).  The
model(pad) column shows what padding would have cost, for reference.

    python compare_primitives_to_model.py
"""

from __future__ import annotations

import os
import sys

_PAPER = os.environ.get(
    "XPRIZE_PAPER_SCRIPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "XPRIZE_paper", "script"),
)
sys.path.insert(0, os.path.abspath(_PAPER))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import resource_estimates as model  # noqa: E402
from toffoli_cost import toffoli_count as tof  # noqa: E402

from block_isometry_column_synthesis_QROAM import (  # noqa: E402
    ColumnIsometryRectangularBlockEncoding,
)
from block_unitary_interferometer_QROAM import (  # noqa: E402
    BlockUnitaryInterferometerSynthesisQROAM,
)
from diagonal_kernel_block_encoding import DiagonalCoulombKernelBlockEncoding  # noqa: E402
from eigendecomposition_block_encoding import EigendecompositionBlockEncoding  # noqa: E402
from load_all_state_preparation_QROAM import LoadAllStatePreparationQROAM  # noqa: E402

B = 32
MODE = "T-opt"          # matches optimal_T=True on the bloq side


def pow2(n: int) -> int:
    return 1 << max(1, (int(n) - 1).bit_length())


def is_pow2(n: int) -> bool:
    return int(n) >= 1 and (int(n) & (int(n) - 1)) == 0


def fnp(n: int) -> dict:
    """``force_non_power`` only when it is actually needed.

    Passing it for an already-power-of-two dimension selects the model's range-safe
    (P-13) branch and inflates the number, which is not the quantity to compare against.
    """
    return {} if is_pow2(n) else {'force_non_power': True}


def m_(fn, *a, **kw) -> int | None:
    try:
        return int(fn(*a, **kw).toffolis)
    except Exception as e:                                  # noqa: BLE001
        return None


def hdr(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'params':<26} {'model(N)':>11} {'model(pad)':>11} {'qualtran':>11} "
          f"{'vs model(N)':>12} {'vs model(pad)':>14}")
    print("  " + "-" * 90)


def row(params, th, th_pad, im) -> None:
    def r(a, bq):
        return f"{bq/a:>11.2f}x" if (a and bq is not None) else f"{'-':>12}"
    s_th = f"{th:>11,}" if th is not None else f"{'ERR':>11}"
    s_tp = f"{th_pad:>11,}" if th_pad is not None else f"{'-':>11}"
    s_im = f"{im:>11,}" if im is not None else f"{'CANNOT BUILD':>11}"
    print(f"  {params:<26} {s_th} {s_tp} {s_im} {r(th, im):>12} {r(th_pad, im):>14}")


def main() -> int:
    print(__doc__.split('\n')[0])
    print(f"b = {B}, mode = {MODE}, complex data\n")

    # ---- P-06 unitary synthesis -------------------------------------------------
    hdr("P-06  unitary synthesis (interferometer)   [N_o=4, N_v=22, N_THC=208 are the physical dims]")
    for N, N_k in [(4, 8), (22, 8), (22, 27), (208, 8), (208, 27), (32, 8), (256, 8)]:
        th = m_(model.unitary_synthesis, N, N_k, B, mode=MODE, **fnp(N))
        Np = pow2(max(N, 4))
        th_pad = m_(model.unitary_synthesis, Np, N_k, B, mode=MODE) if Np != N else None
        try:
            im = tof(BlockUnitaryInterferometerSynthesisQROAM(
                n_blocks=N_k, n_rows=N, phase_bitsize=B, optimal_T=True))
        except Exception:                                   # noqa: BLE001
            im = None
        row(f"N={N}, N_k={N_k}", th, th_pad, im)

    # ---- P-12 eigendecomposition ------------------------------------------------
    hdr("P-12  eigendecomposition block encoding (U D U^dag)")
    for N, N_k in [(4, 8), (22, 27), (208, 27)]:
        th = m_(model.eigendecomposition_block_encoding, N, N_k, B, mode=MODE, **fnp(N))
        Np = pow2(max(N, 4))
        th_pad = m_(model.eigendecomposition_block_encoding, Np, N_k, B, mode=MODE) \
            if Np != N else None
        try:
            im = tof(EigendecompositionBlockEncoding(
                n_blocks=N_k, n_rows=N, phase_bitsize=B, optimal_T=True))
        except Exception:                                   # noqa: BLE001
            im = None
        row(f"N={N}, N_k={N_k}", th, th_pad, im)

    # ---- P-09 diagonal ----------------------------------------------------------
    hdr("P-09  diagonal matrix block encoding")
    # The model's ``N`` here is the matrix dimension N_IP; it accounts internally for
    # the (Q, I, J) address, which is what the bloq's (N_k, N_IP, N_IP) table holds.
    for N, N_k in [(208, 8), (208, 27), (256, 8), (512, 8)]:
        th = m_(model.diagonal_matrix_block_encoding, N, N_k, B, mode=MODE, **fnp(N))
        try:
            im = tof(DiagonalCoulombKernelBlockEncoding(
                N_k=N_k, N_IP=N, phase_bitsize=B, optimal_T=True))
        except Exception:                                   # noqa: BLE001
            im = None
        row(f"N_IP={N}, N_k={N_k}", th, None, im)

    # ---- P-07 isometry synthesis ------------------------------------------------
    hdr("P-07  isometry synthesis, column-by-column (forward only)")
    for N, M, N_k in [(208, 4, 8), (208, 22, 8), (208, 22, 27), (256, 22, 8)]:
        th = m_(model.forward_isometry_synthesis, N, M, N_k, B, mode=MODE)
        Np = pow2(N)
        th_pad = m_(model.forward_isometry_synthesis, Np, M, N_k, B, mode=MODE) \
            if Np != N else None
        try:
            im = tof(ColumnIsometryRectangularBlockEncoding(
                n_blocks=N_k, n_rows=N, phase_bitsize=B, n_reflections=M,
                optimal_T=True, real_data=False))
        except Exception:                                   # noqa: BLE001
            im = None
        row(f"N={N}, M={M}, N_k={N_k}", th, th_pad, im)

    # ---- P-08 load-all state preparation ---------------------------------------
    hdr("P-08  state preparation, load-all (forward only)")
    for N, N_k in [(22, 8), (22, 27), (32, 8), (22, 8 * 208)]:
        th = m_(model.forward_state_preparation_load_all, N, N_k, B, mode=MODE)
        Np = pow2(N)
        th_pad = m_(model.forward_state_preparation_load_all, Np, N_k, B, mode=MODE) \
            if Np != N else None
        try:
            im = tof(LoadAllStatePreparationQROAM(
                n_addr=N_k, n_rows=N, phase_bitsize=B, real_data=False))
        except Exception:                                   # noqa: BLE001
            im = None
        row(f"N={N}, addr={N_k}", th, th_pad, im)

    print("\n  Ratios > 1: the Qualtran bloq is dearer than the analytic model.")
    print("  'CANNOT BUILD': the bloq rejects a non-power-of-two dimension.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
