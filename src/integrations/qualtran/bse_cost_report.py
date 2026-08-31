#!/usr/bin/env python3
r"""Block-encoding Toffoli cost of the BSE walk operator for diamond.

Reports ``C_walk`` and its per-template breakdown across the ``[n,n,n]`` momentum meshes
for ``n = 2..6``, at ``N_o = 4``, ``N_v = 22``, ``N_THC = 8 * n_ao`` with
``n_ao = N_o + N_v = 26`` (so ``N_THC = 208``), for both central options on the ``ov``
exchange template.

This reports numbers only.  It does **not** compare them against the analytic model in
``XPRIZE_paper/script/`` or against any table in the manuscript.

    python bse_cost_report.py
"""

from __future__ import annotations

import sys

try:
    from .bse_walk_operator import BSEWalkOperator
except ImportError:  # pragma: no cover - script execution
    from bse_walk_operator import BSEWalkOperator

N_O = 4
N_V = 22
N_AO = N_O + N_V           # 26
N_THC = 8 * N_AO           # 208
MESHES = [2, 3, 4, 5, 6]
PHASE_BITSIZE = 32

_COLS = ['C_0', 'C_oo', 'C_vv', 'C_ov_ex', 'C_ov_dir', 'C_route', 'C_reflect', 'C_walk']


def walk(n: int, central: str, m: int = 1) -> BSEWalkOperator:
    return BSEWalkOperator(
        m=m, N_o=N_O, N_v=N_V, N_IP=N_THC, N_k=n ** 3,
        phase_bitsize=PHASE_BITSIZE, optimal_T=True,
        exchange_central=central, real_data=False,
    )


def table(central: str, m: int = 1) -> None:
    print(f"\n### exchange central = {central}   (m = {m}, b = {PHASE_BITSIZE})")
    head = f"{'mesh':>9} {'N_k':>5} " + " ".join(f"{c:>10}" for c in _COLS)
    print(head)
    print("-" * len(head))
    for n in MESHES:
        w = walk(n, central, m)
        d = w.template_costs()
        row = f"{f'[{n},{n},{n}]':>9} {n**3:>5} " + " ".join(f"{d[c]:>10,}" for c in _COLS)
        print(row)


def main() -> int:
    print(__doc__.split('\n')[0])
    print(f"N_o = {N_O}, N_v = {N_V}, n_ao = {N_AO}, N_THC = 8*n_ao = {N_THC}")
    print("Toffoli-equivalent (n_ccz + n_t/4); C_walk is one walk-operator query.")
    print("Complex data. THC rank padded to a power of two where a primitive requires it.")
    for central in ("eigendecomposition", "frobenius"):
        table(central)
    print(
        "\nC_oo and C_vv are INCREMENTAL: the shared zeta^W and two chi factors are"
        "\ncounted once, inside C_ov_dir."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
