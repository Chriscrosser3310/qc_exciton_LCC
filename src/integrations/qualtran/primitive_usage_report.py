#!/usr/bin/env python3
r"""How many times each cost primitive is used in the BSE walk operator.

Walks the Qualtran call graph of :class:`BSEWalkOperator` and counts occurrences of each
``P-nn`` primitive, per template and in total.  Two levels are reported:

* **direct** -- what each template calls itself (P-12 counted as one eigendecomposition);
* **expanded** -- with the composite primitives unfolded (P-12 -> 2 x P-06 + 1 diagonal,
  P-07 -> its layers), which is the level the manuscript's component counts are stated at.

    python primitive_usage_report.py
"""

from __future__ import annotations

import sys
from collections import Counter

from block_isometry_column_synthesis_QROAM import (
    BlockIsometryColumnSynthesisQROAM, ColumnIsometryRectangularBlockEncoding,
)
from block_unitary_interferometer_QROAM import BlockUnitaryInterferometerSynthesisQROAM
from classical_matrix_block_encoding_QROAM import (
    BlockDiagonalClassicalMatrixBlockEncoding, DirectHermitianBlockEncoding,
)
from diagonal_kernel_block_encoding import DiagonalCoulombKernelBlockEncoding
from eigendecomposition_block_encoding import EigendecompositionBlockEncoding
from load_all_state_preparation_QROAM import LoadAllStatePreparationQROAM
from bse_walk_operator import BSEWalkOperator

from qualtran.bloqs.arithmetic import Subtract
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.qft import QFTTextBook
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.resource_counting import SympySymbolAllocator

# primitive id -> (label, classes)
PRIMS = [
    ('P-06', 'unitary synthesis (interferometer)', (BlockUnitaryInterferometerSynthesisQROAM,)),
    ('P-07', 'isometry synthesis (column-by-column)',
     (ColumnIsometryRectangularBlockEncoding, BlockIsometryColumnSynthesisQROAM)),
    ('P-08', 'state preparation (load-all)', (LoadAllStatePreparationQROAM,)),
    ('P-09', 'diagonal-matrix BE', (DiagonalCoulombKernelBlockEncoding,)),
    ('P-11', 'Frobenius BE (Hermitian)',
     (BlockDiagonalClassicalMatrixBlockEncoding, DirectHermitianBlockEncoding)),
    ('P-12', 'eigendecomposition BE', (EigendecompositionBlockEncoding,)),
    ('--QFT', 'Fourier transform', (QFTTextBook,)),
    ('--PRP', 'uniform superposition (PREPARE / momentum isometry)', (PrepareUniformSuperposition,)),
    ('--SWP', 'controlled SWAP (routing)', (CSwap,)),
    ('--SUB', 'modular subtract (momentum arithmetic)', (Subtract,)),
]
_ALL = tuple(c for _, _, cs in PRIMS for c in cs)

ssa = SympySymbolAllocator()


def unwrap(b):
    while hasattr(b, 'subbloq'):
        b = b.subbloq
    while hasattr(b, 'inner') and type(b).__name__.startswith('_Controlled'):
        b = b.inner
    return b


def count(bloq, mult=1, acc=None, depth=0, stop_at_prims=True):
    """Count primitive occurrences reachable from ``bloq``."""
    acc = Counter() if acc is None else acc
    b = unwrap(bloq)
    for pid, _, classes in PRIMS:
        if isinstance(b, classes):
            acc[(pid, type(b).__name__)] += mult
            if stop_at_prims:
                return acc
            break
    if depth > 8:
        return acc
    if not stop_at_prims:
        try:
            callees = b.build_call_graph(ssa)
        except Exception:                                    # noqa: BLE001
            return acc
        if not isinstance(callees, dict):
            callees = dict(callees)
        for c, n in callees.items():
            count(unwrap(c), mult * n, acc, depth + 1, stop_at_prims)
        return acc
    try:
        callees = b.build_call_graph(ssa)
    except Exception:                                        # noqa: BLE001
        return acc
    if not isinstance(callees, dict):
        callees = dict(callees)
    for c, n in callees.items():
        cu = unwrap(c)
        if stop_at_prims and isinstance(cu, _ALL) and cu is not b:
            count(cu, mult * n, acc, depth + 1, stop_at_prims)
        elif not isinstance(cu, _ALL):
            count(cu, mult * n, acc, depth + 1, stop_at_prims)
    return acc


def show(title, acc):
    print(f"\n  {title}")
    if not acc:
        print("      (none)")
        return
    order = {p: i for i, (p, _, _) in enumerate(PRIMS)}
    for (pid, cls), n in sorted(acc.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        label = next(l for p, l, _ in PRIMS if p == pid)
        print(f"      {n:>4} x  {pid}  {label:<42} [{cls}]")


def main() -> int:
    print(__doc__.split('\n')[0])
    for central in ('eigendecomposition', 'frobenius'):
        w = BSEWalkOperator(m=1, N_o=4, N_v=22, N_IP=208, N_k=216,
                            phase_bitsize=32, optimal_T=True, exchange_central=central)
        be = w.block_encoding
        print(f"\n{'='*84}\nexchange central = {central}   (m=1, N_o=4, N_v=22, N_THC=208, N_k=216)")
        templates = [
            ('C_0      Fock, occupied', be.fock_occ),
            ('C_0      Fock, virtual', be.fock_virt),
            ('C_ov_ex  exchange', be.exchange),
            ('C_ov_dir direct', be.direct),
            ('C_oo     same-spin occ (incremental)', be.oo),
            ('C_vv     same-spin virt (incremental)', be.vv),
        ]
        total = Counter()
        for name, t in templates:
            acc = count(t)
            total.update(acc)
            show(name, acc)
        show('TOTAL over one walk step (direct level)', total)

        exp = Counter()
        for _, t in templates:
            exp.update(count(t, stop_at_prims=False))
        show('TOTAL, composites expanded (P-12 -> 2 x P-06 + diagonal)', exp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
