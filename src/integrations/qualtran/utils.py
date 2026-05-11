"""Small resource-counting helpers for local Qualtran integrations."""

from typing import Optional

from qualtran import Bloq
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting.generalizers import generalize_cswap_approx
from qualtran.symbolics import smax, SymbolicInt


def get_Toffoli_counts(bloq: Bloq, *, ts_per_rotation: int = 0) -> SymbolicInt:
    """Return the T/CCZ-count converted to Toffoli-style counts.

    This preserves the convention used in `QROAM.ipynb`:
    `total_t_and_ccz_count(ts_per_rotation=0)['n_ccz']`.
    """
    gate_counts = get_cost_value(
        bloq, QECGatesCost(), generalizer=generalize_cswap_approx
    )
    return gate_counts.total_t_and_ccz_count(ts_per_rotation=ts_per_rotation)["n_ccz"]


def get_qubit_counts(bloq: Bloq) -> SymbolicInt:
    """Return Qualtran's peak logical qubit estimate for a bloq."""
    return get_cost_value(bloq, QubitCount())


def get_ancilla_counts(bloq: Bloq, *, external_qubits: Optional[SymbolicInt] = None) -> SymbolicInt:
    """Return estimated intermediate ancillas used by a bloq.

    This is computed as peak logical qubits minus the external logical qubits implied by the
    bloq signature. Override `external_qubits` if some signature registers should be treated as
    reusable external workspace instead of algorithm registers.
    """
    if external_qubits is None:
        external_qubits = bloq.signature.n_qubits()
    return smax(0, get_qubit_counts(bloq) - external_qubits)


# Backwards-compatible lowercase alias for notebook/code style.
get_toffoli_counts = get_Toffoli_counts
