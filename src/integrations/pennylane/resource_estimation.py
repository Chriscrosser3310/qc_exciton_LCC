from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml


CLIFFORD_NAMES = {
    "CNOT",
    "CY",
    "CZ",
    "SWAP",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "Adjoint(S)",
}
ROTATION_NAMES = {"RX", "RY", "RZ", "Rot", "CRX", "CRY", "CRZ", "ControlledPhaseShift"}
PHASE_NAMES = {"GlobalPhase", "C(GlobalPhase)", "Adjoint(GlobalPhase)"}


@dataclass(frozen=True)
class CompiledCircuitCounts:
    total_gates: int
    depth: int
    t_count: int
    t_dagger_count: int
    toffoli_count: int
    cnot_count: int
    clifford_count: int
    rotation_count: int

    @property
    def total_t(self) -> int:
        return self.t_count + self.t_dagger_count

    @property
    def toffoli_equiv_7t(self) -> float:
        return self.total_t / 7.0


def _zero_counts() -> CompiledCircuitCounts:
    return CompiledCircuitCounts(
        total_gates=0,
        depth=0,
        t_count=0,
        t_dagger_count=0,
        toffoli_count=0,
        cnot_count=0,
        clifford_count=0,
        rotation_count=0,
    )


def _add_counts(a: CompiledCircuitCounts, b: CompiledCircuitCounts) -> CompiledCircuitCounts:
    return CompiledCircuitCounts(
        total_gates=a.total_gates + b.total_gates,
        depth=a.depth + b.depth,
        t_count=a.t_count + b.t_count,
        t_dagger_count=a.t_dagger_count + b.t_dagger_count,
        toffoli_count=a.toffoli_count + b.toffoli_count,
        cnot_count=a.cnot_count + b.cnot_count,
        clifford_count=a.clifford_count + b.clifford_count,
        rotation_count=a.rotation_count + b.rotation_count,
    )


def _counts_for_basic_name(name: str) -> CompiledCircuitCounts | None:
    if name in PHASE_NAMES:
        return _zero_counts()
    if name == "T":
        return CompiledCircuitCounts(1, 1, 1, 0, 0, 0, 0, 0)
    if name == "Adjoint(T)":
        return CompiledCircuitCounts(1, 1, 0, 1, 0, 0, 0, 0)
    if name == "Toffoli":
        # Use PennyLane's standard exact Toffoli decomposition counts.
        return CompiledCircuitCounts(15, 11, 4, 3, 1, 6, 8, 0)
    if name in CLIFFORD_NAMES:
        cnot = 1 if name == "CNOT" else 0
        return CompiledCircuitCounts(1, 1, 0, 0, 0, cnot, 1, 0)
    if name in ROTATION_NAMES:
        return CompiledCircuitCounts(1, 1, 0, 0, 0, 0, 0, 1)
    return None


def _compiled_counts_for_op(op: qml.operation.Operator, cache: dict[int, CompiledCircuitCounts]) -> CompiledCircuitCounts:
    basic = _counts_for_basic_name(op.name)
    if basic is not None:
        return basic

    key = int(op.hash)
    if key in cache:
        return cache[key]

    total = _zero_counts()
    try:
        decomp = op.decomposition()
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Cannot decompose PennyLane op {op.name!r} for resource estimation.") from exc

    for subop in decomp:
        total = _add_counts(total, _compiled_counts_for_op(subop, cache))

    cache[key] = total
    return total


def compiled_counts_from_qnode(qnode: qml.QNode) -> CompiledCircuitCounts:
    """Estimate Clifford/T/rotation counts by recursively decomposing each tape op.

    This avoids flattening the entire circuit into a single giant tape, which is too
    expensive for the current block-encoding circuits. The returned depth is a serial
    upper bound obtained by summing per-operation compiled depths.
    """
    tape = qml.workflow.construct_tape(qnode)()
    cache: dict[int, CompiledCircuitCounts] = {}
    total = _zero_counts()
    for op in tape.operations:
        total = _add_counts(total, _compiled_counts_for_op(op, cache))
    return total


__all__ = ["CompiledCircuitCounts", "compiled_counts_from_qnode"]
