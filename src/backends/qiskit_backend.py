from __future__ import annotations

from dataclasses import dataclass

from .base import BackendAdapter, BackendProgram


@dataclass
class QiskitBackendAdapter(BackendAdapter):
    """Maps abstract operations into a Qiskit-friendly payload."""

    emit_barriers: bool = False

    def compile_operations(self, operations: list[object]) -> BackendProgram:
        # Intentionally keeps payload neutral until concrete gate translators are added.
        return BackendProgram(
            payload={"sdk": "qiskit", "ops": operations},
            metadata={"emit_barriers": self.emit_barriers},
        )