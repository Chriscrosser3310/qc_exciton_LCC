from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PennyLaneBackendAdapter:
    """Maps abstract operations into a PennyLane-friendly payload.

    This adapter stays dependency-light: it does not import PennyLane directly.
    """

    interface: str = "autograd"

    def compile_operations(self, operations: list[object]) -> dict[str, object]:
        return {
            "payload": {"sdk": "pennylane", "ops": operations},
            "metadata": {"interface": self.interface},
        }
