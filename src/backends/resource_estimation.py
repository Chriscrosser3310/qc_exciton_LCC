from __future__ import annotations

from dataclasses import dataclass

from .base import BackendProgram


@dataclass
class ResourceEstimatorAdapter:
    """Converts backend programs into estimator-ready intermediate data."""

    target: str = "logical"

    def export(self, program: BackendProgram) -> dict[str, object]:
        return {
            "target": self.target,
            "sdk": program.metadata.get("sdk", "unknown"),
            "summary": {
                "n_operations": len(program.payload.get("ops", []))
                if isinstance(program.payload, dict)
                else None
            },
            "payload": program.payload,
        }