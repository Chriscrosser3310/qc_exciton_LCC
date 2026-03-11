from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualtranBackendAdapter:
    """Maps abstract operations into a Qualtran Bloq-construction payload."""

    def compile_operations(self, operations: list[object]) -> dict[str, object]:
        return {"payload": {"sdk": "qualtran", "ops": operations}, "metadata": {}}
