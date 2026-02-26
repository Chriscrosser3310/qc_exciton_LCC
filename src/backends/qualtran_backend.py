from __future__ import annotations

from dataclasses import dataclass

from .base import BackendAdapter, BackendProgram


@dataclass
class QualtranBackendAdapter(BackendAdapter):
    """Maps abstract operations into a Qualtran Bloq-construction payload."""

    def compile_operations(self, operations: list[object]) -> BackendProgram:
        return BackendProgram(payload={"sdk": "qualtran", "ops": operations}, metadata={})