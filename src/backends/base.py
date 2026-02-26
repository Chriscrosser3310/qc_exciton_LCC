from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BackendProgram:
    """SDK-specific payload plus metadata."""

    payload: object
    metadata: dict[str, object] = field(default_factory=dict)


class BackendAdapter(ABC):
    @abstractmethod
    def compile_operations(self, operations: list[object]) -> BackendProgram:
        raise NotImplementedError