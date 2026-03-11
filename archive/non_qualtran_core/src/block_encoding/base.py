from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockEncodingMetadata:
    name: str
    alpha: float
    ancilla_qubits: int
    logical_cost_hint: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockEncodingQuery:
    """A query instance can carry runtime knobs for non-stationary algorithms."""

    step: int
    parameters: dict[str, float] = field(default_factory=dict)


class BlockEncoding(ABC):
    """Interface independent from any concrete quantum SDK."""

    @abstractmethod
    def metadata(self) -> BlockEncodingMetadata:
        raise NotImplementedError

    @abstractmethod
    def query(self, request: BlockEncodingQuery) -> object:
        """Return an SDK-neutral operation description or backend node handle."""
        raise NotImplementedError

    @abstractmethod
    def adjoint_query(self, request: BlockEncodingQuery) -> object:
        raise NotImplementedError