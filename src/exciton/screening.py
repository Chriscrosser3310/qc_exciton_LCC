from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreeningQuery:
    """Query in an LMO basis for screened interaction matrix elements."""

    p: int
    q: int
    r: int
    s: int
    omega: float | None = None


class ScreenedInteractionProvider(ABC):
    """Interface for obtaining screened Coulomb terms W_pqrs (optionally frequency dependent)."""

    @abstractmethod
    def matrix_element(self, query: ScreeningQuery) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantScreening(ScreenedInteractionProvider):
    """Minimal placeholder screening model for early algorithm prototyping."""

    value: float = 1.0

    def matrix_element(self, query: ScreeningQuery) -> float:
        _ = query
        return self.value