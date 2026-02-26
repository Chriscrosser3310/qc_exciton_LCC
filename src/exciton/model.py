from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class OrbitalPartition:
    """Partition of localized molecular orbitals into occupied and virtual sets."""

    occupied: tuple[int, ...]
    virtual: tuple[int, ...]

    def validate(self) -> None:
        overlap = set(self.occupied).intersection(self.virtual)
        if overlap:
            raise ValueError(f"Occupied/virtual overlap detected: {sorted(overlap)}")


@dataclass(frozen=True)
class ExcitonTerm:
    """One term in an exciton Hamiltonian decomposition."""

    label: str
    coefficient: float
    modes: tuple[int, ...]


@dataclass
class ExcitonModel:
    """Exciton Hamiltonian container in an LMO basis."""

    n_orbitals: int
    partition: OrbitalPartition
    terms: list[ExcitonTerm] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def add_term(self, term: ExcitonTerm) -> None:
        if any(m < 0 or m >= self.n_orbitals for m in term.modes):
            raise ValueError("Term mode index out of range.")
        self.terms.append(term)

    def extend_terms(self, terms: Iterable[ExcitonTerm]) -> None:
        for term in terms:
            self.add_term(term)

    def validate(self) -> None:
        self.partition.validate()
        if len(set(self.partition.occupied + self.partition.virtual)) != self.n_orbitals:
            raise ValueError("Orbital partition does not cover n_orbitals uniquely.")