from __future__ import annotations

from dataclasses import dataclass

from .model import ExcitonModel, ExcitonTerm, OrbitalPartition
from .screening import ScreenedInteractionProvider, ScreeningQuery


@dataclass
class ExcitonBuilder:
    """Builds exciton models from LMO quantities and screening providers."""

    n_orbitals: int
    partition: OrbitalPartition
    screening: ScreenedInteractionProvider

    def build_minimal(self) -> ExcitonModel:
        model = ExcitonModel(n_orbitals=self.n_orbitals, partition=self.partition)
        model.validate()

        # Placeholder coupling term from a screening query.
        w = self.screening.matrix_element(ScreeningQuery(0, 0, 0, 0, omega=0.0))
        model.add_term(ExcitonTerm(label="W_0000", coefficient=w, modes=(0,)))
        model.metadata["builder"] = "minimal"
        return model