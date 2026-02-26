from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LMOData:
    """Container for localized orbital quantities required by exciton builders."""

    fock_lmo: np.ndarray
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]


class PySCFExcitonDataBuilder:
    """Stub for molecule -> LMO-domain quantities.

    Planned capabilities:
    - run SCF in PySCF
    - localize occupied and virtual spaces separately
    - transform one-/two-electron quantities to LMO basis
    - expose terms needed for exciton model assembly
    """

    def build(self, *args, **kwargs) -> LMOData:
        raise NotImplementedError("PySCF integration is not implemented in this scaffold yet.")