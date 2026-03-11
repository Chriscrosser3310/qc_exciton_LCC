"""Backend adapters for circuit generation and resource export."""

from .base import BackendAdapter, BackendProgram
from .qiskit_backend import QiskitBackendAdapter
from .resource_estimation import ResourceEstimatorAdapter

__all__ = [
    "BackendAdapter",
    "BackendProgram",
    "QiskitBackendAdapter",
    "ResourceEstimatorAdapter",
]

try:  # optional dependency: qualtran
    from .qualtran_backend import QualtranBackendAdapter

    __all__.append("QualtranBackendAdapter")
except ImportError:  # pragma: no cover
    pass

try:
    from .pennylane_backend import PennyLaneBackendAdapter

    __all__.append("PennyLaneBackendAdapter")
except ImportError:  # pragma: no cover
    pass
