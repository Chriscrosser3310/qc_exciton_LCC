"""Algorithm-level query abstractions."""

from .query_model import QueryCall, QuerySchedule
from .generalized_query_algorithm import GeneralizedQueryAlgorithm

try:  # optional dependency: qualtran
    from integrations.qualtran.algorithms import build_qsvt_composite, make_query_schedule
except ImportError:  # pragma: no cover - exercised when qualtran extra is absent
    pass

__all__ = [
    "QueryCall",
    "QuerySchedule",
    "GeneralizedQueryAlgorithm",
]

if "make_query_schedule" in globals():
    __all__.append("make_query_schedule")
if "build_qsvt_composite" in globals():
    __all__.append("build_qsvt_composite")
