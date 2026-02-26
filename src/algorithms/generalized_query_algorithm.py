from __future__ import annotations

from dataclasses import dataclass

from .query_model import QuerySchedule


@dataclass
class QueryExecutionResult:
    operations: list[object]


class GeneralizedQueryAlgorithm:
    """Skeleton that consumes a schedule with potentially heterogeneous block encodings."""

    def run(self, schedule: QuerySchedule) -> QueryExecutionResult:
        ops: list[object] = []
        for call in schedule:
            ops.append(call.encoding.query(call.request))
        return QueryExecutionResult(operations=ops)