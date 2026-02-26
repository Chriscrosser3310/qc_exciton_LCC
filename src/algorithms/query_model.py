from __future__ import annotations

from dataclasses import dataclass

from block_encoding.base import BlockEncoding, BlockEncodingQuery


@dataclass(frozen=True)
class QueryCall:
    index: int
    encoding: BlockEncoding
    request: BlockEncodingQuery


class QuerySchedule:
    """Allows per-query changes in block-encoding strategy or parameters."""

    def __init__(self) -> None:
        self._calls: list[QueryCall] = []

    def append(self, call: QueryCall) -> None:
        self._calls.append(call)

    def __iter__(self):
        return iter(self._calls)

    def __len__(self) -> int:
        return len(self._calls)
