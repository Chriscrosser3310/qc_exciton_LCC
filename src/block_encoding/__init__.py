"""Block-encoding contracts."""

from .base import BlockEncoding, BlockEncodingMetadata, BlockEncodingQuery
from .sparse_matrix import (
    AmplitudeEncoding,
    ColAccessOracle,
    EntryBinaryOracle,
    FullDataLoadingAmplitudeOracle,
    RowAccessOracle,
    SparseMatrixBlockEncoding,
    SparseOracleBundle,
)

__all__ = [
    "BlockEncoding",
    "BlockEncodingMetadata",
    "BlockEncodingQuery",
    "AmplitudeEncoding",
    "ColAccessOracle",
    "EntryBinaryOracle",
    "FullDataLoadingAmplitudeOracle",
    "RowAccessOracle",
    "SparseMatrixBlockEncoding",
    "SparseOracleBundle",
]
