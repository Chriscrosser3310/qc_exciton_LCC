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
from .qualtran_sparse_bench import (
    PolylogSparseIndex,
    build_all_sparse_oracles,
    build_product_matrices_from_tensors,
    build_sparse_oracle_from_matrix,
    build_thresholded_benchmark_tensors,
)
from .qualtran_lattice_index_oracles import (
    SingleParticleSparseIndexOracle,
    TwoParticleSparseIndexOracle,
    build_lattice_sparse_index_oracles,
)
from .exciton_hamiltonian_encoding import (
    ExcitonHamiltonianEncoding,
    ExcitonLCUTerm,
    build_exciton_hamiltonian_encoding,
    build_f_and_v_block_encodings,
)
from .qrom_sparse import SparseTensorCOO, qrom_build_from_sparse_data, sparse_coo_from_items

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
    "build_thresholded_benchmark_tensors",
    "build_product_matrices_from_tensors",
    "build_sparse_oracle_from_matrix",
    "build_all_sparse_oracles",
    "PolylogSparseIndex",
    "SingleParticleSparseIndexOracle",
    "TwoParticleSparseIndexOracle",
    "build_lattice_sparse_index_oracles",
    "ExcitonLCUTerm",
    "ExcitonHamiltonianEncoding",
    "build_f_and_v_block_encodings",
    "build_exciton_hamiltonian_encoding",
    "SparseTensorCOO",
    "qrom_build_from_sparse_data",
    "sparse_coo_from_items",
]
