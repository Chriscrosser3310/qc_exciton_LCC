from math import isclose, pi

from block_encoding.base import BlockEncodingQuery
from block_encoding.sparse_matrix import (
    EntryBinaryOracle,
    SparseMatrixBlockEncoding,
    SparseOracleBundle,
)


def test_row_col_oracles_compile_from_functions():
    row_to_col = lambda row, l_pos: (row + l_pos) % 3
    col_to_row = lambda col, l_pos: (col + 2 * l_pos) % 3
    entry_fn = lambda i, j: 0.25 if i == j else -0.125

    bundle = SparseOracleBundle.from_functions(
        n_rows=3,
        n_cols=3,
        max_row_nnz=2,
        max_col_nnz=2,
        row_to_col_fn=row_to_col,
        col_to_row_fn=col_to_row,
        entry_fn=entry_fn,
        value_bits=8,
        frac_bits=6,
        alpha=1.0,
    )

    assert bundle.row_oracle.lookup(2, 1) == 0
    assert bundle.col_oracle.lookup(1, 1) == 0
    compiled = bundle.row_oracle.compile_truth_table()
    assert len(compiled) == 6


def test_entry_oracle_full_data_loading_and_amplitude():
    matrix = [
        [0.5, -0.25],
        [0.0, 0.125],
    ]
    entry_oracle = EntryBinaryOracle.from_dense(matrix, value_bits=10, frac_bits=8)

    bits = entry_oracle.lookup_bits(0, 1)
    assert bits[0] == "1"
    assert isclose(entry_oracle.lookup_value(1, 1), 0.125, rel_tol=0, abs_tol=1e-9)

    bundle = SparseOracleBundle.from_functions(
        n_rows=2,
        n_cols=2,
        max_row_nnz=2,
        max_col_nnz=2,
        row_to_col_fn=lambda row, l_pos: l_pos,
        col_to_row_fn=lambda col, l_pos: l_pos,
        entry_fn=lambda i, j: matrix[i][j],
        value_bits=10,
        frac_bits=8,
        alpha=1.0,
    )
    amp = bundle.amplitude_oracle.encode(0, 1)
    assert isclose(amp.normalized_abs, 0.25, rel_tol=0, abs_tol=1e-12)
    assert isclose(amp.phase, pi, rel_tol=0, abs_tol=1e-12)


def test_sparse_block_encoding_query_payload():
    bundle = SparseOracleBundle.from_functions(
        n_rows=2,
        n_cols=2,
        max_row_nnz=2,
        max_col_nnz=2,
        row_to_col_fn=lambda row, l_pos: l_pos,
        col_to_row_fn=lambda col, l_pos: l_pos,
        entry_fn=lambda i, j: 0.5 if i == j else 0.0,
        value_bits=10,
        frac_bits=8,
        alpha=1.0,
    )
    encoding = SparseMatrixBlockEncoding(bundle=bundle)

    payload = encoding.query(BlockEncodingQuery(step=3, parameters={"row": 1, "l_pos": 1}))
    assert payload["op"] == "sparse_block_encoding_query"
    assert payload["row"] == 1
    assert payload["col"] == 1
    assert isclose(payload["value"], 0.5, rel_tol=0, abs_tol=1e-9)
