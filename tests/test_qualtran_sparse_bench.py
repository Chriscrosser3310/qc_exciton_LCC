from __future__ import annotations

import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran

from integrations.qualtran.block_encoding.sparse_bench import (
    build_all_sparse_oracles,
    build_product_matrices_from_tensors,
    build_thresholded_benchmark_tensors,
)


def test_thresholded_tensor_shapes():
    f, v = build_thresholded_benchmark_tensors(shape=(2,), epsilon=0.15)
    assert f.shape == (2, 2)
    assert v.shape == (2, 2, 2, 2)


def test_product_layout_shapes():
    f, v = build_thresholded_benchmark_tensors(shape=(2,), epsilon=0.15)
    m1, m2 = build_product_matrices_from_tensors(f, v)
    assert m1.shape == (4, 4)
    assert m2.shape == (4, 4)


def test_build_all_sparse_oracles():
    bundles = build_all_sparse_oracles(shape=(2,), epsilon=0.2)
    assert set(bundles.keys()) == {"F_pq", "FV_pq_rs", "FV_pr_qs"}
    assert bundles["F_pq"].matrix.shape == (2, 2)
    assert bundles["FV_pq_rs"].matrix.shape == (4, 4)
    assert bundles["FV_pr_qs"].matrix.shape == (4, 4)
    idx = bundles["F_pq"].polylog_index
    assert idx.row_lth_nonzero(0, 0) in (0, 1)
    assert idx.col_lth_nonzero(0, 0) in (0, 1)
