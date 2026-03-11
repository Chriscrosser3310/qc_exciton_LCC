from __future__ import annotations

import numpy as np
import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran

from integrations.qualtran.block_encoding.qrom_sparse import SparseTensorCOO, qrom_build_from_sparse_data, sparse_coo_from_items


def test_sparse_tensor_coo_to_dense():
    sparse = sparse_coo_from_items(
        shape=(4, 4),
        items=[((0, 1), 5), ((2, 3), 7)],
    )
    dense = sparse.to_dense(fill_value=0)
    assert dense.shape == (4, 4)
    assert dense[0, 1] == 5
    assert dense[2, 3] == 7
    assert dense[1, 1] == 0


def test_qrom_build_from_sparse_matches_dense():
    dense = np.zeros((4, 4), dtype=int)
    dense[0, 1] = 5
    dense[2, 3] = 7
    sparse = SparseTensorCOO.from_dense(dense)

    q_sparse = qrom_build_from_sparse_data(sparse)
    from qualtran.bloqs.data_loading.qrom import QROM

    q_dense = QROM.build_from_data(dense)
    assert np.array_equal(q_sparse.data[0], q_dense.data[0])


def test_qrom_build_from_mixed_sparse_and_dense():
    sparse = sparse_coo_from_items(shape=(8,), items=[((3,), 9), ((5,), 2)])
    dense = np.arange(8, dtype=int)
    qrom = qrom_build_from_sparse_data(sparse, dense)
    assert len(qrom.data) == 2
    assert qrom.data[0].shape == (8,)
    assert qrom.data[1].shape == (8,)
    assert qrom.data[0][3] == 9
    assert qrom.data[0][5] == 2
    assert qrom.data[0][0] == 0
