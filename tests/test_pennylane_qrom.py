from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


qml = pytest.importorskip("pennylane")

from integrations.pennylane import qrom_table_2d


def _bits(value: int, width: int) -> np.ndarray:
    return np.array([int(b) for b in format(value, f"0{width}b")], dtype=int)


def test_qrom_table_2d_loads_value_on_zero_data_register():
    table = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    i_wires = [0]
    j_wires = [1, 2]
    data_wires = [3, 4, 5]

    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def circuit(i_val: int, j_val: int):
        qml.BasisState(_bits(i_val, len(i_wires)), wires=i_wires)
        qml.BasisState(_bits(j_val, len(j_wires)), wires=j_wires)
        qrom_table_2d(table, i_wires=i_wires, j_wires=j_wires, data_wires=data_wires)
        return qml.probs(wires=data_wires)

    probs = circuit(1, 2)
    loaded = int(np.argmax(probs))
    assert loaded == 6


def test_qrom_table_2d_supports_n_by_1_with_no_j_wires():
    table = np.array([[1], [3], [2], [7]], dtype=int)  # shape (4, 1)
    i_wires = [0, 1]       # ceil(log2(4)) = 2
    j_wires: list[int] = []  # allowed since M = 1
    data_wires = [2, 3, 4]   # enough for value 7

    dev = qml.device("default.qubit", wires=5)

    @qml.qnode(dev)
    def circuit(i_val: int):
        qml.BasisState(_bits(i_val, len(i_wires)), wires=i_wires)
        qrom_table_2d(table, i_wires=i_wires, j_wires=j_wires, data_wires=data_wires)
        return qml.probs(wires=data_wires)

    for i_val in range(table.shape[0]):
        probs = circuit(i_val)
        loaded = int(np.argmax(probs))
        assert loaded == int(table[i_val, 0])
