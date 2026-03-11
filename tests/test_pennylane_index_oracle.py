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

from integrations.pennylane.block_encoding import (
    one_particle_index_oracle,
    two_particle_index_oracle,
)


def _bits(value: int, width: int) -> np.ndarray:
    return np.array([int(b) for b in format(value, f"0{width}b")], dtype=int)


def test_one_particle_index_oracle_d1():
    # i' = i - R_loc + m (mod L)
    L = 8
    R_loc = 2
    n = 3
    m_val = 1
    i_val = 6
    expected = (i_val - R_loc + m_val) % L

    # wires: m[0:3], i[3:6], work[6:8]
    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(_bits(m_val, n), wires=[0, 1, 2])
        qml.BasisState(_bits(i_val, n), wires=[3, 4, 5])
        one_particle_index_oracle(
            i_wires=[3, 4, 5],
            m_wires=[0, 1, 2],
            L=L,
            R_loc=R_loc,
            work_wires=[6, 7],
        )
        return qml.probs(wires=[3, 4, 5])

    out = int(np.argmax(circuit()))
    assert out == expected


def test_two_particle_index_oracle_d1():
    # i' = i - R_c + m
    # j' = j - R_c + m - R_loc + l
    L = 8
    R_c = 1
    R_loc = 2
    n = 3

    m_val = 3
    l_val = 1
    i_val = 5
    j_val = 4
    expected_i = (i_val - R_c + m_val) % L
    expected_j = (j_val - R_c + m_val - R_loc + l_val) % L

    # wires: m[0:3], l[3:6], i[6:9], j[9:12], work[12:14]
    dev = qml.device("default.qubit", wires=14)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(_bits(m_val, n), wires=[0, 1, 2])
        qml.BasisState(_bits(l_val, n), wires=[3, 4, 5])
        qml.BasisState(_bits(i_val, n), wires=[6, 7, 8])
        qml.BasisState(_bits(j_val, n), wires=[9, 10, 11])
        two_particle_index_oracle(
            i_wires=[6, 7, 8],
            j_wires=[9, 10, 11],
            m_wires=[0, 1, 2],
            l_wires=[3, 4, 5],
            L=L,
            R_c=R_c,
            R_loc=R_loc,
            work_wires=[12, 13],
        )
        return qml.probs(wires=[6, 7, 8]), qml.probs(wires=[9, 10, 11])

    p_i, p_j = circuit()
    out_i = int(np.argmax(p_i))
    out_j = int(np.argmax(p_j))
    assert out_i == expected_i
    assert out_j == expected_j
