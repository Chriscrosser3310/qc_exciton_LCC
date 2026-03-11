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
    one_particle_sparse_block_encoding,
    two_particle_sparse_block_encoding,
)


def _bits(value: int, width: int) -> np.ndarray:
    return np.array([int(b) for b in format(value, f"0{width}b")], dtype=int)


def test_one_particle_sparse_block_encoding_r0_matches_entry_column0():
    # D=1, L=4, R_loc=0 => m has a single logical state.
    L = 4
    R_loc = 0
    table = np.array(
        [
            [0.10],
            [0.25],
            [0.55],
            [0.80],
        ],
        dtype=float,
    )
    i_val = 2

    # wires: m[0,1], i[2,3], anc[4], angle[5..9], work[10]
    dev = qml.device("default.qubit", wires=11)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(_bits(i_val, 2), wires=[2, 3])
        # m register left at |00>, valid for single-state R_loc=0 convention.
        one_particle_sparse_block_encoding(
            table=table,
            i_wires=[[2, 3]],
            m_wires=[[0, 1]],
            ancilla_wire=4,
            angle_wires=[5, 6, 7, 8, 9],
            L=L,
            R_loc=R_loc,
            index_work_wires=[10],
            entry_work_wires=[],
        )
        return qml.probs(wires=[4])

    p = circuit()
    assert abs(float(p[0]) - float(table[i_val, 0] ** 2)) < 0.06


def test_two_particle_sparse_block_encoding_r0_matches_entry_column0():
    # D=1, L=4, R_c=R_loc=0 => m,l each have a single logical state.
    L = 4
    R_c = 0
    R_loc = 0

    # Row index is packed as [i_bits, j_bits] => row = i*L + j for 2-bit i,j.
    table = np.linspace(0.05, 0.95, num=16, dtype=float).reshape(16, 1)
    i_val = 1
    j_val = 3
    row = i_val * L + j_val

    # wires: m[0,1], l[2,3], i[4,5], j[6,7], anc[8], angle[9..13], work[14]
    dev = qml.device("default.qubit", wires=15)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(_bits(i_val, 2), wires=[4, 5])
        qml.BasisState(_bits(j_val, 2), wires=[6, 7])
        two_particle_sparse_block_encoding(
            table=table,
            i_wires=[[4, 5]],
            j_wires=[[6, 7]],
            m_wires=[[0, 1]],
            l_wires=[[2, 3]],
            ancilla_wire=8,
            angle_wires=[9, 10, 11, 12, 13],
            L=L,
            R_c=R_c,
            R_loc=R_loc,
            index_work_wires=[14],
            entry_work_wires=[],
        )
        return qml.probs(wires=[8])

    p = circuit()
    assert abs(float(p[0]) - float(table[row, 0] ** 2)) < 0.06

