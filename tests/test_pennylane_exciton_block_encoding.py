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

from integrations.pennylane.block_encoding import exciton_block_encoding


def test_exciton_block_encoding_smoke_small():
    # Smallest meaningful setup with m=1 (2 particle registers), D=1, L=2.
    # R_c=R_loc=0 keeps selector spaces small and oracle tables compact.
    F_table = np.array([[0.2], [0.7]], dtype=float)      # shape (L^D, 1) = (2,1)
    W_table = np.array([[0.1], [0.3], [0.4], [0.8]], dtype=float)  # (L^(2D),1) = (4,1)
    V_table = np.array([[0.2], [0.1], [0.9], [0.6]], dtype=float)  # (4,1)

    # Wires layout
    # particle regs: r0=[0], r1=[1] (2m=2 registers, D=1, 1 bit per index)
    particle_registers = [
        [[0]],
        [[1]],
    ]
    # selectors
    h_sel = [2, 3]    # 2 bits for 3 terms
    f_sel = [4]       # 1 bit for 2 terms
    w_sel: list[int] = []  # 1 term only
    v_sel: list[int] = []  # 1 term only
    # local index registers
    f_m = [[5]]
    w_m = [[6]]
    w_l = [[7]]
    v_m = [[8]]
    v_l = [[9]]
    anc = 10
    angle = [11, 12, 13, 14]
    work = []  # 1-bit index registers need no SemiAdder work wires

    dev = qml.device("default.qubit", wires=15)

    @qml.qnode(dev)
    def circuit():
        exciton_block_encoding(
            F_table=F_table,
            W_table=W_table,
            V_table=V_table,
            particle_registers=particle_registers,
            h_sel_wires=h_sel,
            f_sel_wires=f_sel,
            w_sel_wires=w_sel,
            v_sel_wires=v_sel,
            f_m_wires=f_m,
            w_m_wires=w_m,
            w_l_wires=w_l,
            v_m_wires=v_m,
            v_l_wires=v_l,
            ancilla_wire=anc,
            angle_wires=angle,
            L=2,
            R_c=0,
            R_loc=0,
            index_work_wires=work,
            entry_work_wires=[],
        )
        return qml.probs(wires=[anc])

    p = circuit()
    assert np.isclose(float(np.sum(p)), 1.0)
    assert p.shape == (2,)

