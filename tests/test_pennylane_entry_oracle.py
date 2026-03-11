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

from integrations.pennylane.block_encoding import controlled_entry_oracle_2d, entry_oracle_2d


def _bits(value: int, width: int) -> np.ndarray:
    if width == 0:
        return np.array([], dtype=int)
    return np.array([int(b) for b in format(value, f"0{width}b")], dtype=int)


def test_entry_oracle_2d_probability_matches_aij_squared():
    # Keep values in [0,1] for simple probability check.
    table = np.array([[0.1, 0.4], [0.7, 0.9]], dtype=float)

    i_wires = [0]
    j_wires = [1]
    ancilla = 2
    angle_wires = [3, 4, 5, 6]  # angle precision bits

    dev = qml.device("default.qubit", wires=7)

    @qml.qnode(dev)
    def circuit(i_val: int, j_val: int):
        qml.BasisState(_bits(i_val, len(i_wires)), wires=i_wires)
        qml.BasisState(_bits(j_val, len(j_wires)), wires=j_wires)
        entry_oracle_2d(
            table=table,
            i_wires=i_wires,
            j_wires=j_wires,
            ancilla_wire=ancilla,
            angle_wires=angle_wires,
            work_wires=[],
        )
        return qml.probs(wires=[ancilla])

    # Quantization error tolerance from 4-bit fixed-point angle.
    tol = 0.12
    for i in range(2):
        for j in range(2):
            probs = circuit(i, j)
            p0 = float(probs[0])
            expected = float(table[i, j] ** 2)
            assert abs(p0 - expected) < tol


def test_controlled_entry_oracle_2d_only_controls_rotation_ladder():
    table = np.array([[0.6, 0.2], [0.1, 0.8]], dtype=float)

    control_wire = 0
    i_wires = [1]
    j_wires = [2]
    ancilla = 3
    angle_wires = [4, 5, 6, 7]

    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev)
    def circuit(ctrl_val: int, i_val: int, j_val: int):
        qml.BasisState(_bits(ctrl_val, 1), wires=[control_wire])
        qml.BasisState(_bits(i_val, len(i_wires)), wires=i_wires)
        qml.BasisState(_bits(j_val, len(j_wires)), wires=j_wires)
        controlled_entry_oracle_2d(
            table=table,
            i_wires=i_wires,
            j_wires=j_wires,
            ancilla_wire=ancilla,
            angle_wires=angle_wires,
            control_wires=[control_wire],
            control_values=[1],
            work_wires=[],
        )
        return qml.probs(wires=[ancilla]), qml.probs(wires=angle_wires)

    tol = 0.12
    for i in range(2):
        for j in range(2):
            anc_off, angle_off = circuit(0, i, j)
            assert float(anc_off[0]) == pytest.approx(1.0, abs=1e-9)
            assert float(angle_off[0]) == pytest.approx(1.0, abs=1e-9)

            anc_on, angle_on = circuit(1, i, j)
            expected = float(table[i, j] ** 2)
            assert abs(float(anc_on[0]) - expected) < tol
            assert float(angle_on[0]) == pytest.approx(1.0, abs=1e-9)
