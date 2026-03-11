from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

qml = pytest.importorskip("pennylane")

from integrations.pennylane.resource_estimation import compiled_counts_from_qnode


def test_compiled_counts_from_qnode_toffoli():
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.Toffoli(wires=[0, 1, 2])
        return qml.probs(wires=[0])

    counts = compiled_counts_from_qnode(circuit)
    assert counts.total_t == 7
    assert counts.cnot_count == 6
    assert counts.clifford_count == 8
    assert counts.rotation_count == 0
