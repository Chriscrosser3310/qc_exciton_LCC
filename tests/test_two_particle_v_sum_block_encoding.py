from __future__ import annotations

import numpy as np
import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran
cirq = pytest.importorskip("cirq")
_ = cirq

from qualtran.cirq_interop import BloqAsCirqGate, get_named_qubits

from block_encoding.two_particle_v_sum_block_encoding import (
    TwoParticleControlledVSumBlockEncoding,
    TwoParticleVSumBlockEncoding,
    build_two_particle_controlled_v_sum_block_encoding,
    build_two_particle_v_sum_block_encoding,
)


def test_two_particle_v_sum_block_encoding_signature():
    v = 0.2 * np.ones((2, 2, 1, 1), dtype=np.float64)
    bloq = build_two_particle_v_sum_block_encoding(
        num_pairs=2,
        D=1,
        L=2,
        R_c=0,
        R_loc=0,
        V_table=v,
        entry_bitsize=8,
    )
    assert isinstance(bloq, TwoParticleVSumBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["q", "sel", "m", "l", "r0", "r1", "r2", "r3"]
    assert bloq.term_count == 4  # m*m


def test_two_particle_v_sum_block_encoding_m1_reduces_to_v():
    # m=1 => only cross term (0,1), anchors are (0,1), no routing/no sign.
    v = np.array(
        [
            [[[0.2]], [[0.3]]],
            [[[0.4]], [[0.7]]],
        ],
        dtype=np.float64,
    )
    bloq = build_two_particle_v_sum_block_encoding(
        num_pairs=1,
        D=1,
        L=2,
        R_c=0,
        R_loc=0,
        V_table=v,
        entry_bitsize=14,
    )

    named = get_named_qubits(bloq.signature.lefts())
    reg_qubits = {}
    qorder = []
    for reg in bloq.signature.lefts():
        arr = np.array(named[reg.name], dtype=object).reshape(-1)
        reg_qubits[reg.name] = arr
        qorder.extend(arr.tolist())

    u = cirq.unitary(BloqAsCirqGate(bloq).on(*qorder))

    def _bits_of(val: int, n: int) -> list[int]:
        return [int((val >> (n - 1 - k)) & 1) for k in range(n)]

    def _basis_index(q: int, sel: int, m: int, l: int, r0: int, r1: int) -> int:
        bits = []
        bits.extend(_bits_of(q, len(reg_qubits["q"])))
        bits.extend(_bits_of(sel, len(reg_qubits["sel"])))
        bits.extend(_bits_of(m, len(reg_qubits["m"])))
        bits.extend(_bits_of(l, len(reg_qubits["l"])))
        bits.extend(_bits_of(r0, len(reg_qubits["r0"])))
        bits.extend(_bits_of(r1, len(reg_qubits["r1"])))
        return int(cirq.big_endian_bits_to_int(bits))

    block = np.zeros((4, 4), dtype=np.complex128)
    for out_idx in range(4):
        i_out, j_out = (out_idx >> 1) & 1, out_idx & 1
        for in_idx in range(4):
            i_in, j_in = (in_idx >> 1) & 1, in_idx & 1
            r = _basis_index(0, 0, 0, 0, i_out, j_out)
            c = _basis_index(0, 0, 0, 0, i_in, j_in)
            block[out_idx, in_idx] = u[r, c]

    expected_diag = np.array([v[0, 0, 0, 0], v[0, 1, 0, 0], v[1, 0, 0, 0], v[1, 1, 0, 0]])
    expected = np.diag(expected_diag.astype(np.complex128))
    assert np.allclose(block, expected, atol=1e-4)


def test_two_particle_v_sum_block_encoding_controlled_specialization():
    v = 0.2 * np.ones((2, 2, 1, 1), dtype=np.float64)
    bloq = build_two_particle_v_sum_block_encoding(
        num_pairs=1, D=1, L=2, R_c=0, R_loc=0, V_table=v, entry_bitsize=8
    )
    ctrl = bloq.controlled()
    assert isinstance(ctrl, TwoParticleControlledVSumBlockEncoding)

    direct_ctrl = build_two_particle_controlled_v_sum_block_encoding(
        num_pairs=1, D=1, L=2, R_c=0, R_loc=0, V_table=v, entry_bitsize=8
    )
    assert isinstance(direct_ctrl, TwoParticleControlledVSumBlockEncoding)
