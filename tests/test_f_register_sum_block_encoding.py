from __future__ import annotations

import numpy as np
import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran
cirq = pytest.importorskip("cirq")
_ = cirq

from qualtran.cirq_interop import BloqAsCirqGate, get_named_qubits

from integrations.qualtran.block_encoding.f_register_sum_block_encoding import (
    OneParticleControlledFSumBlockEncoding,
    OneParticleFSumBlockEncoding,
    build_one_particle_controlled_f_sum_block_encoding,
    build_one_particle_f_sum_block_encoding,
)


def test_signed_f_register_sum_block_encoding_signature():
    f_table = np.full((2, 1), 0.5, dtype=np.float64)
    bloq = build_one_particle_f_sum_block_encoding(
        num_pairs=1,
        D=1,
        L=2,
        R_loc=0,
        F_table=f_table,
        entry_bitsize=8,
    )
    assert isinstance(bloq, OneParticleFSumBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["q", "sel", "m", "r0", "r1"]


def test_signed_f_register_sum_block_encoding_matrix_block_small():
    # m=1 -> two registers, target block is (F(r0) - F(r1)) / 2.
    f_table = np.array([[0.2], [0.7]], dtype=np.float64)
    bloq = build_one_particle_f_sum_block_encoding(
        num_pairs=1,
        D=1,
        L=2,
        R_loc=0,
        F_table=f_table,
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

    def _basis_index(q: int, sel: int, m: int, r0: int, r1: int) -> int:
        bits = []
        bits.extend(_bits_of(q, len(reg_qubits["q"])))
        bits.extend(_bits_of(sel, len(reg_qubits["sel"])))
        bits.extend(_bits_of(m, len(reg_qubits["m"])))
        bits.extend(_bits_of(r0, len(reg_qubits["r0"])))
        bits.extend(_bits_of(r1, len(reg_qubits["r1"])))
        return int(cirq.big_endian_bits_to_int(bits))

    block = np.zeros((4, 4), dtype=np.complex128)
    for out_idx in range(4):
        r0_out, r1_out = (out_idx >> 1) & 1, out_idx & 1
        for in_idx in range(4):
            r0_in, r1_in = (in_idx >> 1) & 1, in_idx & 1
            r = _basis_index(0, 0, 0, r0_out, r1_out)
            c = _basis_index(0, 0, 0, r0_in, r1_in)
            block[out_idx, in_idx] = u[r, c]

    fdiag = [0.2, 0.7]
    expected_diag = np.array(
        [
            0.5 * (fdiag[0] - fdiag[0]),
            0.5 * (fdiag[0] - fdiag[1]),
            0.5 * (fdiag[1] - fdiag[0]),
            0.5 * (fdiag[1] - fdiag[1]),
        ],
        dtype=np.complex128,
    )
    expected = np.diag(expected_diag)
    assert np.allclose(block, expected, atol=1e-4)


def test_one_particle_f_sum_block_encoding_controlled_specialization():
    f_table = np.full((2, 1), 0.5, dtype=np.float64)
    bloq = build_one_particle_f_sum_block_encoding(
        num_pairs=1, D=1, L=2, R_loc=0, F_table=f_table, entry_bitsize=8
    )
    ctrl = bloq.controlled()
    assert isinstance(ctrl, OneParticleControlledFSumBlockEncoding)

    direct_ctrl = build_one_particle_controlled_f_sum_block_encoding(
        num_pairs=1, D=1, L=2, R_loc=0, F_table=f_table, entry_bitsize=8
    )
    assert isinstance(direct_ctrl, OneParticleControlledFSumBlockEncoding)
