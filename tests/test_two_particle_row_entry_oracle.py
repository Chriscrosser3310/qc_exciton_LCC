from __future__ import annotations

import numpy as np
import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran

from block_encoding.two_particle_row_oracles import (
    OneParticleControlledSparseBlockEncoding,
    OneParticleRowEntryOracle,
    OneParticleRowIndexOracle,
    OneParticleSparseBlockEncoding,
    TwoParticleControlledSparseBlockEncoding,
    TwoParticleRowEntryOracle,
    TwoParticleRowIndexOracle,
    TwoParticleSparseBlockEncoding,
    build_one_particle_controlled_sparse_block_encoding,
    build_one_particle_row_entry_oracle,
    build_one_particle_row_index_oracle,
    build_one_particle_sparse_block_encoding,
    build_two_particle_controlled_sparse_block_encoding,
    build_two_particle_sparse_block_encoding,
    build_two_particle_row_entry_oracle,
    build_two_particle_row_index_oracle,
)


def test_two_particle_row_entry_oracle_shape_and_value():
    d, l, rc, rloc = 2, 3, 1, 1
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)
    m[1, 2, 0, 1, 1, 2, 0, 1] = 0.3

    oracle = build_two_particle_row_entry_oracle(m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=8)
    assert isinstance(oracle, TwoParticleRowEntryOracle)
    assert oracle.entry_value(i=(1, 2), j=(0, 1), m=(1, 2), l=(0, 1)) == pytest.approx(0.3)


def test_two_particle_row_entry_oracle_signature_names():
    d, l, rc, rloc = 2, 3, 1, 1
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)
    oracle = build_two_particle_row_entry_oracle(m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=8)

    reg_names = [r.name for r in oracle.signature]
    assert reg_names == ["q", "m", "l", "i", "j"]


def test_two_particle_row_index_oracle_classical_map():
    # L=8 so decomposition modulus aligns with exact modulus.
    oracle = build_two_particle_row_index_oracle(D=2, L=8, R_c=2, R_loc=1)
    assert isinstance(oracle, TwoParticleRowIndexOracle)

    m = np.array([0, 4], dtype=int)
    l = np.array([1, 2], dtype=int)
    i = np.array([3, 7], dtype=int)
    j = np.array([6, 0], dtype=int)

    m_o, l_o, i_p, j_p = oracle.call_classically(m=m, l=l, i=i, j=j)
    assert np.array_equal(m_o, m)
    assert np.array_equal(l_o, l)

    # i' = i - R_c + m mod L
    assert np.array_equal(i_p, np.array([(3 - 2 + 0) % 8, (7 - 2 + 4) % 8], dtype=int))
    # j' = j - R_c + m - R_loc + l mod L
    assert np.array_equal(
        j_p,
        np.array([(6 - 2 + 0 - 1 + 1) % 8, (0 - 2 + 4 - 1 + 2) % 8], dtype=int),
    )


def test_two_particle_sparse_block_encoding_signature_and_cost():
    # L=4 keeps index-oracle decomposition valid (power-of-two modulus path).
    d, l, rc, rloc = 1, 4, 1, 1
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_two_particle_sparse_block_encoding(
        m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=6
    )
    assert isinstance(bloq, TwoParticleSparseBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["q", "m", "l", "i", "j"]

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    qec = get_cost_value(bloq, QECGatesCost())
    assert qec is not None


def test_two_particle_controlled_sparse_block_encoding_signature_and_cost():
    d, l, rc, rloc = 1, 4, 1, 1
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_two_particle_controlled_sparse_block_encoding(
        m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=6
    )
    assert isinstance(bloq, TwoParticleControlledSparseBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["ctrl", "q", "m", "l", "i", "j"]


def test_two_particle_sparse_block_encoding_uses_specialized_control():
    d, l, rc, rloc = 1, 4, 1, 1
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_two_particle_sparse_block_encoding(
        m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=6
    )
    ctrl = bloq.controlled()
    assert isinstance(ctrl, TwoParticleControlledSparseBlockEncoding)


def test_two_particle_sparse_block_encoding_supports_zero_radii():
    # Exercise R_c=0 and R_loc=0 edge case.
    d, l, rc, rloc = 1, 4, 0, 0
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_two_particle_sparse_block_encoding(
        m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=6
    )
    assert isinstance(bloq, TwoParticleSparseBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["q", "m", "l", "i", "j"]

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    qec = get_cost_value(bloq, QECGatesCost())
    assert qec is not None


def test_entry_oracle_theta_quantization_uses_rounding():
    # Regression: theta quantization should use nearest integer, not floor.
    d, l, rc, rloc = 1, 2, 1, 1
    shape = (l,) * d + (l,) * d + (2 * rc + 1,) * d + (2 * rloc + 1,) * d
    m = np.full(shape, 0.5, dtype=np.float64)
    target_val = 0.13533528
    m[0, 0, 0, 0] = target_val
    bits = 8

    oracle = build_two_particle_row_entry_oracle(
        m, D=d, L=l, R_c=rc, R_loc=rloc, entry_bitsize=bits
    )
    qrom = oracle._qrom_theta
    theta_loaded = int(qrom.data[0][0, 0, 0, 0])
    theta_expected = int(np.rint(np.arccos(target_val) / np.pi * (2**bits)))

    assert theta_loaded == theta_expected


def test_one_particle_row_index_oracle_classical_map():
    oracle = build_one_particle_row_index_oracle(D=2, L=8, R_loc=2)
    assert isinstance(oracle, OneParticleRowIndexOracle)

    m = np.array([0, 3], dtype=int)
    i = np.array([4, 7], dtype=int)
    m_o, i_p = oracle.call_classically(m=m, i=i)

    assert np.array_equal(m_o, m)
    assert np.array_equal(i_p, np.array([(4 - 2 + 0) % 8, (7 - 2 + 3) % 8], dtype=int))


def test_one_particle_row_entry_oracle_signature_names():
    d, l, rloc = 2, 4, 1
    shape = (l,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)
    oracle = build_one_particle_row_entry_oracle(m, D=d, L=l, R_loc=rloc, entry_bitsize=8)

    assert isinstance(oracle, OneParticleRowEntryOracle)
    reg_names = [r.name for r in oracle.signature]
    assert reg_names == ["q", "m", "i"]


def test_one_particle_sparse_block_encoding_signature_and_cost():
    d, l, rloc = 1, 4, 1
    shape = (l,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_one_particle_sparse_block_encoding(
        m, D=d, L=l, R_loc=rloc, entry_bitsize=6
    )
    assert isinstance(bloq, OneParticleSparseBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["q", "m", "i"]

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    qec = get_cost_value(bloq, QECGatesCost())
    assert qec is not None


def test_one_particle_controlled_sparse_block_encoding_signature_and_cost():
    d, l, rloc = 1, 4, 1
    shape = (l,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_one_particle_controlled_sparse_block_encoding(
        m, D=d, L=l, R_loc=rloc, entry_bitsize=6
    )
    assert isinstance(bloq, OneParticleControlledSparseBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["ctrl", "q", "m", "i"]

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    qec = get_cost_value(bloq, QECGatesCost())
    assert qec is not None


def test_one_particle_sparse_block_encoding_uses_specialized_control():
    d, l, rloc = 1, 4, 1
    shape = (l,) * d + (2 * rloc + 1,) * d
    m = 0.1 * np.ones(shape, dtype=np.float64)

    bloq = build_one_particle_sparse_block_encoding(
        m, D=d, L=l, R_loc=rloc, entry_bitsize=6
    )
    ctrl = bloq.controlled()
    assert isinstance(ctrl, OneParticleControlledSparseBlockEncoding)


def test_one_particle_sparse_block_encoding_matrix_block():
    cirq = pytest.importorskip("cirq")
    from qualtran.cirq_interop import BloqAsCirqGate, get_named_qubits

    # Small exact check: with R_loc=0, m is fixed at |0> and index map is identity.
    # The postselected block <0_q,0_m|U|0_q,0_m> on i should match diag(M[:, 0]).
    d, l, rloc = 1, 2, 0
    m_table = np.array([[0.2], [0.7]], dtype=np.float64)
    bloq = build_one_particle_sparse_block_encoding(
        m_table, D=d, L=l, R_loc=rloc, entry_bitsize=16
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

    def _basis_index(q_val: int, m_val: int, i_val: int) -> int:
        bits = []
        bits.extend(_bits_of(q_val, len(reg_qubits["q"])))
        bits.extend(_bits_of(m_val, len(reg_qubits["m"])))
        bits.extend(_bits_of(i_val, len(reg_qubits["i"])))
        return int(cirq.big_endian_bits_to_int(bits))

    block = np.zeros((l, l), dtype=np.complex128)
    for i_out in range(l):
        for i_in in range(l):
            r = _basis_index(0, 0, i_out)
            c = _basis_index(0, 0, i_in)
            block[i_out, i_in] = u[r, c]

    expected = np.diag([0.2, 0.7]).astype(np.complex128)
    assert np.allclose(block, expected, atol=5e-5)
