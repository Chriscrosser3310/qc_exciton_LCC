from __future__ import annotations

import numpy as np

from block_encoding.exciton_hamiltonian_encoding import (
    build_exciton_hamiltonian_encoding,
    build_f_and_v_block_encodings,
)


def test_build_f_and_v_block_encodings_shapes():
    f_be, vd_be, vx_be = build_f_and_v_block_encodings(
        lattice_shape=(3,),
        r_loc=1,
        r_c=1,
        epsilon=0.0,
    )
    assert f_be.matrix_unpadded_dim == 3
    assert vd_be.matrix_unpadded_dim == 9
    assert vx_be.matrix_unpadded_dim == 9


def test_exciton_term_counts_for_m2():
    # m=2:
    # +F first: 2, -F second: 2
    # direct first pairs: C(2,2)=1
    # direct second pairs: C(2,2)=1
    # cross pairs: 2*2=4 each contributes (+exchange) + (-direct) => 8
    # total = 2 + 2 + 1 + 1 + 8 = 14
    enc = build_exciton_hamiltonian_encoding(
        m=2,
        lattice_shape=(3,),
        r_loc=1,
        r_c=1,
        epsilon=0.0,
    )
    assert len(enc.terms) == 14

    first_f = [t for t in enc.terms if t.op_kind == "F" and t.set_label == "first"]
    second_f = [t for t in enc.terms if t.op_kind == "F" and t.set_label == "second"]
    cross_ex = [t for t in enc.terms if t.op_kind == "V_exchange" and t.set_label == "cross"]
    cross_dir = [
        t
        for t in enc.terms
        if t.op_kind == "V_direct" and t.set_label == "cross" and t.coefficient < 0
    ]
    assert len(first_f) == 2
    assert all(t.coefficient > 0 for t in first_f)
    assert len(second_f) == 2
    assert all(t.coefficient < 0 for t in second_f)
    assert len(cross_ex) == 4
    assert len(cross_dir) == 4


def test_exciton_builder_defaults_match_explicit_maxdist():
    # shape=(3,) => max_dist=2 (non-periodic)
    enc_default = build_exciton_hamiltonian_encoding(
        m=1,
        lattice_shape=(3,),
        epsilon=0.0,
    )
    enc_explicit = build_exciton_hamiltonian_encoding(
        m=1,
        lattice_shape=(3,),
        r_loc=2,
        r_c=2,
        epsilon=0.0,
    )
    assert np.allclose(enc_default.f_bundle.matrix, enc_explicit.f_bundle.matrix)
    assert np.allclose(enc_default.v_direct_bundle.matrix, enc_explicit.v_direct_bundle.matrix)
