from __future__ import annotations

import numpy as np
import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran

from integrations.qualtran.block_encoding.exciton_hamiltonian_encoding import (
    ExcitonHamiltonianBlockEncoding,
    build_exciton_hamiltonian_block_encoding,
)
from integrations.qualtran.block_encoding.f_register_sum_block_encoding import OneParticleFSumBlockEncoding
from integrations.qualtran.block_encoding.two_particle_v_sum_block_encoding import TwoParticleVSumBlockEncoding
from integrations.qualtran.block_encoding.two_particle_w_sum_block_encoding import TwoParticleWSumBlockEncoding


def _demo_tables(l: int = 2):
    f = 0.2 * np.ones((l, 1), dtype=np.float64)  # D=1, R_loc=0
    w = 0.3 * np.ones((l, l, 1, 1), dtype=np.float64)  # D=1, R_c=R_loc=0
    v = 0.4 * np.ones((l, l, 1, 1), dtype=np.float64)
    return f, w, v


def test_exciton_hamiltonian_block_encoding_signature_and_components():
    f, w, v = _demo_tables()
    bloq = build_exciton_hamiltonian_block_encoding(
        num_pairs=1,
        D=1,
        L=2,
        R_c=0,
        R_loc=0,
        F=f,
        W=w,
        V=v,
        entry_bitsize=8,
    )
    assert isinstance(bloq, ExcitonHamiltonianBlockEncoding)
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == [
        "q",
        "h_sel",
        "f_sel",
        "f_m",
        "w_sel",
        "w_m",
        "w_l",
        "v_sel",
        "v_m",
        "v_l",
        "r0",
        "r1",
    ]

    assert isinstance(bloq.f_bloq, OneParticleFSumBlockEncoding)
    assert isinstance(bloq.w_bloq, TwoParticleWSumBlockEncoding)
    assert isinstance(bloq.v_bloq, TwoParticleVSumBlockEncoding)


def test_exciton_hamiltonian_block_encoding_builds():
    f, w, v = _demo_tables()
    bloq = build_exciton_hamiltonian_block_encoding(
        num_pairs=1,
        D=1,
        L=2,
        R_c=0,
        R_loc=0,
        F=f,
        W=w,
        V=v,
        entry_bitsize=6,
    )

    # Basic decomposition smoke-test for the new 3-term composition.
    cb = bloq.decompose_bloq()
    assert cb is not None
