from __future__ import annotations

import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran

from block_encoding.qualtran_lattice_index_oracles import (
    SingleParticleSparseIndexOracle,
    TwoParticleSparseIndexOracle,
    build_lattice_sparse_index_oracles,
)


def test_single_particle_oracle_basic():
    oracle = SingleParticleSparseIndexOracle(lattice_shape=(4,), r_loc=1)
    # For 1D chain with periodic offsets [-1,0,1], l=0 should map to i-1.
    j, i_out = oracle.call_classically(l=0, i=0)
    assert i_out == 0
    assert j == 3
    j1, _ = oracle.call_classically(l=1, i=0)
    assert j1 == 0
    j2, _ = oracle.call_classically(l=2, i=0)
    assert j2 == 1


def test_single_particle_oracle_2d_num_nonzero():
    oracle = SingleParticleSparseIndexOracle(lattice_shape=(4, 4), r_loc=1)
    assert int(oracle.num_nonzero) == 9


def test_two_particle_oracle_pair_packing():
    oracle = TwoParticleSparseIndexOracle(lattice_shape=(4,), r_loc=1, r_c=1)
    packed, i_out, ip_out = oracle.call_classically(l=0, i=0, i_prime=1)
    assert i_out == 0
    assert ip_out == 1
    j, jp = oracle.decode_pair(packed)
    # l=0 chooses left-most offsets for both levels in this ordering.
    assert j == 3
    assert jp == 2


def test_factory():
    single, two = build_lattice_sparse_index_oracles(lattice_shape=(4, 4), r_loc=1, r_c=2)
    assert isinstance(single, SingleParticleSparseIndexOracle)
    assert isinstance(two, TwoParticleSparseIndexOracle)
