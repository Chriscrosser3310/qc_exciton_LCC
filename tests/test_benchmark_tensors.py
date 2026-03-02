from __future__ import annotations

import numpy as np

from exciton.benchmark_tensors import generate_f_tensor, generate_v_tensor


def test_generate_f_tensor_1d_chain():
    f = generate_f_tensor(shape=(3,), a=1.0, metric="manhattan")
    expected = np.array(
        [
            [1.0, np.exp(-1.0), np.exp(-2.0)],
            [np.exp(-1.0), 1.0, np.exp(-1.0)],
            [np.exp(-2.0), np.exp(-1.0), 1.0],
        ]
    )
    assert f.shape == (3, 3)
    assert np.allclose(f, expected)


def test_generate_v_tensor_formula_samples():
    a, b, c = 1.0, 1.0, 1.0
    v = generate_v_tensor(shape=(3,), a=a, b=b, c=c, metric="manhattan")
    assert v.shape == (3, 3, 3, 3)

    # p=0, q=1, r=0, s=2:
    # R_pr=0, R_qs=1, R_pq=1, delta_pr=1, delta_qs=0
    # => exp(-a*0)*exp(-b*1)*(c*1)^(2) = exp(-1)
    assert np.isclose(v[0, 1, 0, 2], np.exp(-1.0))

    # If R_pq = 0 and exponent > 0, contribution is zero.
    assert np.isclose(v[0, 0, 1, 2], 0.0)


def test_generate_f_tensor_hard_cutoff():
    # 1D non-periodic chain: site 0 to site 2 has max distance 2.
    f = generate_f_tensor(shape=(3,), a=1.0, metric="manhattan", r_cut=1, periodic_cutoff=False)
    assert np.isclose(f[0, 1], np.exp(-1.0))
    assert np.isclose(f[0, 2], 0.0)


def test_generate_v_tensor_hard_cutoff():
    # Enforce:
    # max_dist(p,r) <= r_c=1
    # max_dist(p,q), max_dist(r,s) <= r_loc=1
    v = generate_v_tensor(
        shape=(4,),
        a=1.0,
        b=1.0,
        c=1.0,
        metric="manhattan",
        r_loc=1,
        r_c=1,
        periodic_cutoff=False,
    )
    # Violates p-r cutoff: |0-2|=2 > 1.
    assert np.isclose(v[0, 1, 2, 1], 0.0)
    # Violates p-q cutoff: |0-3|=3 > 1.
    assert np.isclose(v[0, 3, 0, 1], 0.0)
    # Violates r-s cutoff: |0-2|=2 > 1.
    assert np.isclose(v[0, 1, 0, 2], 0.0)


def test_default_cutoff_equals_explicit_maxdist():
    # For shape=(4,), max_dist=3 in non-periodic metric.
    f_default = generate_f_tensor(shape=(4,), a=1.0, metric="manhattan")
    f_explicit = generate_f_tensor(shape=(4,), a=1.0, metric="manhattan", r_cut=3)
    assert np.allclose(f_default, f_explicit)

    v_default = generate_v_tensor(shape=(4,), a=1.0, b=1.0, c=1.0, metric="manhattan")
    v_explicit = generate_v_tensor(
        shape=(4,), a=1.0, b=1.0, c=1.0, metric="manhattan", r_loc=3, r_c=3
    )
    assert np.allclose(v_default, v_explicit)
