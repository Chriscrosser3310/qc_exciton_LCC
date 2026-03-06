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
    # R_pr=0, R_qs=1, delta_pr=1, delta_qs=0.
    # Center(p,r)=0.0, Center(q,s)=1.5 -> R_(p,r),(q,s)=1.5.
    # exponent=-(3-1-0)=-2
    # => exp(-1) * (1.5)^(-2)
    assert np.isclose(v[0, 1, 0, 2], np.exp(-1.0) * (1.5 ** -2))

    # If center distance is zero and exponent < 0, contribution is set to zero_distance_value.
    # Default zero_distance_value is 2*c, and prefactor is exp(-a*2) * exp(-b*0) = exp(-2).
    # p=0,r=2 -> center=1 ; q=1,s=1 -> center=1 ; delta_pr=0, delta_qs=1 => exponent=-2.
    assert np.isclose(v[0, 1, 2, 1], (2.0 * c) * np.exp(-2.0))

    # Tunable override for center distance zero.
    v2 = generate_v_tensor(
        shape=(3,),
        a=a,
        b=b,
        c=c,
        metric="manhattan",
        zero_distance_value=5.0,
    )
    assert np.isclose(v2[0, 1, 2, 1], 5.0 * np.exp(-2.0))


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


def test_generate_tensors_accept_l_and_d():
    f_shape = generate_f_tensor(shape=(3,), a=1.0, metric="manhattan")
    f_ld = generate_f_tensor(L=3, D=1, a=1.0, metric="manhattan")
    assert np.allclose(f_shape, f_ld)

    v_shape = generate_v_tensor(shape=(3,), a=1.0, b=1.0, c=1.0, metric="manhattan")
    v_ld = generate_v_tensor(L=3, D=1, a=1.0, b=1.0, c=1.0, metric="manhattan")
    assert np.allclose(v_shape, v_ld)


def test_generate_f_tensor_row_oracle_convention_shape_and_values():
    l = 4
    rloc = 1
    f_full = generate_f_tensor(shape=(l,), a=1.0, metric="manhattan", r_cut=rloc)
    f_row = generate_f_tensor(
        shape=(l,),
        a=1.0,
        metric="manhattan",
        r_cut=rloc,
        oracle_convention="row_oracle",
    )

    assert f_row.shape == (l, 2 * rloc + 1)

    # p=2, m=0 => q=(2-1+0)%4 = 1
    p = 2
    m = 0
    q = (p - rloc + m) % l
    assert np.isclose(f_row[p, m], f_full[p, q])


def test_generate_f_tensor_row_oracle_convention_cutoff_default_matches_explicit():
    l = 4
    # r_cut defaults to max distance for non-periodic, which is 3 in 1D with L=4.
    f_default = generate_f_tensor(shape=(l,), a=1.0, metric="manhattan", oracle_convention="row_oracle")
    f_explicit = generate_f_tensor(
        shape=(l,),
        a=1.0,
        metric="manhattan",
        r_cut=3,
        oracle_convention="row_oracle",
    )
    assert np.allclose(f_default, f_explicit)


def test_generate_v_tensor_oracle_conventions_shapes_and_values():
    # D=1 keeps checks simple while exercising i', j' remapping.
    l = 4
    rc = 1
    rloc = 1
    v_pqrs = generate_v_tensor(shape=(l,), a=1.0, b=1.0, c=1.0, metric="manhattan", r_loc=rloc, r_c=rc)

    v_direct = generate_v_tensor(
        shape=(l,),
        a=1.0,
        b=1.0,
        c=1.0,
        metric="manhattan",
        r_loc=rloc,
        r_c=rc,
        oracle_convention="direct",
    )
    v_exchange = generate_v_tensor(
        shape=(l,),
        a=1.0,
        b=1.0,
        c=1.0,
        metric="manhattan",
        r_loc=rloc,
        r_c=rc,
        oracle_convention="exchange",
    )

    assert v_direct.shape == (l, l, 2 * rc + 1, 2 * rloc + 1)
    assert v_exchange.shape == (l, l, 2 * rc + 1, 2 * rloc + 1)

    # Sample index check against explicit formulas:
    i = 2
    j = 1
    m = 0
    ll = 2
    ip = (i - rc + m) % l
    jp = (j - rc + m - rloc + ll) % l

    # direct = V_{i j i' j'}
    assert np.isclose(v_direct[i, j, m, ll], v_pqrs[i, j, ip, jp])
    # exchange = V_{i i' j j'}
    assert np.isclose(v_exchange[i, j, m, ll], v_pqrs[i, ip, j, jp])
