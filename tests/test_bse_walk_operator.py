#!/usr/bin/env python3
"""Structural checks for the Sec.-2-form BSE walk operator and its new primitives.

Runnable as a plain script (``python tests/test_bse_walk_operator.py``) because
``pytest`` is not installed in the shared ``py312`` env.  These are *structure* checks --
they assert the construction is what ``main.tex`` Sec. 2 describes and that the
self-inverse property holds.  They deliberately do **not** assert any Toffoli value.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'integrations', 'qualtran'))

from bse_block_encoding import (  # noqa: E402
    BSEBlockEncoding, DirectTemplate, ExchangeTemplate, FockTemplate,
)
from bse_walk_operator import BSEWalkOperator  # noqa: E402
from block_isometry_column_synthesis_QROAM import (  # noqa: E402
    ColumnIsometryRectangularBlockEncoding,
)
from eigendecomposition_block_encoding import EigendecompositionBlockEncoding  # noqa: E402
from load_all_state_preparation_QROAM import LoadAllStatePreparationQROAM  # noqa: E402

SMALL = dict(m=1, N_o=4, N_v=8, N_IP=32, N_k=8, phase_bitsize=16, optimal_T=True)


from toffoli_cost import toffoli_count as tof  # noqa: E402  (n_ccz + n_t/4)


def test_ZRy_is_an_involution():
    """The load-bearing self-inverse fact: ``Z R_y(2 theta)`` squares to the identity.

    This is what makes every congruence template ``M Z M^dag`` an involution, and it is
    Clifford -- so the walk's ``U_A^2 = I`` requirement costs nothing.
    """
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    for theta in np.linspace(0, np.pi, 17):
        Ry = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        M = Z @ Ry
        assert np.allclose(M @ M, np.eye(2), atol=1e-12), theta
        assert np.allclose(M, M.T, atol=1e-12), theta          # Hermitian (real symmetric)
        assert np.isclose(M[0, 0], np.cos(theta))              # still encodes cos(theta)
    print("ok  Z R_y(2t) is a real symmetric involution encoding cos(t)")


def test_templates_match_the_manuscript_component_counts():
    """Each template emits exactly the components Sec. 2 lists for it."""
    ex = ExchangeTemplate(N_o=4, N_v=8, N_IP=32, N_k=8, phase_bitsize=16, optimal_T=True)
    calls = {}
    ex._calls(calls := __import__('collections').Counter(), controlled=False)
    n_iso = sum(n for b, n in calls.items()
                if isinstance(b, ColumnIsometryRectangularBlockEncoding))
    n_prep = sum(n for b, n in calls.items()
                 if isinstance(b, LoadAllStatePreparationQROAM))
    assert n_iso == 2, f"exchange: expected 2 isometry syntheses, got {n_iso}"
    assert n_prep == 2, f"exchange: expected 2 state preparations, got {n_prep}"

    di = DirectTemplate(N_o=4, N_v=8, N_IP=32, N_k=8, phase_bitsize=16, optimal_T=True)
    dcalls = __import__('collections').Counter()
    di._calls(dcalls, controlled=False)
    n_iso_d = sum(n for b, n in dcalls.items()
                  if isinstance(b, ColumnIsometryRectangularBlockEncoding))
    assert n_iso_d == 4, f"ov-direct: expected 4 isometry syntheses, got {n_iso_d}"

    for spin in ("oo", "vv"):
        sib = __import__('attrs').evolve(di, same_spin=spin)
        scalls = __import__('collections').Counter()
        sib._calls(scalls, controlled=False)
        n_iso_s = sum(n for b, n in scalls.items()
                      if isinstance(b, ColumnIsometryRectangularBlockEncoding))
        assert n_iso_s == 2, f"{spin}: expected 2 incremental isometries, got {n_iso_s}"
    print("ok  component counts: exchange 2+2, direct 4, oo/vv 2 each (incremental)")


def test_fock_uses_eigendecomposition_not_svd():
    f = FockTemplate(N_k=8, N=4, phase_bitsize=16, optimal_T=True)
    assert isinstance(f.eig, EigendecompositionBlockEncoding)
    # The congruence keeps U/U^dag uncontrolled, so the controlled form is barely dearer.
    assert tof(f.eig.controlled()) < 1.2 * tof(f.eig)
    print("ok  Fock template is U D U^dag, and its controlled form stays cheap")


def test_real_data_is_cheaper_and_default_is_complex():
    kw = dict(n_blocks=8, n_rows=64, phase_bitsize=16, n_reflections=8, optimal_T=True)
    complex_cost = tof(ColumnIsometryRectangularBlockEncoding(**kw))
    real_cost = tof(ColumnIsometryRectangularBlockEncoding(**kw, real_data=True))
    assert real_cost < complex_cost, (real_cost, complex_cost)
    # default must be the general complex path (backward compatible)
    assert tof(ColumnIsometryRectangularBlockEncoding(**kw, real_data=False)) == complex_cost
    print(f"ok  rotation-only synthesis {real_cost} < complex {complex_cost}; default unchanged")


def test_lambda_uses_the_manuscript_multiplicities():
    for m in (1, 2, 3):
        b = BSEBlockEncoding(**{**SMALL, 'm': m},
                             lambda_0_o=1.0, lambda_0_v=2.0, lambda_oo=3.0,
                             lambda_vv=4.0, lambda_ov_ex=5.0, lambda_ov_dir=6.0)
        want = m * (1.0 + 2.0) + (m * (m - 1) / 2) * (3.0 + 4.0) + m ** 2 * (5.0 + 6.0)
        assert np.isclose(b.alpha, want), (m, b.alpha, want)
    print("ok  lambda = m l_0 + m(m-1)/2 (l_oo+l_vv) + m^2 (l_ex+l_dir)")


def test_walk_operator_builds_both_centrals():
    for central in ("eigendecomposition", "frobenius"):
        w = BSEWalkOperator(**SMALL, exchange_central=central)
        d = w.template_costs()
        assert d['C_walk'] > 0
        assert set(d) >= {'C_0', 'C_oo', 'C_vv', 'C_ov_ex', 'C_ov_dir', 'C_route', 'C_walk'}
    print("ok  walk operator builds with both central options")


def test_mcg_is_a_real_rotation():
    """Iten's subleading fix-up needs only a rotation, even for COMPLEX isometries.

    It has to annihilate one of two amplitudes, so only ``c1/c0`` real matters -- the
    relative phase, not either absolute phase.  That relative phase folds into the
    preceding uniformly-controlled layer (which already carries a per-control phase
    pair, so it is free), and it always lands on a control gate at ``l1 >= lo``, never
    on the identity-pinned gates that protect already-disentangled columns.

    Asserts the disentangler still gives ``G V = I`` up to the final diagonal.
    """
    import block_isometry_column_synthesis_QROAM as CS

    rng = np.random.default_rng(3)
    L2 = CS.lemma2_gate

    def real_rot(a, b):
        r = float(np.hypot(a, b))
        if r < 1e-300:
            return np.eye(2, dtype=complex)
        return np.array([[a / r, b / r], [-b / r, a / r]], dtype=complex)

    def col_op(Mx, k, n, stats):
        N = 1 << n
        Gk = np.eye(N, dtype=complex)
        cur = Mx.copy()
        prev = None
        for s in range(n):
            ks = (k >> s) & 1
            a_s1 = k >> (s + 1)
            b_s1 = k & ((1 << (s + 1)) - 1)
            place = 1 << s
            f = k & (place - 1)
            if ks == 0 and b_s1 != 0:
                stats['total'] += 1
                i0 = a_s1 * (2 * place) + f
                i1 = i0 + place
                s_p, cg_p, (cur_b, Gk_b) = prev
                l0, l1 = i0 >> (s_p + 1), i1 >> (s_p + 1)
                rel = np.angle(cur[i1, k]) - np.angle(cur[i0, k])
                cg_new = dict(cg_p)
                if l1 in cg_p:
                    cg_new[l1] = np.exp(-1j * rel) * cg_p[l1]
                    stats['absorbed'] += 1
                elif l0 in cg_p:
                    cg_new[l0] = np.exp(1j * rel) * cg_p[l0]
                    stats['absorbed'] += 1
                else:
                    stats['blocked'] += 1
                Gu_new = CS._embed_ucg(n, s_p, cg_new)
                cur, Gk = Gu_new @ cur_b, Gu_new @ Gk_b
                c0, c1 = cur[i0, k], cur[i1, k]
                g = np.exp(-1j * np.angle(c0))
                U = real_rot(abs(c0), (c1 * g).real) @ np.diag([g, g])
                Gm = CS._embed_two_level(n, i0, i1, U)
                cur, Gk = Gm @ cur, Gm @ Gk
            n_ctrl = n - 1 - s
            lo = a_s1 + 1 if b_s1 != 0 else a_s1
            cg = {}
            for l in range(1 << n_ctrl):
                if l < lo:
                    continue
                j0 = l * (2 * place) + f
                cg[l] = L2(cur[j0, k], cur[j0 + place, k], ks)
            prev = (s, cg, (cur.copy(), Gk.copy()))
            Gu = CS._embed_ucg(n, s, cg)
            cur, Gk = Gu @ cur, Gu @ Gk
        return Gk

    stats = {'total': 0, 'absorbed': 0, 'blocked': 0}
    worst = 0.0
    for _ in range(20):
        n, K = 4, 7
        N = 1 << n
        A = rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))
        V, _ = np.linalg.qr(A)
        cur = V.copy()
        for k in range(K):
            cur = col_op(cur, k, n, stats) @ cur
        worst = max(worst, np.abs(np.abs(cur) - np.eye(N, K)).max())

    assert stats['blocked'] == 0, stats
    assert stats['total'] > 0
    assert worst < 1e-10, worst
    print(f"ok  mcg = real rotation on COMPLEX isometries: "
          f"{stats['absorbed']}/{stats['total']} absorbed, |G V| - I = {worst:.1e}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} structural checks passed")
