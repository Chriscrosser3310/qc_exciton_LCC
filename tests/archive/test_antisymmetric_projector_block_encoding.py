"""Tests for the antisymmetric/symmetric projector block encoding (arXiv:2407.17563).

Pins both the *math* (dense numpy references) and the *qualtran realization*:

Numpy reference
  * ``permutation_sign`` / ``permutation_unitary`` are the standard ``S_k`` representation (Eq. 7-8).
  * ``antisymmetric_projector`` / ``symmetric_projector`` are genuine projectors with
    ``tr = C(d, k)`` and ``tr = (number of symmetric basis states)`` respectively.
  * ``plus_sk_amplitudes`` is the Eq. 10-15 control state: ``k!`` equal-weight strings whose
    Hamming-weight parity equals the permutation sign.

Qualtran bloqs
  * ``PrepareSymmetricGroupControl`` maps ``|0> -> |+_{S_k}>`` exactly and is invertible.
  * ``SelectSymmetricGroupPermutation`` permutes subsystems classically per control block.
  * ``AntisymmetricProjectorBlockEncoding`` has unitary ``W`` whose ``<0|.|0>`` ancilla block
    equals the dense projector (Eq. 22 / 44), with ``alpha = 1``, ``epsilon = 0``.
  * ``build_call_graph`` and ``build_composite_bloq`` report identical ``QECGatesCost``.
"""

from __future__ import annotations

import itertools
import math
import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for envs without pytest

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__}, got nothing")
            return issubclass(exc_type, self.exc)

    class _PytestShim:
        @staticmethod
        def raises(exc):
            return _Raises(exc)

        @staticmethod
        def importorskip(name):
            return __import__(name)

    pytest = _PytestShim()  # type: ignore[assignment]
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

import numpy as np

qualtran = pytest.importorskip("qualtran")

from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.antisymmetric_projector_block_encoding import (
    AntisymmetricProjectorBlockEncoding,
    PrepareSymmetricGroupControl,
    SelectSymmetricGroupPermutation,
    antisymmetric_projector,
    block_offsets,
    control_bitsize,
    permutation_sign,
    permutation_unitary,
    plus_sk_amplitudes,
    symmetric_projector,
)

ATOL = 1e-12


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost())


# ---------------------------------------------------------------------------
# Numpy reference: permutation representation
# ---------------------------------------------------------------------------


def test_permutation_sign_known_values():
    assert permutation_sign((0, 1, 2)) == 1  # identity
    assert permutation_sign((1, 0, 2)) == -1  # one transposition
    assert permutation_sign((1, 2, 0)) == 1  # 3-cycle = two transpositions
    assert permutation_sign((0, 2, 1, 3)) == -1
    # sign is a homomorphism under composition for a couple of random pairs
    for k in (3, 4):
        perms = list(itertools.permutations(range(k)))
        for p in perms[:6]:
            for q in perms[:6]:
                comp = tuple(p[q[a]] for a in range(k))
                assert permutation_sign(comp) == permutation_sign(p) * permutation_sign(q)


def test_permutation_unitary_is_representation():
    d, k = 3, 3
    for perm in itertools.permutations(range(k)):
        u = permutation_unitary(perm, d)
        assert u.shape == (d**k, d**k)
        assert np.allclose(u.conj().T @ u, np.eye(d**k), atol=ATOL)
    # homomorphism: U(p)U(q) == U(p o q)
    perms = list(itertools.permutations(range(k)))
    for p in perms:
        for q in perms:
            comp = tuple(p[q[a]] for a in range(k))
            assert np.allclose(
                permutation_unitary(p, d) @ permutation_unitary(q, d),
                permutation_unitary(comp, d),
                atol=ATOL,
            )


def test_permutation_unitary_swap_action():
    # transposition (0 1) on two qutrits swaps the two digits.
    d = 3
    u = permutation_unitary((1, 0), d)
    v = np.zeros(d * d)
    v[1 * d + 2] = 1.0  # |1>|2>
    out = u @ v
    expected = np.zeros(d * d)
    expected[2 * d + 1] = 1.0  # |2>|1>
    assert np.allclose(out, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Numpy reference: projectors
# ---------------------------------------------------------------------------


def test_projectors_are_projectors():
    for k, d in [(2, 2), (2, 4), (3, 4), (4, 4), (3, 3)]:
        for proj in (antisymmetric_projector(k, d), symmetric_projector(k, d)):
            assert np.allclose(proj, proj.conj().T, atol=ATOL)
            assert np.allclose(proj @ proj, proj, atol=ATOL)


def test_antisymmetric_trace_is_binomial():
    # dim of antisymmetric subspace of (C^d)^{⊗k} is C(d, k).
    for k, d in [(2, 2), (2, 4), (3, 3), (3, 4), (4, 4), (2, 8)]:
        proj = antisymmetric_projector(k, d)
        assert abs(np.trace(proj).real - math.comb(d, k)) < 1e-9
        assert abs(np.trace(proj).imag) < 1e-9


def test_symmetric_trace_is_binomial():
    # dim of symmetric subspace is C(d + k - 1, k).
    for k, d in [(2, 2), (2, 4), (3, 3), (3, 4)]:
        proj = symmetric_projector(k, d)
        assert abs(np.trace(proj).real - math.comb(d + k - 1, k)) < 1e-9


def test_sym_and_anti_orthogonal_and_swap_decomposition():
    # For k = 2: SWAP = Pi_sym - Pi_anti, and I = Pi_sym + Pi_anti.
    for d in (2, 3, 4):
        ps = symmetric_projector(2, d)
        pa = antisymmetric_projector(2, d)
        assert np.allclose(ps + pa, np.eye(d * d), atol=ATOL)
        assert np.allclose(ps - pa, permutation_unitary((1, 0), d), atol=ATOL)
        assert np.allclose(ps @ pa, np.zeros((d * d, d * d)), atol=ATOL)


# ---------------------------------------------------------------------------
# Numpy reference: control plus-state
# ---------------------------------------------------------------------------


def test_plus_sk_amplitudes_structure():
    for k in (2, 3, 4):
        amps = plus_sk_amplitudes(k)
        nc = control_bitsize(k)
        assert amps.shape == (2**nc,)
        assert abs(np.linalg.norm(amps) - 1.0) < ATOL
        support = np.where(np.abs(amps) > 1e-12)[0]
        assert len(support) == math.factorial(k)
        # equal weights
        assert np.allclose(np.abs(amps[support]), 1.0 / math.sqrt(math.factorial(k)), atol=ATOL)


def test_plus_sk_support_parity_matches_sign():
    # Each valid control string has Hamming-weight parity equal to a permutation's sign,
    # and the multiset of parities matches the multiset of S_k signs (half +, half -).
    for k in (2, 3, 4):
        amps = plus_sk_amplitudes(k)
        support = np.where(np.abs(amps) > 1e-12)[0]
        parities = [bin(int(c)).count("1") % 2 for c in support]
        n_odd = sum(parities)
        n_even = len(parities) - n_odd
        signs = [permutation_sign(p) for p in itertools.permutations(range(k))]
        assert n_even == signs.count(1)
        assert n_odd == signs.count(-1)


# ---------------------------------------------------------------------------
# PrepareSymmetricGroupControl bloq
# ---------------------------------------------------------------------------


def test_prepare_matches_plus_state():
    for k in (2, 3, 4):
        prep = PrepareSymmetricGroupControl(k)
        u = prep.tensor_contract()
        got = u[:, 0]  # PREP|0...0>
        assert np.allclose(got, plus_sk_amplitudes(k), atol=1e-10)


def test_prepare_adjoint_round_trip():
    for k in (2, 3, 4):
        prep = PrepareSymmetricGroupControl(k)
        nc = control_bitsize(k)
        u = prep.tensor_contract()
        uadj = prep.adjoint().tensor_contract()
        assert np.allclose(uadj @ u, np.eye(2**nc), atol=1e-10)
        assert prep.adjoint().uncompute is True
        assert prep.adjoint().adjoint() == prep


def test_prepare_requires_two_subsystems():
    with pytest.raises(AssertionError):
        PrepareSymmetricGroupControl(1)


# ---------------------------------------------------------------------------
# SelectSymmetricGroupPermutation bloq
# ---------------------------------------------------------------------------


def test_select_classical_action_k2():
    # k=2: single control qubit; |1> swaps the two subsystems, |0> leaves them.
    sel = SelectSymmetricGroupPermutation(2, subsystem_bitsize=2)
    ctrl, system = sel.call_classically(control=0, system=np.array([1, 2]))
    assert ctrl == 0 and list(system) == [1, 2]
    ctrl, system = sel.call_classically(control=1, system=np.array([1, 2]))
    assert ctrl == 1 and list(system) == [2, 1]


def test_select_realizes_permutations_on_valid_strings():
    # On each valid control string, SELECT applies the corresponding permutation to the
    # subsystem labels. We verify against the dense permutation map by reading the support
    # bits of the control string and composing the encoded transpositions.
    k, s = 3, 2
    sel = SelectSymmetricGroupPermutation(k, subsystem_bitsize=s)
    nc = control_bitsize(k)
    offsets = block_offsets(k)
    init = list(range(k))
    for control in range(2**nc):
        # decode the transpositions selected by this control string
        labels = list(init)
        valid = True
        for j in range(2, k + 1):
            m = j - 1
            block_bits = [(control >> (nc - 1 - (offsets[j] + t))) & 1 for t in range(m)]
            if sum(block_bits) > 1:
                valid = False
                break
            if sum(block_bits) == 1:
                t = block_bits.index(1)
                i = m - t  # one-hot value 2^{i-1} at MSB-first position t = m - i
                labels[i - 1], labels[j - 1] = labels[j - 1], labels[i - 1]
        if not valid:
            continue
        ctrl_out, sys_out = sel.call_classically(control=control, system=np.array(init))
        assert ctrl_out == control
        assert list(sys_out) == labels


# ---------------------------------------------------------------------------
# AntisymmetricProjectorBlockEncoding
# ---------------------------------------------------------------------------


def _top_left_block(be):
    """Extract the ancilla=|0> block of the block-encoding unitary W."""
    w = be.tensor_contract()
    ds = (2**be.subsystem_bitsize) ** be.n_subsystems
    nanc = 2**be.ancilla_bitsize
    assert w.shape == (ds * nanc, ds * nanc)
    return w.reshape(ds, nanc, ds, nanc)[:, 0, :, 0]


def test_block_encoding_matches_antisymmetric_projector():
    for k, s in [(2, 1), (2, 2), (3, 2), (4, 2)]:
        be = AntisymmetricProjectorBlockEncoding(k, s, signed=True)
        block = _top_left_block(be)
        assert np.allclose(block, antisymmetric_projector(k, 2**s), atol=1e-10)
        assert np.allclose(block, be.projector(), atol=1e-10)


def test_block_encoding_matches_symmetric_projector():
    for k, s in [(2, 1), (2, 2), (3, 2)]:
        be = AntisymmetricProjectorBlockEncoding(k, s, signed=False)
        block = _top_left_block(be)
        assert np.allclose(block, symmetric_projector(k, 2**s), atol=1e-10)


def test_block_encoding_W_is_unitary():
    for k, s, signed in [(2, 1, True), (3, 2, True), (3, 2, False), (4, 2, True)]:
        be = AntisymmetricProjectorBlockEncoding(k, s, signed=signed)
        w = be.tensor_contract()
        assert np.allclose(w.conj().T @ w, np.eye(w.shape[0]), atol=1e-10)


def test_block_encoding_interface_metadata():
    be = AntisymmetricProjectorBlockEncoding(3, 2, signed=True)
    assert be.alpha == 1.0
    assert be.epsilon == 0.0
    assert be.system_bitsize == 3 * 2
    assert be.ancilla_bitsize == control_bitsize(3) == 3
    assert be.resource_bitsize == 0
    names = [r.name for r in be.signature]
    assert names == ['system', 'ancilla']
    # signal state is |0> on the ancilla register
    assert be.signal_state.tensor_contract().shape == (2**be.ancilla_bitsize,) * 2


def test_block_encoding_cost_consistency():
    for k, s in [(2, 1), (3, 2), (4, 2)]:
        for signed in (True, False):
            be = AntisymmetricProjectorBlockEncoding(k, s, signed=signed)
            assert _cost(be) == _cost(be.decompose_bloq())


def test_block_encoding_requires_valid_params():
    with pytest.raises(AssertionError):
        AntisymmetricProjectorBlockEncoding(1, 2)
    with pytest.raises(AssertionError):
        AntisymmetricProjectorBlockEncoding(3, 0)


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
