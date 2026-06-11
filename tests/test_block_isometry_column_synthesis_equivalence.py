"""Correctness/equivalence tests for the column-by-column isometry synthesizer.

These pin the *algorithm* behind ``BlockIsometryColumnSynthesisQROAM``:

* Berry Eq. 24 (:func:`eq24_angles`) reconstructs an arbitrary single-qubit unitary to
  machine precision.
* The Iten column-by-column disentangler (:func:`column_by_column_disentangler`) produces a
  unitary ``G`` with ``G V = I_{2^n x K}`` for random ``m -> n`` isometries (real and complex),
  i.e. it synthesizes the isometry ``V = G^dagger I``.
* The multi-controlled-gate count matches Iten Cor. 1 ``Q(m,n)`` for ``K = 2^m``.
* The bloq's shape inference (``from_isometry`` / default ``n_cols``) is correct.
"""

from __future__ import annotations

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
_ = qualtran

from integrations.qualtran.block_isometry_column_synthesis_QROAM import (
    BlockIsometryColumnSynthesisQROAM,
    column_by_column_disentangler,
    eq24_angles,
    eq24_reconstruct,
    num_mcgs,
)


def _haar_u2(rng) -> np.ndarray:
    X = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    return Q @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 2)))


def _random_isometry(n: int, m: int, rng, real: bool = False) -> np.ndarray:
    N, K = 1 << n, 1 << m
    X = rng.standard_normal((N, K))
    if not real:
        X = X + 1j * rng.standard_normal((N, K))
    Q, _ = np.linalg.qr(X)
    return Q[:, :K].astype(complex)


def test_eq24_reconstructs_random_and_edge_u2():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3000):
        U = _haar_u2(rng)
        phi0, phi1, theta, phi = eq24_angles(U)
        worst = max(worst, float(np.linalg.norm(eq24_reconstruct(phi0, phi1, theta, phi) - U)))
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    edges = [
        np.eye(2, dtype=complex),
        H,
        np.diag([np.exp(0.7j), np.exp(-1.3j)]),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, np.exp(0.3j)], [np.exp(-1.1j), 0]], dtype=complex),
    ]
    for U in edges:
        phi0, phi1, theta, phi = eq24_angles(U)
        worst = max(worst, float(np.linalg.norm(eq24_reconstruct(phi0, phi1, theta, phi) - U)))
    assert worst < 1e-10, worst


def test_disentangler_synthesizes_isometry():
    """G V = I_{2^n x K} (up to per-column phase) and G is unitary, for random isometries."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in range(1, 6):
        for m in range(0, n + 1):
            for real in (False, True):
                V = _random_isometry(n, m, rng, real=real)
                G, _ = column_by_column_disentangler(V)
                worst = max(worst, float(np.linalg.norm(G.conj().T @ G - np.eye(1 << n))))
                GV = G @ V
                target = np.eye(1 << n, dtype=complex)[:, : (1 << m)]
                for k in range(1 << m):
                    # column k must be e_k up to a global phase
                    worst = max(worst, float(np.linalg.norm(np.abs(GV[:, k]) - target[:, k])))
                    worst = max(worst, abs(abs(GV[k, k]) - 1.0))
    assert worst < 1e-7, worst


def test_num_mcgs_matches_corollary1():
    """For K = 2^m, the multi-controlled-gate total equals Iten Cor. 1 Q(m,n)."""
    for n in range(1, 8):
        for m in range(0, n + 1):
            q = (2 ** m) * (n - m / 2 - 1) - n + m + 1
            assert num_mcgs(1 << n, 1 << m) == round(q), (n, m, num_mcgs(1 << n, 1 << m), q)


def test_first_column_needs_no_mcg():
    for n in range(1, 8):
        assert num_mcgs(1 << n, 1) == 0


def test_from_isometry_and_default_n_cols():
    rng = np.random.default_rng(2)
    V = _random_isometry(4, 2, rng)  # 16 x 4
    bloq = BlockIsometryColumnSynthesisQROAM.from_isometry(V, phase_bitsize=6)
    assert (bloq.n_blocks, bloq.n_rows, bloq.n_cols) == (1, 16, 4)
    full = BlockIsometryColumnSynthesisQROAM.from_shape(2, 8, 6)
    assert full.n_cols == 8  # defaults to n_rows (full unitary)


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
