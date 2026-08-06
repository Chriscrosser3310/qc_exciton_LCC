"""Tests for ``ParticleNumberCounter``.

Counts the input registers that are NOT in the flagged ``vacuum`` state (default N-1) and
adds that number into the ``count`` register, in place.  Verified:

* the classical action ``c -> c + #{i : x_i != vacuum}`` across vacuum choices, count
  offsets, and the all/none-vacuum extremes, with the input registers preserved (THRU);
* the signature (m shaped input registers + a count register sized to hold 0..m);
* the call-graph cost matches the actual decomposition (consistency);
* input validation (N >= 2, vacuum in [0, N)).
"""

from __future__ import annotations

import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover

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

qualtran = pytest.importorskip("qualtran")

import numpy as np
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import bit_length

from integrations.qualtran.particle_number_counter import ParticleNumberCounter


def _ccz(bloq) -> int:
    return int(get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"])


# ------------------------------- correctness -------------------------------


def _check(m, N, vac, inputs, c0):
    b = ParticleNumberCounter(m=m, N=N, vacuum_state=vac)
    regs_out, count_out = b.call_classically(registers=np.array(inputs), count=c0)
    expected = c0 + sum(1 for x in inputs if x != b.vacuum)
    assert int(count_out) == expected, (inputs, b.vacuum, int(count_out), expected)
    assert list(np.asarray(regs_out)) == list(inputs)  # inputs preserved


def test_default_vacuum_is_N_minus_1():
    assert ParticleNumberCounter(m=4, N=8).vacuum == 7
    _check(4, 8, None, [7, 3, 7, 5], 0)  # two non-vacuum


def test_all_and_none_vacuum():
    _check(4, 8, 7, [7, 7, 7, 7], 0)   # all vacuum -> 0
    _check(4, 8, 7, [0, 1, 2, 3], 0)   # none vacuum -> m


def test_custom_vacuum_and_offset():
    _check(3, 4, 0, [0, 1, 0], 0)           # vacuum=0 -> one particle
    _check(5, 16, 15, [15, 1, 2, 15, 9], 2)  # offset c0=2 + 3 particles = 5


def test_single_register():
    _check(1, 2, 1, [0], 0)
    _check(1, 2, 1, [1], 0)


def test_exhaustive_small():
    m, N, vac = 3, 4, 3
    b = ParticleNumberCounter(m=m, N=N, vacuum_state=vac)
    for a in range(N):
        for bb in range(N):
            for c in range(N):
                inp = [a, bb, c]
                _, cnt = b.call_classically(registers=np.array(inp), count=0)
                assert int(cnt) == sum(1 for x in inp if x != vac)


# --------------------------------- structure --------------------------------


def test_signature():
    b = ParticleNumberCounter(m=5, N=16)
    sig = {r.name: r for r in b.signature}
    assert set(sig) == {"registers", "count"}
    assert sig["registers"].shape == (5,)
    assert sig["registers"].dtype.num_qubits == bit_length(16 - 1)  # 4
    assert sig["count"].dtype.num_qubits == bit_length(5)           # holds 0..5 -> 3


def test_call_graph_matches_decomposition():
    for m, N in [(4, 8), (5, 16), (8, 32)]:
        b = ParticleNumberCounter(m=m, N=N)
        assert _ccz(b) == _ccz(b.decompose_bloq())


# --------------------------------- validation -------------------------------


def test_validation():
    with pytest.raises(ValueError):
        ParticleNumberCounter(m=4, N=1)            # N must be >= 2
    with pytest.raises(ValueError):
        ParticleNumberCounter(m=0, N=8)            # m must be >= 1
    with pytest.raises(ValueError):
        ParticleNumberCounter(m=4, N=8, vacuum_state=8)   # vacuum out of [0, N)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All particle-number-counter tests passed.")
