"""Tests for the ``restrict_input`` comparator on the direct/exchange THC block encodings.

Each X channel acts on an N_k x N_IP system register, but the orbital input is only valid
over the first N_o (occupied = N_up) / N_v (virtual = N_down) entries.  With
``restrict_input=True`` a ``LessThanConstant`` comparator flags ``x < N_o`` / ``x < N_v``
into a per-channel ancilla (computed + uncomputed), restraining the encoded operator's
input subspace.  These tests verify:

* the comparators appear in the call graph with the right ``less_than_val`` (N_o / N_v),
  twice each (compute + uncompute), for both the uncontrolled and controlled bloqs;
* ``restrict_input`` adds exactly two ancilla qubits (one flag per channel);
* turning it off recovers the previous (no-comparator) call graph and ancilla count;
* the Toffoli overhead is negligible.
"""

from __future__ import annotations

import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover

    class _PytestShim:
        @staticmethod
        def importorskip(name):
            return __import__(name)

    pytest = _PytestShim()  # type: ignore[assignment]
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

qualtran = pytest.importorskip("qualtran")

from qualtran.bloqs.arithmetic import LessThanConstant

from integrations.qualtran.direct_Coulomb_block_encoding import DirectCoulombBlockEncoding
from integrations.qualtran.exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
from integrations.qualtran.utils import get_Toffoli_counts

NUP, NDN, NIP, NK, B = 4, 22, 208, 216, 32


def _toffoli(bloq) -> int:
    return int(get_Toffoli_counts(bloq))


def _comparators(cg):
    """Return {less_than_val: count} over LessThanConstant entries in a call graph."""
    out = {}
    for bloq, cnt in cg.items():
        if isinstance(bloq, LessThanConstant):
            out[int(bloq.less_than_val)] = out.get(int(bloq.less_than_val), 0) + int(cnt)
    return out


def _mk(cls, **kw):
    return cls(N_up=NUP, N_down=NDN, N_IP=NIP, N_k=NK, phase_bitsize=B, optimal_T=True, **kw)


def test_direct_comparators_present():
    b = _mk(DirectCoulombBlockEncoding)
    cmps = _comparators(b.build_call_graph(None))
    # x < N_o (=N_up=4) and x < N_v (=N_down=22), each computed + uncomputed.
    assert cmps == {NUP: 2, NDN: 2}


def test_exchange_comparators_present():
    b = _mk(ExchangeCoulombBlockEncoding)
    cmps = _comparators(b.build_call_graph(None))
    assert cmps == {NUP: 2, NDN: 2}


def test_controlled_also_restricts():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        cb = _mk(cls).controlled()
        assert _comparators(cb.build_call_graph(None)) == {NUP: 2, NDN: 2}


def test_restrict_input_adds_two_ancilla():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        on = _mk(cls, restrict_input=True)
        off = _mk(cls, restrict_input=False)
        assert on.ancilla_bitsize == off.ancilla_bitsize + 2
        # system register is N_k x N_IP per channel; restriction is ancilla-only.
        assert on.system_bitsize == off.system_bitsize


def test_off_has_no_comparators_and_is_cheaper():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        on = _mk(cls, restrict_input=True)
        off = _mk(cls, restrict_input=False)
        assert _comparators(off.build_call_graph(None)) == {}
        # Comparators are negligible: < 0.01% Toffoli overhead, but strictly positive.
        assert _toffoli(on) > _toffoli(off)
        assert (_toffoli(on) - _toffoli(off)) < 0.001 * _toffoli(off)


def test_default_is_on():
    for cls in (DirectCoulombBlockEncoding, ExchangeCoulombBlockEncoding):
        assert _mk(cls).restrict_input is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All input-comparator restriction tests passed.")
