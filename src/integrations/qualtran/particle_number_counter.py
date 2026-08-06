r"""Particle-number counter: count registers that are NOT in the flagged ``vacuum`` state.

Given ``m`` input registers each of dimension ``N`` (so each holds a value in
``[0, N)``) and a ``count`` ancilla register, this reversible operator computes

    |x_0, ..., x_{m-1}>|c>  -->  |x_0, ..., x_{m-1}>|c + #{i : x_i != vacuum}> ,

i.e. it adds, into the ``count`` register, the number of input registers that are *not*
in the flagged ``vacuum`` state (default ``vacuum = N - 1``).  A register equal to the
vacuum carries "no particle"; any other value is "one particle".

Circuit (per input register ``x_i``, using one borrowed flag qubit):

  1. ``EqualsAConstant`` writes ``flag = (x_i == vacuum)`` into the flag qubit.
  2. A controlled ``AddK(+1)`` increments ``count`` when the register is *not* the vacuum
     (the increment is conditioned on ``flag == 0``, realized by an ``X`` sandwich).
  3. ``EqualsAConstant`` (its own inverse) uncomputes the flag back to ``|0>``.

The ``count`` register is updated *in place* (``c -> c + ...``), so the operator composes
with any other Qualtran bloq / block encoding: feed the same ``count`` register through,
or initialize it to ``|0>`` to read off the particle number directly.  ``count`` is sized
to hold ``0..m`` (``ceil(log2(m + 1))`` qubits).

The flag qubits are borrowed internally (allocated and freed inside the decomposition);
they are not part of the signature.  All Toffoli / qubit counts come from Qualtran's
resource counter walking the decomposition.
"""

from __future__ import annotations

from functools import cached_property
from typing import Dict, Optional, TYPE_CHECKING

import attrs
import numpy as np

from qualtran import Bloq, BloqBuilder, GateWithRegisters, QAny, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic import AddK, EqualsAConstant
from qualtran.bloqs.basic_gates import XGate
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class ParticleNumberCounter(GateWithRegisters):
    r"""Count the input registers that are NOT in the flagged ``vacuum`` state.

    Adds ``#{i : x_i != vacuum}`` into the ``count`` register, in place.

    Attributes:
        m: number of input registers.
        N: dimension of each input register (each holds a value in ``[0, N)``).
        vacuum_state: the flagged ("vacuum") value; a register holding it is *not*
            counted.  Defaults to ``N - 1``.

    Registers:
        registers: ``m`` input registers, ``QUInt(ceil(log2 N))`` each (shape ``(m,)``).
        count: the count ancilla register, ``QUInt(ceil(log2(m + 1)))``; updated in place
            (``c -> c + #non-vacuum``).
    """

    m: SymbolicInt
    N: SymbolicInt
    vacuum_state: Optional[SymbolicInt] = None

    def __attrs_post_init__(self):
        if not is_symbolic(self.N) and int(self.N) < 2:
            raise ValueError(f"N must be >= 2 (got {self.N})")
        if not is_symbolic(self.m) and int(self.m) < 1:
            raise ValueError(f"m must be >= 1 (got {self.m})")
        v = self.vacuum
        if not is_symbolic(v, self.N) and not (0 <= int(v) < int(self.N)):
            raise ValueError(f"vacuum_state must be in [0, N); got {v} for N={self.N}")

    @property
    def vacuum(self) -> SymbolicInt:
        return self.N - 1 if self.vacuum_state is None else self.vacuum_state

    @cached_property
    def reg_bitsize(self) -> SymbolicInt:
        """Qubits per input register (= ceil(log2 N))."""
        return bit_length(self.N - 1)

    @cached_property
    def count_bitsize(self) -> SymbolicInt:
        """Qubits in the count register, enough to hold ``0..m``."""
        return bit_length(self.m)

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('registers', QUInt(self.reg_bitsize), shape=(self.m,)),
            Register('count', QUInt(self.count_bitsize)),
        ])

    # --------------------------- Sub-bloq factories ----------------------------

    @cached_property
    def _equals_vacuum(self) -> Bloq:
        return EqualsAConstant(bitsize=self.reg_bitsize, val=self.vacuum)

    @cached_property
    def _ctrl_increment(self) -> Bloq:
        # Increment ``count`` by 1, controlled on a single qubit (cv = 1 here; the caller
        # X-sandwiches the flag so the active condition is "register != vacuum").
        return AddK(QUInt(self.count_bitsize), 1).controlled()

    # ----------------------------- Resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        # Per register: compute + uncompute the equality flag, one controlled increment,
        # and two (Clifford) X gates around the increment control.
        return {
            self._equals_vacuum: 2 * self.m,
            self._ctrl_increment: self.m,
            XGate(): 2 * self.m,
        }

    # --------------------------- Composite circuit -----------------------------

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.m, self.N):
            raise NotImplementedError("build_composite_bloq requires concrete m, N")

        regs = np.asarray(soqs['registers'], dtype=object).copy()
        count = soqs['count']
        eq = self._equals_vacuum
        cadd = self._ctrl_increment
        ctrl_name = cadd.signature[0].name  # control register name from .controlled()

        for i in range(int(self.m)):
            flag = bb.allocate(1)
            # flag = (x_i == vacuum)
            regs[i], flag = bb.add(eq, x=regs[i], target=flag)
            # flip so the control is active iff x_i != vacuum
            flag = bb.add(XGate(), q=flag)
            # count += 1 controlled on (x_i != vacuum)
            inc_out = bb.add_d(cadd, **{ctrl_name: flag, 'x': count})
            flag = inc_out[ctrl_name]
            count = inc_out['x']
            flag = bb.add(XGate(), q=flag)
            # uncompute the flag (EqualsAConstant is its own inverse) and release it
            regs[i], flag = bb.add(eq, x=regs[i], target=flag)
            bb.free(flag)

        return {'registers': regs, 'count': count}
