r"""Clifford-controlled phase-gradient rotation (arXiv:2007.07391, Appendix A).

A rotation :math:`e^{-2\pi i\,\ell Z/2^{b}}` **controlled** on a qubit, with the angle
:math:`\ell` held in a ``b``-bit register, costs only ``b - 2`` Toffolis -- *not* the
``~2(b-1)`` of a naively controlled adder.

Mechanism (App. A, Eq. (A5)): the control does not gate the addition.  Instead it drives
``cnot`` gates on the addend register *before and after* a single **uncontrolled** addition
into the phase-gradient state -- flipping add :math:`\leftrightarrow` subtract via the two's
-complement identity ``NOT(v) = -v-1``.  Those ``cnot``s are Clifford, so the only Toffoli cost
is the one uncontrolled addition.  That addition is ``b - 2`` (not ``b - 1``) because the most
significant qubit of the phase-gradient state is in :math:`|+\rangle`, so its ``NOT`` becomes a
phase gate and the qubit is discarded, and the final carry Toffoli is replaced by a controlled
phase.  (A classical angle would be ``b - 3``.)

Qualtran's ``AddIntoPhaseGrad(b, b)`` (uncontrolled) already realises the ``b - 2`` addition;
its ``.controlled()`` however charges ``~2(b-1)``.  This bloq is the drop-in replacement, with
matching registers ``(ctrl, x, phase_grad)``, that charges the App. A cost instead.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING

import attrs

from qualtran import Bloq, QAny, QBit, QUInt, Register, Signature
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class SignedCtrlAddIntoPhaseGrad(Bloq):
    r"""Controlled phase-gradient rotation via Clifford add/subtract (2007.07391 App. A).

    Registers match ``AddIntoPhaseGrad(b, b).controlled()`` -- ``(ctrl, x, phase_grad)`` -- so it
    is a drop-in replacement.  Toffoli cost ``b - 2`` (one uncontrolled addition; the control is
    Clifford), i.e. *half* the ``~2(b-1)`` Qualtran charges for the naive controlled adder.
    """

    phase_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        b = int(self.phase_bitsize)
        return Signature([
            Register('ctrl', QBit()),
            Register('x', QUInt(b)),
            Register('phase_grad', QAny(b)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        b = int(self.phase_bitsize)
        ret: "Counter[Bloq]" = Counter()
        ret[AddIntoPhaseGrad(b, b)] += 1     # one UNcontrolled add == b-2 Toffoli (MSB-|+> trick)
        ret[CNOT()] += 2 * b                 # Clifford control (add<->subtract), 0 Toffoli
        return dict(ret)
