r"""Qubitization *walk operator* for the BSE Hamiltonian.

This is an alternative block encoding of the same BSE operator built by
:class:`~.bse_block_encoding.BSEBlockEncoding`, assembled as a *qubitized walk*
``W = (2\Pi - I) U_A`` instead of the ``<psi| S P A D P S |psi>`` sandwich.  Where the
plain block encoding returns ``U_A`` (with subnormalization ``Lambda``), the walk operator
turns that Hermitian-unitary ``U_A`` into a rotation whose spectrum encodes the operator's
eigenvalues, so phase estimation on ``W`` reads them off directly.

Construction (the standard LCU / qubitization reflection-walk)
-------------------------------------------------------------
The target Hermitian operator is a real linear combination of block-encoded pieces

.. math::  A = \sum_i c_i\, K^\dagger A_i K, \qquad c_i \in \mathbb{R},

where ``K`` is a fixed conjugation (here the particle-number counting ``P``,
antisymmetrizers ``S``, and the diagonal ``D``, with ``D`` commuting with the ``A_i``) and
each ``A_i`` has a **Hermitian-unitary** block encoding ``U_{A_i}``:

.. math::
    \Pi_a U_{A_i} \Pi_a = \Pi_a\, A_i/\alpha_i, \qquad
    U_{A_i} = U_{A_i}^\dagger, \qquad U_{A_i}^2 = I,
    \qquad \Pi_a = |0^a\rangle\langle 0^a| \otimes I.

With the LCU normalization ``Lambda = sum_i |c_i| alpha_i`` and

.. math::
    \mathrm{PREP}\,|0\rangle = \sum_i \sqrt{|c_i|\alpha_i/\Lambda}\, |i\rangle,
    \qquad \widetilde U_i = \operatorname{sgn}(c_i)\, U_{A_i},
    \qquad \mathrm{SELECT} = \sum_i |i\rangle\langle i| \otimes \widetilde U_i,

the sign of each real coefficient is absorbed into the selected unitary, so ``SELECT`` stays
Hermitian and involutive (``SELECT^dag = SELECT``, ``SELECT^2 = I``).  The block encoding is

.. math::  U_A = (\mathrm{PREP}^\dagger \otimes I)\, K^\dagger\, \mathrm{SELECT}\, K\, (\mathrm{PREP} \otimes I),

which is itself Hermitian and involutive (``U_A = U_A^\dagger``, ``U_A^2 = I``) because
``SELECT`` is and ``K``, ``PREP`` are unitary.  With the total projector
``\Pi = |0\rangle\langle 0|_{\mathrm{LCU}} \otimes \Pi_a`` one has
``\Pi U_A \Pi = \Pi\, A/\Lambda``, so the qubitized walk operator is simply

.. math::  W = (2\Pi - I)\, U_A.

On the invariant 2-D subspace of an eigenvector ``A|k> = E_k|k>`` (with ``x_k = E_k/Lambda``)
the walk acts as the rotation ``[[x_k, sqrt(1-x_k^2)], [-sqrt(1-x_k^2), x_k]]``, so the
eigenvalues of ``W`` are ``e^{+- i arccos(E_k/Lambda)}`` and phase estimation recovers
``E_k = Lambda cos(theta_k)``.

Hermitian-unitary terms
-----------------------
``U_A`` is produced by :class:`~.bse_block_encoding.BSEBlockEncoding` in its ``hermitian``
mode.  The Fock, direct-Coulomb, and combined-term central pieces are already involutive --
their diagonal / SVD sub-encodings apply ``Z R_y`` rather than ``R_y``, which makes the
block-encoding ancilla a *reflection* (``Z R_y(2 theta)`` is a Hermitian unitary).  The only
term that needs adapting is the exchange Coulomb sandwich ``B C B^dag``: it is Hermitian and
involutive exactly when its full-matrix central ``C`` is, so ``hermitian`` mode switches that
central to the Hermitian Frobenius-norm encoding
(``ExchangeCoulombBlockEncoding.hermitian_fro_central``).

Data-free convention
--------------------
Like the rest of this package the construction is *data-free*: ``build_call_graph`` emits the
inner block encoding ``U_A`` and the reflection ``2\Pi - I`` (a
:class:`ReflectionUsingPrepare` about ``|0>`` on the full ancilla register, which carries the
LCU selection bits), and all Toffoli / qubit counts come from Qualtran's resource counter
walking that graph.  No real coefficient / angle data is populated; in particular ``PREP`` is
modeled by the same data-free coefficient-state proxy as the plain block encoding and the
``sgn(c_i)`` sign flips are Clifford (free) bookkeeping inside ``SELECT``.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.symbolics import SymbolicFloat, SymbolicInt

try:
    from .bse_block_encoding import BSEBlockEncoding
except ImportError:
    from bse_block_encoding import BSEBlockEncoding

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class BSEWalkOperator(Bloq):
    r"""Qubitization walk operator ``W = (2\Pi - I) U_A`` for the BSE Hamiltonian.

    ``U_A`` is the Hermitian-unitary BSE block encoding
    (:class:`~.bse_block_encoding.BSEBlockEncoding` in ``hermitian`` mode) of
    ``A/Lambda``; ``2\Pi - I`` reflects about ``|0>`` on the full ancilla register (which
    includes the LCU selection bits), realized by :class:`ReflectionUsingPrepare`.  The walk
    acts in place on the same ``(system, ancilla, resource)`` registers as ``U_A``; its
    eigenvalues are ``e^{+- i arccos(E_k/Lambda)}``, so phase estimation on ``W`` recovers
    the BSE eigenvalues ``E_k = Lambda cos(theta_k)``.

    Attributes mirror :class:`~.bse_block_encoding.BSEBlockEncoding`:
        m, N_o, N_v, N_IP, N_k, phase_bitsize, optimal_T -- see that class.
    """

    m: int
    N_o: int
    N_v: int
    N_IP: int
    N_k: int
    phase_bitsize: int = 32
    optimal_T: bool = False
    # Rectangular X-tensor synthesis, forwarded to the inner block encoding:
    # "reflection" (LKS Householder, default) or "column" (Iten/Berry, ~2x cheaper).
    outer_synthesis: str = "reflection"

    # ------------------------------ inner pieces ------------------------------

    @cached_property
    def block_encoding(self) -> BSEBlockEncoding:
        """The Hermitian-unitary BSE block encoding ``U_A`` (``U_A = U_A^dag``, ``U_A^2 = I``)."""
        return BSEBlockEncoding(
            m=self.m, N_o=self.N_o, N_v=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T, hermitian=True,
            outer_synthesis=self.outer_synthesis,
        )

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        """``2\\Pi - I``: reflection about ``|0>`` on the full ancilla register."""
        return ReflectionUsingPrepare(self.block_encoding.signal_state, global_phase=-1)

    # ------------------------------- interface --------------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.block_encoding.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.block_encoding.ancilla_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.block_encoding.resource_bitsize

    @property
    def Lambda(self) -> SymbolicFloat:
        r"""LCU normalization ``Lambda = sum_i |c_i| alpha_i`` (data-free: sum of term alphas)."""
        return self.block_encoding.alpha

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    # ----------------------------- resource counts ----------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        return {self.block_encoding: 1, self.reflect: 1}

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Single-qubit control: only the reflection gains the control.

        ``.controlled()`` returns :class:`_ControlledBSEWalkOperator`, which leaves the
        Hermitian-unitary ``U_A`` uncontrolled and promotes the reflection to its controlled
        form -- the standard cheap controlled-walk primitive used inside qubitization phase
        estimation.
        """
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBSEWalkOperator(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledBSEWalkOperator(Bloq):
    """Singly-controlled :class:`BSEWalkOperator` (controls the reflection only).

    The external control reaches only the reflection ``2\\Pi - I`` (via
    :class:`ReflectionUsingPrepare`'s ``control_val``); the Hermitian-unitary ``U_A`` is
    applied unconditionally.  This is the standard qubitization controlled-walk used for
    phase estimation, at essentially the uncontrolled cost.
    """

    inner: BSEWalkOperator

    @cached_property
    def reflect_ctrl(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(
            self.inner.block_encoding.signal_state, control_val=1, global_phase=-1
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner.block_encoding] += 1   # U_A applied unconditionally
        ret[self.reflect_ctrl] += 1           # only the reflection gains the control
        return ret
