r"""Qubitization walk operator for the BSE Hamiltonian, in the form of ``main.tex`` Sec. 2.

.. math::  W = (2\Pi - I)\, U_A ,

with :math:`U_A` the five-template block encoding of :class:`BSEBlockEncoding` and
:math:`\Pi = |0\rangle\langle 0|` on the full ancilla register (which carries the LCU
selection bits).  On the invariant two-dimensional subspace of an eigenvector
:math:`H_{\mathrm{BSE}}|j\rangle = E_j|j\rangle` the walk acts as a rotation by
:math:`\arccos(E_j/\lambda)`, so phase estimation on :math:`W` returns
:math:`E_j = \lambda\cos\theta_j` and

.. math::
    C_{\mathrm{QPE}} = \left\lceil\frac{\pi\lambda}{2\epsilon_{\mathrm{QPE}}}\right\rceil
                       C_{\mathrm{walk}} .

Why :math:`U_A` is already self-inverse
---------------------------------------
The walk requires :math:`U_A^2 = I`.  The manuscript asserts this ("we need to make our
:math:`U_i` self-inverse throughout the construction") without discharging it; here it is
discharged structurally, and for free:

* every template is a **congruence** :math:`M Z M^\dagger`, so
  :math:`(MZM^\dagger)^2 = M Z^2 M^\dagger = I` as soon as :math:`Z^2 = I`;
* every central :math:`Z` applies :math:`Z R_y(2\theta)` rather than :math:`R_y(2\theta)`.
  :math:`Z R_y(2\theta)` is real symmetric and squares to the identity, and the extra
  :math:`Z` is **Clifford** -- zero Toffolis;
* the signs of the real LCU coefficients are absorbed into the selected unitaries, so
  SELECT stays Hermitian; PREPARE and the routing SWAP network are unitary conjugations
  and cancel in the square.

The single exception is the Frobenius central (``exchange_central="frobenius"``), which
is *not* a congruence.  There the Hermitian variant is used -- the inner Frobenius
encoding wrapped in ``DirectHermitianBlockEncoding``, i.e. the four-preparation form --
at roughly twice the bare central cost.

Data-free: structure only; all counts come from Qualtran's resource counter.
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
except ImportError:  # pragma: no cover - script/direct execution
    from bse_block_encoding import BSEBlockEncoding

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@attrs.frozen
class BSEWalkOperator(Bloq):
    r"""``W = (2\Pi - I) U_A`` for the BSE Hamiltonian.

    Attributes mirror :class:`BSEBlockEncoding`: ``m, N_o, N_v, N_IP, N_k,
    phase_bitsize, optimal_T, exchange_central, real_data`` and the six ``lambda_*``
    component subnormalizations.
    """

    m: int
    N_o: int
    N_v: int
    N_IP: int
    N_k: int
    phase_bitsize: int = 32
    optimal_T: bool = False
    exchange_central: str = "eigendecomposition"
    ex_density_fitting: bool = False
    real_data: bool = False
    lambda_0_o: SymbolicFloat = 1.0
    lambda_0_v: SymbolicFloat = 1.0
    lambda_oo: SymbolicFloat = 1.0
    lambda_vv: SymbolicFloat = 1.0
    lambda_ov_ex: SymbolicFloat = 1.0
    lambda_ov_dir: SymbolicFloat = 1.0

    # ------------------------------ inner pieces ------------------------------

    @cached_property
    def block_encoding(self) -> BSEBlockEncoding:
        r"""The self-inverse block encoding :math:`U_A` (:math:`U_A^2 = I`)."""
        return BSEBlockEncoding(
            m=self.m, N_o=self.N_o, N_v=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
            exchange_central=self.exchange_central, real_data=self.real_data,
            ex_density_fitting=self.ex_density_fitting,
            lambda_0_o=self.lambda_0_o, lambda_0_v=self.lambda_0_v,
            lambda_oo=self.lambda_oo, lambda_vv=self.lambda_vv,
            lambda_ov_ex=self.lambda_ov_ex, lambda_ov_dir=self.lambda_ov_dir,
        )

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        r""":math:`2\Pi - I` about :math:`|0\rangle` on the full ancilla register."""
        return ReflectionUsingPrepare(
            self.block_encoding.signal_state, global_phase=-1
        )

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
        r""":math:`\lambda = m\lambda_0 + \frac{m(m-1)}{2}(\lambda_{oo}+\lambda_{vv})
        + m^2(\lambda_{ov}^{\mathrm{ex}}+\lambda_{ov}^{\mathrm{dir}})`."""
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

    def walk_cost(self) -> int:
        """``C_walk`` in Toffolis."""
        try:
            from .toffoli_cost import toffoli_count
        except ImportError:
            from toffoli_cost import toffoli_count
        return toffoli_count(self)

    def template_costs(self) -> dict:
        """``C_walk`` broken out by the manuscript's per-template symbols."""
        try:
            from .toffoli_cost import toffoli_count
        except ImportError:
            from toffoli_cost import toffoli_count
        d = dict(self.block_encoding.template_costs())
        d['C_reflect'] = toffoli_count(self.reflect)
        d['C_walk'] = self.walk_cost()
        return d

    def qpe_cost(self, epsilon_qpe: float) -> float:
        r""":math:`\lceil \pi\lambda / 2\epsilon\rceil \, C_{\mathrm{walk}}`."""
        import math

        return math.ceil(math.pi * float(self.Lambda) / (2.0 * epsilon_qpe)) * self.walk_cost()

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Standard cheap controlled walk: only the reflection gains the control."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBSEWalkOperator(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledBSEWalkOperator(Bloq):
    r"""Singly-controlled :class:`BSEWalkOperator`.

    :math:`U_A` is applied unconditionally and only :math:`2\Pi - I` is controlled -- the
    standard qubitization controlled-walk, at essentially the uncontrolled cost.
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
        ret[self.inner.block_encoding] += 1
        ret[self.reflect_ctrl] += 1
        return ret
