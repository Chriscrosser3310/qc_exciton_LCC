r"""BSE (Bethe--Salpeter-equation) block encoding.

A data-free *structural* block encoding of the operator

    BSE = <psi| S . P . A . D . P . S |psi> ,

assembled from the THC component block encodings in this package.  It follows the same
data-free convention as :class:`ExchangeCoulombBlockEncoding` / :class:`FockBlockEncoding`:
the circuit *structure* and sub-bloq multiplicities are described in ``build_call_graph``,
and all Toffoli / qubit counts come from Qualtran's resource counter walking that graph
(no real angle/coefficient data is populated -- "fake data").

Register layout
---------------
``2m`` system registers, each a two-component register:

  * a momentum component of dimension ``N_k``       (``ceil(log2 N_k)`` qubits),
  * an orbital/interpolation component of dimension ``N_IP + 1`` (``ceil(log2(N_IP+1))``
    qubits) -- the ``+1`` is the flagged "vacuum" state (index ``N_IP``) used by the
    particle-number counter.

The ``2m`` registers are partitioned: the first ``m`` are **occupied**, the last ``m`` are
**virtual**.

The linear-combination operator ``A`` (LCU SELECT over 5 terms)
--------------------------------------------------------------
``A`` applies, multiplexed on a coefficient ("LCU") ancilla, the following terms (each a
block encoding acting on a few of the ``2m`` registers, identity elsewhere):

  1. ``Fock`` on the first occupied register  -- ``N_o x N_o`` inside an ``N_k x N_IP`` reg.
  2. ``Fock`` on the first virtual register    -- ``N_v x N_v``.
  3. ``ExchangeCoulomb`` on (first occupied, first virtual) -- ``N_up=N_o, N_down=N_v``.
  4. ``DirectCoulomb``   on (first occupied, first virtual) -- same structure.
  5. A *combined* term = sum of four direct-Coulomb-like operators that share one central
     kernel ``W``:
       (a) direct Coulomb on the first two occupied registers,
       (b) (a) with the two output registers swapped (controlled swaps),
       (c) direct Coulomb on the first two virtual registers, reusing ``W`` from (a),
       (d) (c) with outputs swapped.
     Implemented as: controlled versions of the four ``X`` tensors of (a) and (c); the
     central ``W`` synthesized *once* and sandwiched by swaps to attach to the leading two
     occupied / virtual registers; and a small identity/swap LCU that generates (b) and (d).

The full sandwich
-----------------
With ``A`` as above (coefficient state not yet baked in):

  * ``P`` -- a :class:`ParticleNumberCounter` on the occupied registers and another on the
    virtual registers (vacuum = orbital index ``N_IP``), writing the occupied/virtual
    particle numbers into two ancilla registers; applied on both sides of ``A``.
  * ``D`` -- a diagonal (QROAM -> phase -> QROAM^dagger) acting on the joint
    (LCU coefficient, occupied count, virtual count) ancilla register.
  * ``S`` -- antisymmetrizers controlled on the particle number: for occupied count
    ``k_o`` apply an :class:`AntisymmetricProjectorBlockEncoding` to the first ``k_o``
    occupied registers, and likewise for the virtual count ``k_v``; applied on both sides.
  * ``|psi>`` -- the coefficient state on the LCU ancilla, sandwiching the whole product
    (this supplies the LCU PREPARE / PREPARE^dagger; coefficients are data-free here).

This module is data-free; a single-qubit controlled version is provided
(:class:`_ControlledBSEBlockEncoding`) that promotes the operator-defining pieces (the
``A`` terms and ``D``) to controlled form while the symmetric ``P`` / ``S`` / ``|psi>``
pairs stay uncontrolled.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .antisymmetric_projector_block_encoding import AntisymmetricProjectorBlockEncoding
    from .block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from .direct_Coulomb_block_encoding import (
        DiagonalCoulombKernelBlockEncoding,
        DirectCoulombBlockEncoding,
    )
    from .exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
    from .fock_block_encoding import FockBlockEncoding
    from .particle_number_counter import ParticleNumberCounter
    from .rectangular_block_encoding_reflection import ReflectionRectangularBlockEncoding
except ImportError:
    from antisymmetric_projector_block_encoding import AntisymmetricProjectorBlockEncoding
    from block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from direct_Coulomb_block_encoding import (
        DiagonalCoulombKernelBlockEncoding,
        DirectCoulombBlockEncoding,
    )
    from exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
    from fock_block_encoding import FockBlockEncoding
    from particle_number_counter import ParticleNumberCounter
    from rectangular_block_encoding_reflection import ReflectionRectangularBlockEncoding

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _next_power_of_two(n: int) -> int:
    return 1 << max(1, (int(n) - 1).bit_length())


# Number of LCU terms in A (Fock_occ, Fock_virt, Exchange, Direct, combined).
_N_TERMS = 5


@attrs.frozen
class BSEBlockEncoding(BlockEncoding):
    r"""Data-free BSE block encoding ``<psi| S P A D P S |psi>`` (see module docstring).

    Attributes:
        m: number of registers per partition (so ``2m`` system registers total).
        N_o: occupied count (Fock/exchange/direct ``N_up``; antisymmetrizer extent).
        N_v: virtual count (``N_down``).
        N_IP: number of interpolation points (each orbital register has dimension
            ``N_IP + 1``; the extra state is the vacuum).
        N_k: number of momenta.
        phase_bitsize: bitsize of phase / angle registers in the sub-encodings.
        optimal_T: forward Toffoli-optimal QROAM blocking to the sub-encodings.
        lambd_terms: number of LCU terms (fixed at 5; exposed for clarity).
        hermitian: when True, every LCU SELECT term is a *Hermitian-unitary* block encoding
            (``U_{A_i} = U_{A_i}^dag``, ``U_{A_i}^2 = I``), so the whole encoding is a
            Hermitian-unitary ``U_A`` suitable for the qubitization walk operator
            :class:`~.bse_walk_operator.BSEWalkOperator`.  The Fock / direct-Coulomb /
            combined-term central pieces are already involutive (their diagonal / SVD
            sub-encodings apply ``Z R_y`` rather than ``R_y``); the only change is that the
            exchange term's full-matrix central switches to the Hermitian Frobenius-norm
            encoding (``hermitian_fro_central=True``).  Off by default (the plain block
            encoding does not require involutive terms).
    """

    m: int
    N_o: int
    N_v: int
    N_IP: int
    N_k: int
    phase_bitsize: int = 32
    optimal_T: bool = False
    hermitian: bool = False
    # Encode the exchange term's full-matrix central A'_Q via the Clader-Frobenius dense
    # block encoding (``ExchangeCoulombBlockEncoding.use_fro_BE``) instead of the default SVD
    # interferometer.  This is the cost-optimal central choice (and the one the qubitization
    # walk needs, in its Hermitian-wrapped form).  When ``hermitian`` is also set, the central
    # is additionally wrapped to be Hermitian-unitary (``hermitian_fro_central``); the bare
    # ``use_fro_BE`` path here is the *non-Hermitian* Frobenius central used by the plain
    # block encoding.  Fock / direct / combined centrals are unaffected.
    use_fro_BE: bool = False
    # Synthesis method for every rectangular X-tensor isometry (the exchange / direct
    # B_up,B_down and the combined term's x_occ,x_virt): "reflection" (LKS Householder,
    # default) or "column" (column-by-column isometry synthesis, Iten 1501.06911 + Berry
    # Eq. 24).  Column synthesis is ~2x cheaper in Toffoli and is the cost-optimal choice;
    # "reflection" is kept as the default for backward compatibility.  The Fock / diagonal
    # central pieces are unaffected.
    outer_synthesis: str = "reflection"

    def __attrs_post_init__(self):
        if self.outer_synthesis not in ("reflection", "column"):
            raise ValueError(
                f"outer_synthesis must be 'reflection' or 'column', got {self.outer_synthesis!r}"
            )
        for name in ("m", "N_o", "N_v", "N_IP", "N_k"):
            v = getattr(self, name)
            if not is_symbolic(v) and int(v) < 1:
                raise ValueError(f"{name} must be >= 1 (got {v})")
        if not is_symbolic(self.N_o, self.N_IP) and int(self.N_o) > int(self.N_IP):
            raise ValueError("N_o must be <= N_IP")
        if not is_symbolic(self.N_v, self.N_IP) and int(self.N_v) > int(self.N_IP):
            raise ValueError("N_v must be <= N_IP")

    # ------------------------------ register sizes ------------------------------

    @cached_property
    def k_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_k - 1)

    @cached_property
    def orbital_dim(self) -> SymbolicInt:
        return self.N_IP + 1  # the +1 is the vacuum state

    @cached_property
    def orbital_bitsize(self) -> SymbolicInt:
        return bit_length(self.orbital_dim - 1)

    @cached_property
    def reg_bitsize(self) -> SymbolicInt:
        # one two-component system register: momentum + orbital/interpolation.
        return self.k_bitsize + self.orbital_bitsize

    @cached_property
    def n_registers(self) -> SymbolicInt:
        return 2 * self.m

    @cached_property
    def pn_bitsize(self) -> SymbolicInt:
        # particle-number register: counts 0..m (one for occupied, one for virtual).
        return bit_length(self.m)

    @cached_property
    def lcu_sel_bitsize(self) -> SymbolicInt:
        return bit_length(_N_TERMS - 1)

    # --------------------------- BlockEncoding interface ------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.n_registers * self.reg_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        # LCU coefficient ancilla + two particle-number registers + the largest term's
        # own block-encoding ancilla (LCU-style: max of constituents + selection bits).
        max_term_ancilla = max(
            self.fock_occ.ancilla_bitsize,
            self.fock_virt.ancilla_bitsize,
            self.exchange.ancilla_bitsize,
            self.direct.ancilla_bitsize,
            self.central_W.ancilla_bitsize + 2,  # +flag(s) for the combined-term sub-LCU
        )
        return self.lcu_sel_bitsize + 2 * self.pn_bitsize + max_term_ancilla

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        # LCU subnormalization: sum of the constituent term alphas.  (The structural
        # P / S / D / |psi> factors are omitted from this nominal value.)
        return float(
            self.fock_occ.alpha + self.fock_virt.alpha + self.exchange.alpha
            + self.direct.alpha + self.combined_term_alpha
        )

    @property
    def epsilon(self) -> SymbolicFloat:
        return 2.0 ** (-self.phase_bitsize)

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return PrepareIdentity.from_bitsizes((self.ancilla_bitsize,))

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    # ------------------------------ shape helpers -------------------------------

    @cached_property
    def n_rows_occ(self) -> int:
        return _next_power_of_two(max(self.N_o, self.N_IP))

    @cached_property
    def n_rows_virt(self) -> int:
        return _next_power_of_two(max(self.N_v, self.N_IP))

    # ----------------------- A: the five LCU terms ------------------------------

    @cached_property
    def fock_occ(self) -> FockBlockEncoding:
        # Term 1: Fock on the first occupied register (N_o x N_o in an N_k x (N_IP+1) reg).
        return FockBlockEncoding(N_k=self.N_k, N_IP=self.orbital_dim, N=self.N_o,
                                 phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)

    @cached_property
    def fock_virt(self) -> FockBlockEncoding:
        # Term 2: Fock on the first virtual register (N_v x N_v).
        return FockBlockEncoding(N_k=self.N_k, N_IP=self.orbital_dim, N=self.N_v,
                                 phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T)

    @cached_property
    def exchange(self) -> ExchangeCoulombBlockEncoding:
        # Term 3: exchange Coulomb on (first occupied, first virtual).  In ``hermitian``
        # mode the central full-matrix kernel switches to the Hermitian Frobenius-norm
        # encoding so the whole ``B C B^dag`` sandwich is Hermitian-unitary.
        return ExchangeCoulombBlockEncoding(
            N_up=self.N_o, N_down=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
            outer_synthesis=self.outer_synthesis,
            use_fro_BE=self.use_fro_BE and not self.hermitian,
            hermitian_fro_central=self.hermitian)

    @cached_property
    def direct(self) -> DirectCoulombBlockEncoding:
        # Term 4: direct Coulomb on (first occupied, first virtual).
        return DirectCoulombBlockEncoding(
            N_up=self.N_o, N_down=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
            outer_synthesis=self.outer_synthesis)

    # ----------------------- term 5: combined direct ----------------------------

    def _x_tensor(self, n_rows: int, n_reflections: int) -> Bloq:
        """A combined-term X-tensor isometry BE, per ``outer_synthesis`` (reflection / column)."""
        if self.outer_synthesis == "column":
            return ColumnIsometryRectangularBlockEncoding(
                n_blocks=self.N_k, n_rows=n_rows, phase_bitsize=self.phase_bitsize,
                n_reflections=n_reflections, optimal_T=self.optimal_T)
        return ReflectionRectangularBlockEncoding(
            n_blocks=self.N_k, n_rows=n_rows, phase_bitsize=self.phase_bitsize,
            n_reflections=n_reflections, optimal_T=self.optimal_T)

    @cached_property
    def x_occ(self) -> Bloq:
        # An X tensor (interpolating vectors) on an occupied register (N_o x N_IP).
        return self._x_tensor(self.n_rows_occ, min(self.N_o, self.N_IP))

    @cached_property
    def x_virt(self) -> Bloq:
        # An X tensor on a virtual register (N_v x N_IP).
        return self._x_tensor(self.n_rows_virt, min(self.N_v, self.N_IP))

    @cached_property
    def central_W(self) -> DiagonalCoulombKernelBlockEncoding:
        # The single shared central kernel W of the combined term.
        return DiagonalCoulombKernelBlockEncoding(
            N_k=self.N_k, N_IP=self.N_IP, phase_bitsize=self.phase_bitsize,
            optimal_T=self.optimal_T)

    @cached_property
    def output_swap(self) -> Bloq:
        # Controlled swap of two output (system) registers -- generates the swapped (2nd /
        # 4th) members of the combined term from an identity/swap LCU.
        return CSwap(self.reg_bitsize)

    @cached_property
    def combined_sub_lcu_prep(self) -> Bloq:
        # Small identity-vs-swap LCU prepare for occupied / virtual swap selection.
        return PrepareUniformSuperposition(n=2)

    @property
    def combined_term_alpha(self) -> float:
        # Four direct-like operators sharing one W; nominal LCU alpha.
        return 4.0 * float(self.direct.alpha)

    # --------------------------- P, D, S, |psi> ---------------------------------

    @cached_property
    def particle_counter_occ(self) -> ParticleNumberCounter:
        # Counts non-vacuum occupied registers (vacuum = orbital index N_IP).
        return ParticleNumberCounter(m=self.m, N=self.orbital_dim, vacuum_state=self.N_IP)

    @cached_property
    def particle_counter_virt(self) -> ParticleNumberCounter:
        return ParticleNumberCounter(m=self.m, N=self.orbital_dim, vacuum_state=self.N_IP)

    @cached_property
    def diag_D(self) -> Bloq:
        # Diagonal over the joint (LCU coeff, occ count, virt count) ancilla:
        # QROAMClean load -> phase -> QROAMCleanAdjoint.  Returned as the forward QROAM;
        # the rotation and adjoint are emitted alongside in build_call_graph.
        return QROAMClean.build_from_bitsize(
            (_N_TERMS, self.m + 1, self.m + 1),
            target_bitsizes=(self.phase_bitsize,),
        )

    @cached_property
    def diag_D_adjoint(self) -> Bloq:
        return QROAMCleanAdjoint.build_from_bitsize(
            (_N_TERMS, self.m + 1, self.m + 1), target_bitsizes=(self.phase_bitsize,))

    @cached_property
    def diag_D_rotation(self) -> Bloq:
        return AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)

    def antisymmetrizer(self, k: int, *, virtual: bool) -> AntisymmetricProjectorBlockEncoding:
        # Antisymmetrize the first ``k`` registers of one partition.
        return AntisymmetricProjectorBlockEncoding(
            n_subsystems=k, subsystem_bitsize=int(self.reg_bitsize), signed=True)

    @cached_property
    def psi_prep(self) -> Bloq:
        # |psi>: the LCU coefficient state on the selection ancilla (data-free proxy).
        return PrepareUniformSuperposition(n=_N_TERMS)

    # ------------------------------ resource counts -----------------------------

    def _combined_term_calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        """Emit the combined (term 5) sub-bloqs into ``ret``."""
        x_occ = self.x_occ.controlled() if controlled else self.x_occ
        x_virt = self.x_virt.controlled() if controlled else self.x_virt
        # Four controlled X tensors of operators (a) [occ pair] and (c) [virt pair],
        # each applied forward + adjoint.
        ret[x_occ] += 4   # X on occ[0], occ[1], forward + adjoint
        ret[x_virt] += 4  # X on virt[0], virt[1], forward + adjoint
        # The single shared central kernel W.
        ret[self.central_W.controlled() if controlled else self.central_W] += 1
        # Identity/swap LCU generating the swapped operators (b), (d): a prepare pair plus
        # the controlled output swaps that attach W to the leading occ / virt registers.
        ret[self.combined_sub_lcu_prep] += 2
        ret[self.output_swap] += 4

    def _A_select_calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        """Emit the LCU SELECT of A (the five terms, multiplexed / controlled)."""
        ret[self.fock_occ.controlled() if controlled else self.fock_occ] += 1
        ret[self.fock_virt.controlled() if controlled else self.fock_virt] += 1
        ret[self.exchange.controlled() if controlled else self.exchange] += 1
        ret[self.direct.controlled() if controlled else self.direct] += 1
        self._combined_term_calls(ret, controlled=controlled)

    def _D_calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        ret[self.diag_D] += 1
        ret[self.diag_D_rotation.controlled().controlled() if controlled
            else self.diag_D_rotation.controlled()] += 1
        ret[self.diag_D_adjoint] += 1

    def _PS_psi_calls(self, ret: "Counter[Bloq]") -> None:
        """Symmetric structural pieces (uncontrolled in the controlled variant)."""
        # |psi> prepare + unprepare on the LCU coefficient ancilla.
        ret[self.psi_prep] += 2
        # P on both sides: one counter per partition, each appearing twice.
        ret[self.particle_counter_occ] += 2
        ret[self.particle_counter_virt] += 2
        # S on both sides: for each particle number k = 2..m, a controlled antisymmetrizer
        # on the first k occupied registers and on the first k virtual registers.
        if not is_symbolic(self.m):
            for k in range(2, int(self.m) + 1):
                anti = self.antisymmetrizer(k, virtual=False).controlled()
                ret[anti] += 2  # occupied side, both sandwich sides
                ret[anti] += 2  # virtual side, both sandwich sides

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        self._PS_psi_calls(ret)
        self._A_select_calls(ret, controlled=False)
        self._D_calls(ret, controlled=False)
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Single-qubit control: only the operator-defining ``A`` terms and ``D`` gain the
        control; the symmetric ``P`` / ``S`` / ``|psi>`` pairs stay uncontrolled."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBSEBlockEncoding(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledBSEBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`BSEBlockEncoding`.

    The external control reaches the operator-defining pieces -- the five ``A`` SELECT
    terms (including the combined term's X tensors and central ``W``) and the diagonal
    ``D`` -- through their own cheap controlled forms.  The symmetric ``|psi>`` / ``P`` /
    ``S`` pairs are left uncontrolled (they cancel / are structural at ``ctrl = 0``).
    """

    inner: BSEBlockEncoding

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.inner.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.inner.ancilla_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.inner.resource_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.inner.alpha

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.inner.epsilon

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return self.inner.signal_state

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('ctrl', QBit()),
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        self.inner._PS_psi_calls(ret)                       # uncontrolled structural pieces
        self.inner._A_select_calls(ret, controlled=True)    # controlled SELECT terms
        self.inner._D_calls(ret, controlled=True)           # controlled diagonal
        return ret
