r"""BSE block encoding in the form of ``XPRIZE_paper/main.tex`` Sec. 2.

Replaces the archived ``<psi| S P A D P S |psi>`` construction
(``archive/bse_block_encoding_v1.py``).  The manuscript's walk cost is

.. math::
    C_{\mathrm{walk}} = C_0 + C_{oo} + C_{vv} + C_{ov}^{\mathrm{ex}}
                        + C_{ov}^{\mathrm{dir}} + C_{\mathrm{route}},

six circuit templates in five cost categories, each compiled **once**, with a
controlled-SWAP network routing the selected particle registers through them.  The
subnormalization is

.. math::
    \lambda = m\lambda_0 + \tfrac{m(m-1)}{2}(\lambda_{oo}+\lambda_{vv})
              + m^2(\lambda_{ov}^{\mathrm{ex}} + \lambda_{ov}^{\mathrm{dir}}).

The registers are treated as **distinguishable**: every term carries the same coefficient
tensor for every register (pair), so the encoding commutes with permutations within each
partition and preserves the antisymmetric sector.  Antisymmetry is imposed on the input
state and is *not* re-enforced by the walk -- which is why no antisymmetrizer or
particle-number counter appears here (they did, wrongly, in v1).

Every template is a congruence
------------------------------
Reading the THC index placement off the manuscript's Eqs. for :math:`V` and :math:`W`:

======================  ==============================  ================================
template                matrix form                     primitives
======================  ==============================  ================================
Fock (:math:`C_0`)      :math:`\mathcal U \mathcal D
                        \mathcal U^\dagger`             2 x eigendecomposition
:math:`ov` exchange     :math:`M \mathcal Z M^\dagger`  2 x column isometry,
                                                        2 x load-all state prep,
                                                        1 x central (eigendec. / Frobenius)
:math:`ov` direct       :math:`\mathcal X \Delta_\zeta
                        \mathcal X^\dagger`             4 x column isometry, 1 x diagonal
:math:`oo`, :math:`vv`  same as direct                  2 x column isometry (incremental)
======================  ==============================  ================================

In the exchange term each of :math:`\mu, \nu` must reach **both** particle registers, so
it can be consumed by an isometry on only one side; the other side keeps the index alive
with a controlled state preparation.  That asymmetry is the manuscript's, and it is why
the virtual side carries the :math:`\lVert\cdot\rVert_{2\to\infty}` norm.  In the direct
term each register owns one index outright, so all four factors are isometries and
:math:`\zeta` degrades from a matrix to a diagonal.

Self-inverse throughout
-----------------------
Every template is :math:`M Z M^\dagger`, so :math:`(MZM^\dagger)^2 = M Z^2 M^\dagger = I`
whenever :math:`Z^2 = I`.  The central pieces all apply :math:`Z R_y` rather than
:math:`R_y` (:math:`Z R_y(2\theta)` is real symmetric and squares to the identity), so
the whole SELECT is an involution at **no Toffoli cost** -- the :math:`Z` is Clifford.
The one exception is the Frobenius alternative, which is not a congruence and therefore
needs the explicit :class:`DirectHermitianBlockEncoding` wrapper (2x the central).

Data-free: structure only; all Toffoli / qubit counts come from Qualtran's resource
counter walking ``build_call_graph``.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.arithmetic import Subtract
from qualtran.bloqs.basic_gates import ZGate
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.qft import QFTTextBook
from qualtran.symbolics import SymbolicInt as _SI
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt
from qualtran import QUInt

try:
    from .block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from .classical_matrix_block_encoding_QROAM import (
        BlockDiagonalClassicalMatrixBlockEncoding,
        DirectHermitianBlockEncoding,
    )
    from .diagonal_kernel_block_encoding import DiagonalCoulombKernelBlockEncoding
    from .block_unitary_interferometer_QROAM import BlockUnitaryInterferometerSynthesisQROAM
    from .berry_isometry_synthesis_QROAM import BerryIsometrySynthesisQROAM
    from .eigendecomposition_block_encoding import EigendecompositionBlockEncoding
    from .load_all_state_preparation_QROAM import LoadAllStatePreparationQROAM
except ImportError:  # pragma: no cover - script/direct execution
    from block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from classical_matrix_block_encoding_QROAM import (
        BlockDiagonalClassicalMatrixBlockEncoding,
        DirectHermitianBlockEncoding,
    )
    from diagonal_kernel_block_encoding import DiagonalCoulombKernelBlockEncoding
    from block_unitary_interferometer_QROAM import BlockUnitaryInterferometerSynthesisQROAM
    from berry_isometry_synthesis_QROAM import BerryIsometrySynthesisQROAM
    from eigendecomposition_block_encoding import EigendecompositionBlockEncoding
    from load_all_state_preparation_QROAM import LoadAllStatePreparationQROAM

try:
    from .range_safe_qroam import emit_range_safety
except ImportError:  # pragma: no cover
    from range_safe_qroam import emit_range_safety

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _pow2(n: int) -> int:
    """Round up to a power of two.

    Needed **only** for state preparation: a dense rotation tree has one layer per
    address bit, so its amplitude count must be a power of two and a shorter state is
    zero-padded.  The analytic model imposes the same requirement
    (``_require_state_dimension``).  Synthesis primitives (interferometer, column
    isometry, diagonal) do *not* need it and are given the physical dimension.
    """
    return 1 << max(1, (int(n) - 1).bit_length())


# The five cost categories of C_walk (the two one-particle templates share C_0).
_N_TERMS = 5


# ===========================================================================
# Template 1 -- the Fock operator (C_0)
# ===========================================================================


@attrs.frozen
class FockTemplate(BlockEncoding):
    r"""``C_0``: the one-particle template, one sector.

    Encodes :math:`\sum_{\mathbf k} |\mathbf k\rangle\langle\mathbf k| \otimes f_{\mathbf
    k}` with :math:`f_{\mathbf k} = U_{\mathbf k} D_{\mathbf k} U_{\mathbf k}^\dagger`.
    The manuscript specifies **eigendecomposition**, not SVD: the blocks are Hermitian,
    the eigenvalues are real (so one :math:`R_y` suffices), and the congruence form is an
    involution for free.

    :math:`\lambda_{0,o/v} = \max_{\mathbf k}\lVert f_{\mathbf k}\rVert` -- the operator
    norm, which the eigendecomposition attains exactly.

    ``C_0`` for the whole walk is **two** of these (occupied and virtual), i.e. four
    unitary syntheses and two diagonal block encodings.
    """

    N_k: SymbolicInt
    N: SymbolicInt              # N_o or N_v
    phase_bitsize: SymbolicInt = 32
    optimal_T: bool = False
    alpha_val: SymbolicFloat = 1.0
    real_data: bool = False     # Bloch orbitals are complex at general k

    @cached_property
    def n_rows_inner(self) -> SymbolicInt:
        # No power-of-two padding: the interferometer runs on the physical dimension.
        return self.N

    @cached_property
    def eig(self) -> EigendecompositionBlockEncoding:
        return EigendecompositionBlockEncoding(
            n_blocks=self.N_k,
            n_rows=self.n_rows_inner,
            phase_bitsize=self.phase_bitsize,
            alpha_val=self.alpha_val,
            optimal_T=self.optimal_T,
            real_data=self.real_data,
        )

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.eig.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.eig.ancilla_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.alpha_val

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

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        return Counter({self.eig: 1})

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledTemplate(self), ctrl_reg_name='ctrl',
        )


# ===========================================================================
# Template 2 -- the ov exchange term
# ===========================================================================


@attrs.frozen
class ExchangeTemplate(BlockEncoding):
    r"""``C_ov^ex``: :math:`M \mathcal Z M^\dagger` with :math:`\mathcal Z = \oplus_Q \zeta^V_Q`.

    Both :math:`\mu` (ket) and :math:`\nu` (bra) sit on **both** particle registers, so
    :math:`\zeta^V` survives as a genuine matrix and the central needs a matrix block
    encoding.  Each index is consumed by an isometry on the hole side and kept alive by a
    controlled state preparation on the electron side.

    Components (matching the manuscript's count):
      * 2 x column-by-column isometry synthesis -- the hole factors :math:`\chi^{V,o}`;
      * 2 x load-all controlled state preparation -- the electron factors
        :math:`\chi^{V,v}` (the ``2->inf`` norm side);
      * 1 x central :math:`\zeta^V_Q`: **eigendecomposition** (operator norm, and a
        congruence so self-inverse for free) or the **Hermitian Frobenius** alternative
        (larger :math:`\lambda`, more aggressive space-Toffoli tradeoff, and needs the
        explicit Hermitian wrapper because it is not a congruence);
      * 2 x momentum-transfer isometry
        :math:`N_k^{-1/2}\sum_{\mathbf k,\mathbf Q}(|\mathbf k\rangle\otimes|\mathbf k
        \ominus\mathbf Q\rangle)\langle \mathbf Q|` and its adjoint -- isometries, so no
        extra subnormalization.

    :math:`\lambda_{ov}^{\mathrm{ex}} = 2\,\lVert\chi^{V,o}\rVert^2\,
    \lVert\chi^{V,v}\rVert_{2\to\infty}^2\,\lVert\zeta^V\rVert_*`; the 2 is the operator
    norm of the pair-spin operator.
    """

    N_o: SymbolicInt
    N_v: SymbolicInt
    N_IP: SymbolicInt           # N_THC
    N_k: SymbolicInt
    phase_bitsize: SymbolicInt = 32
    optimal_T: bool = False
    #: "eigendecomposition" (operator norm, default) or "frobenius" (Hermitian variant).
    central: str = "eigendecomposition"
    alpha_val: SymbolicFloat = 1.0
    real_data: bool = False
    #: How the THC factor chi is embedded as an isometry.  chi is NOT an isometry --
    #: measured on the ISDF data, ||X^dag X - I|| reaches 31 and the singular values span
    #: 0.002 to 9.42 -- so it cannot be synthesized directly as one.  Two correct routes:
    #:   "dilation" (default) -- complete it: V = [A ; sqrt(I - A^dag A)] has orthonormal
    #:      columns, needs row dimension N_THC + M, and encodes A/||A|| at the operator
    #:      norm.  Because N_THC + M still fits the same register (208+22 = 230 <= 256),
    #:      this costs ~0.1-0.3 % more than the bare (wrong) synthesis and NO extra qubits.
    #:   "svd" -- A = U Sigma V^dag: an isometry U, a real diagonal Sigma (one rotation,
    #:      singular values are real and non-negative), and a small M x M unitary V.
    #: Measured: dilation is 0.91-0.94x the SVD route, the gap being the separate V
    #: synthesis (20,518 Toffoli at M=22, N_k=216).  Same subnormalization either way.
    chi_embedding: str = "dilation"

    def __attrs_post_init__(self):
        if self.central not in ("eigendecomposition", "frobenius"):
            raise ValueError(
                f"central must be 'eigendecomposition' or 'frobenius', got {self.central!r}"
            )
        if self.chi_embedding not in ("dilation", "svd"):
            raise ValueError(
                f"chi_embedding must be 'dilation' or 'svd', got {self.chi_embedding!r}"
            )

    # ------------------------------ shape helpers ------------------------------

    @cached_property
    def k_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_k - 1)

    @cached_property
    def n_rows_thc(self) -> SymbolicInt:
        return self.N_IP

    @cached_property
    def n_rows_occ(self) -> SymbolicInt:
        return self.N_o

    @cached_property
    def n_rows_virt(self) -> SymbolicInt:
        return self.N_v

    # -------------------------------- sub-bloqs --------------------------------

    @cached_property
    def hole_isometry(self) -> ColumnIsometryRectangularBlockEncoding:
        r"""``chi^{V,o}``: an ``N_o x N_THC`` isometry, k-multiplexed, column-by-column."""
        m = min(int(self.N_o), int(self.N_IP))
        n_rows = int(self.n_rows_thc) + (m if self.chi_embedding == "dilation" else 0)
        return ColumnIsometryRectangularBlockEncoding(
            n_blocks=self.N_k,
            n_rows=n_rows,
            phase_bitsize=self.phase_bitsize,
            n_reflections=m,
            optimal_T=self.optimal_T,
            real_data=self.real_data,
        )

    @cached_property
    def electron_state_prep(self) -> LoadAllStatePreparationQROAM:
        r"""``chi^{V,v}``: the ``mu``-indexed family of virtual-side states, load-all."""
        return LoadAllStatePreparationQROAM(
            n_addr=self.N_k * self.N_IP,
            n_rows=_pow2(self.n_rows_virt),   # rotation tree: zero-padded state
            phase_bitsize=self.phase_bitsize,
            real_data=self.real_data,
        )

    @cached_property
    def central_BE(self) -> BlockEncoding:
        r"""``zeta^V_Q``: eigendecomposition (a congruence) or Hermitian Frobenius."""
        if self.central == "frobenius":
            # Not a congruence -> wrap to make the central unitary Hermitian.  This is the
            # 4-preparation Hermitian Frobenius variant.
            return DirectHermitianBlockEncoding(
                inner=BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(
                    n_blocks=self.N_k,
                    # Frobenius is built from row/column state preparations, so its
                    # dimension is zero-padded to a power of two.  The eigendecomposition
                    # alternative below needs no padding.
                    n_rows=_pow2(self.n_rows_thc),
                    phase_bitsize=self.phase_bitsize,
                    optimal_T=self.optimal_T,
                )
            )
        # zeta^V_Q is Hermitian (verified on the ISDF data), so U D U^dag applies and
        # gives the operator norm.  Self-inverse with no wrapper.
        return EigendecompositionBlockEncoding(
            n_blocks=self.N_k,
            n_rows=self.n_rows_thc,
            phase_bitsize=self.phase_bitsize,
            optimal_T=self.optimal_T,
            real_data=self.real_data,
        )

    @cached_property
    def momentum_prep(self) -> Bloq:
        """The ``N_k^{-1/2} sum_{k,Q}`` isometry's uniform superposition over k."""
        return PrepareUniformSuperposition(n=self.N_k)

    @cached_property
    def momentum_sub(self) -> Bloq:
        """``|k> -> |k - Q>`` on the momentum register."""
        return Subtract(QUInt(self.k_bitsize))

    # --------------------------- BlockEncoding interface ------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        # two particle registers, each (momentum, orbital)
        return 2 * (self.k_bitsize + bit_length(max(self.n_rows_occ, self.n_rows_virt) - 1))

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return (
            self.hole_isometry.ancilla_bitsize
            + self.central_BE.ancilla_bitsize
            + 2 * bit_length(int(self.hole_isometry.n_rows) - 1)  # mu, nu (dilated)
            + self.k_bitsize                        # the Q register
            + 2                                     # the two spin qubits
        )

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.alpha_val

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

    def _calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        """Emit the template's components.

        With ``controlled``, **every** operator-defining piece is controlled, including
        the isometries and state preparations.  The congruence ``M ... M^dag`` does cancel
        at ``ctrl = 0``, so controlling only the central would also be correct -- but that
        argument has to hold exactly, and controlling the legs costs almost nothing:
        a control is absorbed into the unary-iteration tree for **+1 Toffoli per QROAM
        lookup**, independent of table size, and additive with the P-13 range-safety
        surcharge.  Not worth relying on a cancellation argument to save.
        """
        iso = self.hole_isometry.controlled() if controlled else self.hole_isometry
        sp_f = self.electron_state_prep
        sp_i = self.electron_state_prep.adjoint()
        ret[iso] += 2                                        # mu and nu, hole side
        ret[sp_f.controlled() if controlled else sp_f] += 1   # S_chi   (mu)
        ret[sp_i.controlled() if controlled else sp_i] += 1   # S_chi^dag (nu)
        ret[self.central_BE.controlled() if controlled else self.central_BE] += 1
        ret[self.momentum_prep] += 2          # the isometry and its adjoint
        ret[self.momentum_sub] += 2

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        self._calls(ret, controlled=False)
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Only the central gains the control -- the ``M ... M^dag`` pair cancels."""
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledTemplate(self), ctrl_reg_name='ctrl',
        )


# ===========================================================================
# Template 2b -- the ov exchange term via DENSITY FITTING
# ===========================================================================


@attrs.frozen
class ExchangeDensityFittingTemplate(BlockEncoding):
    r"""``C_ov^ex`` built from **density fitting** instead of THC: :math:`L L^\dagger`.

    Density fitting factorizes the bare exchange integral as
    :math:`V_{ov,o'v'} = \sum_P L_{ov,P}\,L^*_{o'v',P}`, so the template is a plain
    congruence with **no central matrix at all** -- the whole operator is
    :math:`L L^\dagger`.

    :math:`L` carries six indices: :math:`\mathbf k`, :math:`\mathbf k'`, the momentum
    transfer :math:`\mathbf Q = \mathbf k \ominus \mathbf k'`, an occupied index, a
    virtual index, and the auxiliary index :math:`P` (dimension
    :math:`N_{\mathrm{THC}}`).  It is compiled as a **single isometry synthesis**:

    * **address** :math:`N_k^2` -- multiplexed on the pair :math:`(\mathbf k, \mathbf k')`;
    * **shape** :math:`N_o N_v \times N_{\mathrm{THC}}` -- the composite ``ov`` index
      against the auxiliary index;
    * :math:`\mathbf Q` is then a modular subtraction on the two momentum registers,
      not a separate lookup.

    :math:`L^\dagger` is the same synthesis inverted.

    Embedding: **SVD**, not dilation.  :math:`L` is not an isometry, so it must be
    completed or decomposed, and here the two routes come out the *opposite* way round
    from the THC template.  With :math:`n_{\mathrm{cols}} = N_oN_v = 88`, dilation pushes
    the row count :math:`208 \to 296`, across a power-of-two boundary -- an extra register
    qubit and an extra synthesis layer.  Measured, dilation is **1.17--1.19x** the SVD
    route at every mesh, so ``chi_embedding='svd'`` is the default *here* while dilation
    remains the default for the THC templates (where it wins 0.91--0.94x).
    """

    N_o: SymbolicInt
    N_v: SymbolicInt
    N_IP: SymbolicInt
    N_k: SymbolicInt
    phase_bitsize: SymbolicInt = 32
    optimal_T: bool = False
    alpha_val: SymbolicFloat = 1.0
    real_data: bool = False
    #: "svd" (default, cheaper here) or "dilation".
    chi_embedding: str = "svd"
    #: Isometry synthesis for L: "berry" (arXiv:2409.11748 Sec. III B, P-07 construction
    #: C) or "iten" (column-by-column).  Berry runs ``N_un/2`` layers once and multiplies
    #: by ``d-1`` with ``d = N_un/chi``; Iten runs ``chi`` column ops of ``n-1`` layers.
    #: So Berry wins for MANY specified columns and loses badly for few.  Measured at
    #: N_THC=208: chi=4 -> 311x worse, chi=22 -> 12.2x worse, chi=88 -> 0.95x BETTER.
    #: DF has chi = N_o*N_v = 88, so "berry"; the THC templates have chi = 4 or 22 and
    #: keep Iten.  Berry needs the rows padded to a multiple of chi (208 -> 264, d = 3).
    synthesis: str = "berry"

    def __attrs_post_init__(self):
        if self.chi_embedding not in ("dilation", "svd"):
            raise ValueError(
                f"chi_embedding must be 'dilation' or 'svd', got {self.chi_embedding!r}"
            )
        if self.synthesis not in ("berry", "iten"):
            raise ValueError(f"synthesis must be 'berry' or 'iten', got {self.synthesis!r}")

    # ------------------------------ shape helpers ------------------------------

    @cached_property
    def k_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_k - 1)

    @cached_property
    def n_addr(self) -> SymbolicInt:
        """``N_k^2`` -- the isometry is multiplexed on the pair ``(k, k')``."""
        return int(self.N_k) ** 2

    @cached_property
    def n_ov(self) -> SymbolicInt:
        """``N_o * N_v`` -- the composite index the isometry maps onto."""
        return int(self.N_o) * int(self.N_v)

    # -------------------------------- sub-bloqs --------------------------------

    @cached_property
    def L_isometry(self):
        r"""``L``: an ``N_o N_v x N_THC`` isometry, multiplexed on ``(k, k')``."""
        n_rows = int(self.N_IP) + (self.n_ov if self.chi_embedding == "dilation" else 0)
        if self.synthesis == "berry":
            return BerryIsometrySynthesisQROAM(
                n_blocks=self.n_addr, n_rows=n_rows, n_cols=self.n_ov,
                phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
            )
        return ColumnIsometryRectangularBlockEncoding(
            n_blocks=self.n_addr, n_rows=n_rows, phase_bitsize=self.phase_bitsize,
            n_reflections=self.n_ov, optimal_T=self.optimal_T, real_data=self.real_data,
        )

    @cached_property
    def sv_diag_shape(self) -> Tuple[SymbolicInt, ...]:
        r"""``(N_k^2, N_oN_v)`` -- ONE singular value per ``(k, k')`` and per mode.

        Not ``(N_k^2, N_oN_v, N_oN_v)``: :math:`L` is :math:`N_oN_v \times
        N_{\mathrm{THC}}`, so it has :math:`\min(N_oN_v, N_{\mathrm{THC}}) = N_oN_v`
        singular values per momentum pair, not a full matrix of them.  An earlier version
        here reused the Coulomb-kernel diagonal, whose shape is
        :math:`(N_k, N_{\mathrm{IP}}, N_{\mathrm{IP}})`, and so loaded a 361-million-entry
        table where 4.1 million is correct.
        """
        return (self.n_addr, self.n_ov)

    @cached_property
    def _sv_diag_lbs(self):
        try:
            from .qroam_block_sizes import optimal_log_block_sizes_measured as _opt
        except ImportError:
            from qroam_block_sizes import optimal_log_block_sizes_measured as _opt
        b = int(self.phase_bitsize)
        return (_opt(self.sv_diag_shape, (b,)), _opt(self.sv_diag_shape, (b,), adjoint=True))

    @cached_property
    def sv_diag_qroam(self) -> Bloq:
        """Load ``theta_s = arccos(sigma_s)``; real singular values -> one b-bit word."""
        return QROAMClean.build_from_bitsize(
            self.sv_diag_shape, target_bitsizes=(self.phase_bitsize,),
            log_block_sizes=self._sv_diag_lbs[0])

    @cached_property
    def sv_diag_qroam_adjoint(self) -> Bloq:
        return QROAMCleanAdjoint.build_from_bitsize(
            self.sv_diag_shape, target_bitsizes=(self.phase_bitsize,),
            log_block_sizes=self._sv_diag_lbs[1])

    @cached_property
    def sv_unitary(self) -> Optional[Bloq]:
        r"""``V``: the small ``N_oN_v x N_oN_v`` right-singular unitary of ``L``.

        **It does not cancel.**  For a bare product one would have
        :math:`L L^\dagger = U\Sigma V^\dagger V \Sigma U^\dagger = U\Sigma^2U^\dagger`,
        but that needs :math:`V^\dagger` and :math:`V` to be the *same* matrix.  They are
        not: the synthesis is multiplexed on the momentum registers, and the modular
        subtraction :math:`\mathbf Q = \mathbf k \ominus \mathbf k'` sits between
        :math:`L` and :math:`L^\dagger` and **rewrites that address**
        (:math:`(\mathbf k,\mathbf k') \to (\mathbf k,\mathbf Q)`), so the second
        synthesis is addressed differently from the first and the product does not
        telescope.  An earlier version dropped ``V`` on the bare-product argument and
        under-counted the template by ~17 %.
        """
        if self.chi_embedding != "svd":
            return None
        return BlockUnitaryInterferometerSynthesisQROAM(
            n_blocks=self.n_addr, n_rows=max(2, self.n_ov),
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
        )

    @cached_property
    def momentum_sub(self) -> Bloq:
        r"""``Q = k - k'`` -- modular subtraction, no lookup."""
        return Subtract(QUInt(self.k_bitsize))

    # --------------------------- BlockEncoding interface ------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        rows = getattr(self.L_isometry, 'n_un', None) or int(self.L_isometry.n_rows)
        return 2 * (self.k_bitsize + bit_length(int(rows) - 1))

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return getattr(self.L_isometry, 'ancilla_bitsize', 1) + bit_length(self.n_ov - 1) + 2

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.alpha_val

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

    def _calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        """``L`` and its inverse, plus the SVD pieces and the momentum subtraction."""
        iso = self.L_isometry
        ret[iso.controlled() if controlled and hasattr(iso, 'controlled') else iso] += 2
        if self.chi_embedding == "svd":
            # Sigma and V on BOTH sides: the modular subtraction between L and L^dag
            # rewrites the multiplexer address, so nothing telescopes.  See sv_unitary.
            for _ in range(2):
                ret[self.sv_diag_qroam] += 1
                emit_range_safety(ret, self.sv_diag_shape)     # P-13
                ret[AddIntoPhaseGrad(self.phase_bitsize,
                                     self.phase_bitsize).controlled()] += 1
                ret[ZGate()] += 1
                ret[self.sv_diag_qroam_adjoint] += 1
            ret[self.sv_unitary] += 2                          # V and V^dag
        ret[self.momentum_sub] += 2                            # Q = k - k', and inverse

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        self._calls(ret, controlled=False)
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledTemplate(self), ctrl_reg_name='ctrl',
        )


# ===========================================================================
# Templates 3-5 -- the ov direct term and its oo / vv siblings
# ===========================================================================


@attrs.frozen
class DirectTemplate(BlockEncoding):
    r"""``C_ov^dir``: :math:`\mathcal X \Delta_\zeta \mathcal X^\dagger` in real space.

    Here :math:`\mu` sits on the electron register (ket *and* bra) and :math:`\nu` on the
    hole register, so each side collapses to :math:`X \,\mathrm{diag}\, X^\dagger` and
    **all four** THC factors are isometries.  The momentum transfer crosses the
    contraction, so both momentum registers are Fourier-transformed to real space and
    combined into :math:`\mathbf R = \mathbf R_1 \ominus \mathbf R_2`; the central then
    reduces to a diagonal addressed by :math:`(\mathbf R, \mu, \nu)`.

    That step is exact and consumes the :math:`1/N_k` of the THC expression -- verified
    numerically: conjugating both momentum registers by the unitary DFT makes the operator
    block diagonal in :math:`(\mathbf R_1,\mathbf R_2)` with entry
    :math:`\tilde\zeta^W_{\mathbf R_1 \ominus \mathbf R_2}` and no residual :math:`N_k`.

    Components: 4 x column isometry + 1 x diagonal block encoding, plus the (lower-order)
    Fourier transforms and modular adder.  ``same_spin`` marks the ``oo`` / ``vv``
    siblings, which share :math:`\zeta^W` and two :math:`\chi` factors with this template
    and therefore contribute only **2** additional isometries each.
    """

    N_o: SymbolicInt
    N_v: SymbolicInt
    N_IP: SymbolicInt
    N_k: SymbolicInt
    phase_bitsize: SymbolicInt = 32
    optimal_T: bool = False
    #: How the THC factor chi is embedded as an isometry.  chi is NOT an isometry --
    #: measured on the ISDF data, ||X^dag X - I|| reaches 31 and the singular values span
    #: 0.002 to 9.42 -- so it cannot be synthesized directly as one.  Two correct routes:
    #:   "dilation" (default) -- complete it: V = [A ; sqrt(I - A^dag A)] has orthonormal
    #:      columns, needs row dimension N_THC + M, and encodes A/||A|| at the operator
    #:      norm.  Because N_THC + M still fits the same register (208+22 = 230 <= 256),
    #:      this costs ~0.1-0.3 % more than the bare (wrong) synthesis and NO extra qubits.
    #:   "svd" -- A = U Sigma V^dag: an isometry U, a real diagonal Sigma (one rotation,
    #:      singular values are real and non-negative), and a small M x M unitary V.
    #: Measured: dilation is 0.91-0.94x the SVD route, the gap being the separate V
    #: synthesis (20,518 Toffoli at M=22, N_k=216).  Same subnormalization either way.
    chi_embedding: str = "dilation"
    alpha_val: SymbolicFloat = 1.0
    real_data: bool = False
    #: None = the ov-direct template (4 isometries + diagonal + FTs);
    #: "oo" / "vv" = the incremental sibling (2 isometries only).
    same_spin: Optional[str] = None

    def __attrs_post_init__(self):
        if self.chi_embedding not in ("dilation", "svd"):
            raise ValueError(
                f"chi_embedding must be 'dilation' or 'svd', got {self.chi_embedding!r}"
            )
        if self.same_spin not in (None, "oo", "vv"):
            raise ValueError(f"same_spin must be None, 'oo' or 'vv', got {self.same_spin!r}")

    @cached_property
    def k_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_k - 1)

    @cached_property
    def n_rows_thc(self) -> SymbolicInt:
        return self.N_IP

    def _isometry(self, n_cols: int) -> ColumnIsometryRectangularBlockEncoding:
        m = min(int(n_cols), int(self.N_IP))
        n_rows = int(self.n_rows_thc) + (m if self.chi_embedding == "dilation" else 0)
        return ColumnIsometryRectangularBlockEncoding(
            n_blocks=self.N_k,
            n_rows=n_rows,
            phase_bitsize=self.phase_bitsize,
            n_reflections=m,
            optimal_T=self.optimal_T,
            real_data=self.real_data,
        )

    @cached_property
    def occ_isometry(self) -> ColumnIsometryRectangularBlockEncoding:
        return self._isometry(self.N_o)

    @cached_property
    def virt_isometry(self) -> ColumnIsometryRectangularBlockEncoding:
        return self._isometry(self.N_v)

    @cached_property
    def diagonal_central(self) -> DiagonalCoulombKernelBlockEncoding:
        r"""``Delta_zeta``: diagonal over ``(R, mu, nu)``, QROAM -> Z R_y -> QROAM^dag."""
        return DiagonalCoulombKernelBlockEncoding(
            N_k=self.N_k, N_IP=self.N_IP,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
        )

    @cached_property
    def fourier(self) -> Bloq:
        """The unitary DFT on one momentum register (drops out of ``lambda``)."""
        return QFTTextBook(bitsize=self.k_bitsize)

    @cached_property
    def momentum_sub(self) -> Bloq:
        """``R = R_1 - R_2``, reversible."""
        return Subtract(QUInt(self.k_bitsize))

    # --------------------------- BlockEncoding interface ------------------------

    @cached_property
    def working_dim(self) -> SymbolicInt:
        """States the THC-side register must hold: the widest isometry's row count.

        With ``chi_embedding='dilation'`` that is ``N_THC + max(N_o, N_v)``, not
        ``N_THC``.  Sizing from ``N_THC`` happens to fit at the current parameters
        (``bit_length(229) == bit_length(207) == 8``) but is wrong in general.
        """
        return max(int(self.occ_isometry.n_rows), int(self.virt_isometry.n_rows))

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return 2 * (self.k_bitsize + bit_length(self.working_dim - 1))

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        # Follows the build order in ``_calls``: oo now carries the shared diagonal, and
        # ov-direct keeps only the momentum register its arithmetic needs.
        if self.same_spin == "oo":
            return self.occ_isometry.ancilla_bitsize + self.diagonal_central.ancilla_bitsize
        if self.same_spin == "vv":
            return self.virt_isometry.ancilla_bitsize
        return self.occ_isometry.ancilla_bitsize + self.k_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.alpha_val

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

    def _calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        """See :meth:`ExchangeTemplate._calls` -- the isometries are controlled too."""
        # Build order: the SAME-SPIN templates are built first and the ov-direct term
        # routes its factors in from them, which is the manuscript's convention.  The oo
        # term needs four occupied factors and the vv term four virtual ones; the direct
        # term needs two of each, and the SWAP network supplies them, so it synthesizes
        # nothing and pays only its Fourier transforms and momentum arithmetic.  The
        # shared central diagonal is charged to oo, where it first appears.
        #
        # This is a reattribution, not a redefinition: the walk still contains
        # 4 occupied + 4 virtual isometries, one diagonal, four Fourier transforms and one
        # subtraction, exactly as before.  The previous code built ov-direct first and made
        # oo / vv incremental at two isometries each; both orders give the same C_walk.
        # No primitive is constructed, resized or retuned by this choice.
        if self.same_spin is not None:
            iso = self.occ_isometry if self.same_spin == "oo" else self.virt_isometry
            ret[iso.controlled() if controlled else iso] += 4
            if self.same_spin == "oo":
                ret[self.diagonal_central.controlled() if controlled
                    else self.diagonal_central] += 1
            return
        ret[self.fourier] += 4                # two registers, forward + inverse
        ret[self.momentum_sub] += 1           # R_1 - R_2

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        self._calls(ret, controlled=False)
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledTemplate(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledTemplate(BlockEncoding):
    """Singly-controlled template: only the central (operator-defining) piece is controlled.

    Each template is a congruence ``M Z M^dag``; the ``M`` / ``M^dag`` pair is applied
    unconditionally because it cancels when ``ctrl = 0``.
    """

    inner: BlockEncoding

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
        return Signature([Register('ctrl', QBit()), *self.inner.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if isinstance(self.inner, FockTemplate):
            ret[self.inner.eig.controlled()] += 1
        else:
            self.inner._calls(ret, controlled=True)
        return ret


# ===========================================================================
# The assembled block encoding
# ===========================================================================


@attrs.frozen
class BSEBlockEncoding(BlockEncoding):
    r"""``U_A``: the manuscript's five-template BSE block encoding.

    Attributes:
        m: exciton number (``2m`` particle registers: ``m`` electrons, ``m`` holes).
        N_o, N_v: occupied / virtual bands per momentum.
        N_IP: THC rank ``N_THC``.
        N_k: number of momenta.
        phase_bitsize: angle bitsize ``b``.
        optimal_T: Toffoli-optimal QROAM blocking throughout.
        exchange_central: ``"eigendecomposition"`` (operator norm, default) or
            ``"frobenius"`` (the Hermitian Frobenius alternative).
        real_data: THC factors and Fock eigenvectors are real -> rotation-only layers.
        lambda_*: per-template subnormalizations; defaults of 1.0 are the data-free
            convention.  ``alpha`` combines them with the manuscript's ``m`` multiplicities.
    """

    m: int
    N_o: int
    N_v: int
    N_IP: int
    N_k: int
    phase_bitsize: int = 32
    optimal_T: bool = False
    exchange_central: str = "eigendecomposition"
    #: Build the ov exchange term by DENSITY FITTING (``L L^dag``, one isometry
    #: multiplexed on ``(k, k')`` with address ``N_k^2``) instead of THC.  Replaces the
    #: whole exchange template, including its central; ``exchange_central`` is then unused.
    ex_density_fitting: bool = False
    real_data: bool = False
    lambda_0_o: SymbolicFloat = 1.0
    lambda_0_v: SymbolicFloat = 1.0
    lambda_oo: SymbolicFloat = 1.0
    lambda_vv: SymbolicFloat = 1.0
    lambda_ov_ex: SymbolicFloat = 1.0
    lambda_ov_dir: SymbolicFloat = 1.0

    def __attrs_post_init__(self):
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
    def orbital_bitsize(self) -> SymbolicInt:
        return bit_length(max(self.N_o, self.N_v) - 1)

    @cached_property
    def reg_bitsize(self) -> SymbolicInt:
        """One particle register: momentum + band."""
        return self.k_bitsize + self.orbital_bitsize

    @cached_property
    def lcu_sel_bitsize(self) -> SymbolicInt:
        return bit_length(_N_TERMS - 1)

    # -------------------------------- templates ---------------------------------

    @cached_property
    def fock_occ(self) -> FockTemplate:
        return FockTemplate(N_k=self.N_k, N=self.N_o, phase_bitsize=self.phase_bitsize,
                            optimal_T=self.optimal_T, alpha_val=self.lambda_0_o,
                            real_data=self.real_data)

    @cached_property
    def fock_virt(self) -> FockTemplate:
        return FockTemplate(N_k=self.N_k, N=self.N_v, phase_bitsize=self.phase_bitsize,
                            optimal_T=self.optimal_T, alpha_val=self.lambda_0_v,
                            real_data=self.real_data)

    @cached_property
    def exchange(self):
        if self.ex_density_fitting:
            return ExchangeDensityFittingTemplate(
                N_o=self.N_o, N_v=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
                phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
                alpha_val=self.lambda_ov_ex, real_data=self.real_data,
            )
        return ExchangeTemplate(
            N_o=self.N_o, N_v=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
            central=self.exchange_central, alpha_val=self.lambda_ov_ex,
            real_data=self.real_data,
        )

    @cached_property
    def direct(self) -> DirectTemplate:
        return DirectTemplate(
            N_o=self.N_o, N_v=self.N_v, N_IP=self.N_IP, N_k=self.N_k,
            phase_bitsize=self.phase_bitsize, optimal_T=self.optimal_T,
            alpha_val=self.lambda_ov_dir, real_data=self.real_data,
        )

    @cached_property
    def oo(self) -> DirectTemplate:
        return attrs.evolve(self.direct, same_spin="oo", alpha_val=self.lambda_oo)

    @cached_property
    def vv(self) -> DirectTemplate:
        return attrs.evolve(self.direct, same_spin="vv", alpha_val=self.lambda_vv)

    # ------------------------------- routing ------------------------------------

    @cached_property
    def route_swap(self) -> Bloq:
        """One controlled swap of a whole particle register."""
        return CSwap(self.reg_bitsize)

    @cached_property
    def n_route_swaps(self) -> int:
        r"""``O(m log[N_k(N_o+N_v)])`` -- linear in ``m``, not polylog.

        A unary-iterated controlled-SWAP network selects one of ``m`` registers per
        partition; the two-body templates need two routed registers and the ``ov`` terms
        route one from each partition.  Four routed slots, each an ``m``-way select, each
        applied forward and back around the templates.
        """
        return 0 if is_symbolic(self.m) else 4 * 2 * int(self.m)

    @cached_property
    def lcu_prep(self) -> Bloq:
        """PREPARE over the five cost categories (data-free proxy)."""
        return PrepareUniformSuperposition(n=_N_TERMS)

    # --------------------------- BlockEncoding interface ------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return 2 * self.m * self.reg_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        max_term = max(
            self.fock_occ.ancilla_bitsize,
            self.fock_virt.ancilla_bitsize,
            self.exchange.ancilla_bitsize,
            self.direct.ancilla_bitsize,
        )
        return self.lcu_sel_bitsize + 2 * bit_length(self.m) + max_term

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        r"""``lambda = m l_0 + m(m-1)/2 (l_oo + l_vv) + m^2 (l_ex + l_dir)``."""
        m = self.m
        return (
            m * (self.lambda_0_o + self.lambda_0_v)
            + (m * (m - 1) / 2) * (self.lambda_oo + self.lambda_vv)
            + m ** 2 * (self.lambda_ov_ex + self.lambda_ov_dir)
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

    # ------------------------------ resource counts -----------------------------

    def _select_calls(self, ret: "Counter[Bloq]", *, controlled: bool) -> None:
        """SELECT: each template compiled once (C_0 = the two one-particle templates).

        The templates are **always** emitted in controlled form.  A SELECT applies each
        term conditioned on the LCU index, so the control is intrinsic to the block
        encoding, not an optional extra -- and every operator-defining piece inside a
        template carries it, isometries and state preparations included, rather than
        leaning on the ``M ... M^dag`` congruence cancelling at ``ctrl = 0``.

        This is affordable because a control is absorbed into the unary-iteration tree at
        **+1 Toffoli per QROAM lookup**, independent of table size and additive with the
        P-13 range-safety surcharge.  Across the whole walk it is +800 Toffolis, flat in
        ``N_k`` (the lookup *count* does not depend on the mesh).

        **Cheaper convention, selected here.**  The congruence ``M ... M^dag`` inside each
        template cancels at ``ctrl = 0``, so only the operator-defining central actually
        needs the index control; the legs may be applied unconditionally.  That is what
        ``controlled=False`` emits, and it is the default.  Controlling every leg instead
        costs +800 Toffolis over the whole walk (flat in ``N_k``) -- see the templates'
        ``_calls``, which still support it via ``controlled=True``.
        """
        for t in (self.fock_occ, self.fock_virt, self.exchange, self.direct,
                  self.oo, self.vv):
            ret[t.controlled() if controlled else t] += 1

    def _route_calls(self, ret: "Counter[Bloq]") -> None:
        """``C_route``: the controlled-SWAP network plus the LCU PREPARE pair."""
        ret[self.lcu_prep] += 2
        if self.n_route_swaps:
            ret[self.route_swap] += self.n_route_swaps

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        self._route_calls(ret)
        self._select_calls(ret, controlled=False)
        return ret

    def template_costs(self) -> dict:
        """Per-template Toffoli counts, keyed by the manuscript's ``C_*`` symbols."""
        try:
            from .toffoli_cost import toffoli_count as tof
        except ImportError:
            from toffoli_cost import toffoli_count as tof

        route = Counter()
        self._route_calls(route)

        return {
            'C_0': tof(self.fock_occ) + tof(self.fock_virt),
            'C_oo': tof(self.oo),
            'C_vv': tof(self.vv),
            'C_ov_ex': tof(self.exchange),
            'C_ov_dir': tof(self.direct),
            'C_route': sum(tof(b) * n for b, n in route.items()),
        }

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBSEBlockEncoding(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledBSEBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`BSEBlockEncoding`; the PREPARE / routing pairs stay free."""

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
        self.inner._route_calls(ret)
        self.inner._select_calls(ret, controlled=True)
        return ret
