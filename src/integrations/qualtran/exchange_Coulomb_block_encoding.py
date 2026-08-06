r"""Data-free block encoding of the periodic THC two-electron integral tensor.

This mirrors the tensor-hypercontraction (THC) factorization of arXiv:2601.16379
Eq. (9):

    (mu_{k_mu} nu_{k_nu} | lam_{k_lam} sig_{k_sig})
        = sum_{IJ}  X_I^{mu, k_mu}  X_I^{nu, -k_nu}  X_J^{lam, k_lam}  X_J^{sig, -k_sig}  W_{IJ}^{q}

with momentum transfer  q = k_mu - k_nu (+ B_n).  The "other tensors" are the
interpolating vectors X (shape N_IP x N_AO per k-point, block index k); the
"central block" is the Coulomb kernel W (shape N_IP x N_IP per q, block index q).

Block-encoding map (B . C . B^dagger sandwich):

  1.  Two reflection-rectangular block encodings load the bra-pair X tensors:
        B_up   = BE of sum_{k1} |k1><k1| (x) X^{mu, +k1}   [isometry N_up  x N_IP]
        B_down = BE of sum_{k2} |k2><k2| (x) X^{nu, -k2}   [isometry N_down x N_IP]
      i.e. mu carries momentum +k1 and nu carries -k2 (the -k_nu of Eq. 9; the
      negation is the ``mod_neg`` step below).  M = B_up (x) B_down keeps the two
      interpolation registers I_up, I_down independent (the "two-register" model),
      and two block registers |k1>, |k2> of bitsize ceil(log2 N_k).
  2.  Negate the nu momentum:  |k2> -> |-k2 mod N_k>   (``mod_neg``), realizing
      the -k_nu argument of X_I^{nu, -k_nu} in Eq. 9.
  3.  Form the momentum transfer  q = k1 - k2 mod N_k  (``mod_sub``):
        |k1>|k2>  -->  |k1>|q>.
  4.  Post-select |k1> against the uniform superposition (1/sqrt(N_k)) sum_k |k>
      (``PrepareUniformSuperposition(N_k).adjoint()`` projected onto |0>).  The
      surviving momentum register holds q.  Everything above is ``B``.
  5.  Apply the central Coulomb-kernel block encoding, block-indexed by q:
        C = BE of sum_{q} |q><q| (x) W^{q}            [W is N_IP x N_IP]
      (N_q = N_k distinct momentum transfers).
  6.  Apply B^{dagger} (the ket pair X^{lam,+k}, X^{sig,-k} sharing J).  Full:

        B_total = B . C . B^{dagger}.

  Because of the sandwich structure, the externally-controlled version only has
  to control C: the B / B^{dagger} pair self-cancels when the external control
  is 0, so the controlled block encoding costs essentially the uncontrolled cost
  plus the small controlled-C overhead.

This module is *data-free*: it only describes the structural circuit and emits
the right sub-bloqs in ``build_call_graph``.  All Toffoli and qubit counts come
from Qualtran's resource counter walking that call graph.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from math import ceil, log2
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, QUInt, Register, Signature
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.arithmetic import LessThanConstant, Negate, Subtract  # ModSub has Qualtran bugs; Subtract/Negate as cost proxies

from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, SymbolicFloat, SymbolicInt

try:
    from .block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from .block_unitary_interferometer_QROAM import optimal_interferometer_log_block_sizes
    from .block_unitary_reflection_QROAM import (
        optimal_reflection_adjoint_log_block_sizes,
        optimal_reflection_log_block_sizes,
    )
    from .classical_matrix_block_encoding_QROAM import (
        BlockDiagonalClassicalMatrixBlockEncoding,
        DirectHermitianBlockEncoding,
    )
    from .rectangular_block_encoding_reflection import (
        ReflectionRectangularBlockEncoding,
    )
    from .svd_block_encoding_interferometer import SVDBlockEncodingInterferometer
except ImportError:
    from block_isometry_column_synthesis_QROAM import ColumnIsometryRectangularBlockEncoding
    from block_unitary_interferometer_QROAM import optimal_interferometer_log_block_sizes
    from block_unitary_reflection_QROAM import (
        optimal_reflection_adjoint_log_block_sizes,
        optimal_reflection_log_block_sizes,
    )
    from classical_matrix_block_encoding_QROAM import (
        BlockDiagonalClassicalMatrixBlockEncoding,
        DirectHermitianBlockEncoding,
    )
    from rectangular_block_encoding_reflection import (
        ReflectionRectangularBlockEncoding,
    )
    from svd_block_encoding_interferometer import SVDBlockEncodingInterferometer

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _next_power_of_two(n: int) -> int:
    return 1 << max(1, (int(n) - 1).bit_length())


@attrs.frozen
class ExchangeCoulombBlockEncoding(BlockEncoding):
    """Data-free exchange-Coulomb block encoding.

    The encoded operator acts on the joint (k1, sys_up, k2, sys_down) register.
    "Data-free" means the QROAM angle tables inside each component are sized
    correctly but never populated with real numbers; resource counts come from
    Qualtran's call-graph traversal.

    Construction policy:
      * Rectangular matrices (A_{k}^{up}: N_up x N_IP, A_{k}^{down}: N_down x N_IP)
        are block encoded by :class:`ReflectionRectangularBlockEncoding` (Householder
        isometry synthesis -- only ``n_reflections = N_IP`` columns are synthesized
        instead of a full N x N unitary, so cost scales with the rectangle's column
        count rather than the padded square size).
      * Square matrices (A'_Q: N_IP x N_IP, sitting in the middle of the sandwich)
        are block encoded by :class:`SVDBlockEncodingInterferometer` (interferometer
        synthesis of U_k and V_k plus a QROAM-loaded diagonal Sigma_k), which is the
        cheapest known full-rank N x N block encoding.

    Attributes:
        N_up: row count of A_{k1}^{up}.
        N_down: row count of A_{k2}^{down}.
        N_IP: shared column count for A^{up}, A^{down}, and A'_Q.
        N_k: number of blocks (k1, k2, Q each range over [0, N_k)).
        phase_bitsize: bitsize of phase / angle registers used inside each
            sub-block encoding.
        outer_amp_log_block_sizes / outer_phase_log_block_sizes /
        outer_phase_adjoint_log_block_sizes:
            log_block_sizes for the amplitude staircase, final phase layer forward,
            and final phase layer adjoint QROAMs inside each rectangular reflection
            block encoding (B_up and B_down).
        inner_intf_log_block_sizes / inner_intf_final_log_block_sizes /
        inner_intf_final_adjoint_log_block_sizes:
            log_block_sizes for the U_k / V_k phase-layer / final-phase / final-phase
            adjoint QROAMs inside the middle SVD block encoding.
        inner_diag_log_block_sizes / inner_diag_adjoint_log_block_sizes:
            log_block_sizes for the diagonal Sigma_k QROAM forward / adjoint inside
            the middle SVD block encoding.
    """

    N_up: int
    N_down: int
    N_IP: int
    N_k: int
    phase_bitsize: int = 32
    outer_amp_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    outer_amp_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    outer_phase_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    outer_phase_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    inner_intf_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    inner_intf_final_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    inner_intf_final_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    inner_diag_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    inner_diag_adjoint_log_block_sizes: Optional[Tuple[int, ...]] = (0, 0)
    optimal_T: bool = False
    # When True, encode the central A'_Q via :class:`ReflectionRectangularBlockEncoding`
    # (Householder isometry, n_reflections = N_IP) instead of the SVD interferometer.
    # The outer reflection lbs (``outer_amp_*`` / ``outer_phase_*``) are reused for the
    # central reflection's QROAM staircase since the central register has the same
    # padded dimension as the outer ones (next_pow2(N_IP)).  Toggled in callers; the
    # SVD path remains the default.
    central_via_reflection: bool = False
    # When True, encode the central W^q tensor (just W, NOT the X matrices) via the
    # Clader-Frobenius block-encoding scheme for block matrices
    # (:class:`BlockDiagonalClassicalMatrixBlockEncoding`, ``U_A = U_L^dag U_R``) instead of
    # the SVD interferometer ("unitary synthesis").  Its QROAM batching follows its own
    # ``optimal_T``; the ``inner_*`` lbs attributes are unused on this path.  Takes
    # precedence over ``central_via_reflection``.
    use_fro_BE: bool = False
    # Synthesis method for the rectangular A^{up}/A^{down} matrices (B_up, B_down):
    # "reflection" (LKS Householder, default) or "column" (column-by-column isometry
    # synthesis, Iten 1501.06911 + Berry Eq. 24).  The central A'_Q is unaffected.
    outer_synthesis: str = "reflection"
    # Restrict each orbital input to its physical range with a comparator.  Each outer X
    # channel acts on an N_k x N_IP system register (block k + matrix sized to
    # ceil(log2 N_IP)), and the full construction on two such registers (double).  When
    # True a ``LessThanConstant`` flags ``mu < N_up`` (= N_o) and ``nu < N_down`` (= N_v)
    # into a 1-qubit ancilla per channel, restraining the encoded operator's input to the
    # physical orbital subspace (computed + uncomputed; O(log N_IP) Toffoli).
    restrict_input: bool = True
    # When True, wrap the central W^q block encoding in
    # :class:`DirectHermitianBlockEncoding` (W = (H (x) I) S (H (x) I), S = [[0,U],[U^dag,0]])
    # so the central *unitary* is Hermitian (W = W^dag, W^2 = I) while block-encoding the SAME
    # Hermitian Coulomb kernel A'_Q.  Costs ~2x the base central block encoding and one extra
    # ancilla qubit (accounted for in ``ancilla_bitsize``).  Requires A'_Q = (A'_Q)^dag.
    hermitian_central: bool = False
    # When True, encode the central A'_Q via the *Hermitian Frobenius-norm* block encoding:
    # the Clader-Frobenius dense block-matrix BE (:class:`BlockDiagonalClassicalMatrixBlockEncoding`,
    # ``U_A = U_L^dag U_R``) wrapped in :class:`DirectHermitianBlockEncoding` so the central
    # *unitary* is Hermitian and involutive (W = W^dag, W^2 = I).  This is the single switch the
    # qubitization *BSE walk operator* uses to make the whole exchange sandwich ``B C B^dag``
    # Hermitian-unitary (``B`` is a genuine unitary, so ``B C B^dag`` is an involution exactly
    # when the central ``C`` is).  Selects the Frobenius base (like ``use_fro_BE``) AND forces
    # the Hermitian wrap (like ``hermitian_central``) in one toggle; it takes precedence over
    # ``use_fro_BE`` / ``central_via_reflection`` for the base choice and adds the one Hermitian
    # flag ancilla.  The SVD / diagonal block encodings used elsewhere are already involutive
    # (they apply ``Z R_y`` rather than ``R_y``), so only the full-matrix central needs this.
    hermitian_fro_central: bool = False

    def __attrs_post_init__(self):
        if self.outer_synthesis not in ("reflection", "column"):
            raise ValueError(
                f"outer_synthesis must be 'reflection' or 'column', got {self.outer_synthesis!r}"
            )
        if self.optimal_T:
            # Outer (rectangular reflection) sub-bloqs B_up and B_down have different
            # padded n_rows; pick the larger for the stored attribute. The sub-bloqs
            # themselves still re-optimize for their own shape via optimal_T propagation.
            n_rows_outer = max(self.n_rows_up, self.n_rows_down)
            opt_outer_fwd = optimal_reflection_log_block_sizes(
                int(self.N_k), int(n_rows_outer), int(self.phase_bitsize)
            )
            opt_outer_adj = optimal_reflection_adjoint_log_block_sizes(
                int(self.N_k), int(n_rows_outer)
            )
            opt_inner = optimal_interferometer_log_block_sizes(
                int(self.N_k), int(self.n_rows_inner), int(self.phase_bitsize)
            )
            object.__setattr__(self, 'outer_amp_log_block_sizes', opt_outer_fwd)
            object.__setattr__(self, 'outer_amp_adjoint_log_block_sizes', opt_outer_adj)
            object.__setattr__(self, 'outer_phase_log_block_sizes', opt_outer_fwd)
            object.__setattr__(self, 'outer_phase_adjoint_log_block_sizes', opt_outer_adj)
            for field in (
                'inner_intf_log_block_sizes',
                'inner_intf_final_log_block_sizes',
                'inner_intf_final_adjoint_log_block_sizes',
                'inner_diag_log_block_sizes',
                'inner_diag_adjoint_log_block_sizes',
            ):
                object.__setattr__(self, field, opt_inner)

    # ------------------------- shape helpers -------------------------

    @cached_property
    def n_rows_up(self) -> int:
        return _next_power_of_two(max(self.N_up, self.N_IP))

    @cached_property
    def n_rows_down(self) -> int:
        return _next_power_of_two(max(self.N_down, self.N_IP))

    @cached_property
    def n_rows_inner(self) -> int:
        # A'_Q acts on the matrix register of one of the outer block encodings;
        # in practice we size it to N_IP (since the outer BEs' system bitsize
        # is determined by N_IP after padding).
        return _next_power_of_two(self.N_IP)

    @cached_property
    def k_bitsize(self) -> int:
        return bit_length(self.N_k - 1)

    # --------------------- BlockEncoding interface ---------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        # k1 + sys_up + k2(=Q) + sys_down  (all retained throughout the encoding)
        return (
            self.k_bitsize + bit_length(self.n_rows_up - 1)
            + self.k_bitsize + bit_length(self.n_rows_down - 1)
        )

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        # Two outer BEs' ancillas (matrix-row aux + flag) + middle BE's ancilla, plus
        # (when restrict_input) one input-range comparator flag per orbital channel.
        n_up = bit_length(self.n_rows_up - 1)
        n_down = bit_length(self.n_rows_down - 1)
        n_mid = bit_length(self.n_rows_inner - 1)
        cmp_flags = 2 if self.restrict_input else 0
        # DirectHermitianBlockEncoding adds one Hermitian-flag ancilla to the central BE.
        herm_flag = 1 if (self.hermitian_central or self.hermitian_fro_central) else 0
        return (n_up + 1) + (n_down + 1) + (n_mid + 1) + cmp_flags + herm_flag

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        # Single phase-gradient workspace shared across all sub-bloqs.
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        # B carries alpha = N_up * N_down (reflection-BE subnormalization for each),
        # multiplied by N_k from the uniform postselect.  C contributes another factor
        # equal to its alpha (N_IP after padding).  B^dagger contributes the same as B.
        return (
            float(self.n_rows_up) * float(self.n_rows_down)
            * float(self.N_k)
            * float(self.n_rows_inner)
            * float(self.n_rows_up) * float(self.n_rows_down)
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

    # --------------------- Sub-bloq factories ---------------------

    def _outer_be(self, n_rows: int, n_reflections: int) -> BlockEncoding:
        """Rectangular X-matrix isometry block encoding, per ``outer_synthesis``.

        Both choices share the same (system = block + matrix, ancilla = 1, resource =
        phase-gradient, alpha = 1) interface so they are drop-in interchangeable; only
        the internal synthesis (Householder reflections vs column-by-column) differs.
        """
        if self.outer_synthesis == "column":
            return ColumnIsometryRectangularBlockEncoding(
                n_blocks=self.N_k,
                n_rows=n_rows,
                phase_bitsize=self.phase_bitsize,
                n_reflections=n_reflections,
                optimal_T=self.optimal_T,
            )
        return ReflectionRectangularBlockEncoding(
            n_blocks=self.N_k,
            n_rows=n_rows,
            phase_bitsize=self.phase_bitsize,
            n_reflections=n_reflections,
            amp_log_block_sizes=self.outer_amp_log_block_sizes,
            amp_adjoint_log_block_sizes=self.outer_amp_adjoint_log_block_sizes,
            phase_log_block_sizes=self.outer_phase_log_block_sizes,
            phase_adjoint_log_block_sizes=self.outer_phase_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    @property
    def B_up(self) -> BlockEncoding:
        # Rectangular A^{up} (N_up x N_IP):  isometry on an n_rows_up-dim register
        # (= pad to power-of-two of max(N_up, N_IP)).  A^{up} has rank <= min(N_up, N_IP),
        # so we synthesize along the *short* direction (min(N_up, N_IP) vectors).
        return self._outer_be(self.n_rows_up, min(self.N_up, self.N_IP))

    @property
    def B_down(self) -> BlockEncoding:
        # Rectangular A^{down} (N_down x N_IP):  min(N_down, N_IP) synthesized vectors.
        return self._outer_be(self.n_rows_down, min(self.N_down, self.N_IP))

    @property
    def C_inner(self) -> BlockEncoding:
        """Central block encoding of ``A'_Q``; Hermitian-unitary-wrapped if requested.

        When ``hermitian_central`` is set, the chosen base block encoding (SVD / reflection
        / Clader-Frobenius) is wrapped in :class:`DirectHermitianBlockEncoding` so the
        central unitary is Hermitian (``W = W^dag``) while encoding the SAME ``A'_Q``; this
        adds one ancilla and doubles the central cost.
        """
        base = self._base_C_inner
        if self.hermitian_central or self.hermitian_fro_central:
            return DirectHermitianBlockEncoding(base)
        return base

    @property
    def _base_C_inner(self) -> BlockEncoding:
        # Central block encoding of A'_Q (square, N_IP x N_IP).  Two choices:
        #   * SVD interferometer (default, ``central_via_reflection=False``): cheapest
        #     known full-rank N x N block encoding with alpha = 1.
        #   * Householder reflection isometry (``central_via_reflection=True``): uses
        #     n_reflections = N_IP block reflections on the padded n_rows_inner-dim
        #     register, sharing the outer reflection lbs (same padded dim).
        #   * Clader-Frobenius block-matrix BE (``use_fro_BE=True``, or the Hermitian
        #     ``hermitian_fro_central=True`` which additionally wraps it in
        #     :class:`DirectHermitianBlockEncoding` -- see ``C_inner``): U_A = U_L^dag U_R
        #     column/row state preparation; alpha = F_max.
        if self.use_fro_BE or self.hermitian_fro_central:
            return BlockDiagonalClassicalMatrixBlockEncoding.from_bitsize(
                n_blocks=self.N_k,
                n_rows=self.n_rows_inner,
                phase_bitsize=self.phase_bitsize,
                optimal_T=self.optimal_T,
            )
        if self.central_via_reflection:
            return ReflectionRectangularBlockEncoding(
                n_blocks=self.N_k,
                n_rows=self.n_rows_inner,
                phase_bitsize=self.phase_bitsize,
                n_reflections=self.N_IP,
                amp_log_block_sizes=self.outer_amp_log_block_sizes,
                amp_adjoint_log_block_sizes=self.outer_amp_adjoint_log_block_sizes,
                phase_log_block_sizes=self.outer_phase_log_block_sizes,
                phase_adjoint_log_block_sizes=self.outer_phase_adjoint_log_block_sizes,
                optimal_T=self.optimal_T,
            )
        return SVDBlockEncodingInterferometer(
            n_blocks=self.N_k,
            n_rows=self.n_rows_inner,
            phase_bitsize=self.phase_bitsize,
            interferometer_log_block_sizes=self.inner_intf_log_block_sizes,
            interferometer_final_log_block_sizes=self.inner_intf_final_log_block_sizes,
            interferometer_final_adjoint_log_block_sizes=self.inner_intf_final_adjoint_log_block_sizes,
            diag_log_block_sizes=self.inner_diag_log_block_sizes,
            diag_adjoint_log_block_sizes=self.inner_diag_adjoint_log_block_sizes,
            optimal_T=self.optimal_T,
        )

    @property
    def mod_neg(self) -> Bloq:
        # |k2> -> |-k2 mod N_k>, realizing the -k_nu argument of X_I^{nu,-k_nu} in Eq. 9.
        # Two's-complement negation on QUInt(n_k); ~O(n_k) Toffoli -- negligible.
        return Negate(QUInt(self.k_bitsize))

    @property
    def mod_sub(self) -> Bloq:
        # |k1, k2> -> |k1, k1 - k2 mod N_k> = |k1, q>.  q = k_mu - k_nu is the momentum
        # transfer indexing W in Eq. 9.  Modeled by Subtract on QUInt(n_k); a true
        # modular subtraction is Subtract + comparison + conditional add, all O(n_k)
        # Toffoli -- negligible compared to the QROAMClean costs that dominate.
        return Subtract(a_dtype=QUInt(self.k_bitsize))

    @property
    def uniform_prep(self) -> Bloq:
        return PrepareUniformSuperposition(n=self.N_k)

    @property
    def input_comparator_up(self) -> Bloq:
        # Restrain the mu input (occupied): flag x < N_up (= N_o) on the N_IP-sized matrix
        # register of the up channel.  Negligible O(log N_IP) Toffoli.
        return LessThanConstant(bitsize=bit_length(self.n_rows_up - 1), less_than_val=self.N_up)

    @property
    def input_comparator_down(self) -> Bloq:
        # Restrain the nu input (virtual): flag x < N_down (= N_v) on the down channel.
        return LessThanConstant(bitsize=bit_length(self.n_rows_down - 1), less_than_val=self.N_down)

    # --------------------- Resource counts ---------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        # Full = B . C . B^dagger.  For data-free cost counting each forward bloq
        # and its adjoint contribute the same Toffoli / qubit cost, so emit each
        # outer component twice rather than emitting an explicit .adjoint().
        # (Qualtran's QROAMClean has an unhashable internal field that breaks the
        # default Adjoint resource walk, and we don't need the structural label.)
        # Restrain each orbital input to its physical N_o / N_v range (compute + uncompute).
        if self.restrict_input:
            ret[self.input_comparator_up] += 2
            ret[self.input_comparator_down] += 2
        ret[self.B_up] += 2
        ret[self.B_down] += 2
        if self.N_k > 1:
            ret[self.mod_neg] += 2   # -k_nu momentum negation (Eq. 9), B and B^dagger
            ret[self.mod_sub] += 2   # q = k1 - k2 momentum transfer, B and B^dagger
            ret[self.uniform_prep] += 2
        ret[self.C_inner] += 1       # central Coulomb kernel W^q, block-indexed by q
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Cheap single-qubit control via the sandwich identity.

        ``.controlled()`` returns :class:`_ControlledExchangeCoulombBlockEncoding`,
        which only controls the middle ``C`` block encoding (itself controlled cheaply
        through ``SVDBlockEncodingInterferometer.get_ctrl_system``).  The outer
        ``B``/``B^dagger`` reflection pair and the modular-subtraction / uniform-prep
        bookkeeping stay uncontrolled because they cancel pairwise when ``ctrl = 0``.
        """
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledExchangeCoulombBlockEncoding(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledExchangeCoulombBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`ExchangeCoulombBlockEncoding`.

    Exploits the sandwich structure ``B . C . B^dagger``: only the middle ``C``
    needs an external control, because the outer ``B``/``B^dagger`` pair acts as
    identity when the external control is 0.  This adds the small overhead of
    the controlled middle block encoding on top of the uncontrolled cost.
    """

    inner: ExchangeCoulombBlockEncoding

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
        # Sandwich identity: only C is promoted to a controlled BE.  Outer B and
        # B^dagger cancel pairwise on ext_ctrl = 0 so they stay uncontrolled.
        if self.inner.restrict_input:
            ret[self.inner.input_comparator_up] += 2
            ret[self.inner.input_comparator_down] += 2
        ret[self.inner.B_up] += 2
        ret[self.inner.B_down] += 2
        if self.inner.N_k > 1:
            ret[self.inner.mod_neg] += 2
            ret[self.inner.mod_sub] += 2
            ret[self.inner.uniform_prep] += 2
        # Central C (Coulomb kernel W^q) is the only piece that gets the external control.
        ret[self.inner.C_inner.controlled()] += 1
        return ret
