r"""Frobenius block-encoding of a dense matrix of classical data (Clader et al., 2022).

Implements the *minimal-T-count* block-encoding of a dense ``N x N`` real matrix ``A``
of classical data from

    [Quantum Resources Required to Block-Encode a Matrix of Classical Data]
    (https://arxiv.org/abs/2206.03505).
    Clader, Dalzell, Stamatopoulos, Salton, Berta, Zeng. 2022.

Only the T-count-optimal construction is provided here (the paper's "fixed-precision"
state-preparation route backed by select-swap / QROAM data loading).  The minimal-*depth*
"pre-rotated" route -- which carries an unavoidable ``Omega(N^2)`` T-count -- is *not*
implemented.

Construction (paper Sec. III D / IV, Eqs. (4)-(7) and Fig. 1)
------------------------------------------------------------
The block-encoding unitary is the product of a pair of (controlled) state preparations

.. math::  U_A = U_R^\dagger \, U_L,

acting on two ``n``-qubit registers (``N = 2^n``) -- a *prep* register and an *index*
register -- plus a shared phase-gradient workspace.  With

.. math::
    |\psi_j\rangle = \sum_k \frac{A_{jk}}{\lVert A_{j,\cdot}\rVert} |k\rangle, \qquad
    |\varphi\rangle = \sum_j \frac{\lVert A_{j,\cdot}\rVert}{\lVert A\rVert_F} |j\rangle,

``U_R`` is the controlled state preparation that builds ``|psi_j>`` in the prep register
controlled on the index register ``|j>``, while ``U_L`` prepares the *index-independent*
state ``|varphi>`` and swaps the two registers.  One verifies

.. math::  (\langle 0|_n \otimes \langle j|_n)\, U_A\, (|0\rangle_n \otimes |k\rangle_n)
           = A_{jk} / \lVert A\rVert_F,

so ``U_A`` is a ``(\lVert A\rVert_F, n, \epsilon)``-block-encoding of ``A``.  Because
``|varphi>`` is independent of the control register, ``U_L`` is *plain* (uncontrolled)
state preparation rather than controlled state preparation -- the resource-saving
optimization noted in Sec. IV A.

The two state preparations reuse the project's QROAM-rotation primitives, which load the
``t``-bit rotation angles with select-swap QROAM and apply them through a phase gradient
(Low-Kliuchnikov-Schaeffer, arXiv:1812.00954).  Choosing the QROAM block sizes at their
analytic ``lambda*`` optimum is exactly the minimal-T-count regime of the paper.

Registers (the standard ``qualtran`` :class:`BlockEncoding` interface)
----------------------------------------------------------------------
* ``system``   -- ``n``-qubit index register that ``A`` acts on (``|k> -> |j>``).
* ``ancilla``  -- ``n``-qubit prep register; the block-encoding flag (signal ``|0>^n``).
* ``resource`` -- ``phase_bitsize``-qubit phase-gradient workspace (shared by both preps).

QROAM data-load ancillas are internal to the state-preparation sub-bloqs.

``build_call_graph`` returns exactly the two state preparations and the register swap, so
the Qualtran resource counter walks the real sub-bloq graph (no analytic formulas here).
A data-free constructor (:meth:`from_bitsize`) supports resource estimates from sizes
alone; :meth:`from_matrix` builds the data-bearing, decomposable version.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property, lru_cache
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, Swap
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, HasLength, is_symbolic, Shaped, SymbolicFloat, SymbolicInt

try:
    from .block_state_preparation_QROAM import BlockStatePreparationViaQROAMRotations
    from .state_prep_QROAM import StatePreparationViaQROAMRotations
    from .three_phase_layer_state_prep_QROAM import ThreePhaseLayerStatePreparation
except ImportError:  # pragma: no cover - script / notebook execution
    from block_state_preparation_QROAM import BlockStatePreparationViaQROAMRotations
    from state_prep_QROAM import StatePreparationViaQROAMRotations
    from three_phase_layer_state_prep_QROAM import ThreePhaseLayerStatePreparation

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length() if x > 1 else 1


@lru_cache(maxsize=None)
def _min_T_block_prep_lbs(
    n_blocks: int, n_coeff: int, phase_bitsize: int, uncompute: bool = True
) -> Tuple[int, int]:
    r"""Block sizes ``(log_k_block, log_k_row)`` minimizing the Toffoli count of a block prep.

    For a block state preparation with ``n_blocks`` addresses, the dominant QROAM is the
    ``n_blocks x n_coeff`` phase table, so the T-minimum lives at a large select-swap batch
    -- which the per-layer ``lambda*`` heuristic never reaches and which QROAMClean's own
    ``None`` default (a qubit-conscious point) also undershoots.  We search a uniform literal
    ``(log_k_block, log_k_row)`` applied to every QROAM table (forward + adjoint, amplitude
    + phase, each capped per layer) and pick the Toffoli-minimal choice.  This is the
    minimal-T *extreme* of the qubit/T trade-off (it spends many ancillas); the qubit-minimal
    extreme is ``lambda = 1``.  Costs are shape-only, so the result is cached.
    """
    try:  # local import: avoids a module-load cycle, works as package or script
        from .utils import get_Toffoli_counts
    except ImportError:
        from utils import get_Toffoli_counts

    nb = int(bit_length(n_blocks - 1))
    nc = int(bit_length(n_coeff - 1))
    best: Optional[Tuple[int, int, int]] = None
    for lbb in range(0, nb + 1):
        for lbr in range(0, nc + 1):
            t = (lbb, lbr)
            try:
                bsp = BlockStatePreparationViaQROAMRotations.from_bitsize(
                    n_blocks,
                    n_coeff,
                    phase_bitsize,
                    uncompute=uncompute,
                    amp_log_block_sizes=t,
                    amp_adjoint_log_block_sizes=t,
                    phase_log_block_sizes=t,
                    phase_adjoint_log_block_sizes=t,
                    per_layer_optimal=False,
                    measure_reset=True,  # forward layers erase by X-measurement (0 Toffoli)
                )
                cost = int(get_Toffoli_counts(bsp))
            except Exception:  # noqa: BLE001 - skip block sizes invalid for this shape
                continue
            if best is None or cost < best[0]:
                best = (cost, lbb, lbr)
    return (best[1], best[2]) if best is not None else (0, 0)


def _min_T_row_prep_lbs(n_blocks: int, n_coeff: int, phase_bitsize: int) -> Tuple[int, int]:
    """Toffoli-minimal block sizes for the adjoint row prep ``U_R^dag`` (uncompute=True)."""
    return _min_T_block_prep_lbs(n_blocks, n_coeff, phase_bitsize, True)


def _coeffs_from_matrix(
    A: NDArray[np.floating],
) -> Tuple[NDArray[np.complex128], Tuple[complex, ...], float, int]:
    r"""Return ``(psi_rows, phi_col, alpha, N)`` for the dense block-encoding of ``A``.

    ``A`` is zero-padded to ``N x N`` with ``N`` a power of two.  ``psi_rows[j]`` is row
    ``j`` normalized to unit Euclidean norm (the state ``|psi_j>``); ``phi_col`` holds the
    row norms divided by the Frobenius norm (the state ``|varphi>``); ``alpha`` is the
    Frobenius norm ``\lVert A\rVert_F``.  Degenerate (all-zero) rows are mapped to the
    ``|0>`` basis state, which is harmless because their ``phi`` weight is zero.
    """
    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    n_rows, n_cols = A.shape
    N = _next_pow2(max(n_rows, n_cols))
    Apad = np.zeros((N, N), dtype=np.complex128)
    Apad[:n_rows, :n_cols] = A

    row_norms = np.linalg.norm(Apad, axis=1)
    alpha = float(np.linalg.norm(Apad))  # Frobenius norm
    if alpha == 0.0:
        raise ValueError("cannot block-encode the all-zero matrix (alpha = 0)")

    psi = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        if row_norms[j] > 0:
            psi[j] = Apad[j] / row_norms[j]
        else:
            psi[j, 0] = 1.0  # arbitrary unit state; phi[j] = 0 so it never contributes
    phi = tuple((row_norms / alpha).astype(np.complex128))
    return psi, phi, alpha, N


@attrs.frozen
class ClassicalMatrixBlockEncoding(BlockEncoding):
    r"""``(\lVert A\rVert_F, n, \epsilon)`` block-encoding of a dense classical matrix ``A``.

    Minimal-T-count construction of Clader et al. (arXiv:2206.03505): ``U_A = U_R^\dag U_L``
    with ``U_L`` a plain state preparation of the row-norm state ``|varphi>`` plus a
    register swap, and ``U_R`` the controlled state preparation of the normalized rows
    ``|psi_j>``.  Build with :meth:`from_matrix` (data-bearing, decomposable) or
    :meth:`from_bitsize` (data-free, for resource estimates).

    Attributes:
        n_rows: matrix dimension ``N = 2^n`` (after power-of-two padding).
        phase_bitsize: bitsize ``b`` of the shared phase-gradient / angle registers.
        psi_rows: ``(N, N)`` normalized rows (``Shaped`` when data-free).
        phi_col: length-``N`` row-norm amplitudes (``HasLength`` when data-free).
        alpha_val: Frobenius norm ``\lVert A\rVert_F`` (a sympy symbol when data-free).
    """

    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    psi_rows: Union[Shaped, NDArray[np.complex128]] = attrs.field(eq=False)
    phi_col: Union[HasLength, Tuple[complex, ...]] = attrs.field(eq=False)
    alpha_val: SymbolicFloat
    optimal_T: bool = True
    three_phase_layer_prep: bool = False

    @psi_rows.validator
    def _check_psi(self, attribute, value):
        if not isinstance(value, (Shaped, np.ndarray)):
            raise TypeError("psi_rows must be a numpy array or Shaped")

    # ------------------------------- Constructors -------------------------------

    @classmethod
    def from_matrix(
        cls,
        A: NDArray[np.floating],
        phase_bitsize: SymbolicInt,
        *,
        optimal_T: bool = True,
        three_phase_layer_prep: bool = False,
    ) -> "ClassicalMatrixBlockEncoding":
        """Data-bearing block-encoding of a concrete (real or complex) matrix ``A``.

        Uses the ``U_A = U_L^dag U_R`` ordering so the dominant block prep ``U_R`` runs
        FORWARD (cheap X-measurement uncompute).  ``U_L^dag U_R`` built from a matrix ``B``
        encodes ``B^T``; we pass ``B = A^T`` so the encoded operator is ``A`` itself --
        i.e. ``|psi_l>`` is the normalized column ``l`` of ``A`` and ``|phi>`` carries the
        column norms.
        """
        psi, phi, alpha, N = _coeffs_from_matrix(np.asarray(A).T)
        return cls(
            n_rows=N,
            phase_bitsize=phase_bitsize,
            psi_rows=psi,
            phi_col=phi,
            alpha_val=alpha,
            optimal_T=optimal_T,
            three_phase_layer_prep=three_phase_layer_prep,
        )

    @classmethod
    def from_bitsize(
        cls,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        alpha: Optional[SymbolicFloat] = None,
        optimal_T: bool = True,
        three_phase_layer_prep: bool = False,
    ) -> "ClassicalMatrixBlockEncoding":
        """Data-free block-encoding for resource estimation from sizes alone.

        ``n_rows`` must be a power of two for concrete values.  The returned bloq supports
        call-graph / resource estimates but cannot be decomposed into a concrete circuit.

        ``optimal_T`` selects the QROAM block-size regime: ``True`` (default) targets the
        analytic ``lambda*`` minimal-T-count point of the paper; ``False`` uses un-batched
        (``lambda = 1``) QROAM for the minimal-qubit point.

        ``three_phase_layer_prep`` replaces both state preparations with the "three diagonal
        phase layers + Hadamards" ansatz (arXiv:2409.11748 p.14).
        """
        # The row/column state preparations run on a ceil(log2 n_rows)-qubit register
        # and the Shaped tables take any n_rows, so a power of two is not required.
        # Requiring it forced callers to pad (208 -> 256), which the analytic model
        # does not do and which costs ~1.35x here.
        if not is_symbolic(n_rows) and n_rows < 2:
            raise ValueError("n_rows must be >= 2")
        if alpha is None:
            alpha = sympy.Symbol(r"\|A\|_F", positive=True)
        return cls(
            n_rows=n_rows,
            phase_bitsize=phase_bitsize,
            psi_rows=Shaped((n_rows, n_rows)),
            phi_col=HasLength(n_rows),
            alpha_val=alpha,
            optimal_T=optimal_T,
            three_phase_layer_prep=three_phase_layer_prep,
        )

    # ----------------------------- Shape helpers --------------------------------

    @property
    def matrix_bitsize(self) -> SymbolicInt:
        """``n = log2(N)`` -- size of each of the two n-qubit registers."""
        return bit_length(self.n_rows - 1)

    @property
    def is_data_free(self) -> bool:
        return isinstance(self.psi_rows, Shaped) or is_symbolic(self.phase_bitsize)

    # ------------------------- BlockEncoding interface --------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.matrix_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.matrix_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.alpha_val

    @property
    def epsilon(self) -> SymbolicFloat:
        # Dominated by the t-bit angle rounding / phase-gradient discretization.
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

    # --------------------------- Sub-bloq factories -----------------------------

    def _prep_psi_fwd(self, control_bitsize: int = 0) -> Bloq:
        """``U_R``: FORWARD controlled state preparation of the column states ``|psi_l>``.

        Running this (the dominant ``N x N`` block prep) forward lets every per-layer QROAM
        unload be a 0-Toffoli X-measurement (``measure_reset``).  T-opt -> Toffoli-minimal
        select-swap batch from :func:`_min_T_block_prep_lbs`; minimal-qubit -> ``(0, 0)``
        un-batched ``lambda = 1`` QROAM.

        With ``three_phase_layer_prep`` the ``N`` column states are prepared by the
        block-diagonal three-phase-layer ansatz (``n_blocks = N`` index addresses), in its
        own ``optimal_T`` blocking regime (auto-optimal split, or ``lambda = 1``).
        """
        if self.three_phase_layer_prep:
            lbs = None if self.optimal_T else (0, 0)
            return ThreePhaseLayerStatePreparation.from_bitsize(
                n_coeff=self.n_rows,
                phase_bitsize=self.phase_bitsize,
                n_blocks=self.n_rows,
                control_bitsize=control_bitsize,
                uncompute=False,
                measure_reset=True,
                log_block_sizes=lbs,
                adjoint_log_block_sizes=lbs,
            )
        if self.optimal_T:
            lbs = (
                None
                if is_symbolic(self.n_rows, self.phase_bitsize)
                else _min_T_block_prep_lbs(
                    int(self.n_rows), int(self.n_rows), int(self.phase_bitsize), False
                )
            )
        else:
            lbs = (0, 0)
        return BlockStatePreparationViaQROAMRotations(
            state_coefficients=self.psi_rows,
            phase_bitsize=self.phase_bitsize,
            control_bitsize=control_bitsize,
            uncompute=False,
            amp_log_block_sizes=lbs,
            amp_adjoint_log_block_sizes=lbs,
            phase_log_block_sizes=lbs,
            phase_adjoint_log_block_sizes=lbs,
            per_layer_optimal=False,
            measure_reset=True,
        )

    def _prep_phi_adj(self, control_bitsize: int = 0) -> Bloq:
        """``U_L^\\dagger``: ADJOINT plain state preparation of the column-norm state ``|phi>``.

        The cheap 1-D prep; kept as a full coherent adjoint (no measurement shortcut).
        ``optimal_T`` -> QROAMClean's own T-optimal block size; minimal-qubit -> ``(0,)``.

        With ``three_phase_layer_prep`` the (single, non-block) state is prepared by the
        three-phase-layer ansatz run in uncompute mode.
        """
        if self.three_phase_layer_prep:
            lbs = None if self.optimal_T else (0,)
            return ThreePhaseLayerStatePreparation.from_bitsize(
                n_coeff=self.n_rows,
                phase_bitsize=self.phase_bitsize,
                n_blocks=1,
                control_bitsize=control_bitsize,
                uncompute=True,
                log_block_sizes=lbs,
                adjoint_log_block_sizes=lbs,
            )
        lbs = None if self.optimal_T else (0,)
        return StatePreparationViaQROAMRotations(
            state_coefficients=self.phi_col,
            phase_bitsize=self.phase_bitsize,
            control_bitsize=control_bitsize,
            uncompute=True,
            log_block_sizes=lbs,
            adjoint_log_block_sizes=lbs,
        )

    @property
    def swap(self) -> Swap:
        return Swap(self.matrix_bitsize)

    # ------------------------------ Resource counts -----------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self._prep_psi_fwd()] += 1         # U_R: forward column prep |psi_l>
        ret[self.swap] += 1                    # register swap
        ret[self._prep_phi_adj()] += 1         # U_L^dagger: uncompute |phi>
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Cheap single-qubit control via :class:`_ControlledClassicalMatrixBlockEncoding`.

        The control propagates into both state preparations (which natively accept a
        control register) and turns the register swap into a Fredkin (controlled-swap), so
        the whole block encoding reduces to the identity when ``ctrl = 0``.
        """
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledClassicalMatrixBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )

    # ---------------------------- Composite circuit -----------------------------

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Wire ``U_A = U_L^\dagger U_R``: forward column prep -> swap -> adjoint ``|phi>``.

        ``ancilla`` is the prep register (starts ``|0>^n``); ``system`` is the index
        register (input ``|l>``, output ``|j>``).  ``U_R`` runs forward so its per-layer
        QROAM unloads are 0-Toffoli X-measurements.

        The wiring only uses sub-bloq *signatures*, so it works data-free (shape-concrete)
        too -- enabling qubit/T resource estimates via :meth:`from_bitsize`.  The data-free
        leaves themselves raise ``DecomposeTypeError`` when pushed further, which the cost
        protocols handle by falling back to their call graphs.
        """
        system = soqs['system']     # index register R2 (|l>)
        prep = soqs['ancilla']      # prep register R1 (|0>^n)
        pg = soqs['resource']       # phase gradient

        # ---- U_R : forward column prep |psi_l> on R1 (controlled on R2 = |l>) ----
        out = bb.add_d(
            self._prep_psi_fwd(), block=system, target_state=prep, phase_gradient=pg
        )
        system = out['block']
        prep = out['target_state']
        pg = out['phase_gradient']

        # ---- swap R1 <-> R2, then U_L^dagger : uncompute |phi> on R1 ----
        prep, system = bb.add(self.swap, x=prep, y=system)
        out = bb.add_d(self._prep_phi_adj(), target_state=prep, phase_gradient=pg)
        prep = out['target_state']
        pg = out['phase_gradient']

        return {'system': system, 'ancilla': prep, 'resource': pg}


@attrs.frozen
class _ControlledClassicalMatrixBlockEncoding(BlockEncoding):
    r"""Singly-controlled :class:`ClassicalMatrixBlockEncoding`.

    Adds one ``ctrl`` qubit that controls both state preparations and the register swap
    (Fredkin).  With ``ctrl = 0`` every sub-bloq acts trivially, so the block encoding is
    the identity; with ``ctrl = 1`` it is the full ``U_A``.  This realizes
    ``CU_A = |0><0| \otimes I + |1><1| \otimes U_A`` (paper Sec. IV D) at essentially the
    cost of the uncontrolled bloq.
    """

    inner: "ClassicalMatrixBlockEncoding"

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

    @property
    def cswap(self) -> CSwap:
        return CSwap(self.inner.matrix_bitsize)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner._prep_psi_fwd(control_bitsize=1)] += 1
        ret[self.cswap] += 1
        ret[self.inner._prep_phi_adj(control_bitsize=1)] += 1
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        ctrl = soqs['ctrl']
        system = soqs['system']
        prep = soqs['ancilla']
        pg = soqs['resource']

        out = bb.add_d(
            self.inner._prep_psi_fwd(control_bitsize=1),
            prepare_control=ctrl,
            block=system,
            target_state=prep,
            phase_gradient=pg,
        )
        ctrl = out['prepare_control']
        system = out['block']
        prep = out['target_state']
        pg = out['phase_gradient']

        ctrl, prep, system = bb.add(self.cswap, ctrl=ctrl, x=prep, y=system)

        out = bb.add_d(
            self.inner._prep_phi_adj(control_bitsize=1),
            prepare_control=ctrl,
            target_state=prep,
            phase_gradient=pg,
        )
        ctrl = out['prepare_control']
        prep = out['target_state']
        pg = out['phase_gradient']

        return {'ctrl': ctrl, 'system': system, 'ancilla': prep, 'resource': pg}


# =============================================================================
# Block-diagonal version:  A_block = sum_k |k><k| (x) A_k
# =============================================================================


def _block_coeffs_from_matrices(
    mats,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], float, int, int]:
    r"""Coefficient tables for the block-diagonal Frobenius block-encoding.

    Given a list of ``K`` matrices ``A_k`` (zero-padded to a common ``N x N`` with ``N`` a
    power of two; ``K`` rounded up to a power of two ``K_pad`` with zero blocks appended),
    returns ``(flag, phi, psi, F_max, N, K_pad)`` where

      * ``F_k   = ||A_k||_F``,  ``F_max = max_k F_k``;
      * ``flag[k]      = (F_k/F_max, sqrt(1 - F_k^2/F_max^2))``  -- the 1-qubit flag state;
      * ``phi[k][j]    = ||(A_k)_{j,.}|| / F_k``                 -- the row-norm state |phi_k>;
      * ``psi[k*N+j]   = (A_k)_{j,.} / ||(A_k)_{j,.}||``          -- the row state |psi_{k,j}>.

    Zero rows / zero (or padded) blocks map to the ``|0>`` basis state; their flag weight on
    ``|0>_f`` is zero, so they contribute nothing to the encoded ``f = 0`` branch.
    """
    mats = [np.asarray(m, dtype=np.complex128) for m in mats]
    if not mats:
        raise ValueError("need at least one block A_k")
    K = len(mats)
    N = _next_pow2(max(max(m.shape) for m in mats))
    K_pad = _next_pow2(K)

    Apads, Fs = [], []
    for m in mats:
        if m.ndim != 2:
            raise ValueError("each A_k must be a 2D matrix")
        ap = np.zeros((N, N), dtype=np.complex128)
        ap[: m.shape[0], : m.shape[1]] = m
        Apads.append(ap)
        Fs.append(float(np.linalg.norm(ap)))
    F_max = max(Fs)
    if F_max == 0.0:
        raise ValueError("cannot block-encode all-zero blocks (F_max = 0)")

    flag = np.zeros((K_pad, 2), dtype=np.complex128)
    phi = np.zeros((K_pad, N), dtype=np.complex128)
    psi = np.zeros((K_pad * N, N), dtype=np.complex128)
    for k in range(K_pad):
        if k < K and Fs[k] > 0:
            ap, Fk = Apads[k], Fs[k]
            ck = Fk / F_max
            flag[k] = (ck, np.sqrt(max(0.0, 1.0 - ck * ck)))
            row_norms = np.linalg.norm(ap, axis=1)
            phi[k] = row_norms / Fk
            for j in range(N):
                if row_norms[j] > 0:
                    psi[k * N + j] = ap[j] / row_norms[j]
                else:
                    psi[k * N + j, 0] = 1.0
        else:  # zero or padded block: flag weight 0, harmless |0> placeholders
            flag[k] = (0.0, 1.0)
            phi[k, 0] = 1.0
            for j in range(N):
                psi[k * N + j, 0] = 1.0
    return flag, phi, psi, float(F_max), N, K_pad


@attrs.frozen
class BlockDiagonalClassicalMatrixBlockEncoding(BlockEncoding):
    r"""``(F_max, n+1, \epsilon)`` block-encoding of ``sum_k |k><k| (x) A_k`` (Clader-Frobenius).

    Block-diagonal generalization of :class:`ClassicalMatrixBlockEncoding`.  For each block
    ``A_k`` (``2^n x 2^n``) with Frobenius norm ``F_k`` and ``F_max = max_k F_k``, the
    encoded operator is ``sum_k |k><k| (x) A_k / F_max``.

    Construction ``U_A = U_L^dag U_R`` per block (read-only ``k`` addresses every QROAM;
    ``k`` is never changed).  Coefficients are built from ``A_k^T`` so ``|psi_{k,l}>`` is the
    normalized column ``l`` of ``A_k`` and ``|phi_k>`` carries its column norms:

    1. **Flag** (new): controlled on ``k``, rotate a fresh flag qubit ``f``

       .. math:: |0>_f \mapsto (F_k/F_max)|0>_f + \sqrt{1 - F_k^2/F_max^2}\,|1>_f,

       a 1-qubit block state preparation with coefficients ``(F_k/F_max, .)``.
    2. **U_R** (forward): controlled on ``(k, l)``, prepare ``|psi_{k,l}>`` in the prep
       register, then swap it with the index register.  Running this dominant block prep
       FORWARD lets every per-layer QROAM unload be a 0-Toffoli X-measurement.
    3. **U_L^dag**: controlled on ``k``, uncompute ``|phi_k> = sum_l ||(A_k)_{.,l}||/F_k |l>``.

    Post-selecting ``f = 0`` together with the prep register on ``|0>^n`` leaves amplitude
    ``(F_k/F_max)(||(A_k)_{.,l}||/F_k)((A_k)_{j,l}/||(A_k)_{.,l}||) = (A_k)_{j,l}/F_max``.

    Registers (standard ``qualtran`` :class:`BlockEncoding` interface):
      * ``system``   -- ``(k, l)``: ``block_bitsize + n`` qubits; the encoded operator's space.
      * ``ancilla``  -- ``(prep, flag)``: ``n + 1`` qubits; block-encoding flag (signal ``|0>``).
      * ``resource`` -- ``phase_bitsize``-qubit phase-gradient workspace (shared by all preps).

    Attributes:
        n_blocks: number of blocks ``K`` (rounded up to a power of two).
        n_rows: block dimension ``N = 2^n``.
        phase_bitsize: bitsize ``b`` of the phase-gradient / angle registers.
        flag_coeffs: ``(K, 2)`` flag-qubit states (``Shaped`` when data-free).
        phi_col: ``(K, N)`` row-norm states ``|phi_k>`` (``Shaped`` when data-free).
        psi_rows: ``(K*N, N)`` row states ``|psi_{k,j}>`` (``Shaped`` when data-free).
        alpha_val: ``F_max`` (a sympy symbol when data-free).
        optimal_T: T-count-optimal QROAM batching (True) vs qubit-minimal ``lambda=1`` (False).
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    phase_bitsize: SymbolicInt
    flag_coeffs: Union[Shaped, NDArray[np.complex128]] = attrs.field(eq=False)
    phi_col: Union[Shaped, NDArray[np.complex128]] = attrs.field(eq=False)
    psi_rows: Union[Shaped, NDArray[np.complex128]] = attrs.field(eq=False)
    alpha_val: SymbolicFloat
    optimal_T: bool = True
    three_phase_layer_prep: bool = False

    # ------------------------------- Constructors -------------------------------

    @classmethod
    def from_matrices(
        cls,
        mats,
        phase_bitsize: SymbolicInt,
        *,
        optimal_T: bool = True,
        three_phase_layer_prep: bool = False,
    ) -> "BlockDiagonalClassicalMatrixBlockEncoding":
        """Data-bearing block-encoding of a list of concrete blocks ``A_k``.

        Uses the per-block ``U_A = U_L^dag U_R`` ordering so the dominant block prep ``U_R``
        runs FORWARD (cheap X-measurement uncompute).  Coefficients are built from
        ``A_k^T`` so the encoded operator is ``A_k`` itself (``|psi_{k,l}>`` is the
        normalized column ``l`` of ``A_k``; ``|phi_k>`` carries its column norms).
        """
        flag, phi, psi, alpha, N, K_pad = _block_coeffs_from_matrices(
            [np.asarray(m).T for m in mats]
        )
        return cls(
            n_blocks=K_pad,
            n_rows=N,
            phase_bitsize=phase_bitsize,
            flag_coeffs=flag,
            phi_col=phi,
            psi_rows=psi,
            alpha_val=alpha,
            optimal_T=optimal_T,
            three_phase_layer_prep=three_phase_layer_prep,
        )

    @classmethod
    def from_bitsize(
        cls,
        n_blocks: SymbolicInt,
        n_rows: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        alpha: Optional[SymbolicFloat] = None,
        optimal_T: bool = True,
        three_phase_layer_prep: bool = False,
    ) -> "BlockDiagonalClassicalMatrixBlockEncoding":
        """Data-free block-encoding for resource estimation from sizes alone.

        ``n_rows`` is the block dimension (need not be a power of two); ``n_blocks`` is
        the number of blocks ``K``.  The returned bloq supports call-graph / resource estimates.

        ``three_phase_layer_prep`` replaces every state preparation (flag, ``U_R``,
        ``U_L^dag``) with the "three diagonal phase layers + Hadamards" ansatz.
        """
        # Not required: the state preparations run on a ceil(log2 n_rows)-qubit
        # register and the Shaped tables take any n_rows.  See the sibling method.
        if not is_symbolic(n_rows) and n_rows < 2:
            raise ValueError("n_rows must be >= 2")
        if alpha is None:
            alpha = sympy.Symbol(r"F_\max", positive=True)
        nb_x_nr = n_blocks * n_rows
        return cls(
            n_blocks=n_blocks,
            n_rows=n_rows,
            phase_bitsize=phase_bitsize,
            flag_coeffs=Shaped((n_blocks, 2)),
            phi_col=Shaped((n_blocks, n_rows)),
            psi_rows=Shaped((nb_x_nr, n_rows)),
            alpha_val=alpha,
            optimal_T=optimal_T,
            three_phase_layer_prep=three_phase_layer_prep,
        )

    # ----------------------------- Shape helpers --------------------------------

    @property
    def block_bitsize(self) -> SymbolicInt:
        """``log2(K)`` -- size of the read-only block (``k``) register."""
        return bit_length(self.n_blocks - 1)

    @property
    def matrix_bitsize(self) -> SymbolicInt:
        """``n = log2(N)`` -- size of the index / prep registers."""
        return bit_length(self.n_rows - 1)

    @property
    def is_data_free(self) -> bool:
        return isinstance(self.psi_rows, Shaped) or is_symbolic(self.phase_bitsize)

    # ------------------------- BlockEncoding interface --------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.block_bitsize + self.matrix_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.matrix_bitsize + 1  # prep register + flag qubit

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

    # --------------------------- Sub-bloq factories -----------------------------

    def _lbs_for(self, n_blocks: SymbolicInt, n_coeff: SymbolicInt, uncompute: bool):
        """QROAM block sizes: Toffoli-minimal search (optimal_T) or ``(0,0)`` (qubit-min)."""
        if not self.optimal_T:
            return (0, 0)
        if is_symbolic(n_blocks, n_coeff, self.phase_bitsize):
            return None
        return _min_T_block_prep_lbs(
            int(n_blocks), int(n_coeff), int(self.phase_bitsize), uncompute
        )

    def _block_prep(
        self, coeffs, n_blocks: SymbolicInt, n_coeff: SymbolicInt, uncompute: bool,
        control_bitsize: int,
    ) -> Bloq:
        if self.three_phase_layer_prep:
            # All three preps (flag, U_R, U_L^dag) route through here, so this single branch
            # swaps every state preparation for the block-diagonal three-phase-layer ansatz in
            # its own optimal_T regime (auto-optimal split, or lambda=1).
            lbs = None if self.optimal_T else (0, 0)
            return ThreePhaseLayerStatePreparation.from_bitsize(
                n_coeff=n_coeff,
                phase_bitsize=self.phase_bitsize,
                n_blocks=n_blocks,
                control_bitsize=control_bitsize,
                uncompute=uncompute,
                measure_reset=True,
                log_block_sizes=lbs,
                adjoint_log_block_sizes=lbs,
            )
        lbs = self._lbs_for(n_blocks, n_coeff, uncompute)
        return BlockStatePreparationViaQROAMRotations(
            state_coefficients=coeffs,
            phase_bitsize=self.phase_bitsize,
            control_bitsize=control_bitsize,
            uncompute=uncompute,
            amp_log_block_sizes=lbs,
            amp_adjoint_log_block_sizes=lbs,
            phase_log_block_sizes=lbs,
            phase_adjoint_log_block_sizes=lbs,
            per_layer_optimal=False,
            measure_reset=True,  # forward layers erase by X-measurement; gated off for adjoints
        )

    def _prep_flag(self, control_bitsize: int = 0) -> Bloq:
        """``k``-controlled 1-qubit flag rotation ``|0>_f -> (F_k/F_max)|0> + (.)|1>`` (forward)."""
        return self._block_prep(self.flag_coeffs, self.n_blocks, 2, False, control_bitsize)

    def _prep_psi_fwd(self, control_bitsize: int = 0) -> Bloq:
        """``U_R``: ``(k, l)``-controlled FORWARD state prep of the column states ``|psi_{k,l}>``.

        The dominant ``K*N x N`` block prep; running it forward makes every per-layer QROAM
        unload a 0-Toffoli X-measurement.
        """
        return self._block_prep(
            self.psi_rows, self.n_blocks * self.n_rows, self.n_rows, False, control_bitsize
        )

    def _prep_phi_adj(self, control_bitsize: int = 0) -> Bloq:
        """``U_L^dag``: ``k``-controlled ADJOINT state prep of ``|phi_k>`` (kept coherent)."""
        return self._block_prep(self.phi_col, self.n_blocks, self.n_rows, True, control_bitsize)

    @property
    def swap(self) -> Swap:
        return Swap(self.matrix_bitsize)

    # ------------------------------ Resource counts -----------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self._prep_flag()] += 1            # k-controlled flag rotation (forward)
        ret[self._prep_psi_fwd()] += 1         # U_R: forward column prep |psi_{k,l}>
        ret[self.swap] += 1                    # register swap
        ret[self._prep_phi_adj()] += 1         # U_L^dagger: uncompute |phi_k>
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledBlockDiagonalClassicalMatrixBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )

    # --------------------------- Composite circuit ------------------------------

    def _split_system(self, bb, system):
        """system -> (k, R2).  Returns (k_or_None, R2)."""
        if self.block_bitsize == 0:
            return None, system
        arr = bb.split(system)
        nk = int(self.block_bitsize)
        k = bb.join(arr[:nk], dtype=QAny(self.block_bitsize))
        r2 = bb.join(arr[nk:], dtype=QAny(self.matrix_bitsize))
        return k, r2

    def _split_ancilla(self, bb, ancilla):
        """ancilla -> (R1, flag).  R1 is the prep register (MSBs), flag is the LSB qubit."""
        arr = bb.split(ancilla)
        r1 = bb.join(arr[:-1], dtype=QAny(self.matrix_bitsize))
        return r1, arr[-1]

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Wire flag -> forward column prep ``|psi_{k,l}>`` -> swap -> adjoint ``|phi_k>``.

        ``U_A = U_L^dag U_R`` per block: the dominant ``U_R`` runs forward (0-Toffoli
        X-measurement uncompute).  ``k`` is read-only throughout.
        """
        system = soqs['system']
        ancilla = soqs['ancilla']
        pg = soqs['resource']

        k, r2 = self._split_system(bb, system)
        r1, flag = self._split_ancilla(bb, ancilla)
        has_k = k is not None

        # ---- 1) k-controlled flag rotation ----
        fin = {'target_state': flag, 'phase_gradient': pg}
        if has_k:
            fin['block'] = k
        fout = bb.add_d(self._prep_flag(), **fin)
        flag = fout['target_state']
        pg = fout['phase_gradient']
        if has_k:
            k = fout['block']

        # ---- 2) U_R: forward column prep |psi_{k,l}> on R1, addressed by combined (k, l=R2) ----
        if has_k:
            block_addr = bb.join(
                np.concatenate([bb.split(k), bb.split(r2)]), dtype=QAny(self.system_bitsize)
            )
        else:
            block_addr = r2
        rout = bb.add_d(
            self._prep_psi_fwd(), block=block_addr, target_state=r1, phase_gradient=pg
        )
        block_addr = rout['block']
        r1 = rout['target_state']
        pg = rout['phase_gradient']
        if has_k:
            barr = bb.split(block_addr)
            nk = int(self.block_bitsize)
            k = bb.join(barr[:nk], dtype=QAny(self.block_bitsize))
            r2 = bb.join(barr[nk:], dtype=QAny(self.matrix_bitsize))
        else:
            r2 = block_addr

        # ---- swap R1 <-> R2 ----
        r1, r2 = bb.add(self.swap, x=r1, y=r2)

        # ---- 3) U_L^dag: uncompute |phi_k> on R1 (k-controlled) ----
        lin = {'target_state': r1, 'phase_gradient': pg}
        if has_k:
            lin['block'] = k
        lout = bb.add_d(self._prep_phi_adj(), **lin)
        r1 = lout['target_state']
        pg = lout['phase_gradient']
        if has_k:
            k = lout['block']

        # ---- rejoin system = (k, R2) and ancilla = (R1, flag) ----
        if has_k:
            system = bb.join(
                np.concatenate([bb.split(k), bb.split(r2)]), dtype=QAny(self.system_bitsize)
            )
        else:
            system = r2
        ancilla = bb.join(
            np.concatenate([bb.split(r1), [flag]]), dtype=QAny(self.ancilla_bitsize)
        )
        return {'system': system, 'ancilla': ancilla, 'resource': pg}


@attrs.frozen
class _ControlledBlockDiagonalClassicalMatrixBlockEncoding(BlockEncoding):
    r"""Singly-controlled :class:`BlockDiagonalClassicalMatrixBlockEncoding`.

    One ``ctrl`` qubit controls the flag rotation, both state preparations, and the register
    swap (Fredkin); with ``ctrl = 0`` every sub-bloq is trivial, so the block encoding is the
    identity.  Realizes ``|0><0| (x) I + |1><1| (x) U`` at essentially the uncontrolled cost.
    """

    inner: "BlockDiagonalClassicalMatrixBlockEncoding"

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

    @property
    def cswap(self) -> CSwap:
        return CSwap(self.inner.matrix_bitsize)

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        ret[self.inner._prep_flag(control_bitsize=1)] += 1
        ret[self.inner._prep_psi_fwd(control_bitsize=1)] += 1
        ret[self.cswap] += 1
        ret[self.inner._prep_phi_adj(control_bitsize=1)] += 1
        return ret

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        inner = self.inner
        ctrl = soqs['ctrl']
        system = soqs['system']
        ancilla = soqs['ancilla']
        pg = soqs['resource']

        k, r2 = inner._split_system(bb, system)
        r1, flag = inner._split_ancilla(bb, ancilla)
        has_k = k is not None

        fin = {'prepare_control': ctrl, 'target_state': flag, 'phase_gradient': pg}
        if has_k:
            fin['block'] = k
        fout = bb.add_d(inner._prep_flag(control_bitsize=1), **fin)
        ctrl, flag, pg = fout['prepare_control'], fout['target_state'], fout['phase_gradient']
        if has_k:
            k = fout['block']

        # U_R: forward column prep, addressed by combined (k, l=R2)
        if has_k:
            block_addr = bb.join(
                np.concatenate([bb.split(k), bb.split(r2)]), dtype=QAny(inner.system_bitsize)
            )
        else:
            block_addr = r2
        rout = bb.add_d(
            inner._prep_psi_fwd(control_bitsize=1),
            prepare_control=ctrl, block=block_addr, target_state=r1, phase_gradient=pg,
        )
        ctrl, block_addr, r1, pg = (
            rout['prepare_control'], rout['block'], rout['target_state'], rout['phase_gradient']
        )
        if has_k:
            barr = bb.split(block_addr)
            nk = int(inner.block_bitsize)
            k = bb.join(barr[:nk], dtype=QAny(inner.block_bitsize))
            r2 = bb.join(barr[nk:], dtype=QAny(inner.matrix_bitsize))
        else:
            r2 = block_addr

        ctrl, r1, r2 = bb.add(self.cswap, ctrl=ctrl, x=r1, y=r2)

        # U_L^dag: uncompute |phi_k> (k-controlled)
        lin = {'prepare_control': ctrl, 'target_state': r1, 'phase_gradient': pg}
        if has_k:
            lin['block'] = k
        lout = bb.add_d(inner._prep_phi_adj(control_bitsize=1), **lin)
        ctrl, r1, pg = lout['prepare_control'], lout['target_state'], lout['phase_gradient']
        if has_k:
            k = lout['block']

        if has_k:
            system = bb.join(
                np.concatenate([bb.split(k), bb.split(r2)]), dtype=QAny(inner.system_bitsize)
            )
        else:
            system = r2
        ancilla = bb.join(
            np.concatenate([bb.split(r1), [flag]]), dtype=QAny(inner.ancilla_bitsize)
        )
        return {'ctrl': ctrl, 'system': system, 'ancilla': ancilla, 'resource': pg}


# =============================================================================
# Off-diagonal Hermitian block-encoding via the off-diagonal dilation
#   S = |0><1| (x) U + |1><0| (x) U^dag = [[0, U], [U^dag, 0]]   (Clader et al. App. A 1)
# =============================================================================


@attrs.frozen
class HermitianOffDiagonalBlockEncoding(BlockEncoding):
    r"""``(alpha, a, eps)`` block-encoding of the Hermitian dilation ``Abar = [[0, A],[A^dag, 0]]``.

    From ANY inner block-encoding ``U`` of ``A`` (``P U P = A/alpha``, ``P = |0^a><0^a| (x) I``),
    the off-diagonal "qubitization" dilation

    .. math::
        S = |0><1| (x) U + |1><0| (x) U^\dagger = \begin{pmatrix} 0 & U \\ U^\dagger & 0 \end{pmatrix}

    block-encodes ``Abar = [[0, A],[A^dag, 0]]`` at the SAME ``alpha``, projecting only the
    inner ancilla:

    .. math::
        (I_e (x) \langle 0^a| (x) I)\, S\, (I_e (x) |0^a> (x) I)
        = \tfrac{1}{alpha}\begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}.

    The extra qubit ``e`` indexes the two off-diagonal blocks and is part of the *encoded
    system* (no new block-encoding ancilla is needed).  ``Abar`` is Hermitian and ``S`` is
    itself Hermitian and unitary (``S^2 = I``).  This works for ANY ``A`` (Hermitian or not;
    non-square ``A`` is square-padded by the inner encoding).

    Cost: ``C[U] + C[U^dag]`` (identical resources) + one ``X`` -- about ``2x`` the inner
    Toffoli count and only the inner's ancilla.  This is roughly HALF the cost of building a
    fresh controlled state preparation over the doubled space (the literal ``P^dag SWAP P``
    route): it reuses the efficient inner block-encoding twice instead.

    Registers:
      * ``system``   -- 1 off-diagonal-index qubit + the inner system register.
      * ``ancilla``  -- the inner ancilla (signal ``|0^a>``; the extra qubit is NOT projected).
      * ``resource`` -- the inner resource register.

    Note: ``U^dag`` is charged as a second ``C[U]`` (identical Clifford+T cost; the inner
    adjoint is not separately buildable here owing to a ``QROAMClean.adjoint`` shape limit).
    """

    inner: BlockEncoding

    # ------------------------------- Constructors -------------------------------

    @classmethod
    def from_inner(cls, inner: BlockEncoding) -> "HermitianOffDiagonalBlockEncoding":
        """Wrap any block-encoding ``U`` of ``A`` into a block-encoding of ``[[0,A],[A^dag,0]]``."""
        return cls(inner=inner)

    @classmethod
    def from_matrix(
        cls, A: NDArray[np.floating], phase_bitsize: SymbolicInt, *, optimal_T: bool = True
    ) -> "HermitianOffDiagonalBlockEncoding":
        """Data-bearing Hermitian dilation of ``A`` (inner = dense Clader block-encoding)."""
        return cls(inner=ClassicalMatrixBlockEncoding.from_matrix(A, phase_bitsize, optimal_T=optimal_T))

    @classmethod
    def from_bitsize(
        cls,
        n_rows: SymbolicInt,
        n_cols: SymbolicInt,
        phase_bitsize: SymbolicInt,
        *,
        alpha: Optional[SymbolicFloat] = None,
        optimal_T: bool = True,
    ) -> "HermitianOffDiagonalBlockEncoding":
        """Data-free Hermitian dilation for resource estimates.

        ``A`` is square-padded to ``N = max(n_rows, n_cols)`` (a power of two) by the inner
        dense block-encoding; the dilation then acts on ``2N`` dimensions.
        """
        if is_symbolic(n_rows, n_cols):
            N = n_rows
        else:
            N = max(int(n_rows), int(n_cols))
        inner = ClassicalMatrixBlockEncoding.from_bitsize(N, phase_bitsize, alpha=alpha, optimal_T=optimal_T)
        return cls(inner=inner)

    # ------------------------- BlockEncoding interface --------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.inner.system_bitsize + 1  # + off-diagonal index qubit e

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.inner.ancilla_bitsize     # e lives in the system, not the ancilla

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
        return PrepareIdentity.from_bitsizes((self.ancilla_bitsize,))

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    @property
    def ctrl_inner(self) -> Bloq:
        """``C[U]`` -- the singly-controlled inner block-encoding (used for both halves of S)."""
        return self.inner.controlled()

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        from qualtran.bloqs.basic_gates import XGate

        ret: "Counter[Bloq]" = Counter()
        ret[XGate()] += 1                # the X (x) I factor of S (flips the index qubit)
        ret[self.ctrl_inner] += 2        # C[U] and C[U^dagger] (identical resources)
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledHermitianOffDiagonalBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledHermitianOffDiagonalBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`HermitianOffDiagonalBlockEncoding`.

    Controlling ``S = X (x) I . C[U] . C[U^dag]`` fuses the extra control into each half
    (one ``And`` per half) and turns the index flip into a CNOT; the two ``C[U]`` calls are
    otherwise unchanged.
    """

    be: "HermitianOffDiagonalBlockEncoding"

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.be.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.be.ancilla_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.be.resource_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.be.alpha

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.be.epsilon

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return self.be.signal_state

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.be.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        from qualtran.bloqs.basic_gates import XGate
        from qualtran.bloqs.mcmt.and_bloq import And

        ret: "Counter[Bloq]" = Counter()
        ret[self.be.ctrl_inner] += 2     # C[U], C[U^dag] (each now doubly-controlled...)
        ret[And()] += 2                  # ...via one And fusing the extra control per half
        ret[XGate().controlled()] += 1   # controlled index flip (CNOT)
        return ret


# =============================================================================
# Direct Hermitian-unitary block-encoding:  W = (H (x) I) S (H (x) I),
#   S = [[0, U], [U^dag, 0]]  (Hermitian + unitary, W^2 = I)
# =============================================================================


@attrs.frozen
class DirectHermitianBlockEncoding(BlockEncoding):
    r"""Hermitian *unitary* block-encoding of a Hermitian ``A`` from any block-encoding ``U``.

    Given an inner block-encoding ``U`` with ``P U P = A/alpha`` (``P = |0^a><0^a| (x) I``)
    of a Hermitian matrix ``A = A^dag``, ``U`` itself need not satisfy ``U = U^dag``.  Adding
    one ancilla qubit and defining

    .. math::
        S = |0><1| (x) U + |1><0| (x) U^\dagger = \begin{pmatrix} 0 & U \\ U^\dagger & 0 \end{pmatrix},
        \qquad W = (H (x) I)\, S \,(H (x) I),

    yields ``W = W^dag`` and ``W^2 = I`` (a Hermitian unitary / involution), with

    .. math::
        (\langle 0| (x) P)\, W \,(|0> (x) P) = \tfrac12 P(U + U^\dagger)P = \tfrac{A}{alpha}

    when ``A = A^dag``.  (For a general inner matrix ``A`` it block-encodes the Hermitian
    part ``(A + A^dag)/2``.)  This is the standard "make the walk operator Hermitian" trick
    and is distinct from the off-diagonal construction (:class:`HermitianOffDiagonalBlockEncoding`,
    App. A 1), which instead makes the encoded *matrix* Hermitian by doubling its dimension.

    Cost: two controlled calls to the inner block-encoding (``C[U]`` and ``C[U^\dagger]``,
    identical resources) + two Hadamards + one X on the extra ancilla.  So
    ``Toffoli(W) ~ 2 * Toffoli(U)`` with a single extra qubit -- typically much cheaper than
    the dimension-doubling off-diagonal construction.

    Registers:
      * ``system``   -- the inner system register (unchanged).
      * ``ancilla``  -- 1 extra Hermitian-flag qubit + the inner ancilla (signal ``|0>``).
      * ``resource`` -- the inner resource register.

    Note: ``U^\dagger`` is charged as a second ``C[U]`` because the inner adjoint is not
    separately buildable here (a ``QROAMClean.adjoint`` shape limitation), but its
    Clifford+T cost is identical, so the resource estimate is exact.
    """

    inner: BlockEncoding

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.inner.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.inner.ancilla_bitsize + 1  # extra Hermitian-flag qubit

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
        return PrepareIdentity.from_bitsizes((self.ancilla_bitsize,))

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('system', QAny(self.system_bitsize)),
            Register('ancilla', QAny(self.ancilla_bitsize)),
            Register('resource', QAny(self.resource_bitsize)),
        ])

    @property
    def ctrl_inner(self) -> Bloq:
        """``C[U]`` -- the singly-controlled inner block-encoding (used for both halves)."""
        return self.inner.controlled()

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        from qualtran.bloqs.basic_gates import Hadamard, XGate

        ret: "Counter[Bloq]" = Counter()
        ret[Hadamard()] += 2                 # conjugating Hadamards on the extra ancilla
        ret[XGate()] += 1                    # the X (x) I factor of S
        ret[self.ctrl_inner] += 2            # C[U] and C[U^dagger] (identical resources)
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        # C[W] = (H (x) I) C[S] (H (x) I): the conjugating Hadamards stay unconditional
        # (they cancel when ctrl = 0), so controlling W reduces to controlling S.
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledDirectHermitianBlockEncoding(self),
            ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledDirectHermitianBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`DirectHermitianBlockEncoding` (controls ``S``; H stays free)."""

    be: "DirectHermitianBlockEncoding"

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.be.system_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.be.ancilla_bitsize

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.be.resource_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        return self.be.alpha

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.be.epsilon

    @cached_property
    def signal_state(self) -> PrepareOracle:
        return self.be.signal_state

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), *self.be.signature])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        from qualtran.bloqs.basic_gates import Hadamard, XGate
        from qualtran.bloqs.mcmt.and_bloq import And

        ret: "Counter[Bloq]" = Counter()
        ret[Hadamard()] += 2             # unconditional conjugating Hadamards
        ret[self.be.ctrl_inner] += 2     # C[U], C[U^dag] (each doubly-controlled...)
        ret[And()] += 2                  # ...via one And fusing the extra control per half
        ret[XGate().controlled()] += 1   # controlled index flip (CNOT)
        return ret
