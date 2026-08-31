r"""Isometry synthesis by Berry et al. §III.B -- the "columns of a unitary" construction.

An alternative to the Iten column-by-column scheme of
:mod:`block_isometry_column_synthesis_QROAM`.  Ledger primitive **P-07, construction C**.

    Berry, Tong, Khattar, White, Kim, Boixo, Lin, Lee, Chan, Babbush, Rubin,
    "Rapid initial state preparation for the quantum simulation of strongly correlated
    molecules", arXiv:2409.11748, Sec. III B, Eqs. (29), (32)-(36).

The source writes

.. math::
    U = \begin{pmatrix} U' & ? \\ U'' & ? \end{pmatrix} \quad (29), \qquad
    U = \begin{pmatrix} U_1 & ? & \cdots & ? \\ \vdots & & & \end{pmatrix} \quad (36)

with *"blocks that we do not need to specify"* and the input *"restricted to lie within a
subspace"* -- i.e. only some columns are specified, which is the isometry problem.  (The
paper never uses the word "isometry"; arXiv:1812.00954 does, for the same task.)

Cost, verbatim (`R-10` blind, confirmed by direct read of the PDF in `R-17`; `R-16`'s
disagreeing constants were wrong -- see ledger `C-35`).  For :math:`N_{\rm un}` rows:

.. math::
    \underbrace{\left\lceil\tfrac{N_{\rm un}}{2\Lambda}\right\rceil+\Lambda b-2}_{(32)}
  + \underbrace{\tfrac{N_{\rm un}}{2}\left(\left\lceil\tfrac{N_{\rm un}}{2\Lambda}\right\rceil+2\Lambda b-5\right)}_{(33)}
  + \underbrace{(n-3)\left(\tfrac{N_{\rm un}}{2}-1\right)}_{(34)}
  + \underbrace{\left\lceil\tfrac{N_{\rm un}}{\Lambda}\right\rceil+\Lambda b+\left\lceil\tfrac{N_{\rm un}}{\Lambda'}\right\rceil+\Lambda'-6}_{(35)}

Why the rows must divide by the columns
---------------------------------------
Eq. (29) is the :math:`d=2` case: half the columns specified.  Eq. (36) generalises to
local dimension :math:`d` with :math:`N_{\rm un} = d\chi`, so :math:`\chi = N_{\rm un}/d`
columns are specified and **the cost is multiplied by** :math:`d-1`.

That only makes sense for integer :math:`d`, so :math:`N_{\rm un}` must be a multiple of
the column count :math:`\chi`.  When it is not, pad the row dimension up to the next
multiple -- ``n_rows_padded = ceil(N_un / chi) * chi``.  For the density-fitting factor
(:math:`N_{\rm THC}=208` rows, :math:`N_oN_v=88` columns) that is :math:`264 = 3\times88`,
so :math:`d = 3` and the multiplier is 2.

Why it can beat the Iten scheme
-------------------------------
Iten runs :math:`\chi` column operations of :math:`n-1` layers each --
:math:`\chi(n-1)` multiplexed layers.  Berry runs :math:`N_{\rm un}/2` layers once and
multiplies by :math:`d-1`.  For **many** specified columns the second is far cheaper; for
few columns the first wins.  Which applies is a measurement, not a preference.

**The saving is not proportional to the column count.**  `R-16` §4: the QROM-controlled
part drops from :math:`N_{\rm un}` layers to :math:`N_{\rm un}/2`, but the **final phase
term (35) does not shrink** -- it is the same shape as P-09's triple.  So halving the
columns does not halve the cost.

Data-free: the layers are emitted as real Qualtran sub-bloqs so ``QECGatesCost`` does the
counting; :func:`berry_iiib_formula` gives the source's closed form for differencing.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from math import ceil
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, GateWithRegisters, QUInt, Signature
from qualtran.bloqs.arithmetic import AddK
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

try:
    from .block_unitary_interferometer_QROAM import (
        BlockInterferometerFinalPhasesQROAM,
        BlockInterferometerPhaseLayerQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from .range_safe_qroam import emit_range_safety
except ImportError:  # pragma: no cover
    from block_unitary_interferometer_QROAM import (
        BlockInterferometerFinalPhasesQROAM,
        BlockInterferometerPhaseLayerQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from range_safe_qroam import emit_range_safety

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def pad_rows_to_multiple(n_rows: int, n_cols: int) -> int:
    """Smallest ``N_un >= n_rows`` divisible by ``n_cols`` (Eq. 36 needs integer ``d``)."""
    n_rows, n_cols = int(n_rows), int(n_cols)
    if n_cols <= 0:
        raise ValueError("n_cols must be positive")
    return ceil(n_rows / n_cols) * n_cols


def berry_iiib_formula(n_un: int, n_cols: int, b: int,
                       lam: int = 1, lam_p: int = 1) -> int:
    """The source's closed form, Eqs. (32)-(35), times ``d-1`` from Eq. (36).

    ``n_un`` must be divisible by ``n_cols``.  Provided for differencing against the
    emitted call graph -- not used to cost the bloq.
    """
    n_un, n_cols, b = int(n_un), int(n_cols), int(b)
    if n_un % n_cols:
        raise ValueError(f"n_un={n_un} must be divisible by n_cols={n_cols}")
    d = n_un // n_cols
    n = max(1, n_un - 1).bit_length()
    t32 = ceil(n_un / (2 * lam)) + lam * b - 2
    t33 = (n_un // 2) * (ceil(n_un / (2 * lam)) + 2 * lam * b - 5)
    t34 = (n - 3) * (n_un // 2 - 1)
    t35 = ceil(n_un / lam) + lam * b + ceil(n_un / lam_p) + lam_p - 6
    return (d - 1) * (t32 + t33 + t34 + t35)


@attrs.frozen
class BerryIsometrySynthesisQROAM(GateWithRegisters):
    r"""Berry §III.B isometry synthesis, block-multiplexed.

    Attributes:
        n_blocks: block-diagonal multiplexer address (e.g. :math:`N_k` or :math:`N_k^2`).
        n_rows: physical row count; padded internally to a multiple of ``n_cols``.
        n_cols: number of specified columns :math:`\chi`.
        phase_bitsize: angle bitsize :math:`b`.
        optimal_T: choose Toffoli-optimal QROAM blocking.
    """

    n_blocks: SymbolicInt
    n_rows: SymbolicInt
    n_cols: SymbolicInt
    phase_bitsize: SymbolicInt
    optimal_T: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_rows, self.n_cols):
            if int(self.n_cols) < 1 or int(self.n_cols) > int(self.n_rows):
                raise ValueError("need 1 <= n_cols <= n_rows")

    # ------------------------------ shape helpers ------------------------------

    @cached_property
    def n_un(self) -> int:
        """Row dimension padded so that ``d = n_un / n_cols`` is an integer (Eq. 36)."""
        return pad_rows_to_multiple(int(self.n_rows), int(self.n_cols))

    @cached_property
    def d(self) -> int:
        """Local dimension; the cost carries a factor ``d - 1``."""
        return self.n_un // int(self.n_cols)

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_un - 1)

    @cached_property
    def block_bitsize(self) -> SymbolicInt:
        return bit_length(self.n_blocks - 1)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            block=self.block_bitsize, system=self.system_bitsize,
            phase_gradient=self.phase_bitsize,
        )

    # -------------------------------- sub-bloqs --------------------------------

    @cached_property
    def _lbs(self) -> Optional[Tuple[int, ...]]:
        if not self.optimal_T:
            return None
        return optimal_interferometer_log_block_sizes(
            int(self.n_blocks), int(self.n_un), int(self.phase_bitsize)
        )

    @cached_property
    def phase_layer(self) -> BlockInterferometerPhaseLayerQROAM:
        """Eq. (33): one merged multiplexed layer, run ``N_un/2`` times."""
        return BlockInterferometerPhaseLayerQROAM(
            n_blocks=self.n_blocks, n_rows=self.n_un,
            phase_bitsize=self.phase_bitsize, log_block_sizes=self._lbs,
        )

    @cached_property
    def final_phase_layer(self) -> BlockInterferometerFinalPhasesQROAM:
        """Eq. (35): the residual diagonal -- does NOT shrink with the column count."""
        return BlockInterferometerFinalPhasesQROAM(
            n_blocks=self.n_blocks, n_rows=self.n_un,
            phase_bitsize=self.phase_bitsize,
            log_block_sizes=self._lbs, adjoint_log_block_sizes=self._lbs,
        )

    @cached_property
    def _rotation(self) -> Bloq:
        try:
            from .phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        except ImportError:
            from phase_gradient_signed_rotation import SignedCtrlAddIntoPhaseGrad
        return SignedCtrlAddIntoPhaseGrad(self.phase_bitsize)   # App. A (2007.07391): b-2

    # ---------------------------- resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        n = int(self.system_bitsize)
        mult = self.d - 1                       # Eq. (36)
        # (32) the controlled addition/subtraction that sets up the subspace
        ret[self._rotation] += mult
        # (33) N_un/2 multiplexed layers
        ret[self.phase_layer] += mult * (self.n_un // 2)
        # (34) the increment/decrement network
        inc = max(0, (n - 3)) * (self.n_un // 2 - 1)
        if inc > 0:
            ret[AddK(dtype=QUInt(max(1, n)), k=1)] += mult * max(0, n - 3)
        # (35) the residual diagonal phase layer
        ret[self.final_phase_layer] += mult
        emit_range_safety(ret, (int(self.n_blocks),), count=mult)
        return ret
