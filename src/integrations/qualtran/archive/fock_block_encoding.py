r"""Data-free SVD-interferometer block encoding of a block Fock operator.

Encodes

    F = sum_{k=0}^{N_k - 1} |k><k| (x) F_k ,

where each ``F_k`` is an ``N x N`` matrix (``||F_k|| <= 1``) -- the per-momentum Fock /
single-particle block.  The operator acts on an ``N_k x N_IP`` system register (momentum
``k`` plus an ``N_IP``-sized matrix register), but the physical block dimension ``N`` can
be much smaller than ``N_IP``: each ``F_k`` lives in the top-left ``N x N`` corner of the
``N_IP``-sized register.

Construction:

  1.  A :class:`LessThanConstant` comparator cuts the input off to the first ``N`` entries
      of the ``N_IP``-sized matrix register (flags ``p < N`` into a 1-qubit ancilla),
      restraining the encoded operator to the physical ``N``-dimensional subspace.  It is
      computed before and uncomputed after the block encoding.
  2.  The ``N x N`` blocks are block encoded (with *fake / data-free* angle tables) by
      :class:`SVDBlockEncodingInterferometer` on a ``next_pow2(N)``-dim register:
      ``B = (I_a (x) U)(R_y (x) I_s)(I_a (x) V)`` with ``F_k = U_k Sigma_k V_k``.

All Toffoli / qubit counts come from Qualtran's resource counter walking the call graph;
no real matrix data is needed.  A cheap singly-controlled version is provided
(:meth:`FockBlockEncoding.get_ctrl_system` / :class:`_ControlledFockBlockEncoding`): only
the SVD interferometer is promoted to its controlled form, while the comparator pair stays
uncontrolled (it cancels when ``ctrl = 0``).
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import Optional, Tuple, TYPE_CHECKING

import attrs

from qualtran import Bloq, CtrlSpec, QAny, QBit, Register, Signature
from qualtran.bloqs.arithmetic import LessThanConstant
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import bit_length, is_symbolic, SymbolicFloat, SymbolicInt

try:
    from .svd_block_encoding_interferometer import SVDBlockEncodingInterferometer
except ImportError:
    from svd_block_encoding_interferometer import SVDBlockEncodingInterferometer

if TYPE_CHECKING:
    from qualtran import AddControlledT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _next_power_of_two(n: int) -> int:
    return 1 << max(1, (int(n) - 1).bit_length())


@attrs.frozen
class FockBlockEncoding(BlockEncoding):
    r"""$(1, \cdot, \epsilon)$ SVD-interferometer block encoding of $\sum_k |k\rangle\langle k| \otimes F_k$.

    Each $F_k$ is an $N \times N$ contraction embedded in the top-left of an
    $N_\mathrm{IP}$-sized register; the operator acts on an $N_k \times N_\mathrm{IP}$
    system register with $N \le N_\mathrm{IP}$.

    Attributes:
        N_k: number of blocks $F_k$.
        N_IP: dimension of the matrix system register (it holds $N_\mathrm{IP}$ values).
        N: physical block dimension ($F_k$ is $N \times N$); must satisfy $N \le N_\mathrm{IP}$.
        phase_bitsize: bitsize $b$ of the phase / angle registers.
        optimal_T: forwarded to the SVD interferometer (Toffoli-optimal QROAM blocking).
        restrict_input: when True, a ``LessThanConstant`` comparator cuts the input off to
            the first $N$ entries of the $N_\mathrm{IP}$ register (one flag ancilla,
            computed + uncomputed).

    Registers (standard ``BlockEncoding`` interface):
        system: ``k_bitsize + matrix_bitsize`` qubits (momentum + ``N_IP``-sized matrix).
        ancilla: SVD block-encoding ancilla (1 qubit) + comparator flag (1 if restricting).
        resource: ``phase_bitsize``-qubit phase-gradient workspace.
    """

    N_k: SymbolicInt
    N_IP: SymbolicInt
    N: SymbolicInt
    phase_bitsize: SymbolicInt = 32
    optimal_T: bool = False
    restrict_input: bool = True

    def __attrs_post_init__(self):
        if not is_symbolic(self.N, self.N_IP) and int(self.N) > int(self.N_IP):
            raise ValueError(f"N (={self.N}) must be <= N_IP (={self.N_IP})")

    # ----------------------------- shape helpers -----------------------------

    @cached_property
    def k_bitsize(self) -> SymbolicInt:
        return bit_length(self.N_k - 1)

    @cached_property
    def matrix_bitsize(self) -> SymbolicInt:
        # The N_IP-sized matrix register the operator nominally acts on.
        return bit_length(self.N_IP - 1)

    @cached_property
    def n_rows_inner(self) -> SymbolicInt:
        # The N x N block padded to a power of two for the SVD interferometer.  The
        # interferometer pairs modes per layer, so it needs at least 4 dimensions.
        return max(4, _next_power_of_two(self.N))

    # ------------------------- BlockEncoding interface -------------------------

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        # N_k x N_IP register: momentum block + N_IP-sized matrix register.
        return self.k_bitsize + self.matrix_bitsize

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        # SVD block-encoding ancilla + (when restricting) one input-cutoff comparator flag.
        return self.svd.ancilla_bitsize + (1 if self.restrict_input else 0)

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return self.phase_bitsize

    @property
    def alpha(self) -> SymbolicFloat:
        # ||F_k|| <= 1, so the SVD interferometer has subnormalization 1.
        return self.svd.alpha

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

    # --------------------------- Sub-bloq factories ----------------------------

    @cached_property
    def svd(self) -> SVDBlockEncodingInterferometer:
        # Fake-data SVD interferometer of the N x N blocks (padded to next_pow2(N)).
        return SVDBlockEncodingInterferometer(
            n_blocks=self.N_k,
            n_rows=self.n_rows_inner,
            phase_bitsize=self.phase_bitsize,
            optimal_T=self.optimal_T,
        )

    @property
    def input_comparator(self) -> Bloq:
        # Cut the input off to the first N entries of the N_IP-sized matrix register.
        return LessThanConstant(bitsize=self.matrix_bitsize, less_than_val=self.N)

    # ----------------------------- Resource counts ------------------------------

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        ret: "Counter[Bloq]" = Counter()
        if self.restrict_input:
            ret[self.input_comparator] += 2   # cut off input < N (compute + uncompute)
        ret[self.svd] += 1                     # SVD interferometer of the N x N blocks
        return ret

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> "Tuple[Bloq, AddControlledT]":
        """Cheap single-qubit control: only the SVD interferometer gains the control.

        ``.controlled()`` returns :class:`_ControlledFockBlockEncoding`, which promotes the
        SVD interferometer to its (cheap) controlled form; the comparator cutoff pair stays
        uncontrolled because it cancels when ``ctrl = 0``.
        """
        return get_ctrl_system_1bit_cv_from_bloqs(
            self, ctrl_spec, current_ctrl_bit=None,
            bloq_with_ctrl=_ControlledFockBlockEncoding(self), ctrl_reg_name='ctrl',
        )


@attrs.frozen
class _ControlledFockBlockEncoding(BlockEncoding):
    """Singly-controlled :class:`FockBlockEncoding`.

    The external control reaches only the SVD interferometer (via its own cheap controlled
    form); the input-cutoff comparator pair is left uncontrolled (it cancels at ``ctrl = 0``).
    """

    inner: FockBlockEncoding

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
        if self.inner.restrict_input:
            ret[self.inner.input_comparator] += 2   # uncontrolled (cancels at ctrl = 0)
        ret[self.inner.svd.controlled()] += 1       # only the SVD gains the control
        return ret
