r"""Block encoding of the antisymmetric (and symmetric) projector of the symmetric group.

This implements the antisymmetrized-projector construction of

    M. L. LaBorde, S. Rethinasamy, and M. M. Wilde,
    "Quantum Algorithms for Realizing Symmetric, Asymmetric, and Antisymmetric
    Projectors", arXiv:2407.17563.

For the symmetric group :math:`S_k` acting on :math:`k` subsystems of local
dimension :math:`d = 2^s` by permutation (the standard representation, Eq. 7-8),
the antisymmetric projector is (Eq. 18 / 37)

.. math::

    \Pi_{\mathrm{anti}} = \frac{1}{k!} \sum_{\sigma \in S_k}
        \mathrm{sgn}(\sigma)\, U(\sigma).

The paper realizes this projector as a *generalized phase estimation* circuit
(Fig. 3, Eq. 22 / 44): prepare a control register in the symmetric-group plus
state :math:`|+_{S_k}\rangle` (Eq. 10-12), apply the controlled permutation
:math:`\sum_\sigma |\sigma\rangle\langle\sigma| \otimes U(\sigma)`
(``SELECT``), then ``unprepare`` against the signed state :math:`|-_{S_k}\rangle`.
Because :math:`Z^{\otimes n}|+_{S_k}\rangle = |-_{S_k}\rangle` (Eq. 20, 40-41),

.. math::

    W = \mathrm{PREP}^\dagger\, Z^{\otimes n_c}\, \mathrm{SELECT}\, \mathrm{PREP}

is a :math:`(1, n_c, 0)` block encoding of :math:`\Pi_{\mathrm{anti}}`, i.e.

.. math::

    (\langle 0|_C \otimes I)\, W\, (|0\rangle_C \otimes I) = \Pi_{\mathrm{anti}},

where :math:`C` is the control register of :math:`n_c = k(k-1)/2` qubits.

Why the :math:`Z^{\otimes n_c}` layer reproduces the permutation sign: the control
register is laid out in :math:`k-1` blocks (one per :math:`j = 2, \dots, k`), where
block :math:`j` has :math:`j-1` qubits and selects, via a one-hot encoding, either
the identity or one transposition :math:`(i, j)` (Eq. 12-15).  ``PREP`` puts each
block into an equal superposition of its :math:`j` one-hot+vacuum basis states, so
:math:`|+_{S_k}\rangle` is the equal-amplitude superposition over the :math:`k!`
"valid" control strings, one per group element :math:`\sigma`.  A valid string for
:math:`\sigma` has Hamming weight equal to the number of transpositions composing
:math:`\sigma`, so :math:`(-1)^{\text{Hamming weight}} = \mathrm{sgn}(\sigma)`; that
parity is exactly the phase :math:`Z^{\otimes n_c}` imprints.

Dropping the :math:`Z^{\otimes n_c}` layer (``signed=False``) instead block-encodes
the *symmetric* projector :math:`\Pi_{\mathrm{sym}} = \frac{1}{k!}\sum_\sigma U(\sigma)`
(Eq. 4 / 6), since then both sides reduce to :math:`|+_{S_k}\rangle`.

The antisymmetric subspace is nontrivial only when :math:`d \ge k` (its dimension is
:math:`\binom{d}{k}`); for :math:`d < k` the construction faithfully encodes the zero
operator.  All three statements above are verified to machine precision against the
dense numpy references in this module.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Dict, List, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QAny,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, Ry, ZGate
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_block_encoding import PrepareIdentity
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.symbolics import SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


# =============================================================================
# Numpy references (dense, for small k; used for tests and validation)
# =============================================================================


def permutation_sign(perm: Tuple[int, ...]) -> int:
    r"""Return :math:`\mathrm{sgn}(\sigma) \in \{+1, -1\}` for a permutation in one-line form.

    ``perm[a]`` is the image of ``a`` under :math:`\sigma`.
    """
    k = len(perm)
    seen = [False] * k
    sign = 1
    for a in range(k):
        if seen[a]:
            continue
        length = 0
        b = a
        while not seen[b]:
            seen[b] = True
            b = perm[b]
            length += 1
        if length % 2 == 0:  # an L-cycle has sign (-1)^{L-1}
            sign = -sign
    return sign


def permutation_unitary(perm: Tuple[int, ...], subsystem_dim: int) -> NDArray[np.complex128]:
    r"""Standard representation :math:`U(\sigma)` of ``perm`` on :math:`(\mathbb{C}^d)^{\otimes k}`.

    Uses the convention of Eq. 7-8: ``U(perm)`` maps
    :math:`|i_0 \dots i_{k-1}\rangle \mapsto |i_{\sigma^{-1}(0)} \dots i_{\sigma^{-1}(k-1)}\rangle`.
    Subsystem ``0`` is the most-significant digit.
    """
    k = len(perm)
    d = subsystem_dim
    dim = d**k
    inv = [0] * k
    for a in range(k):
        inv[perm[a]] = a
    u = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        digits = [(idx // (d ** (k - 1 - p))) % d for p in range(k)]
        new_digits = [digits[inv[p]] for p in range(k)]
        new_idx = sum(new_digits[p] * d ** (k - 1 - p) for p in range(k))
        u[new_idx, idx] = 1.0
    return u


def antisymmetric_projector(n_subsystems: int, subsystem_dim: int) -> NDArray[np.complex128]:
    r"""Dense :math:`\Pi_{\mathrm{anti}} = \frac{1}{k!}\sum_\sigma \mathrm{sgn}(\sigma) U(\sigma)` (Eq. 18)."""
    k = n_subsystems
    dim = subsystem_dim**k
    proj = np.zeros((dim, dim), dtype=np.complex128)
    for perm in itertools.permutations(range(k)):
        proj += permutation_sign(perm) * permutation_unitary(perm, subsystem_dim)
    return proj / math.factorial(k)


def symmetric_projector(n_subsystems: int, subsystem_dim: int) -> NDArray[np.complex128]:
    r"""Dense :math:`\Pi_{\mathrm{sym}} = \frac{1}{k!}\sum_\sigma U(\sigma)` (Eq. 4 / 6)."""
    k = n_subsystems
    dim = subsystem_dim**k
    proj = np.zeros((dim, dim), dtype=np.complex128)
    for perm in itertools.permutations(range(k)):
        proj += permutation_unitary(perm, subsystem_dim)
    return proj / math.factorial(k)


def control_bitsize(n_subsystems: int) -> int:
    r"""Number of control qubits :math:`n_c = k(k-1)/2` for the :math:`S_k` test (Eq. 10-15)."""
    return n_subsystems * (n_subsystems - 1) // 2


def block_offsets(n_subsystems: int) -> Dict[int, int]:
    r"""Global qubit offset of each control block ``j`` (``j = 2, ..., k``).

    Block ``j`` occupies ``j-1`` control qubits; blocks are concatenated in order
    ``j = 2, 3, ..., k`` (most-significant block first).
    """
    offsets: Dict[int, int] = {}
    cur = 0
    for j in range(2, n_subsystems + 1):
        offsets[j] = cur
        cur += j - 1
    return offsets


def plus_sk_amplitudes(n_subsystems: int) -> NDArray[np.complex128]:
    r"""Dense amplitudes of :math:`|+_{S_k}\rangle` on :math:`n_c` qubits (Eq. 10-15).

    Equal amplitude :math:`1/\sqrt{k!}` on each of the :math:`k!` valid control strings.
    """
    k = n_subsystems
    nc = control_bitsize(k)
    offsets = block_offsets(k)
    amps = np.zeros(2**nc, dtype=np.complex128)
    # Enumerate one choice per block: identity (no bit) or transposition i -> value 2^{i-1}.
    block_choices = []
    for j in range(2, k + 1):
        m = j - 1
        # value within block (m qubits, MSB-first): 0, or one-hot 2^{i-1} for i = 1..m
        choices = [0] + [1 << (i - 1) for i in range(1, m + 1)]
        block_choices.append((j, m, choices))

    def _recurse(bi: int, acc_index: int):
        if bi == len(block_choices):
            amps[acc_index] = 1.0
            return
        j, m, choices = block_choices[bi]
        off = offsets[j]
        for val in choices:
            # place ``val`` (m-bit, MSB-first within block) into global index
            shifted = val << (nc - off - m)
            _recurse(bi + 1, acc_index | shifted)

    if nc == 0:
        return np.array([1.0], dtype=np.complex128)
    _recurse(0, 0)
    amps /= np.sqrt(math.factorial(k))
    return amps


def _wstate_angles(j: int) -> List[float]:
    r"""Angles for the W-state-with-vacuum ladder preparing block ``j`` (``m = j-1`` qubits).

    Step ``t`` (``t = 0 .. m-1``) rotates qubit ``t`` conditioned on qubits ``0..t-1`` being
    ``|0\rangle``, with ``P(q_t = 1 | prev = 0) = 1/(j - t)``.  This yields equal amplitude
    :math:`1/\sqrt{j}` on the vacuum string and each one-hot string.
    """
    return [2.0 * math.asin(math.sqrt(1.0 / (j - t))) for t in range(j - 1)]


# =============================================================================
# Control-state preparation: PREP |0> = |+_{S_k}>
# =============================================================================


@attrs.frozen
class PrepareSymmetricGroupControl(GateWithRegisters):
    r"""Prepare the symmetric-group control state :math:`|+_{S_k}\rangle` (Eq. 10-15).

    Acts on a single ``control`` register of :math:`n_c = k(k-1)/2` qubits, laid out as
    :math:`k-1` blocks (block ``j`` has ``j-1`` qubits, ``j = 2..k``).  Each block is
    prepared with a W-state-with-vacuum ladder of (multi-controlled) ``Ry`` rotations,
    giving an equal superposition over its ``j`` one-hot+vacuum basis states.

    Attributes:
        n_subsystems: the number ``k`` of permuted subsystems.
        uncompute: if True, apply the inverse preparation (maps :math:`|+_{S_k}\rangle \to |0\rangle`).
    """

    n_subsystems: int
    uncompute: bool = False

    def __attrs_post_init__(self):
        assert self.n_subsystems >= 2, "need k >= 2 control blocks"

    @property
    def control_bitsize(self) -> int:
        return control_bitsize(self.n_subsystems)

    @property
    def signature(self) -> Signature:
        return Signature.build(control=self.control_bitsize)

    def adjoint(self) -> 'PrepareSymmetricGroupControl':
        return attrs.evolve(self, uncompute=not self.uncompute)

    def _operations(self) -> List[Tuple[int, Tuple[int, ...], Bloq]]:
        r"""Forward (target_qubit, control_qubits, gate) list, in application order."""
        ops: List[Tuple[int, Tuple[int, ...], Bloq]] = []
        offsets = block_offsets(self.n_subsystems)
        for j in range(2, self.n_subsystems + 1):
            off = offsets[j]
            angles = _wstate_angles(j)
            for t, theta in enumerate(angles):
                target = off + t
                controls = tuple(off + s for s in range(t))  # qubits 0..t-1 of this block
                if controls:
                    gate: Bloq = Ry(theta).controlled(CtrlSpec(QBit(), cvs=(0,) * t))
                else:
                    gate = Ry(theta)
                ops.append((target, controls, gate))
        return ops

    def build_composite_bloq(self, bb: BloqBuilder, control: Soquet) -> Dict[str, SoquetT]:
        qubits = bb.split(control)
        ops = self._operations()
        if self.uncompute:
            ops = [(tgt, ctrls, g.adjoint()) for (tgt, ctrls, g) in reversed(ops)]
        for target, controls, gate in ops:
            if controls:
                ctrl_in = np.array([qubits[c] for c in controls], dtype=object)
                ctrl_out, q_out = bb.add(gate, ctrl=ctrl_in, q=qubits[target])
                for idx, c in enumerate(controls):
                    qubits[c] = ctrl_out[idx]
                qubits[target] = q_out
            else:
                qubits[target] = bb.add(gate, q=qubits[target])
        return {'control': bb.join(qubits)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        for _, _, gate in self._operations():
            ret[gate.adjoint() if self.uncompute else gate] += 1
        return ret


# =============================================================================
# SELECT: controlled permutation sum_sigma |sigma><sigma| (x) U(sigma)
# =============================================================================


@attrs.frozen
class SelectSymmetricGroupPermutation(GateWithRegisters):
    r"""Controlled permutation ``SELECT`` for the :math:`S_k` test (Fig. 2-3).

    For each control block ``j`` and one-hot position ``i`` (``1 <= i < j``), a single
    control qubit triggers the transposition :math:`(i, j)`, realized as a controlled
    swap of subsystems ``i-1`` and ``j-1``.  Because each block is one-hot on the
    prepared subspace, the composed action on a valid control string for :math:`\sigma`
    is the full permutation :math:`U(\sigma)`; on the whole control basis the bloq is
    block-diagonal, :math:`\mathrm{SELECT} = \sum_c |c\rangle\langle c| \otimes U_c`.

    Attributes:
        n_subsystems: number ``k`` of subsystems.
        subsystem_bitsize: qubits ``s`` per subsystem (local dimension ``d = 2^s``).
    """

    n_subsystems: int
    subsystem_bitsize: int

    def __attrs_post_init__(self):
        assert self.n_subsystems >= 2
        assert self.subsystem_bitsize >= 1

    @property
    def control_bitsize(self) -> int:
        return control_bitsize(self.n_subsystems)

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('control', QAny(self.control_bitsize)),
                Register('system', QAny(self.subsystem_bitsize), shape=(self.n_subsystems,)),
            ]
        )

    def _swaps(self) -> List[Tuple[int, int, int]]:
        r"""(control_qubit_index, subsystem_a, subsystem_b) triples, in application order."""
        swaps: List[Tuple[int, int, int]] = []
        offsets = block_offsets(self.n_subsystems)
        for j in range(2, self.n_subsystems + 1):
            m = j - 1
            off = offsets[j]
            for i in range(1, j):
                # one-hot value 2^{i-1} sits at MSB-first local position t = m - i
                t = m - i
                swaps.append((off + t, i - 1, j - 1))
        return swaps

    def build_composite_bloq(
        self, bb: BloqBuilder, control: Soquet, system: NDArray
    ) -> Dict[str, SoquetT]:
        control_qubits = bb.split(control)
        cswap = CSwap(self.subsystem_bitsize)
        for cq_idx, a, b in self._swaps():
            ctrl_out, x_out, y_out = bb.add(
                cswap, ctrl=control_qubits[cq_idx], x=system[a], y=system[b]
            )
            control_qubits[cq_idx] = ctrl_out
            system[a] = x_out
            system[b] = y_out
        return {'control': bb.join(control_qubits), 'system': system}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        ret[CSwap(self.subsystem_bitsize)] += len(self._swaps())
        return ret


# =============================================================================
# Block encoding: W = PREP^dag Z^n SELECT PREP
# =============================================================================


@attrs.frozen
class AntisymmetricProjectorBlockEncoding(BlockEncoding):
    r""":math:`(1, n_c, 0)` block encoding of the antisymmetric/symmetric projector (Eq. 22 / 44).

    Encodes :math:`\Pi_{\mathrm{anti}} = \frac{1}{k!}\sum_\sigma \mathrm{sgn}(\sigma) U(\sigma)`
    (``signed=True``) or :math:`\Pi_{\mathrm{sym}} = \frac{1}{k!}\sum_\sigma U(\sigma)`
    (``signed=False``) of the symmetric group :math:`S_k` acting by permutation on ``k``
    subsystems of ``s`` qubits each.

    The signal state is :math:`|0\rangle` on the ``ancilla`` (control) register, so

    .. math::
        (\langle 0|_{\mathrm{anc}} \otimes I)\, W\, (|0\rangle_{\mathrm{anc}} \otimes I)
            = \Pi,

    with :math:`W = \mathrm{PREP}^\dagger Z^{\otimes n_c} \mathrm{SELECT}\, \mathrm{PREP}`
    (the :math:`Z` layer is dropped for the symmetric projector).

    Registers (``BlockEncoding`` interface):
        system: ``k * s`` qubits holding the ``k`` permuted subsystems.
        ancilla: ``n_c = k(k-1)/2`` control qubits (signal state :math:`|0\rangle`).

    Attributes:
        n_subsystems: number ``k`` of permuted subsystems (``k >= 2``).
        subsystem_bitsize: qubits ``s`` per subsystem (local dimension ``d = 2^s``).
        signed: if True encode the antisymmetric projector, else the symmetric projector.
    """

    n_subsystems: int
    subsystem_bitsize: int
    signed: bool = True

    def __attrs_post_init__(self):
        assert self.n_subsystems >= 2, "need k >= 2 subsystems"
        assert self.subsystem_bitsize >= 1

    # ------------------------- shape helpers -------------------------

    @property
    def control_bitsize(self) -> int:
        return control_bitsize(self.n_subsystems)

    # ------------------------- BlockEncoding interface -------------------------

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.n_subsystems * self.subsystem_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.control_bitsize

    @property
    def resource_bitsize(self) -> SymbolicInt:
        return 0

    @property
    def alpha(self) -> SymbolicFloat:
        return 1.0

    @property
    def epsilon(self) -> SymbolicFloat:
        return 0.0

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('system', QAny(self.system_bitsize)),
                Register('ancilla', QAny(self.ancilla_bitsize)),
            ]
        )

    @property
    def signal_state(self) -> PrepareOracle:
        return PrepareIdentity.from_bitsizes((self.ancilla_bitsize,))

    # --------------------------- sub-bloqs ----------------------------

    @property
    def prepare(self) -> PrepareSymmetricGroupControl:
        return PrepareSymmetricGroupControl(self.n_subsystems)

    @property
    def select(self) -> SelectSymmetricGroupPermutation:
        return SelectSymmetricGroupPermutation(self.n_subsystems, self.subsystem_bitsize)

    # --------------------------- composite circuit -----------------------------

    def _split_system(self, bb: BloqBuilder, system: Soquet) -> NDArray:
        s = self.subsystem_bitsize
        flat = bb.split(system)
        chunks = [
            bb.join(flat[a * s : (a + 1) * s], dtype=QAny(s)) for a in range(self.n_subsystems)
        ]
        return np.array(chunks, dtype=object)

    def _join_system(self, bb: BloqBuilder, chunks: NDArray) -> Soquet:
        flat = np.concatenate([bb.split(c) for c in chunks])
        return bb.join(flat, dtype=QAny(self.system_bitsize))

    def build_composite_bloq(self, bb: BloqBuilder, system: Soquet, ancilla: Soquet) -> Dict[str, SoquetT]:
        # PREP: |0> -> |+_{S_k}> on the control (ancilla) register.
        (ancilla,) = bb.add_t(self.prepare, control=ancilla)

        # SELECT: controlled permutation on the k subsystems.
        sys_chunks = self._split_system(bb, system)
        out = bb.add_d(self.select, control=ancilla, system=sys_chunks)
        ancilla = out['control']
        sys_chunks = out['system']

        # Z^{(x) n_c} on the control register -> turns |+_{S_k}> into |-_{S_k}> on unprepare.
        if self.signed:
            ctrl_qubits = bb.split(ancilla)
            for idx in range(len(ctrl_qubits)):
                ctrl_qubits[idx] = bb.add(ZGate(), q=ctrl_qubits[idx])
            ancilla = bb.join(ctrl_qubits)

        # PREP^dagger.
        (ancilla,) = bb.add_t(self.prepare.adjoint(), control=ancilla)

        system = self._join_system(bb, sys_chunks)
        return {'system': system, 'ancilla': ancilla}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        ret: 'Counter[Bloq]' = Counter()
        ret[self.prepare] += 1
        ret[self.prepare.adjoint()] += 1
        ret[self.select] += 1
        if self.signed:
            ret[ZGate()] += self.control_bitsize
        return ret

    def projector(self) -> NDArray[np.complex128]:
        r"""Dense projector this bloq block-encodes (antisymmetric or symmetric)."""
        if self.signed:
            return antisymmetric_projector(self.n_subsystems, 2**self.subsystem_bitsize)
        return symmetric_projector(self.n_subsystems, 2**self.subsystem_bitsize)
