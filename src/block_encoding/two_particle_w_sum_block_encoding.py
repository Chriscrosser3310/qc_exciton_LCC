from __future__ import annotations

from math import comb

import numpy as np
from attrs import field, frozen

try:
    from qualtran import BQUInt, Bloq, BloqBuilder, CtrlSpec, QBit, QUInt, Register, Signature
    from qualtran.bloqs.basic_gates import Swap, XGate, ZGate
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for two-particle W-sum block-encoding. Install with extra '[qualtran]'."
    ) from exc

from .two_particle_row_oracles import TwoParticleSparseBlockEncoding


def _all_pairs(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def _swap_sequence_for_target_perm(perm: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    n = len(perm)
    if sorted(perm) != list(range(n)):
        raise ValueError("perm must be a permutation of 0..n-1.")
    cur = list(range(n))
    swaps: list[tuple[int, int]] = []
    for i in range(n):
        if cur[i] == perm[i]:
            continue
        j = cur.index(perm[i], i + 1)
        cur[i], cur[j] = cur[j], cur[i]
        swaps.append((i, j))
    return tuple(swaps)


def _perm_for_pair(n: int, a: int, b: int) -> tuple[int, ...]:
    """Permutation such that output[0]=input[a], output[1]=input[b]."""
    rest = [k for k in range(n) if k not in (a, b)]
    return tuple([a, b] + rest)


@frozen
class TwoParticleWSumBlockEncoding(Bloq):
    r"""Signed pair-sum block-encoding built from a two-particle sparse block-encoding W.

    Let n = 2m be the number of particle registers. This bloq iterates over all unordered pairs
    (a, b), a < b, routes that pair to slots (r0, r1), applies W on (r0, r1), and unroutes.

    The first ancilla prepare is a uniform superposition over pair labels with sign:
    - terms with a < m and b >= m (cross-partition pairs) get a minus sign.
    - all other terms get a plus sign.

    The final unprepare is unsigned uniform.
    """

    num_pairs: int
    D: int
    L: int
    R_c: int
    R_loc: int
    W_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    def __attrs_post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be >= 1.")
        expected = (
            (self.L,) * self.D
            + (self.L,) * self.D
            + (2 * self.R_c + 1,) * self.D
            + (2 * self.R_loc + 1,) * self.D
        )
        if self.W_table.shape != expected:
            raise ValueError(f"W_table has shape {self.W_table.shape}, expected {expected}.")

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def term_count(self) -> int:
        return comb(self.n_registers, 2)

    @property
    def term_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(self.term_count))))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        regs = [
            Register("q", QBit()),
            Register("sel", BQUInt(self.term_bitsize, self.term_count)),
            Register("m", BQUInt(self.m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
            Register("l", BQUInt(self.l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
        ]
        regs.extend(
            Register(f"r{k}", BQUInt(self.i_bitsize, self.L), shape=(self.D,))
            for k in range(self.n_registers)
        )
        return Signature(regs)

    @property
    def pair_terms(self) -> tuple[tuple[int, int], ...]:
        return _all_pairs(self.n_registers)

    def _term_control_bits(self, term_idx: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.term_bitsize).to_bits(int(term_idx)))

    def _is_negative_term(self, a: int, b: int) -> bool:
        # Cross-partition terms: first half (0..m-1) vs second half (m..2m-1).
        return (a < self.num_pairs) and (b >= self.num_pairs)

    def build_composite_bloq(self, bb: BloqBuilder, q, sel, m, l, **regs):
        r_regs = [list(regs[f"r{k}"]) for k in range(self.n_registers)]
        pairs = self.pair_terms

        # 1) First (signed) prepare over pair-index ancilla.
        sel_bits = list(bb.split(sel))
        if self.term_count > 1:
            prep_sel = PrepareUniformSuperposition(n=self.term_count)
            sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
            sel_u = bb.add(prep_sel, target=sel_u)
            sel_bits = list(bb.split(sel_u))
        else:
            prep_sel = None

        # Impose minus sign on cross-partition pair terms.
        phase_q = bb.allocate(1)
        phase_q = bb.add(XGate(), q=phase_q)  # |1>, so Z applies phase kickback.
        for t_idx, (a, b) in enumerate(pairs):
            if not self._is_negative_term(a, b):
                continue
            ctrl_spec = CtrlSpec(cvs=self._term_control_bits(t_idx))
            ctrls = np.array(sel_bits, dtype=object)
            ctrls, phase_q = bb.add(ZGate().controlled(ctrl_spec), ctrl=ctrls, q=phase_q)
            sel_bits = list(ctrls)
        phase_q = bb.add(XGate(), q=phase_q)
        bb.free(phase_q)

        # 2) Controlled routing permutations for each pair term.
        for t_idx, (a, b) in enumerate(pairs):
            perm = _perm_for_pair(self.n_registers, a, b)
            swap_seq = _swap_sequence_for_target_perm(perm)
            ctrl_spec = CtrlSpec(cvs=self._term_control_bits(t_idx))
            sw = Swap(self.i_bitsize).controlled(ctrl_spec)
            for u, v in swap_seq:
                for d in range(self.D):
                    ctrls = np.array(sel_bits, dtype=object)
                    ctrls, r_regs[u][d], r_regs[v][d] = bb.add(
                        sw,
                        ctrl=ctrls,
                        x=r_regs[u][d],
                        y=r_regs[v][d],
                    )
                    sel_bits = list(ctrls)

        # 3) Apply W on routed slots (r0, r1).
        w_bloq = TwoParticleSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.W_table,
            entry_bitsize=self.entry_bitsize,
        )
        out_w = bb.add_d(
            w_bloq,
            q=q,
            m=m,
            l=l,
            i=np.array(r_regs[0], dtype=object),
            j=np.array(r_regs[1], dtype=object),
        )
        q = out_w["q"]
        m = out_w["m"]
        l = out_w["l"]
        r_regs[0] = list(out_w["i"])
        r_regs[1] = list(out_w["j"])

        # 4) Undo controlled routing permutations.
        for t_idx in reversed(range(len(pairs))):
            a, b = pairs[t_idx]
            perm = _perm_for_pair(self.n_registers, a, b)
            swap_seq = _swap_sequence_for_target_perm(perm)
            ctrl_spec = CtrlSpec(cvs=self._term_control_bits(t_idx))
            sw = Swap(self.i_bitsize).controlled(ctrl_spec)
            for u, v in reversed(swap_seq):
                for d in range(self.D):
                    ctrls = np.array(sel_bits, dtype=object)
                    ctrls, r_regs[u][d], r_regs[v][d] = bb.add(
                        sw,
                        ctrl=ctrls,
                        x=r_regs[u][d],
                        y=r_regs[v][d],
                    )
                    sel_bits = list(ctrls)

        # 5) Unsigned unprepare.
        if prep_sel is not None:
            sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
            sel_u = bb.add(prep_sel.adjoint(), target=sel_u)
            sel = bb.join(bb.split(sel_u), dtype=BQUInt(self.term_bitsize, self.term_count))
        else:
            sel = bb.join(np.array(sel_bits, dtype=object), dtype=BQUInt(self.term_bitsize, self.term_count))

        out = {"q": q, "sel": sel, "m": m, "l": l}
        out.update({f"r{k}": np.array(r_regs[k], dtype=object) for k in range(self.n_registers)})
        return out

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        ctrl_bloq = TwoParticleControlledWSumBlockEncoding(
            num_pairs=self.num_pairs,
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            W_table=self.W_table,
            entry_bitsize=self.entry_bitsize,
        )
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=None,
            bloq_with_ctrl=ctrl_bloq,
            ctrl_reg_name="ctrl",
        )


@frozen
class TwoParticleControlledWSumBlockEncoding(Bloq):
    """Single-control wrapper for `TwoParticleWSumBlockEncoding`."""

    num_pairs: int
    D: int
    L: int
    R_c: int
    R_loc: int
    W_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def term_count(self) -> int:
        return comb(self.n_registers, 2)

    @property
    def term_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(self.term_count))))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        regs = [
            Register("ctrl", QBit()),
            Register("q", QBit()),
            Register("sel", BQUInt(self.term_bitsize, self.term_count)),
            Register("m", BQUInt(self.m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
            Register("l", BQUInt(self.l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
        ]
        regs.extend(
            Register(f"r{k}", BQUInt(self.i_bitsize, self.L), shape=(self.D,))
            for k in range(self.n_registers)
        )
        return Signature(regs)

    @property
    def pair_terms(self) -> tuple[tuple[int, int], ...]:
        return _all_pairs(self.n_registers)

    def _term_control_bits(self, term_idx: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.term_bitsize).to_bits(int(term_idx)))

    def _is_negative_term(self, a: int, b: int) -> bool:
        return (a < self.num_pairs) and (b >= self.num_pairs)

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, q, sel, m, l, **regs):
        r_regs = [list(regs[f"r{k}"]) for k in range(self.n_registers)]
        pairs = self.pair_terms

        sel_bits = list(bb.split(sel))
        if self.term_count > 1:
            prep_sel = PrepareUniformSuperposition(n=self.term_count)
            sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
            sel_u = bb.add(prep_sel, target=sel_u)
            sel_bits = list(bb.split(sel_u))
        else:
            prep_sel = None

        # Controlled sign phase.
        phase_q = bb.allocate(1)
        phase_q = bb.add(XGate(), q=phase_q)
        for t_idx, (a, b) in enumerate(pairs):
            if not self._is_negative_term(a, b):
                continue
            ctrl_spec = CtrlSpec(cvs=(1,) + self._term_control_bits(t_idx))
            ctrls = np.array([ctrl] + sel_bits, dtype=object)
            ctrls, phase_q = bb.add(ZGate().controlled(ctrl_spec), ctrl=ctrls, q=phase_q)
            ctrl = ctrls[0]
            sel_bits = list(ctrls[1:])
        phase_q = bb.add(XGate(), q=phase_q)
        bb.free(phase_q)

        # Controlled routing.
        for t_idx, (a, b) in enumerate(pairs):
            perm = _perm_for_pair(self.n_registers, a, b)
            swap_seq = _swap_sequence_for_target_perm(perm)
            ctrl_spec = CtrlSpec(cvs=(1,) + self._term_control_bits(t_idx))
            sw1 = Swap(1).controlled(ctrl_spec)
            for u, v in swap_seq:
                for d in range(self.D):
                    xb = list(bb.split(r_regs[u][d]))
                    yb = list(bb.split(r_regs[v][d]))
                    for bi in range(self.i_bitsize):
                        ctrls = np.array([ctrl] + sel_bits, dtype=object)
                        ctrls, xb[bi], yb[bi] = bb.add(
                            sw1, ctrl=ctrls, x=xb[bi], y=yb[bi]
                        )
                        ctrl = ctrls[0]
                        sel_bits = list(ctrls[1:])
                    r_regs[u][d] = bb.join(np.array(xb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))
                    r_regs[v][d] = bb.join(np.array(yb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))

        # Controlled W apply.
        w_bloq = TwoParticleSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.W_table,
            entry_bitsize=self.entry_bitsize,
        )
        out_w = bb.add_d(
            w_bloq.controlled(),
            ctrl=ctrl,
            q=q,
            m=m,
            l=l,
            i=np.array(r_regs[0], dtype=object),
            j=np.array(r_regs[1], dtype=object),
        )
        ctrl = out_w["ctrl"]
        q = out_w["q"]
        m = out_w["m"]
        l = out_w["l"]
        r_regs[0] = list(out_w["i"])
        r_regs[1] = list(out_w["j"])

        # Undo controlled routing.
        for t_idx in reversed(range(len(pairs))):
            a, b = pairs[t_idx]
            perm = _perm_for_pair(self.n_registers, a, b)
            swap_seq = _swap_sequence_for_target_perm(perm)
            ctrl_spec = CtrlSpec(cvs=(1,) + self._term_control_bits(t_idx))
            sw1 = Swap(1).controlled(ctrl_spec)
            for u, v in reversed(swap_seq):
                for d in range(self.D):
                    xb = list(bb.split(r_regs[u][d]))
                    yb = list(bb.split(r_regs[v][d]))
                    for bi in range(self.i_bitsize):
                        ctrls = np.array([ctrl] + sel_bits, dtype=object)
                        ctrls, xb[bi], yb[bi] = bb.add(
                            sw1, ctrl=ctrls, x=xb[bi], y=yb[bi]
                        )
                        ctrl = ctrls[0]
                        sel_bits = list(ctrls[1:])
                    r_regs[u][d] = bb.join(np.array(xb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))
                    r_regs[v][d] = bb.join(np.array(yb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))

        if prep_sel is not None:
            sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
            sel_u = bb.add(prep_sel.adjoint(), target=sel_u)
            sel = bb.join(bb.split(sel_u), dtype=BQUInt(self.term_bitsize, self.term_count))
        else:
            sel = bb.join(np.array(sel_bits, dtype=object), dtype=BQUInt(self.term_bitsize, self.term_count))

        out = {"ctrl": ctrl, "q": q, "sel": sel, "m": m, "l": l}
        out.update({f"r{k}": np.array(r_regs[k], dtype=object) for k in range(self.n_registers)})
        return out

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1,
            bloq_with_ctrl=self,
            ctrl_reg_name="ctrl",
        )


def build_two_particle_w_sum_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    W_table: np.ndarray,
    entry_bitsize: int = 10,
) -> TwoParticleWSumBlockEncoding:
    return TwoParticleWSumBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        W_table=np.asarray(W_table),
        entry_bitsize=int(entry_bitsize),
    )


def build_two_particle_controlled_w_sum_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    W_table: np.ndarray,
    entry_bitsize: int = 10,
) -> TwoParticleControlledWSumBlockEncoding:
    return TwoParticleControlledWSumBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        W_table=np.asarray(W_table),
        entry_bitsize=int(entry_bitsize),
    )
