from __future__ import annotations

import numpy as np
from attrs import field, frozen

try:
    from qualtran import BQUInt, Bloq, BloqBuilder, CtrlSpec, QBit, QUInt, Register, Signature
    from qualtran.bloqs.basic_gates import Swap
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for two-particle V-sum block-encoding. Install with extra '[qualtran]'."
    ) from exc

from .two_particle_row_oracles import TwoParticleSparseBlockEncoding


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


def _perm_map_sources_to_slots(
    n: int,
    src_a: int,
    src_b: int,
    dst_a: int,
    dst_b: int,
) -> tuple[int, ...]:
    """Build permutation with output[dst_a]=input[src_a], output[dst_b]=input[src_b]."""
    if len({src_a, src_b, dst_a, dst_b}) < 4:
        # Allowed only if source and destination pairs are distinct slots.
        # This class always uses distinct destination anchor slots and source terms.
        pass

    out = [-1] * n
    out[dst_a] = src_a
    out[dst_b] = src_b

    rem_src = [k for k in range(n) if k not in (src_a, src_b)]
    rem_dst = [k for k in range(n) if k not in (dst_a, dst_b)]
    for d, s in zip(rem_dst, rem_src):
        out[d] = s

    perm = tuple(out)
    if sorted(perm) != list(range(n)):
        raise ValueError("Internal permutation construction error.")
    return perm


@frozen
class TwoParticleVSumBlockEncoding(Bloq):
    r"""Uniform sum over cross-partition V terms via routed two-particle sparse block-encoding.

    Let n = 2m registers and anchor slots (m-1, m). We iterate over source pairs
    (a, b) with a < m and b >= m, route (a, b) -> (m-1, m), apply two-particle sparse
    block-encoding V on those anchor slots, then unroute.

    First prepare is unsigned uniform over cross-pair terms, final unprepare is unsigned.
    """

    num_pairs: int
    D: int
    L: int
    R_c: int
    R_loc: int
    V_table: np.ndarray = field(
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
        if self.V_table.shape != expected:
            raise ValueError(f"V_table has shape {self.V_table.shape}, expected {expected}.")

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def anchor_left(self) -> int:
        return self.num_pairs - 1

    @property
    def anchor_right(self) -> int:
        return self.num_pairs

    @property
    def cross_terms(self) -> tuple[tuple[int, int], ...]:
        # Ordered by first-half index then second-half index.
        return tuple(
            (a, b) for a in range(self.num_pairs) for b in range(self.num_pairs, self.n_registers)
        )

    @property
    def term_count(self) -> int:
        return len(self.cross_terms)

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

    def _term_control_bits(self, term_idx: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.term_bitsize).to_bits(int(term_idx)))

    def build_composite_bloq(self, bb: BloqBuilder, q, sel, m, l, **regs):
        r_regs = [list(regs[f"r{k}"]) for k in range(self.n_registers)]
        terms = self.cross_terms

        # 1) Unsigned prepare over cross terms.
        sel_bits = list(bb.split(sel))
        if self.term_count > 1:
            prep_sel = PrepareUniformSuperposition(n=self.term_count)
            sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
            sel_u = bb.add(prep_sel, target=sel_u)
            sel_bits = list(bb.split(sel_u))
        else:
            prep_sel = None

        # 2) Controlled route (a,b) -> (m-1,m).
        for t_idx, (a, b) in enumerate(terms):
            perm = _perm_map_sources_to_slots(
                self.n_registers, a, b, self.anchor_left, self.anchor_right
            )
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

        # 3) Apply V sparse block-encoding on anchor slots.
        v_bloq = TwoParticleSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.V_table,
            entry_bitsize=self.entry_bitsize,
        )
        out_v = bb.add_d(
            v_bloq,
            q=q,
            m=m,
            l=l,
            i=np.array(r_regs[self.anchor_left], dtype=object),
            j=np.array(r_regs[self.anchor_right], dtype=object),
        )
        q = out_v["q"]
        m = out_v["m"]
        l = out_v["l"]
        r_regs[self.anchor_left] = list(out_v["i"])
        r_regs[self.anchor_right] = list(out_v["j"])

        # 4) Undo controlled routing.
        for t_idx in reversed(range(len(terms))):
            a, b = terms[t_idx]
            perm = _perm_map_sources_to_slots(
                self.n_registers, a, b, self.anchor_left, self.anchor_right
            )
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
            sel = bb.join(
                np.array(sel_bits, dtype=object), dtype=BQUInt(self.term_bitsize, self.term_count)
            )

        out = {"q": q, "sel": sel, "m": m, "l": l}
        out.update({f"r{k}": np.array(r_regs[k], dtype=object) for k in range(self.n_registers)})
        return out

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        ctrl_bloq = TwoParticleControlledVSumBlockEncoding(
            num_pairs=self.num_pairs,
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            V_table=self.V_table,
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
class TwoParticleControlledVSumBlockEncoding(Bloq):
    """Single-control wrapper for `TwoParticleVSumBlockEncoding`."""

    num_pairs: int
    D: int
    L: int
    R_c: int
    R_loc: int
    V_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def anchor_left(self) -> int:
        return self.num_pairs - 1

    @property
    def anchor_right(self) -> int:
        return self.num_pairs

    @property
    def cross_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (a, b) for a in range(self.num_pairs) for b in range(self.num_pairs, self.n_registers)
        )

    @property
    def term_count(self) -> int:
        return len(self.cross_terms)

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

    def _term_control_bits(self, term_idx: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.term_bitsize).to_bits(int(term_idx)))

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, q, sel, m, l, **regs):
        r_regs = [list(regs[f"r{k}"]) for k in range(self.n_registers)]
        terms = self.cross_terms

        sel_bits = list(bb.split(sel))
        if self.term_count > 1:
            prep_sel = PrepareUniformSuperposition(n=self.term_count)
            sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
            sel_u = bb.add(prep_sel, target=sel_u)
            sel_bits = list(bb.split(sel_u))
        else:
            prep_sel = None

        for t_idx, (a, b) in enumerate(terms):
            perm = _perm_map_sources_to_slots(
                self.n_registers, a, b, self.anchor_left, self.anchor_right
            )
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

        v_bloq = TwoParticleSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.V_table,
            entry_bitsize=self.entry_bitsize,
        )
        out_v = bb.add_d(
            v_bloq.controlled(),
            ctrl=ctrl,
            q=q,
            m=m,
            l=l,
            i=np.array(r_regs[self.anchor_left], dtype=object),
            j=np.array(r_regs[self.anchor_right], dtype=object),
        )
        ctrl = out_v["ctrl"]
        q = out_v["q"]
        m = out_v["m"]
        l = out_v["l"]
        r_regs[self.anchor_left] = list(out_v["i"])
        r_regs[self.anchor_right] = list(out_v["j"])

        for t_idx in reversed(range(len(terms))):
            a, b = terms[t_idx]
            perm = _perm_map_sources_to_slots(
                self.n_registers, a, b, self.anchor_left, self.anchor_right
            )
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
            sel = bb.join(
                np.array(sel_bits, dtype=object), dtype=BQUInt(self.term_bitsize, self.term_count)
            )

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


def build_two_particle_v_sum_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    V_table: np.ndarray,
    entry_bitsize: int = 10,
) -> TwoParticleVSumBlockEncoding:
    return TwoParticleVSumBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        V_table=np.asarray(V_table),
        entry_bitsize=int(entry_bitsize),
    )


def build_two_particle_controlled_v_sum_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    V_table: np.ndarray,
    entry_bitsize: int = 10,
) -> TwoParticleControlledVSumBlockEncoding:
    return TwoParticleControlledVSumBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        V_table=np.asarray(V_table),
        entry_bitsize=int(entry_bitsize),
    )
