from __future__ import annotations

import numpy as np
from attrs import field, frozen

try:
    from qualtran import BQUInt, Bloq, BloqBuilder, CtrlSpec, QBit, QUInt, Register, Signature
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant
    from qualtran.bloqs.basic_gates import Swap, XGate, ZGate
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for F-register-sum block-encoding. Install with extra '[qualtran]'."
    ) from exc

from .two_particle_row_oracles import OneParticleSparseBlockEncoding


@frozen
class OneParticleFSumBlockEncoding(Bloq):
    r"""Linear-combination style block-encoding over ``2m`` system registers.

    Construction:
    1) Prepare signed ancilla state
       ``|chi> = (1/sqrt(2m)) * (sum_{t=0}^{m-1}|t> - sum_{t=m}^{2m-1}|t>)``.
    2) Controlled-swap register ``r0`` with ``rt`` (for each t) based on ancilla value.
    3) Apply one-particle sparse block-encoding ``F`` on ``r0``.
    4) Undo controlled swaps.
    5) Apply unsigned unprepare ``<+|`` on ancilla (uniform, no sign).

    Effective post-selected block:
      ``(1/(2m)) * [sum_{t=0}^{m-1} F_t - sum_{t=m}^{2m-1} F_t]``.
    """

    num_pairs: int
    D: int
    L: int
    R_loc: int
    F_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    def __attrs_post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be >= 1.")
        expected = (self.L,) * self.D + (2 * self.R_loc + 1,) * self.D
        if self.F_table.shape != expected:
            raise ValueError(f"F_table has shape {self.F_table.shape}, expected {expected}.")

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def term_count(self) -> int:
        return 2 * self.num_pairs

    @property
    def term_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.term_count)))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        regs = [
            Register("q", QBit()),
            Register("sel", BQUInt(self.term_bitsize, self.term_count)),
            Register("m", BQUInt(self.m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
        ]
        regs.extend(
            Register(f"r{k}", BQUInt(self.i_bitsize, self.L), shape=(self.D,))
            for k in range(self.n_registers)
        )
        return Signature(regs)

    def _swap_control_bits(self, k: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.term_bitsize).to_bits(int(k)))

    def build_composite_bloq(self, bb: BloqBuilder, q, sel, m, **regs):
        r_regs = [list(regs[f"r{k}"]) for k in range(self.n_registers)]

        # Prepare unsigned uniform over selection register.
        prep_sel = PrepareUniformSuperposition(n=self.term_count)
        sel_bits = bb.split(sel)
        sel_u = bb.join(sel_bits, dtype=QUInt(self.term_bitsize))
        sel_u = bb.add(prep_sel, target=sel_u)

        # Impose sign on terms >= num_pairs: phase -1 on those basis states.
        sign = bb.allocate(1)
        sel_u, sign = bb.add(
            LessThanConstant(self.term_bitsize, self.num_pairs), x=sel_u, target=sign
        )
        sign = bb.add(XGate(), q=sign)  # sign=1 iff sel >= num_pairs
        sign = bb.add(ZGate(), q=sign)
        sign = bb.add(XGate(), q=sign)
        sel_u, sign = bb.add(
            LessThanConstant(self.term_bitsize, self.num_pairs).adjoint(), x=sel_u, target=sign
        )
        bb.free(sign)

        # Controlled permutations: for each k, swap r0 <-> rk when sel == k.
        sel_bits = list(bb.split(sel_u))
        for k in range(1, self.n_registers):
            ctrl_spec = CtrlSpec(cvs=self._swap_control_bits(k))
            sw = Swap(self.i_bitsize).controlled(ctrl_spec)
            for d in range(self.D):
                ctrls = np.array(sel_bits, dtype=object)
                ctrls, r_regs[0][d], r_regs[k][d] = bb.add(
                    sw,
                    ctrl=ctrls,
                    x=r_regs[0][d],
                    y=r_regs[k][d],
                )
                sel_bits = list(ctrls)

        # Apply one-particle sparse F block-encoding on r0.
        f_bloq = OneParticleSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_loc=self.R_loc,
            M=self.F_table,
            entry_bitsize=self.entry_bitsize,
        )
        out_f = bb.add_d(f_bloq, q=q, m=m, i=np.array(r_regs[0], dtype=object))
        q = out_f["q"]
        m = out_f["m"]
        r_regs[0] = list(out_f["i"])

        # Reverse controlled permutations.
        for k in reversed(range(1, self.n_registers)):
            ctrl_spec = CtrlSpec(cvs=self._swap_control_bits(k))
            sw = Swap(self.i_bitsize).controlled(ctrl_spec)
            for d in range(self.D):
                ctrls = np.array(sel_bits, dtype=object)
                ctrls, r_regs[0][d], r_regs[k][d] = bb.add(
                    sw,
                    ctrl=ctrls,
                    x=r_regs[0][d],
                    y=r_regs[k][d],
                )
                sel_bits = list(ctrls)

        # Unsigned unprepare on selection register.
        sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
        sel_u = bb.add(prep_sel.adjoint(), target=sel_u)
        sel = bb.join(bb.split(sel_u), dtype=BQUInt(self.term_bitsize, self.term_count))

        out = {"q": q, "sel": sel, "m": m}
        out.update({f"r{k}": np.array(r_regs[k], dtype=object) for k in range(self.n_registers)})
        return out

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        ctrl_bloq = OneParticleControlledFSumBlockEncoding(
            num_pairs=self.num_pairs,
            D=self.D,
            L=self.L,
            R_loc=self.R_loc,
            F_table=self.F_table,
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
class OneParticleControlledFSumBlockEncoding(Bloq):
    """Single-control wrapper for `OneParticleFSumBlockEncoding`."""

    num_pairs: int
    D: int
    L: int
    R_loc: int
    F_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def term_count(self) -> int:
        return 2 * self.num_pairs

    @property
    def term_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.term_count)))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        regs = [
            Register("ctrl", QBit()),
            Register("q", QBit()),
            Register("sel", BQUInt(self.term_bitsize, self.term_count)),
            Register("m", BQUInt(self.m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
        ]
        regs.extend(
            Register(f"r{k}", BQUInt(self.i_bitsize, self.L), shape=(self.D,))
            for k in range(self.n_registers)
        )
        return Signature(regs)

    def _swap_control_bits(self, k: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.term_bitsize).to_bits(int(k)))

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, q, sel, m, **regs):
        r_regs = [list(regs[f"r{k}"]) for k in range(self.n_registers)]

        sel_bits = list(bb.split(sel))
        prep_sel = PrepareUniformSuperposition(n=self.term_count)
        sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
        sel_u = bb.add(prep_sel, target=sel_u)
        sel_bits = list(bb.split(sel_u))

        # Controlled sign phase for terms >= m.
        sign = bb.allocate(1)
        sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
        sel_u, sign = bb.add(
            LessThanConstant(self.term_bitsize, self.num_pairs), x=sel_u, target=sign
        )
        sel_bits = list(bb.split(sel_u))
        sign = bb.add(XGate(), q=sign)
        ctrls = np.array([ctrl], dtype=object)
        ctrls, sign = bb.add(ZGate().controlled(CtrlSpec(cvs=(1,))), ctrl=ctrls, q=sign)
        ctrl = ctrls[0]
        sign = bb.add(XGate(), q=sign)
        sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
        sel_u, sign = bb.add(
            LessThanConstant(self.term_bitsize, self.num_pairs).adjoint(), x=sel_u, target=sign
        )
        sel_bits = list(bb.split(sel_u))
        bb.free(sign)

        # Controlled routing.
        for k in range(1, self.n_registers):
            ctrl_spec = CtrlSpec(cvs=(1,) + self._swap_control_bits(k))
            sw1 = Swap(1).controlled(ctrl_spec)
            for d in range(self.D):
                xb = list(bb.split(r_regs[0][d]))
                yb = list(bb.split(r_regs[k][d]))
                for bi in range(self.i_bitsize):
                    ctrls = np.array([ctrl] + sel_bits, dtype=object)
                    ctrls, xb[bi], yb[bi] = bb.add(
                        sw1, ctrl=ctrls, x=xb[bi], y=yb[bi]
                    )
                    ctrl = ctrls[0]
                    sel_bits = list(ctrls[1:])
                r_regs[0][d] = bb.join(np.array(xb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))
                r_regs[k][d] = bb.join(np.array(yb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))

        f_bloq = OneParticleSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_loc=self.R_loc,
            M=self.F_table,
            entry_bitsize=self.entry_bitsize,
        )
        out_f = bb.add_d(
            f_bloq.controlled(),
            ctrl=ctrl,
            q=q,
            m=m,
            i=np.array(r_regs[0], dtype=object),
        )
        ctrl = out_f["ctrl"]
        q = out_f["q"]
        m = out_f["m"]
        r_regs[0] = list(out_f["i"])

        for k in reversed(range(1, self.n_registers)):
            ctrl_spec = CtrlSpec(cvs=(1,) + self._swap_control_bits(k))
            sw1 = Swap(1).controlled(ctrl_spec)
            for d in range(self.D):
                xb = list(bb.split(r_regs[0][d]))
                yb = list(bb.split(r_regs[k][d]))
                for bi in range(self.i_bitsize):
                    ctrls = np.array([ctrl] + sel_bits, dtype=object)
                    ctrls, xb[bi], yb[bi] = bb.add(
                        sw1, ctrl=ctrls, x=xb[bi], y=yb[bi]
                    )
                    ctrl = ctrls[0]
                    sel_bits = list(ctrls[1:])
                r_regs[0][d] = bb.join(np.array(xb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))
                r_regs[k][d] = bb.join(np.array(yb, dtype=object), dtype=BQUInt(self.i_bitsize, self.L))

        sel_u = bb.join(np.array(sel_bits, dtype=object), dtype=QUInt(self.term_bitsize))
        sel_u = bb.add(prep_sel.adjoint(), target=sel_u)
        sel = bb.join(bb.split(sel_u), dtype=BQUInt(self.term_bitsize, self.term_count))

        out = {"ctrl": ctrl, "q": q, "sel": sel, "m": m}
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


def build_one_particle_f_sum_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_loc: int,
    F_table: np.ndarray,
    entry_bitsize: int = 10,
) -> OneParticleFSumBlockEncoding:
    return OneParticleFSumBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_loc=int(R_loc),
        F_table=np.asarray(F_table),
        entry_bitsize=int(entry_bitsize),
    )


def build_one_particle_controlled_f_sum_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_loc: int,
    F_table: np.ndarray,
    entry_bitsize: int = 10,
) -> OneParticleControlledFSumBlockEncoding:
    return OneParticleControlledFSumBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_loc=int(R_loc),
        F_table=np.asarray(F_table),
        entry_bitsize=int(entry_bitsize),
    )


# Backward compatibility
SignedFRegisterSumBlockEncoding = OneParticleFSumBlockEncoding
build_signed_f_register_sum_block_encoding = build_one_particle_f_sum_block_encoding
