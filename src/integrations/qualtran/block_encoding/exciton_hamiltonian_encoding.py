from __future__ import annotations

from math import comb

import numpy as np
from attrs import field, frozen

try:
    from qualtran import BQUInt, Bloq, BloqBuilder, CtrlSpec, QBit, QUInt, Register, Signature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for exciton Hamiltonian block-encoding. Install with extra '[qualtran]'."
    ) from exc

from .f_register_sum_block_encoding import OneParticleFSumBlockEncoding
from .two_particle_v_sum_block_encoding import TwoParticleVSumBlockEncoding
from .two_particle_w_sum_block_encoding import TwoParticleWSumBlockEncoding


@frozen
class ExcitonHamiltonianBlockEncoding(Bloq):
    r"""Composite exciton block-encoding from F, W, V components.

    This bloq implements a 3-term selector superposition over:
    1) `OneParticleFSumBlockEncoding` on all 2m registers,
    2) `TwoParticleWSumBlockEncoding`,
    3) `TwoParticleVSumBlockEncoding`.

    With uniform selector prepare/unprepare, the postselected block is proportional to
    `(F + W + V) / 3` (up to each component block-encoding normalization).
    """

    num_pairs: int
    D: int
    L: int
    R_c: int
    R_loc: int
    F_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    W_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    V_table: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    def __attrs_post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be >= 1.")
        f_expected = (self.L,) * self.D + (2 * self.R_loc + 1,) * self.D
        if self.F_table.shape != f_expected:
            raise ValueError(f"F_table has shape {self.F_table.shape}, expected {f_expected}.")
        tw_expected = (
            (self.L,) * self.D
            + (self.L,) * self.D
            + (2 * self.R_c + 1,) * self.D
            + (2 * self.R_loc + 1,) * self.D
        )
        if self.W_table.shape != tw_expected:
            raise ValueError(f"W_table has shape {self.W_table.shape}, expected {tw_expected}.")
        if self.V_table.shape != tw_expected:
            raise ValueError(f"V_table has shape {self.V_table.shape}, expected {tw_expected}.")

    @property
    def n_registers(self) -> int:
        return 2 * self.num_pairs

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def h_term_count(self) -> int:
        return 3

    @property
    def h_sel_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(self.h_term_count))))

    @property
    def f_sel_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.num_pairs))))

    @property
    def w_sel_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(comb(2 * self.num_pairs, 2)))))

    @property
    def v_sel_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(self.num_pairs * self.num_pairs))))

    @property
    def f_m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def two_m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def two_l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        regs = [
            Register("q", QBit()),
            Register("h_sel", BQUInt(self.h_sel_bitsize, self.h_term_count)),
            Register("f_sel", BQUInt(self.f_sel_bitsize, 2 * self.num_pairs)),
            Register("f_m", BQUInt(self.f_m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
            Register("w_sel", BQUInt(self.w_sel_bitsize, comb(2 * self.num_pairs, 2))),
            Register("w_m", BQUInt(self.two_m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
            Register("w_l", BQUInt(self.two_l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
            Register("v_sel", BQUInt(self.v_sel_bitsize, self.num_pairs * self.num_pairs)),
            Register("v_m", BQUInt(self.two_m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
            Register("v_l", BQUInt(self.two_l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
        ]
        regs.extend(
            Register(f"r{k}", BQUInt(self.i_bitsize, self.L), shape=(self.D,))
            for k in range(self.n_registers)
        )
        return Signature(regs)

    @property
    def f_bloq(self) -> OneParticleFSumBlockEncoding:
        return OneParticleFSumBlockEncoding(
            num_pairs=self.num_pairs,
            D=self.D,
            L=self.L,
            R_loc=self.R_loc,
            F_table=self.F_table,
            entry_bitsize=self.entry_bitsize,
        )

    @property
    def w_bloq(self) -> TwoParticleWSumBlockEncoding:
        return TwoParticleWSumBlockEncoding(
            num_pairs=self.num_pairs,
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            W_table=self.W_table,
            entry_bitsize=self.entry_bitsize,
        )

    @property
    def v_bloq(self) -> TwoParticleVSumBlockEncoding:
        return TwoParticleVSumBlockEncoding(
            num_pairs=self.num_pairs,
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            V_table=self.V_table,
            entry_bitsize=self.entry_bitsize,
        )

    def _h_ctrl_bits(self, term_idx: int) -> tuple[int, ...]:
        return tuple(int(b) for b in QUInt(self.h_sel_bitsize).to_bits(int(term_idx)))

    def build_composite_bloq(self, bb: BloqBuilder, q, h_sel, f_sel, f_m, w_sel, w_m, w_l, v_sel, v_m, v_l, **regs):
        h_bits = list(bb.split(h_sel))
        prep = PrepareUniformSuperposition(n=self.h_term_count)
        h_u = bb.join(np.array(h_bits, dtype=object), dtype=QUInt(self.h_sel_bitsize))
        h_u = bb.add(prep, target=h_u)
        h_bits = list(bb.split(h_u))

        r_regs = {f"r{k}": regs[f"r{k}"] for k in range(self.n_registers)}

        def _mark_term(term_idx: int, flag):
            mcx = XGate().controlled(CtrlSpec(cvs=self._h_ctrl_bits(term_idx)))
            ctrls = np.array(h_bits, dtype=object)
            ctrls, flag = bb.add(mcx, ctrl=ctrls, q=flag)
            return list(np.array(ctrls, dtype=object).reshape(-1)), flag

        # Term 0: F (mark term ancilla, single-controlled apply, unmark).
        flag = bb.allocate(1)
        h_bits, flag = _mark_term(0, flag)
        out = bb.add_d(
            self.f_bloq.controlled(),
            ctrl=flag,
            q=q,
            sel=f_sel,
            m=f_m,
            **r_regs,
        )
        flag = out["ctrl"]
        q = out["q"]
        f_sel = out["sel"]
        f_m = out["m"]
        for k in range(self.n_registers):
            r_regs[f"r{k}"] = out[f"r{k}"]
        h_bits, flag = _mark_term(0, flag)
        bb.free(flag)

        # Term 1: W
        flag = bb.allocate(1)
        h_bits, flag = _mark_term(1, flag)
        out = bb.add_d(
            self.w_bloq.controlled(),
            ctrl=flag,
            q=q,
            sel=w_sel,
            m=w_m,
            l=w_l,
            **r_regs,
        )
        flag = out["ctrl"]
        q = out["q"]
        w_sel = out["sel"]
        w_m = out["m"]
        w_l = out["l"]
        for k in range(self.n_registers):
            r_regs[f"r{k}"] = out[f"r{k}"]
        h_bits, flag = _mark_term(1, flag)
        bb.free(flag)

        # Term 2: V
        flag = bb.allocate(1)
        h_bits, flag = _mark_term(2, flag)
        out = bb.add_d(
            self.v_bloq.controlled(),
            ctrl=flag,
            q=q,
            sel=v_sel,
            m=v_m,
            l=v_l,
            **r_regs,
        )
        flag = out["ctrl"]
        q = out["q"]
        v_sel = out["sel"]
        v_m = out["m"]
        v_l = out["l"]
        for k in range(self.n_registers):
            r_regs[f"r{k}"] = out[f"r{k}"]
        h_bits, flag = _mark_term(2, flag)
        bb.free(flag)

        # Unsigned unprepare on h selector.
        h_u = bb.join(np.array(h_bits, dtype=object), dtype=QUInt(self.h_sel_bitsize))
        h_u = bb.add(prep.adjoint(), target=h_u)
        h_sel = bb.join(bb.split(h_u), dtype=BQUInt(self.h_sel_bitsize, self.h_term_count))

        out_final = {
            "q": q,
            "h_sel": h_sel,
            "f_sel": f_sel,
            "f_m": f_m,
            "w_sel": w_sel,
            "w_m": w_m,
            "w_l": w_l,
            "v_sel": v_sel,
            "v_m": v_m,
            "v_l": v_l,
        }
        out_final.update(r_regs)
        return out_final


def build_exciton_hamiltonian_block_encoding(
    *,
    num_pairs: int,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    F: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    entry_bitsize: int = 10,
) -> ExcitonHamiltonianBlockEncoding:
    return ExcitonHamiltonianBlockEncoding(
        num_pairs=int(num_pairs),
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        F_table=np.asarray(F),
        W_table=np.asarray(W),
        V_table=np.asarray(V),
        entry_bitsize=int(entry_bitsize),
    )


# Backward compatibility alias
build_exciton_hamiltonian_encoding = build_exciton_hamiltonian_block_encoding
