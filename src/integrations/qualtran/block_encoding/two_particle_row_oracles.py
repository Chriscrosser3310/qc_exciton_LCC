from __future__ import annotations

import numpy as np
from attrs import field, frozen

try:
    from qualtran import (
        BQUInt,
        Bloq,
        BloqBuilder,
        CtrlSpec,
        DecomposeTypeError,
        QBit,
        Register,
        Signature,
        QUInt,
    )
    from qualtran.bloqs.arithmetic.addition import Add, AddK
    from qualtran.bloqs.basic_gates import Ry
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.symbolics import is_symbolic
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for two-particle row oracles. Install with extra '[qualtran]'."
    ) from exc


@frozen
class TwoParticleRowIndexOracle(Bloq):
    r"""Index oracle mapping |m>|l>|i>|j> -> |m>|l>|i'>|j'>.

    For each coordinate k = 1..D:
      i'_k = i_k - R_c + m_k          (mod L)
      j'_k = j_k - R_c + m_k - R_loc + l_k (mod L)

    Registers are vectorized over D coordinates.

    Notes:
    - Classical action is exact modulo L.
    - Decomposition uses Add/AddK on QUInt registers, which is exact modulo 2^n where
      n = ceil(log2(L)). Therefore decomposition requires L to be a power of two.
    """

    D: int
    L: int
    R_c: int
    R_loc: int

    def __attrs_post_init__(self) -> None:
        if self.D <= 0:
            raise ValueError("D must be >= 1.")
        if self.L <= 1:
            raise ValueError("L must be >= 2.")
        if self.R_c < 0 or self.R_loc < 0:
            raise ValueError("R_c and R_loc must be >= 0.")

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def j_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
                Register("l", BQUInt(self.l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
                Register("j", BQUInt(self.j_bitsize, self.L), shape=(self.D,)),
            ]
        )

    def _is_pow2_l(self) -> bool:
        return self.L > 0 and (self.L & (self.L - 1) == 0)

    def call_classically(self, m, l, i, j):
        m_v = np.asarray(m, dtype=int).reshape(self.D)
        l_v = np.asarray(l, dtype=int).reshape(self.D)
        i_v = np.asarray(i, dtype=int).reshape(self.D)
        j_v = np.asarray(j, dtype=int).reshape(self.D)

        i_p = (i_v - self.R_c + m_v) % self.L
        j_p = (j_v - self.R_c + m_v - self.R_loc + l_v) % self.L
        return (m_v, l_v, i_p, j_p)

    def build_composite_bloq(self, bb: BloqBuilder, m, l, i, j):
        if not self._is_pow2_l():
            raise DecomposeTypeError(
                "TwoParticleRowIndexOracle decomposition currently requires L to be a power of two "
                "to realize exact modular arithmetic via Add/AddK."
            )
        m_regs = list(m)
        l_regs = list(l)
        i_regs = list(i)
        j_regs = list(j)

        for d in range(self.D):
            # Keep current per-register bit soquets; we will consume/recreate them via join/split.
            m_bits = list(bb.split(m_regs[d]))
            l_bits = list(bb.split(l_regs[d]))

            # i_k, j_k: BQUInt -> QUInt views
            i_bits = bb.split(i_regs[d])
            i_u = bb.join(i_bits, dtype=QUInt(self.i_bitsize))
            j_bits = bb.split(j_regs[d])
            j_u = bb.join(j_bits, dtype=QUInt(self.j_bitsize))

            # i'_k = i_k + (m_k mod L) - R_c (mod L).
            # For L=2^n, m_k mod L corresponds to low n bits of m_k.
            if self.m_bitsize <= self.i_bitsize:
                m_u_i = bb.join(np.array(m_bits, dtype=object), dtype=QUInt(self.m_bitsize))
                m_u_i, i_u = bb.add(
                    Add(QUInt(self.m_bitsize), QUInt(self.i_bitsize)), a=m_u_i, b=i_u
                )
                m_bits = list(bb.split(m_u_i))
            else:
                m_low_i = bb.join(
                    np.array(m_bits[-self.i_bitsize :], dtype=object), dtype=QUInt(self.i_bitsize)
                )
                m_low_i, i_u = bb.add(
                    Add(QUInt(self.i_bitsize), QUInt(self.i_bitsize)), a=m_low_i, b=i_u
                )
                m_low_i_bits = list(bb.split(m_low_i))
                m_bits[-self.i_bitsize :] = m_low_i_bits
            i_u = bb.add(AddK(QUInt(self.i_bitsize), k=-self.R_c), x=i_u)

            # j'_k = j_k + (m_k mod L) + (l_k mod L) - R_c - R_loc
            if self.m_bitsize <= self.j_bitsize:
                m_u_j = bb.join(np.array(m_bits, dtype=object), dtype=QUInt(self.m_bitsize))
                m_u_j, j_u = bb.add(
                    Add(QUInt(self.m_bitsize), QUInt(self.j_bitsize)), a=m_u_j, b=j_u
                )
                m_bits = list(bb.split(m_u_j))
            else:
                m_low_j = bb.join(
                    np.array(m_bits[-self.j_bitsize :], dtype=object), dtype=QUInt(self.j_bitsize)
                )
                m_low_j, j_u = bb.add(
                    Add(QUInt(self.j_bitsize), QUInt(self.j_bitsize)), a=m_low_j, b=j_u
                )
                m_low_j_bits = list(bb.split(m_low_j))
                m_bits[-self.j_bitsize :] = m_low_j_bits

            if self.l_bitsize <= self.j_bitsize:
                l_u_j = bb.join(np.array(l_bits, dtype=object), dtype=QUInt(self.l_bitsize))
                l_u_j, j_u = bb.add(
                    Add(QUInt(self.l_bitsize), QUInt(self.j_bitsize)), a=l_u_j, b=j_u
                )
                l_bits = list(bb.split(l_u_j))
            else:
                l_low_j = bb.join(
                    np.array(l_bits[-self.j_bitsize :], dtype=object), dtype=QUInt(self.j_bitsize)
                )
                l_low_j, j_u = bb.add(
                    Add(QUInt(self.j_bitsize), QUInt(self.j_bitsize)), a=l_low_j, b=j_u
                )
                l_low_j_bits = list(bb.split(l_low_j))
                l_bits[-self.j_bitsize :] = l_low_j_bits
            j_u = bb.add(AddK(QUInt(self.j_bitsize), k=-(self.R_c + self.R_loc)), x=j_u)

            # Convert views back to bounded dtypes
            m_u_out = bb.join(np.array(m_bits, dtype=object), dtype=QUInt(self.m_bitsize))
            l_u_out = bb.join(np.array(l_bits, dtype=object), dtype=QUInt(self.l_bitsize))
            m_regs[d] = bb.join(bb.split(m_u_out), dtype=BQUInt(self.m_bitsize, 2 * self.R_c + 1))
            l_regs[d] = bb.join(bb.split(l_u_out), dtype=BQUInt(self.l_bitsize, 2 * self.R_loc + 1))
            i_regs[d] = bb.join(bb.split(i_u), dtype=BQUInt(self.i_bitsize, self.L))
            j_regs[d] = bb.join(bb.split(j_u), dtype=BQUInt(self.j_bitsize, self.L))

        return {
            "m": np.array(m_regs, dtype=object),
            "l": np.array(l_regs, dtype=object),
            "i": np.array(i_regs, dtype=object),
            "j": np.array(j_regs, dtype=object),
        }


@frozen
class TwoParticleRowEntryOracle(Bloq):
    r"""Entry oracle for tensor M[i,j,m,l] with multi-index registers.

    Decomposition:
    1) QROM-load discretized angles theta(i,j,m,l)
    2) Apply controlled-Ry ladder on flag qubit q
    3) Uncompute the QROM load

    Tensor shape convention:
      M.shape == (L,)*D + (L,)*D + (2*R_c+1,)*D + (2*R_loc+1,)*D
    """

    D: int
    L: int
    R_c: int
    R_loc: int
    M: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    def __attrs_post_init__(self) -> None:
        expected = (
            (self.L,) * self.D
            + (self.L,) * self.D
            + (2 * self.R_c + 1,) * self.D
            + (2 * self.R_loc + 1,) * self.D
        )
        if self.M.shape != expected:
            raise ValueError(f"M has shape {self.M.shape}, expected {expected}.")
        if not is_symbolic(self.entry_bitsize) and int(self.entry_bitsize) < 1:
            raise ValueError("entry_bitsize must be >= 1")

        mmin = float(np.min(self.M))
        mmax = float(np.max(self.M))
        if mmin < 0.0 or mmax > 1.0:
            raise ValueError(
                "For this decomposition, all M entries must be in [0, 1]. "
                f"Observed min={mmin}, max={mmax}."
            )

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def j_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("q", QBit()),
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
                Register("l", BQUInt(self.l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
                Register("j", BQUInt(self.j_bitsize, self.L), shape=(self.D,)),
            ]
        )

    @property
    def _qrom_theta(self):
        if is_symbolic(self.entry_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        b = int(self.entry_bitsize)
        theta_table_real = np.arccos(np.asarray(self.M, dtype=np.float64)) / np.pi * (2**b)
        # Quantize using nearest integer (not floor) to reduce systematic bias.
        theta_table = np.rint(theta_table_real).astype(np.int64, copy=False)
        theta_table = np.clip(theta_table, 0, 2**b - 1)
        # Qualtran QROM omits selection registers with iteration_length == 1.
        # Squeeze these singleton selection axes from data to keep indexing aligned.
        # Selection axes are exactly the first 4*D axes of M:
        #   i[0:D], j[D:2D], m[2D:3D], l[3D:4D]
        sel_shape = (
            [self.L] * self.D
            + [self.L] * self.D
            + [2 * self.R_c + 1] * self.D
            + [2 * self.R_loc + 1] * self.D
        )
        drop_axes = tuple(ax for ax, ln in enumerate(sel_shape) if int(ln) <= 1)
        if drop_axes:
            theta_table = np.squeeze(theta_table, axis=drop_axes)
        return QROM.build_from_data(theta_table, target_bitsizes=(int(self.entry_bitsize),))

    def entry_value(self, i, j, m, l) -> float:
        i_t = tuple(int(x) for x in i)
        j_t = tuple(int(x) for x in j)
        m_t = tuple(int(x) for x in m)
        l_t = tuple(int(x) for x in l)
        return float(self.M[i_t + j_t + m_t + l_t])

    def build_composite_bloq(self, bb: BloqBuilder, q, m, l, i, j):
        if is_symbolic(self.entry_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        i_regs = list(i)
        j_regs = list(j)
        m_regs = list(m)
        l_regs = list(l)
        target = bb.allocate(int(self.entry_bitsize))

        dim_lengths = (
            [self.L] * self.D
            + [self.L] * self.D
            + [2 * self.R_c + 1] * self.D
            + [2 * self.R_loc + 1] * self.D
        )
        dim_soqs = i_regs + j_regs + m_regs + l_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        qrom_sel_regs = list(self._qrom_theta.selection_registers)
        if len(kept_soqs) != len(qrom_sel_regs):
            raise DecomposeTypeError(
                "QROM selection-register mismatch for TwoParticleRowEntryOracle."
            )
        sel_in = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}

        out = bb.add_d(self._qrom_theta, **sel_in, target0_=target)

        qrom_sel_names = [reg.name for reg in qrom_sel_regs]
        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            elif idx < 2 * self.D:
                j_regs[idx - self.D] = sq
            elif idx < 3 * self.D:
                m_regs[idx - 2 * self.D] = sq
            else:
                l_regs[idx - 3 * self.D] = sq
        target = out["target0_"]

        target_bits = bb.split(target)
        for k, tbit in enumerate(target_bits):
            tbit, q = bb.add(Ry(2 * np.pi * (2 ** -(k + 1))).controlled(), ctrl=tbit, q=q)
            target_bits[k] = tbit
        target = bb.join(target_bits)

        dim_soqs = i_regs + j_regs + m_regs + l_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        sel_in_adj = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}
        out = bb.add_d(self._qrom_theta.adjoint(), **sel_in_adj, target0_=target)

        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            elif idx < 2 * self.D:
                j_regs[idx - self.D] = sq
            elif idx < 3 * self.D:
                m_regs[idx - 2 * self.D] = sq
            else:
                l_regs[idx - 3 * self.D] = sq
        target = out["target0_"]

        bb.free(target)
        return {
            "q": q,
            "m": np.array(m_regs, dtype=object),
            "l": np.array(l_regs, dtype=object),
            "i": np.array(i_regs, dtype=object),
            "j": np.array(j_regs, dtype=object),
        }


def build_two_particle_row_entry_oracle(
    M: np.ndarray,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    entry_bitsize: int = 10,
) -> TwoParticleRowEntryOracle:
    return TwoParticleRowEntryOracle(
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        M=np.asarray(M),
        entry_bitsize=int(entry_bitsize),
    )


def build_two_particle_row_index_oracle(
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
) -> TwoParticleRowIndexOracle:
    return TwoParticleRowIndexOracle(D=int(D), L=int(L), R_c=int(R_c), R_loc=int(R_loc))


@frozen
class TwoParticleSparseBlockEncoding(Bloq):
    r"""Composite query bloq over registers (q, m, l, i, j).

    Sequence:
    1) Prepare uniform superposition on each coordinate register of m and l.
    2) Apply adjoint of TwoParticleRowIndexOracle on (m, l, i, j).
    3) Apply TwoParticleRowEntryOracle on (q, m, l, i, j).
    4) Unprepare (adjoint prepare) on each coordinate register of m and l.
    """

    D: int
    L: int
    R_c: int
    R_loc: int
    M: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def j_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("q", QBit()),
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
                Register("l", BQUInt(self.l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
                Register("j", BQUInt(self.j_bitsize, self.L), shape=(self.D,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, q, m, l, i, j):
        m_regs = list(m)
        l_regs = list(l)

        n_m = 2 * self.R_c + 1
        n_l = 2 * self.R_loc + 1

        prep_m = PrepareUniformSuperposition(n=n_m)
        prep_l = PrepareUniformSuperposition(n=n_l)

        # 1) Uniform superposition on m and l (coordinate-wise).
        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m, target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        if n_l > 1:
            for d in range(self.D):
                l_bits = bb.split(l_regs[d])
                l_u = bb.join(l_bits, dtype=QUInt(self.l_bitsize))
                l_u = bb.add(prep_l, target=l_u)
                l_regs[d] = bb.join(bb.split(l_u), dtype=BQUInt(self.l_bitsize, n_l))

        # 2) Adjoint of index oracle on (m, l, i, j).
        index_oracle = TwoParticleRowIndexOracle(
            D=self.D, L=self.L, R_c=self.R_c, R_loc=self.R_loc
        )
        out_idx = bb.add_d(
            index_oracle.adjoint(),
            m=np.array(m_regs, dtype=object),
            l=np.array(l_regs, dtype=object),
            i=i,
            j=j,
        )
        m_regs = list(out_idx["m"])
        l_regs = list(out_idx["l"])
        i = out_idx["i"]
        j = out_idx["j"]

        # 3) Entry oracle on all five registers.
        entry_oracle = TwoParticleRowEntryOracle(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.M,
            entry_bitsize=self.entry_bitsize,
        )
        out_ent = bb.add_d(
            entry_oracle,
            q=q,
            m=np.array(m_regs, dtype=object),
            l=np.array(l_regs, dtype=object),
            i=i,
            j=j,
        )
        q = out_ent["q"]
        m_regs = list(out_ent["m"])
        l_regs = list(out_ent["l"])
        i = out_ent["i"]
        j = out_ent["j"]

        # 4) Adjoint uniform superposition on m and l (coordinate-wise).
        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m.adjoint(), target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        if n_l > 1:
            for d in range(self.D):
                l_bits = bb.split(l_regs[d])
                l_u = bb.join(l_bits, dtype=QUInt(self.l_bitsize))
                l_u = bb.add(prep_l.adjoint(), target=l_u)
                l_regs[d] = bb.join(bb.split(l_u), dtype=BQUInt(self.l_bitsize, n_l))

        return {
            "q": q,
            "m": np.array(m_regs, dtype=object),
            "l": np.array(l_regs, dtype=object),
            "i": i,
            "j": j,
        }

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        ctrl_bloq = TwoParticleControlledSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.M,
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
class TwoParticleControlledSparseBlockEncoding(Bloq):
    r"""Single-control wrapper for `TwoParticleSparseBlockEncoding`.

    Action:
      |ctrl>|q,m,l,i,j> -> |ctrl> U^{ctrl} |q,m,l,i,j>
    where U is the two-particle sparse block-encoding query bloq.
    """

    D: int
    L: int
    R_c: int
    R_loc: int
    M: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_c + 1))))

    @property
    def l_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def j_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("ctrl", QBit()),
                Register("q", QBit()),
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_c + 1), shape=(self.D,)),
                Register("l", BQUInt(self.l_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
                Register("j", BQUInt(self.j_bitsize, self.L), shape=(self.D,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, q, m, l, i, j):
        m_regs = list(m)
        l_regs = list(l)
        i_regs = list(i)
        j_regs = list(j)

        n_m = 2 * self.R_c + 1
        n_l = 2 * self.R_loc + 1
        prep_m = PrepareUniformSuperposition(n=n_m)
        prep_l = PrepareUniformSuperposition(n=n_l)

        # Uncontrolled prepare / unprepare.
        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m, target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        if n_l > 1:
            for d in range(self.D):
                l_bits = bb.split(l_regs[d])
                l_u = bb.join(l_bits, dtype=QUInt(self.l_bitsize))
                l_u = bb.add(prep_l, target=l_u)
                l_regs[d] = bb.join(bb.split(l_u), dtype=BQUInt(self.l_bitsize, n_l))

        # Controlled index oracle.
        index_oracle = TwoParticleRowIndexOracle(
            D=self.D, L=self.L, R_c=self.R_c, R_loc=self.R_loc
        )
        out_idx = bb.add_d(
            index_oracle.controlled(),
            ctrl=ctrl,
            m=np.array(m_regs, dtype=object),
            l=np.array(l_regs, dtype=object),
            i=np.array(i_regs, dtype=object),
            j=np.array(j_regs, dtype=object),
        )
        ctrl = out_idx["ctrl"]
        m_regs = list(out_idx["m"])
        l_regs = list(out_idx["l"])
        i_regs = list(out_idx["i"])
        j_regs = list(out_idx["j"])

        # Entry oracle with uncontrolled QROM, controlled rotations only.
        entry_oracle = TwoParticleRowEntryOracle(
            D=self.D,
            L=self.L,
            R_c=self.R_c,
            R_loc=self.R_loc,
            M=self.M,
            entry_bitsize=self.entry_bitsize,
        )
        target = bb.allocate(int(self.entry_bitsize))

        dim_lengths = (
            [self.L] * self.D
            + [self.L] * self.D
            + [2 * self.R_c + 1] * self.D
            + [2 * self.R_loc + 1] * self.D
        )
        dim_soqs = i_regs + j_regs + m_regs + l_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        qrom_sel_regs = list(entry_oracle._qrom_theta.selection_registers)
        if len(kept_soqs) != len(qrom_sel_regs):
            raise DecomposeTypeError(
                "QROM selection-register mismatch for TwoParticleControlledSparseBlockEncoding."
            )
        sel_in = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}

        out = bb.add_d(entry_oracle._qrom_theta, **sel_in, target0_=target)
        qrom_sel_names = [reg.name for reg in qrom_sel_regs]
        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            elif idx < 2 * self.D:
                j_regs[idx - self.D] = sq
            elif idx < 3 * self.D:
                m_regs[idx - 2 * self.D] = sq
            else:
                l_regs[idx - 3 * self.D] = sq
        target = out["target0_"]

        target_bits = bb.split(target)
        two_ctrl_ry = {
            k: Ry(2 * np.pi * (2 ** -(k + 1))).controlled(CtrlSpec(cvs=(1, 1)))
            for k in range(int(self.entry_bitsize))
        }
        for k, tbit in enumerate(target_bits):
            ctrls = np.array([ctrl, tbit], dtype=object)
            ctrls, q = bb.add(two_ctrl_ry[k], ctrl=ctrls, q=q)
            ctrl, tbit = ctrls
            target_bits[k] = tbit
        target = bb.join(target_bits)

        dim_soqs = i_regs + j_regs + m_regs + l_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        sel_in_adj = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}
        out = bb.add_d(entry_oracle._qrom_theta.adjoint(), **sel_in_adj, target0_=target)

        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            elif idx < 2 * self.D:
                j_regs[idx - self.D] = sq
            elif idx < 3 * self.D:
                m_regs[idx - 2 * self.D] = sq
            else:
                l_regs[idx - 3 * self.D] = sq
        target = out["target0_"]
        bb.free(target)

        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m.adjoint(), target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        if n_l > 1:
            for d in range(self.D):
                l_bits = bb.split(l_regs[d])
                l_u = bb.join(l_bits, dtype=QUInt(self.l_bitsize))
                l_u = bb.add(prep_l.adjoint(), target=l_u)
                l_regs[d] = bb.join(bb.split(l_u), dtype=BQUInt(self.l_bitsize, n_l))

        return {
            "ctrl": ctrl,
            "q": q,
            "m": np.array(m_regs, dtype=object),
            "l": np.array(l_regs, dtype=object),
            "i": np.array(i_regs, dtype=object),
            "j": np.array(j_regs, dtype=object),
        }

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1,
            bloq_with_ctrl=self,
            ctrl_reg_name="ctrl",
        )


def build_two_particle_sparse_block_encoding(
    M: np.ndarray,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    entry_bitsize: int = 10,
) -> TwoParticleSparseBlockEncoding:
    return TwoParticleSparseBlockEncoding(
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        M=np.asarray(M),
        entry_bitsize=int(entry_bitsize),
    )


def build_two_particle_controlled_sparse_block_encoding(
    M: np.ndarray,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    entry_bitsize: int = 10,
) -> TwoParticleControlledSparseBlockEncoding:
    return TwoParticleControlledSparseBlockEncoding(
        D=int(D),
        L=int(L),
        R_c=int(R_c),
        R_loc=int(R_loc),
        M=np.asarray(M),
        entry_bitsize=int(entry_bitsize),
    )


@frozen
class OneParticleRowIndexOracle(Bloq):
    r"""Index oracle mapping |m>|i> -> |m>|i'> with i'_k = i_k - R_loc + m_k (mod L)."""

    D: int
    L: int
    R_loc: int

    def __attrs_post_init__(self) -> None:
        if self.D <= 0:
            raise ValueError("D must be >= 1.")
        if self.L <= 1:
            raise ValueError("L must be >= 2.")
        if self.R_loc < 0:
            raise ValueError("R_loc must be >= 0.")

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
            ]
        )

    def _is_pow2_l(self) -> bool:
        return self.L > 0 and (self.L & (self.L - 1) == 0)

    def call_classically(self, m, i):
        m_v = np.asarray(m, dtype=int).reshape(self.D)
        i_v = np.asarray(i, dtype=int).reshape(self.D)
        i_p = (i_v - self.R_loc + m_v) % self.L
        return (m_v, i_p)

    def build_composite_bloq(self, bb: BloqBuilder, m, i):
        if not self._is_pow2_l():
            raise DecomposeTypeError(
                "OneParticleRowIndexOracle decomposition currently requires L to be a power of two "
                "to realize exact modular arithmetic via Add/AddK."
            )

        m_regs = list(m)
        i_regs = list(i)
        for d in range(self.D):
            m_bits = list(bb.split(m_regs[d]))
            i_bits = bb.split(i_regs[d])
            i_u = bb.join(i_bits, dtype=QUInt(self.i_bitsize))

            if self.m_bitsize <= self.i_bitsize:
                m_u = bb.join(np.array(m_bits, dtype=object), dtype=QUInt(self.m_bitsize))
                m_u, i_u = bb.add(Add(QUInt(self.m_bitsize), QUInt(self.i_bitsize)), a=m_u, b=i_u)
                m_bits = list(bb.split(m_u))
            else:
                m_low = bb.join(
                    np.array(m_bits[-self.i_bitsize :], dtype=object), dtype=QUInt(self.i_bitsize)
                )
                m_low, i_u = bb.add(
                    Add(QUInt(self.i_bitsize), QUInt(self.i_bitsize)), a=m_low, b=i_u
                )
                m_low_bits = list(bb.split(m_low))
                m_bits[-self.i_bitsize :] = m_low_bits

            i_u = bb.add(AddK(QUInt(self.i_bitsize), k=-self.R_loc), x=i_u)
            m_u_out = bb.join(np.array(m_bits, dtype=object), dtype=QUInt(self.m_bitsize))
            m_regs[d] = bb.join(
                bb.split(m_u_out), dtype=BQUInt(self.m_bitsize, 2 * self.R_loc + 1)
            )
            i_regs[d] = bb.join(bb.split(i_u), dtype=BQUInt(self.i_bitsize, self.L))

        return {"m": np.array(m_regs, dtype=object), "i": np.array(i_regs, dtype=object)}


@frozen
class OneParticleRowEntryOracle(Bloq):
    r"""Entry oracle for tensor M[i,m] with multi-index registers.

    Tensor shape convention:
      M.shape == (L,)*D + (2*R_loc+1,)*D
    """

    D: int
    L: int
    R_loc: int
    M: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    def __attrs_post_init__(self) -> None:
        expected = (self.L,) * self.D + (2 * self.R_loc + 1,) * self.D
        if self.M.shape != expected:
            raise ValueError(f"M has shape {self.M.shape}, expected {expected}.")
        if not is_symbolic(self.entry_bitsize) and int(self.entry_bitsize) < 1:
            raise ValueError("entry_bitsize must be >= 1")
        mmin = float(np.min(self.M))
        mmax = float(np.max(self.M))
        if mmin < 0.0 or mmax > 1.0:
            raise ValueError(
                "For this decomposition, all M entries must be in [0, 1]. "
                f"Observed min={mmin}, max={mmax}."
            )

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("q", QBit()),
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
            ]
        )

    @property
    def _qrom_theta(self):
        if is_symbolic(self.entry_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        b = int(self.entry_bitsize)
        theta_table_real = np.arccos(np.asarray(self.M, dtype=np.float64)) / np.pi * (2**b)
        theta_table = np.rint(theta_table_real).astype(np.int64, copy=False)
        theta_table = np.clip(theta_table, 0, 2**b - 1)
        sel_shape = [self.L] * self.D + [2 * self.R_loc + 1] * self.D
        drop_axes = tuple(ax for ax, ln in enumerate(sel_shape) if int(ln) <= 1)
        if drop_axes:
            theta_table = np.squeeze(theta_table, axis=drop_axes)
        return QROM.build_from_data(theta_table, target_bitsizes=(int(self.entry_bitsize),))

    def build_composite_bloq(self, bb: BloqBuilder, q, m, i):
        if is_symbolic(self.entry_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        i_regs = list(i)
        m_regs = list(m)
        target = bb.allocate(int(self.entry_bitsize))

        dim_lengths = [self.L] * self.D + [2 * self.R_loc + 1] * self.D
        dim_soqs = i_regs + m_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        qrom_sel_regs = list(self._qrom_theta.selection_registers)
        if len(kept_soqs) != len(qrom_sel_regs):
            raise DecomposeTypeError(
                "QROM selection-register mismatch for OneParticleRowEntryOracle."
            )
        sel_in = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}

        out = bb.add_d(self._qrom_theta, **sel_in, target0_=target)
        qrom_sel_names = [reg.name for reg in qrom_sel_regs]
        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            else:
                m_regs[idx - self.D] = sq
        target = out["target0_"]

        target_bits = bb.split(target)
        for k, tbit in enumerate(target_bits):
            tbit, q = bb.add(Ry(2 * np.pi * (2 ** -(k + 1))).controlled(), ctrl=tbit, q=q)
            target_bits[k] = tbit
        target = bb.join(target_bits)

        dim_soqs = i_regs + m_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        sel_in_adj = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}
        out = bb.add_d(self._qrom_theta.adjoint(), **sel_in_adj, target0_=target)

        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            else:
                m_regs[idx - self.D] = sq
        target = out["target0_"]
        bb.free(target)
        return {"q": q, "m": np.array(m_regs, dtype=object), "i": np.array(i_regs, dtype=object)}


@frozen
class OneParticleSparseBlockEncoding(Bloq):
    r"""Composite one-particle sparse query over (q, m, i)."""

    D: int
    L: int
    R_loc: int
    M: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("q", QBit()),
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, q, m, i):
        m_regs = list(m)
        n_m = 2 * self.R_loc + 1
        prep_m = PrepareUniformSuperposition(n=n_m)

        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m, target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        index_oracle = OneParticleRowIndexOracle(D=self.D, L=self.L, R_loc=self.R_loc)
        out_idx = bb.add_d(index_oracle.adjoint(), m=np.array(m_regs, dtype=object), i=i)
        m_regs = list(out_idx["m"])
        i = out_idx["i"]

        entry_oracle = OneParticleRowEntryOracle(
            D=self.D, L=self.L, R_loc=self.R_loc, M=self.M, entry_bitsize=self.entry_bitsize
        )
        out_ent = bb.add_d(entry_oracle, q=q, m=np.array(m_regs, dtype=object), i=i)
        q = out_ent["q"]
        m_regs = list(out_ent["m"])
        i = out_ent["i"]

        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m.adjoint(), target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        return {"q": q, "m": np.array(m_regs, dtype=object), "i": i}

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        ctrl_bloq = OneParticleControlledSparseBlockEncoding(
            D=self.D,
            L=self.L,
            R_loc=self.R_loc,
            M=self.M,
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
class OneParticleControlledSparseBlockEncoding(Bloq):
    r"""Single-control wrapper for `OneParticleSparseBlockEncoding`.

    Convention matches two-particle controlled sparse block-encoding:
    - prepare / unprepare are left uncontrolled,
    - index oracle is controlled,
    - entry-oracle QROM is uncontrolled, only Ry ladder is controlled.
    """

    D: int
    L: int
    R_loc: int
    M: np.ndarray = field(
        converter=lambda x: np.asarray(x) if not isinstance(x, np.ndarray) else x,
        eq=lambda d: tuple(np.asarray(d).flat),
    )
    entry_bitsize: int = 10

    @property
    def m_bitsize(self) -> int:
        return max(1, int(np.ceil(np.log2(2 * self.R_loc + 1))))

    @property
    def i_bitsize(self) -> int:
        return int(np.ceil(np.log2(self.L)))

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("ctrl", QBit()),
                Register("q", QBit()),
                Register("m", BQUInt(self.m_bitsize, 2 * self.R_loc + 1), shape=(self.D,)),
                Register("i", BQUInt(self.i_bitsize, self.L), shape=(self.D,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, q, m, i):
        m_regs = list(m)
        i_regs = list(i)

        n_m = 2 * self.R_loc + 1
        prep_m = PrepareUniformSuperposition(n=n_m)

        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m, target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        # Controlled index oracle.
        index_oracle = OneParticleRowIndexOracle(D=self.D, L=self.L, R_loc=self.R_loc)
        out_idx = bb.add_d(
            index_oracle.controlled(),
            ctrl=ctrl,
            m=np.array(m_regs, dtype=object),
            i=np.array(i_regs, dtype=object),
        )
        ctrl = out_idx["ctrl"]
        m_regs = list(out_idx["m"])
        i_regs = list(out_idx["i"])

        # Entry oracle with uncontrolled QROM, controlled rotations only.
        entry_oracle = OneParticleRowEntryOracle(
            D=self.D, L=self.L, R_loc=self.R_loc, M=self.M, entry_bitsize=self.entry_bitsize
        )
        target = bb.allocate(int(self.entry_bitsize))

        dim_lengths = [self.L] * self.D + [2 * self.R_loc + 1] * self.D
        dim_soqs = i_regs + m_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        qrom_sel_regs = list(entry_oracle._qrom_theta.selection_registers)
        if len(kept_soqs) != len(qrom_sel_regs):
            raise DecomposeTypeError(
                "QROM selection-register mismatch for OneParticleControlledSparseBlockEncoding."
            )
        sel_in = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}

        out = bb.add_d(entry_oracle._qrom_theta, **sel_in, target0_=target)
        qrom_sel_names = [reg.name for reg in qrom_sel_regs]
        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            else:
                m_regs[idx - self.D] = sq
        target = out["target0_"]

        target_bits = bb.split(target)
        two_ctrl_ry = {
            k: Ry(2 * np.pi * (2 ** -(k + 1))).controlled(CtrlSpec(cvs=(1, 1)))
            for k in range(int(self.entry_bitsize))
        }
        for k, tbit in enumerate(target_bits):
            ctrls = np.array([ctrl, tbit], dtype=object)
            ctrls, q = bb.add(two_ctrl_ry[k], ctrl=ctrls, q=q)
            ctrl, tbit = ctrls
            target_bits[k] = tbit
        target = bb.join(target_bits)

        dim_soqs = i_regs + m_regs
        kept_soqs = [sq for ln, sq in zip(dim_lengths, dim_soqs) if int(ln) > 1]
        sel_in_adj = {reg.name: sq for reg, sq in zip(qrom_sel_regs, kept_soqs)}
        out = bb.add_d(entry_oracle._qrom_theta.adjoint(), **sel_in_adj, target0_=target)

        sel_iter = iter([out[name] for name in qrom_sel_names])
        for idx, ln in enumerate(dim_lengths):
            if int(ln) <= 1:
                continue
            sq = next(sel_iter)
            if idx < self.D:
                i_regs[idx] = sq
            else:
                m_regs[idx - self.D] = sq
        target = out["target0_"]
        bb.free(target)

        if n_m > 1:
            for d in range(self.D):
                m_bits = bb.split(m_regs[d])
                m_u = bb.join(m_bits, dtype=QUInt(self.m_bitsize))
                m_u = bb.add(prep_m.adjoint(), target=m_u)
                m_regs[d] = bb.join(bb.split(m_u), dtype=BQUInt(self.m_bitsize, n_m))

        return {
            "ctrl": ctrl,
            "q": q,
            "m": np.array(m_regs, dtype=object),
            "i": np.array(i_regs, dtype=object),
        }

    def get_ctrl_system(self, ctrl_spec: CtrlSpec):
        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1,
            bloq_with_ctrl=self,
            ctrl_reg_name="ctrl",
        )


def build_one_particle_row_index_oracle(D: int, L: int, R_loc: int) -> OneParticleRowIndexOracle:
    return OneParticleRowIndexOracle(D=int(D), L=int(L), R_loc=int(R_loc))


def build_one_particle_row_entry_oracle(
    M: np.ndarray,
    D: int,
    L: int,
    R_loc: int,
    entry_bitsize: int = 10,
) -> OneParticleRowEntryOracle:
    return OneParticleRowEntryOracle(
        D=int(D),
        L=int(L),
        R_loc=int(R_loc),
        M=np.asarray(M),
        entry_bitsize=int(entry_bitsize),
    )


def build_one_particle_sparse_block_encoding(
    M: np.ndarray,
    D: int,
    L: int,
    R_loc: int,
    entry_bitsize: int = 10,
) -> OneParticleSparseBlockEncoding:
    return OneParticleSparseBlockEncoding(
        D=int(D),
        L=int(L),
        R_loc=int(R_loc),
        M=np.asarray(M),
        entry_bitsize=int(entry_bitsize),
    )


def build_one_particle_controlled_sparse_block_encoding(
    M: np.ndarray,
    D: int,
    L: int,
    R_loc: int,
    entry_bitsize: int = 10,
) -> OneParticleControlledSparseBlockEncoding:
    return OneParticleControlledSparseBlockEncoding(
        D=int(D),
        L=int(L),
        R_loc=int(R_loc),
        M=np.asarray(M),
        entry_bitsize=int(entry_bitsize),
    )


# Backward compatibility
TwoParticleSparseOracle = TwoParticleSparseBlockEncoding
build_two_particle_sparse_oracle = build_two_particle_sparse_block_encoding
TwoParticleRowQueryBloq = TwoParticleSparseBlockEncoding
build_two_particle_row_query_bloq = build_two_particle_sparse_block_encoding
