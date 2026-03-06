from __future__ import annotations

from typing import Sequence

import numpy as np
from attrs import frozen

try:
    from qualtran import Bloq, BloqBuilder, CtrlSpec, QBit, QUInt, Register, Signature
    from qualtran.bloqs.basic_gates import Swap
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for controlled permutation bloqs. Install with extra '[qualtran]'."
    ) from exc


def _validate_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    p = tuple(int(x) for x in perm)
    n = len(p)
    if sorted(p) != list(range(n)):
        raise ValueError("permutation must be a permutation of 0..n-1.")
    return p


def _swap_sequence_for_target_perm(perm: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return swaps that transform registers to output[k] = input[perm[k]]."""
    p = _validate_permutation(perm)
    n = len(p)
    cur = list(range(n))
    swaps: list[tuple[int, int]] = []
    for i in range(n):
        if cur[i] == p[i]:
            continue
        j = cur.index(p[i], i + 1)
        cur[i], cur[j] = cur[j], cur[i]
        swaps.append((i, j))
    return tuple(swaps)


@frozen
class MultiControlledRegisterPermutation(Bloq):
    """Multi-controlled permutation on a small list of registers.

    Register action when control bits match `control_values`:
      output[k] <- input[permutation[k]]

    Control mismatch leaves registers unchanged.
    """

    register_bitsizes: tuple[int, ...]
    permutation: tuple[int, ...]
    control_values: tuple[int, ...] = (1,)

    def __attrs_post_init__(self) -> None:
        bits = tuple(int(b) for b in self.register_bitsizes)
        if len(bits) == 0:
            raise ValueError("register_bitsizes must be non-empty.")
        if any(b <= 0 for b in bits):
            raise ValueError("All register bitsizes must be >= 1.")

        perm = _validate_permutation(self.permutation)
        if len(perm) != len(bits):
            raise ValueError("permutation length must match register_bitsizes length.")

        cvs = tuple(int(v) for v in self.control_values)
        if len(cvs) == 0:
            raise ValueError("control_values must be non-empty.")
        if any(v not in (0, 1) for v in cvs):
            raise ValueError("control_values must be a tuple of 0/1 bits.")

        # For fixed signature dtypes, each destination slot must receive same-sized source.
        for dst, src in enumerate(perm):
            if bits[dst] != bits[src]:
                raise ValueError(
                    f"Invalid permutation for fixed register sizes: bits[{dst}]={bits[dst]} "
                    f"but bits[{src}]={bits[src]}."
                )

    @property
    def n_registers(self) -> int:
        return len(self.register_bitsizes)

    @property
    def n_controls(self) -> int:
        return len(self.control_values)

    @property
    def signature(self) -> Signature:
        regs = [Register("ctrl", QBit(), shape=(self.n_controls,))]
        regs.extend(
            Register(f"r{i}", QUInt(int(bits))) for i, bits in enumerate(self.register_bitsizes)
        )
        return Signature(regs)

    @property
    def swap_sequence(self) -> tuple[tuple[int, int], ...]:
        return _swap_sequence_for_target_perm(self.permutation)

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, **regs):
        ctrl_arr = np.array(ctrl, dtype=object).reshape(self.n_controls)
        reg_list = [regs[f"r{i}"] for i in range(self.n_registers)]

        ctrl_spec = CtrlSpec(cvs=self.control_values)
        swap_bloqs: dict[int, Bloq] = {}
        for i, j in self.swap_sequence:
            bits = int(self.register_bitsizes[i])
            sw = swap_bloqs.get(bits)
            if sw is None:
                sw = Swap(bits).controlled(ctrl_spec)
                swap_bloqs[bits] = sw

            ctrl_arr, reg_list[i], reg_list[j] = bb.add(
                sw, ctrl=ctrl_arr, x=reg_list[i], y=reg_list[j]
            )
            ctrl_arr = np.array(ctrl_arr, dtype=object).reshape(self.n_controls)

        out = {"ctrl": ctrl_arr}
        out.update({f"r{i}": reg_list[i] for i in range(self.n_registers)})
        return out

    def call_classically(self, ctrl, **regs):
        ctrl_bits = tuple(int(x) for x in np.asarray(ctrl, dtype=int).reshape(-1).tolist())
        values = [int(regs[f"r{i}"]) for i in range(self.n_registers)]
        if ctrl_bits == self.control_values:
            values = [values[self.permutation[k]] for k in range(self.n_registers)]
        return (np.array(ctrl_bits, dtype=int), *values)


def build_multi_controlled_register_permutation(
    register_bitsizes: Sequence[int],
    permutation: Sequence[int],
    *,
    control_values: Sequence[int] = (1,),
) -> MultiControlledRegisterPermutation:
    return MultiControlledRegisterPermutation(
        register_bitsizes=tuple(int(b) for b in register_bitsizes),
        permutation=tuple(int(p) for p in permutation),
        control_values=tuple(int(v) for v in control_values),
    )

