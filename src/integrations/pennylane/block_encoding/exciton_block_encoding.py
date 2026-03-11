from __future__ import annotations

from math import comb
from typing import Sequence

import numpy as np

from .sparse_block_encoding import (
    one_particle_sparse_block_encoding,
    two_particle_sparse_block_encoding,
)


Wire = int | str


def _as_particle_registers(
    registers: Sequence[Sequence[Sequence[Wire]]],
) -> list[list[list[Wire]]]:
    out = [[list(dim) for dim in reg] for reg in registers]
    if len(out) == 0:
        raise ValueError("registers must be non-empty.")
    d = len(out[0])
    if d == 0:
        raise ValueError("Each register must have at least one dimension.")
    n_bits = len(out[0][0])
    for reg in out:
        if len(reg) != d:
            raise ValueError("All registers must have same number of dimensions.")
        for dim in reg:
            if len(dim) != n_bits:
                raise ValueError("All per-dimension index registers must have same bit-width.")
    return out


def _uniform_prefix_state(num_states: int, n_qubits: int, signed: np.ndarray | None = None) -> np.ndarray:
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    dim = 2**n_qubits
    if num_states > dim:
        raise ValueError(f"num_states={num_states} exceeds capacity 2^{n_qubits}={dim}.")
    vec = np.zeros(dim, dtype=np.complex128)
    if signed is None:
        vec[:num_states] = 1.0 / np.sqrt(float(num_states))
    else:
        if len(signed) != num_states:
            raise ValueError("signed vector length must match num_states.")
        vec[:num_states] = signed / np.sqrt(float(num_states))
    return vec


def _prepare_state(wires: Sequence[Wire], state: np.ndarray) -> None:
    import pennylane as qml

    if len(wires) == 0:
        if state.shape == (1,):
            return
        raise ValueError("Non-trivial state cannot be prepared on zero wires.")
    qml.MottonenStatePreparation(state, wires=wires)


def _unprepare_state(wires: Sequence[Wire], state: np.ndarray) -> None:
    import pennylane as qml

    if len(wires) == 0:
        if state.shape == (1,):
            return
        raise ValueError("Non-trivial state cannot be unprepared on zero wires.")
    qml.adjoint(qml.MottonenStatePreparation)(state, wires=wires)


def _to_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple(int(b) for b in format(int(value), f"0{int(width)}b"))


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
    rest = [k for k in range(n) if k not in (a, b)]
    return tuple([a, b] + rest)


def _perm_map_sources_to_slots(n: int, src_a: int, src_b: int, dst_a: int, dst_b: int) -> tuple[int, ...]:
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


def _controlled_swap_regs(
    *,
    control_wires: Sequence[Wire],
    control_values: Sequence[int],
    reg_a: Sequence[Sequence[Wire]],
    reg_b: Sequence[Sequence[Wire]],
) -> None:
    import pennylane as qml

    for dim_a, dim_b in zip(reg_a, reg_b):
        for wa, wb in zip(dim_a, dim_b):
            qml.ctrl(qml.SWAP, control=control_wires, control_values=tuple(control_values))(
                wires=[wa, wb]
            )


def _merge_controls(
    control_wires: Sequence[Wire] | None,
    control_values: Sequence[int] | None,
    extra_wires: Sequence[Wire],
    extra_values: Sequence[int],
) -> tuple[tuple[Wire, ...], tuple[int, ...]]:
    base_wires = tuple(control_wires or ())
    if control_values is None:
        base_values = (1,) * len(base_wires)
    else:
        if len(control_values) != len(base_wires):
            raise ValueError("control_values must have same length as control_wires.")
        base_values = tuple(control_values)
    return base_wires + tuple(extra_wires), base_values + tuple(extra_values)


def one_particle_f_sum_block_encoding(
    *,
    F_table: np.ndarray,
    particle_registers: Sequence[Sequence[Sequence[Wire]]],
    sel_wires: Sequence[Wire],
    m_wires: Sequence[Sequence[Wire]],
    ancilla_wire: Wire,
    angle_wires: Sequence[Wire],
    L: int,
    R_loc: int,
    index_work_wires: Sequence[Wire],
    entry_work_wires: Sequence[Wire] | None = None,
    control_wires: Sequence[Wire] | None = None,
    control_values: Sequence[int] | None = None,
) -> None:
    """Signed sum over F acting on 2m particle registers."""
    regs = _as_particle_registers(particle_registers)
    n_regs = len(regs)
    if n_regs % 2 != 0:
        raise ValueError("particle_registers length must be 2m.")
    m_pairs = n_regs // 2
    term_count = n_regs
    if len(sel_wires) < int(np.ceil(np.log2(term_count))):
        raise ValueError("Not enough selector wires for F terms.")

    signed = np.array([1.0 if t < m_pairs else -1.0 for t in range(term_count)], dtype=float)
    psi_signed = _uniform_prefix_state(term_count, len(sel_wires), signed=signed)
    psi_uniform = _uniform_prefix_state(term_count, len(sel_wires))

    _prepare_state(sel_wires, psi_signed)

    for t in range(1, term_count):
        cw, cv = _merge_controls(control_wires, control_values, sel_wires, _to_bits(t, len(sel_wires)))
        _controlled_swap_regs(
            control_wires=cw,
            control_values=cv,
            reg_a=regs[0],
            reg_b=regs[t],
        )

    one_particle_sparse_block_encoding(
        table=F_table,
        i_wires=regs[0],
        m_wires=m_wires,
        ancilla_wire=ancilla_wire,
        angle_wires=angle_wires,
        L=L,
        R_loc=R_loc,
        index_work_wires=index_work_wires,
        entry_work_wires=entry_work_wires or (),
        control_wires=control_wires,
        control_values=control_values,
    )

    for t in reversed(range(1, term_count)):
        cw, cv = _merge_controls(control_wires, control_values, sel_wires, _to_bits(t, len(sel_wires)))
        _controlled_swap_regs(
            control_wires=cw,
            control_values=cv,
            reg_a=regs[0],
            reg_b=regs[t],
        )

    _unprepare_state(sel_wires, psi_uniform)


def two_particle_w_sum_block_encoding(
    *,
    W_table: np.ndarray,
    particle_registers: Sequence[Sequence[Sequence[Wire]]],
    sel_wires: Sequence[Wire],
    m_wires: Sequence[Sequence[Wire]],
    l_wires: Sequence[Sequence[Wire]],
    ancilla_wire: Wire,
    angle_wires: Sequence[Wire],
    L: int,
    R_c: int,
    R_loc: int,
    index_work_wires: Sequence[Wire],
    entry_work_wires: Sequence[Wire] | None = None,
    control_wires: Sequence[Wire] | None = None,
    control_values: Sequence[int] | None = None,
) -> None:
    """Signed sum over all unordered pairs with cross-partition minus sign."""
    regs = _as_particle_registers(particle_registers)
    n_regs = len(regs)
    if n_regs % 2 != 0:
        raise ValueError("particle_registers length must be 2m.")
    m_pairs = n_regs // 2

    pairs = tuple((a, b) for a in range(n_regs) for b in range(a + 1, n_regs))
    term_count = len(pairs)
    if len(sel_wires) < int(np.ceil(np.log2(term_count))):
        raise ValueError("Not enough selector wires for W terms.")

    signed = np.array(
        [(-1.0 if (a < m_pairs and b >= m_pairs) else 1.0) for (a, b) in pairs], dtype=float
    )
    psi_signed = _uniform_prefix_state(term_count, len(sel_wires), signed=signed)
    psi_uniform = _uniform_prefix_state(term_count, len(sel_wires))
    _prepare_state(sel_wires, psi_signed)

    for t, (a, b) in enumerate(pairs):
        cw, cv = _merge_controls(control_wires, control_values, sel_wires, _to_bits(t, len(sel_wires)))
        perm = _perm_for_pair(n_regs, a, b)
        for u, v in _swap_sequence_for_target_perm(perm):
            _controlled_swap_regs(
                control_wires=cw,
                control_values=cv,
                reg_a=regs[u],
                reg_b=regs[v],
            )

    two_particle_sparse_block_encoding(
        table=W_table,
        i_wires=regs[0],
        j_wires=regs[1],
        m_wires=m_wires,
        l_wires=l_wires,
        ancilla_wire=ancilla_wire,
        angle_wires=angle_wires,
        L=L,
        R_c=R_c,
        R_loc=R_loc,
        index_work_wires=index_work_wires,
        entry_work_wires=entry_work_wires or (),
        control_wires=control_wires,
        control_values=control_values,
    )

    for t in reversed(range(term_count)):
        a, b = pairs[t]
        cw, cv = _merge_controls(control_wires, control_values, sel_wires, _to_bits(t, len(sel_wires)))
        perm = _perm_for_pair(n_regs, a, b)
        for u, v in reversed(_swap_sequence_for_target_perm(perm)):
            _controlled_swap_regs(
                control_wires=cw,
                control_values=cv,
                reg_a=regs[u],
                reg_b=regs[v],
            )

    _unprepare_state(sel_wires, psi_uniform)


def two_particle_v_sum_block_encoding(
    *,
    V_table: np.ndarray,
    particle_registers: Sequence[Sequence[Sequence[Wire]]],
    sel_wires: Sequence[Wire],
    m_wires: Sequence[Sequence[Wire]],
    l_wires: Sequence[Sequence[Wire]],
    ancilla_wire: Wire,
    angle_wires: Sequence[Wire],
    L: int,
    R_c: int,
    R_loc: int,
    index_work_wires: Sequence[Wire],
    entry_work_wires: Sequence[Wire] | None = None,
    control_wires: Sequence[Wire] | None = None,
    control_values: Sequence[int] | None = None,
) -> None:
    """Unsigned sum over cross-partition pairs routed to anchors (m-1, m)."""
    regs = _as_particle_registers(particle_registers)
    n_regs = len(regs)
    if n_regs % 2 != 0:
        raise ValueError("particle_registers length must be 2m.")
    m_pairs = n_regs // 2
    anchor_l, anchor_r = m_pairs - 1, m_pairs

    terms = tuple((a, b) for a in range(m_pairs) for b in range(m_pairs, n_regs))
    term_count = len(terms)
    if len(sel_wires) < int(np.ceil(np.log2(term_count))):
        raise ValueError("Not enough selector wires for V terms.")
    psi_uniform = _uniform_prefix_state(term_count, len(sel_wires))
    _prepare_state(sel_wires, psi_uniform)

    for t, (a, b) in enumerate(terms):
        cw, cv = _merge_controls(control_wires, control_values, sel_wires, _to_bits(t, len(sel_wires)))
        perm = _perm_map_sources_to_slots(n_regs, a, b, anchor_l, anchor_r)
        for u, v in _swap_sequence_for_target_perm(perm):
            _controlled_swap_regs(
                control_wires=cw,
                control_values=cv,
                reg_a=regs[u],
                reg_b=regs[v],
            )

    two_particle_sparse_block_encoding(
        table=V_table,
        i_wires=regs[anchor_l],
        j_wires=regs[anchor_r],
        m_wires=m_wires,
        l_wires=l_wires,
        ancilla_wire=ancilla_wire,
        angle_wires=angle_wires,
        L=L,
        R_c=R_c,
        R_loc=R_loc,
        index_work_wires=index_work_wires,
        entry_work_wires=entry_work_wires or (),
        control_wires=control_wires,
        control_values=control_values,
    )

    for t in reversed(range(term_count)):
        a, b = terms[t]
        cw, cv = _merge_controls(control_wires, control_values, sel_wires, _to_bits(t, len(sel_wires)))
        perm = _perm_map_sources_to_slots(n_regs, a, b, anchor_l, anchor_r)
        for u, v in reversed(_swap_sequence_for_target_perm(perm)):
            _controlled_swap_regs(
                control_wires=cw,
                control_values=cv,
                reg_a=regs[u],
                reg_b=regs[v],
            )

    _unprepare_state(sel_wires, psi_uniform)


def exciton_block_encoding(
    *,
    F_table: np.ndarray,
    W_table: np.ndarray,
    V_table: np.ndarray,
    particle_registers: Sequence[Sequence[Sequence[Wire]]],
    h_sel_wires: Sequence[Wire],
    f_sel_wires: Sequence[Wire],
    w_sel_wires: Sequence[Wire],
    v_sel_wires: Sequence[Wire],
    f_m_wires: Sequence[Sequence[Wire]],
    w_m_wires: Sequence[Sequence[Wire]],
    w_l_wires: Sequence[Sequence[Wire]],
    v_m_wires: Sequence[Sequence[Wire]],
    v_l_wires: Sequence[Sequence[Wire]],
    ancilla_wire: Wire,
    angle_wires: Sequence[Wire],
    L: int,
    R_c: int,
    R_loc: int,
    index_work_wires: Sequence[Wire],
    entry_work_wires: Sequence[Wire] | None = None,
) -> None:
    """Top-level exciton block-encoding = LCU over {F-sum, W-sum, V-sum}."""
    import pennylane as qml

    term_count = 3
    if len(h_sel_wires) < int(np.ceil(np.log2(term_count))):
        raise ValueError("Not enough h_sel_wires for 3 exciton terms.")
    psi_uniform = _uniform_prefix_state(term_count, len(h_sel_wires))
    _prepare_state(h_sel_wires, psi_uniform)

    one_particle_f_sum_block_encoding(
        F_table=F_table,
        particle_registers=particle_registers,
        sel_wires=f_sel_wires,
        m_wires=f_m_wires,
        ancilla_wire=ancilla_wire,
        angle_wires=angle_wires,
        L=L,
        R_loc=R_loc,
        index_work_wires=index_work_wires,
        entry_work_wires=entry_work_wires or (),
        control_wires=h_sel_wires,
        control_values=_to_bits(0, len(h_sel_wires)),
    )

    two_particle_w_sum_block_encoding(
        W_table=W_table,
        particle_registers=particle_registers,
        sel_wires=w_sel_wires,
        m_wires=w_m_wires,
        l_wires=w_l_wires,
        ancilla_wire=ancilla_wire,
        angle_wires=angle_wires,
        L=L,
        R_c=R_c,
        R_loc=R_loc,
        index_work_wires=index_work_wires,
        entry_work_wires=entry_work_wires or (),
        control_wires=h_sel_wires,
        control_values=_to_bits(1, len(h_sel_wires)),
    )

    two_particle_v_sum_block_encoding(
        V_table=V_table,
        particle_registers=particle_registers,
        sel_wires=v_sel_wires,
        m_wires=v_m_wires,
        l_wires=v_l_wires,
        ancilla_wire=ancilla_wire,
        angle_wires=angle_wires,
        L=L,
        R_c=R_c,
        R_loc=R_loc,
        index_work_wires=index_work_wires,
        entry_work_wires=entry_work_wires or (),
        control_wires=h_sel_wires,
        control_values=_to_bits(2, len(h_sel_wires)),
    )

    _unprepare_state(h_sel_wires, psi_uniform)


__all__ = [
    "one_particle_f_sum_block_encoding",
    "two_particle_w_sum_block_encoding",
    "two_particle_v_sum_block_encoding",
    "exciton_block_encoding",
]
