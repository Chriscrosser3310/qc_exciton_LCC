from __future__ import annotations

from typing import Any

import numpy as np

try:
    from qualtran import Bloq, BloqBuilder, CompositeBloq, Controlled, CtrlSpec, QAny, QUInt
    from qualtran.bloqs.arithmetic import XorK
    from qualtran.bloqs.basic_gates import GlobalPhase, Rz
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Qualtran is required for algorithms.qsvt_qualtran. Install with extra '[qualtran]'."
    ) from exc


def make_query_schedule(
    query_bloq: Bloq, n_queries: int, alternate_adjoint: bool = True
) -> list[Bloq]:
    """Return a query schedule [U, U^dagger, U, ...] or [U, U, ...]."""
    schedule: list[Bloq] = []
    for k in range(n_queries):
        schedule.append(query_bloq.adjoint() if (alternate_adjoint and (k % 2 == 1)) else query_bloq)
    return schedule


def _apply_phase_single_qubit(
    bb: BloqBuilder, state: dict[str, Any], signal_reg: str, phi: float
) -> None:
    out = bb.add_d(Rz(angle=2.0 * float(phi)), q=state[signal_reg])
    state[signal_reg] = out["q"]


def _apply_phase_projector_zero(
    bb: BloqBuilder,
    state: dict[str, Any],
    signal_reg: str,
    phi: float,
    signal_bitsize: int,
) -> None:
    """Apply exp(i*phi*|0...0><0...0|) on a packed multi-bit signal register."""
    if signal_bitsize < 1:
        raise ValueError("signal_bitsize must be >= 1.")

    sig = state[signal_reg]
    all_ones = (1 << signal_bitsize) - 1

    # Map |0...0> to |1...1> with a packed XOR mask.
    s1 = bb.add_d(XorK(QUInt(signal_bitsize), all_ones), x=sig)["x"]

    # Apply e^{i phi} only when ctrl == |1...1>.
    ctrl_spec = CtrlSpec(qdtypes=QAny(signal_bitsize), cvs=all_ones)
    cond_phase = Controlled(GlobalPhase(exponent=float(phi) / np.pi), ctrl_spec=ctrl_spec)
    s2 = bb.add_d(cond_phase, ctrl=s1)["ctrl"]

    # Undo mapping.
    s3 = bb.add_d(XorK(QUInt(signal_bitsize), all_ones), x=s2)["x"]
    state[signal_reg] = s3


def _apply_phase(
    bb: BloqBuilder,
    state: dict[str, Any],
    signal_reg: str,
    phi: float,
    phase_mode: str,
    signal_bitsize: int,
) -> None:
    if phase_mode == "single_qubit":
        if signal_bitsize != 1:
            raise ValueError(
                "single_qubit phase mode requires signal_bitsize == 1. "
                f"Got signal_bitsize={signal_bitsize}. Use phase_mode='projector_zero'."
            )
        _apply_phase_single_qubit(bb, state, signal_reg, phi)
        return
    if phase_mode == "projector_zero":
        _apply_phase_projector_zero(bb, state, signal_reg, phi, signal_bitsize)
        return
    raise ValueError("phase_mode must be 'single_qubit' or 'projector_zero'.")


def _apply_query(bb: BloqBuilder, state: dict[str, Any], query_bloq: Bloq) -> None:
    in_names = [reg.name for reg in query_bloq.signature.lefts()]
    missing = [name for name in in_names if name not in state]
    if missing:
        raise ValueError(f"State is missing query registers: {missing}")
    out = bb.add_d(query_bloq, **{name: state[name] for name in in_names})
    state.update(out)


def build_qsvt_composite(
    query_schedule: list[Bloq],
    phases: list[float] | np.ndarray,
    register_bitsizes: dict[str, int],
    signal_reg: str = "signal",
    phase_mode: str = "single_qubit",
) -> CompositeBloq:
    """Build a Qualtran CompositeBloq implementing a QSVT-style sequence.

    The built sequence is: Phi_0 U_0 Phi_1 U_1 ... U_{d-1} Phi_d.
    """
    phases = list(phases)
    if len(phases) != len(query_schedule) + 1:
        raise ValueError(
            f"Need len(phases)=len(query_schedule)+1, got {len(phases)} and {len(query_schedule)}"
        )
    if signal_reg not in register_bitsizes:
        raise ValueError(f"signal_reg '{signal_reg}' not in register_bitsizes")

    signal_bitsize = int(register_bitsizes[signal_reg])
    bb = BloqBuilder()
    state: dict[str, Any] = {}
    for name, bitsize in register_bitsizes.items():
        state[name] = bb.add_register(name, bitsize)

    _apply_phase(bb, state, signal_reg, phases[0], phase_mode, signal_bitsize)
    for k, query in enumerate(query_schedule):
        _apply_query(bb, state, query)
        _apply_phase(bb, state, signal_reg, phases[k + 1], phase_mode, signal_bitsize)

    return bb.finalize(**state)
