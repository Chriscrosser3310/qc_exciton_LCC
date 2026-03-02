"""Build a QSVT-style Qualtran composite bloq from a toy block-encoding.

Usage examples:
  python scripts/qsvt_qualtran_demo.py
  python scripts/qsvt_qualtran_demo.py --phase-mode projector_zero --signal-bits 2 --queries 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithms.qsvt_qualtran import build_qsvt_composite, make_query_schedule
from qualtran import BloqBuilder, QUInt
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import CNOT, CRz, Rz
from qualtran.resource_counting import QECGatesCost, get_cost_value


def build_toy_block_encoding(signal_bits: int):
    bb = BloqBuilder()
    signal = bb.add_register("signal", signal_bits)
    system = bb.add_register("system", 1)

    if signal_bits == 1:
        o1 = bb.add_d(CRz(angle=0.7), ctrl=signal, q=system)
        o2 = bb.add_d(CNOT(), ctrl=o1["q"], target=o1["ctrl"])
        return bb.finalize(signal=o2["target"], system=o2["ctrl"])

    signal = bb.add_d(XorK(QUInt(signal_bits), 1), x=signal)["x"]
    system = bb.add_d(Rz(angle=0.25), q=system)["q"]
    return bb.finalize(signal=signal, system=system)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qualtran QSVT-style composite bloq.")
    parser.add_argument("--queries", type=int, default=6, help="Number of query steps.")
    parser.add_argument("--signal-bits", type=int, default=1, help="Signal register bitsize.")
    parser.add_argument(
        "--phase-mode",
        choices=["single_qubit", "projector_zero"],
        default="single_qubit",
        help="Phase mode for QSVT layers.",
    )
    parser.add_argument(
        "--alternate-adjoint",
        action="store_true",
        default=False,
        help="Alternate between U and U^dagger in query schedule.",
    )
    args = parser.parse_args()

    U = build_toy_block_encoding(signal_bits=args.signal_bits)
    schedule = make_query_schedule(
        U, n_queries=args.queries, alternate_adjoint=args.alternate_adjoint
    )
    phases = np.linspace(0.0, np.pi / 2, args.queries + 1)

    qsvt = build_qsvt_composite(
        query_schedule=schedule,
        phases=phases,
        register_bitsizes={"signal": args.signal_bits, "system": 1},
        signal_reg="signal",
        phase_mode=args.phase_mode,
    )

    print("Built QSVT composite bloq.")
    print("Signature:", qsvt.signature)
    try:
        qec = get_cost_value(qsvt, QECGatesCost())
        print("QEC gate counts:", qec.asdict())
        print("T-equivalent:", int(qec.total_t_count()))
    except Exception as exc:
        print("Cost estimation unavailable for this instance:", exc)


if __name__ == "__main__":
    main()
