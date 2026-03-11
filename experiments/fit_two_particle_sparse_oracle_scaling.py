from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qualtran.resource_counting import QECGatesCost, get_cost_value

from integrations.qualtran.block_encoding.two_particle_row_oracles import build_two_particle_sparse_block_encoding
from exciton.benchmark_tensors import generate_v_tensor


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _toffoli_like(qec_dict: dict) -> int:
    return int(qec_dict.get("toffoli", 0) + qec_dict.get("and_bloq", 0) + qec_dict.get("cswap", 0))


def _fit_power_law(
    rows: list[dict],
    target_key: str,
) -> dict[str, float]:
    # target ~ C * L^a * (2R_c+1)^b * (2R_loc+1)^c * D^d
    y = np.array([max(1.0, float(r[target_key])) for r in rows], dtype=float)
    L = np.array([float(r["L"]) for r in rows], dtype=float)
    Rc = np.array([float(r["R_c"]) for r in rows], dtype=float)
    Rloc = np.array([float(r["R_loc"]) for r in rows], dtype=float)
    D = np.array([float(r["D"]) for r in rows], dtype=float)

    X = np.column_stack(
        [
            np.ones_like(y),
            np.log(L),
            np.log(2.0 * Rc + 1.0),
            np.log(2.0 * Rloc + 1.0),
            np.log(D),
        ]
    )
    beta = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    c = float(np.exp(beta[0]))
    return {
        "C": c,
        "a_L": float(beta[1]),
        "b_Rc": float(beta[2]),
        "c_Rloc": float(beta[3]),
        "d_D": float(beta[4]),
    }


def _fit_fixed_structure_constant(
    rows: list[dict],
    target_key: str,
) -> float:
    # target ~ C * ((L^2 * (2R_c+1) * (2R_loc+1))^D)
    x = np.array(
        [
            (float(r["L"]) ** 2 * (2.0 * float(r["R_c"]) + 1.0) * (2.0 * float(r["R_loc"]) + 1.0))
            ** float(r["D"])
            for r in rows
        ],
        dtype=float,
    )
    y = np.array([float(r[target_key]) for r in rows], dtype=float)
    return float((x @ y) / (x @ x))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit TwoParticleSparseBlockEncoding scaling and estimate constant factors."
    )
    parser.add_argument("--L-values", default="2,4,8")
    parser.add_argument("--Rc-values", default="1,2,3")
    parser.add_argument("--Rloc-values", default="1,2,3")
    parser.add_argument("--D-values", default="1,2")
    parser.add_argument("--entry-bitsize", type=int, default=8)
    parser.add_argument("--metric", default="chebyshev")
    parser.add_argument("--convention", choices=["direct", "exchange"], default="direct")
    args = parser.parse_args()

    L_values = _parse_int_list(args.L_values)
    Rc_values = _parse_int_list(args.Rc_values)
    Rloc_values = _parse_int_list(args.Rloc_values)
    D_values = _parse_int_list(args.D_values)

    rows: list[dict] = []
    for d in D_values:
        for l in L_values:
            for rc in Rc_values:
                for rloc in Rloc_values:
                    m = generate_v_tensor(
                        L=l,
                        D=d,
                        r_c=rc,
                        r_loc=rloc,
                        metric=args.metric,
                        oracle_convention=args.convention,
                    )
                    bloq = build_two_particle_sparse_block_encoding(
                        M=m,
                        D=d,
                        L=l,
                        R_c=rc,
                        R_loc=rloc,
                        entry_bitsize=args.entry_bitsize,
                    )
                    qec = get_cost_value(bloq, QECGatesCost())
                    qec_dict = qec.asdict() if hasattr(qec, "asdict") else vars(qec)
                    rows.append(
                        {
                            "D": d,
                            "L": l,
                            "R_c": rc,
                            "R_loc": rloc,
                            "toffoli_like": _toffoli_like(qec_dict),
                            "clifford": int(qec_dict.get("clifford", 0)),
                            "rotation": int(qec_dict.get("rotation", 0)),
                        }
                    )

    print("D,L,R_c,R_loc,toffoli_like,clifford,rotation")
    for r in rows:
        print(
            f"{r['D']},{r['L']},{r['R_c']},{r['R_loc']},"
            f"{r['toffoli_like']},{r['clifford']},{r['rotation']}"
        )

    toff_fit = _fit_power_law(rows, "toffoli_like")
    cliff_fit = _fit_power_law(rows, "clifford")
    rot_fit = _fit_power_law(rows, "rotation")

    c_toff_fixed = _fit_fixed_structure_constant(rows, "toffoli_like")
    c_cliff_fixed = _fit_fixed_structure_constant(rows, "clifford")
    c_rot_fixed = _fit_fixed_structure_constant(rows, "rotation")

    print("\nPower-law fit: target ~ C * L^a * (2R_c+1)^b * (2R_loc+1)^c * D^d")
    print("Toffoli-like:", toff_fit)
    print("Clifford    :", cliff_fit)
    print("Rotation    :", rot_fit)

    print("\nFixed-structure constant fit:")
    print("target ~ C * ((L^2 * (2R_c+1) * (2R_loc+1))^D)")
    print(f"C_toffoli_like = {c_toff_fixed:.8g}")
    print(f"C_clifford     = {c_cliff_fixed:.8g}")
    print(f"C_rotation     = {c_rot_fixed:.8g}")


if __name__ == "__main__":
    main()
