from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exciton.benchmark_tensors import generate_f_tensor, generate_v_tensor
from integrations.qualtran.block_encoding.exciton_hamiltonian_encoding import build_exciton_hamiltonian_block_encoding
from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code.gidney_fowler_model import get_ccz2t_costs


@dataclass
class Row:
    d: int
    R_c: int
    L_c: int
    logical_qubits: int
    ancilla_qubits: int
    single_toffoli_like: int
    single_clifford: int
    single_rotation: int
    qsvt_toffoli_like: int
    qsvt_clifford: int
    qsvt_rotation: int
    qsvt_runtime_hours: float
    single_runtime_hours: float


def _logical_qubits(sig) -> int:
    return int(sum(int(r.total_bits()) for r in sig.lefts()))


def _system_qubits(*, m: int, D: int, L_c: int) -> int:
    n = int(np.ceil(np.log2(L_c)))
    return 2 * m * D * n


def _toffoli_like(qd: dict) -> int:
    return int(qd.get("toffoli", 0) + qd.get("and_bloq", 0) + qd.get("cswap", 0))


def _mc_rz_cost(control_size: int) -> GateCounts:
    # Approximation: C^k(Rz) via AND ladder + one rotation.
    # Toffoli-like cost ~ 2*(k-1), one logical rotation.
    k = max(0, int(control_size))
    toff = max(0, 2 * (k - 1))
    return GateCounts(toffoli=toff, clifford=0, rotation=1)


def _latex_table(rows: list[Row], *, m: int, D: int, r_loc: int, max_logical: int, max_runtime_h: float) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        (
            "\\caption{QSVT-style exciton cost table for "
            f"$m={m}$, $D={D}$, $R_{{\\mathrm{{loc}}}}={r_loc}$, $L_c=dR_c$. "
            f"Filtered to logical qubits $\\leq {max_logical}$ and QSVT runtime $\\leq {max_runtime_h:g}$ h."
            "}"
        ),
        "\\begin{tabularx}{\\linewidth}{rrrrrrrrrrrr}",
        "\\toprule",
        "$d$ & $R_c$ & $L_c$ & logical & ancilla & single Toff-like & single Cliff & single Rot & QSVT Toff-like & QSVT Cliff & QSVT Rot & QSVT runtime [h] \\\\",
        "\\midrule",
    ]
    if not rows:
        lines.append("\\multicolumn{12}{c}{No feasible points under current constraints and tensor-size cap.} \\\\")
    else:
        for r in rows:
            lines.append(
                f"{r.d} & {r.R_c} & {r.L_c} & {r.logical_qubits} & {r.ancilla_qubits} & "
                f"{r.single_toffoli_like} & {r.single_clifford} & {r.single_rotation} & "
                f"{r.qsvt_toffoli_like} & {r.qsvt_clifford} & {r.qsvt_rotation} & {r.qsvt_runtime_hours:.3f} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX resource table for QSVT-style exciton costs with L_c = d R_c."
    )
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--r-loc", type=int, default=1)
    parser.add_argument("--entry-bitsize", type=int, default=10)
    parser.add_argument("--max-logical", type=int, default=200)
    parser.add_argument("--max-runtime-h", type=float, default=24.0)
    parser.add_argument("--max-entries", type=int, default=20_000_000)
    parser.add_argument("--d-max", type=int, default=10)
    parser.add_argument("--rc-max", type=int, default=10)
    parser.add_argument(
        "--out-name",
        type=str,
        default="qsvt_exciton_table_m2_d2_lc_eq_drc.tex",
        help="Output filename under docs/.",
    )
    args = parser.parse_args()

    m = int(args.m)
    D = int(args.D)
    R_loc = int(args.r_loc)
    entry_bitsize = int(args.entry_bitsize)
    max_logical = int(args.max_logical)
    max_runtime_h = float(args.max_runtime_h)
    max_entries = int(args.max_entries)  # guardrail for explicit tensor generation

    factory = MultiFactory(
        base_factory=CCZ2TFactory(distillation_l1_d=19, distillation_l2_d=31),
        n_factories=4,
    )
    data_block = SimpleDataBlock(data_d=31, routing_overhead=0.5)

    rows: list[Row] = []
    skipped_large_tensor = 0

    for d in range(1, args.d_max + 1):
        for R_c in range(1, args.rc_max + 1):
            L_c = d * R_c
            if L_c < 2:
                continue
            if (L_c & (L_c - 1)) != 0:
                # Current arithmetic decomposition requires power-of-two lattice side.
                continue
            # W/V table size estimate for oracle-convention tensors:
            est_entries = (L_c ** (2 * D)) * ((2 * R_c + 1) ** D) * ((2 * R_loc + 1) ** D)
            if est_entries > max_entries:
                skipped_large_tensor += 1
                continue

            shape = (L_c,) * D
            F = generate_f_tensor(
                shape=shape,
                metric="chebyshev",
                r_cut=R_loc,
                oracle_convention="row_oracle",
            )
            W = generate_v_tensor(
                shape=shape,
                metric="chebyshev",
                r_loc=R_loc,
                r_c=R_c,
                oracle_convention="exchange",
            )
            V = generate_v_tensor(
                shape=shape,
                metric="chebyshev",
                r_loc=R_loc,
                r_c=R_c,
                oracle_convention="direct",
            )

            aw = float(np.max(np.abs(W)))
            av = float(np.max(np.abs(V)))
            if aw > 0:
                W = W / aw
            if av > 0:
                V = V / av

            bloq = build_exciton_hamiltonian_block_encoding(
                num_pairs=m,
                D=D,
                L=L_c,
                R_c=R_c,
                R_loc=R_loc,
                F=F,
                W=W,
                V=V,
                entry_bitsize=entry_bitsize,
            )
            lq = _logical_qubits(bloq.signature)
            if lq > max_logical:
                continue

            anc = lq - _system_qubits(m=m, D=D, L_c=L_c)
            single_gc = get_cost_value(bloq, QECGatesCost())
            sdict = single_gc.asdict() if hasattr(single_gc, "asdict") else vars(single_gc)

            mc_rz = _mc_rz_cost(anc)
            qsvt_gc = d * single_gc + d * mc_rz

            single_rt = get_ccz2t_costs(
                n_logical_gates=single_gc,
                n_algo_qubits=lq,
                phys_err=1e-3,
                cycle_time_us=1.0,
                factory=factory,
                data_block=data_block,
            )
            qsvt_rt = get_ccz2t_costs(
                n_logical_gates=qsvt_gc,
                n_algo_qubits=lq,
                phys_err=1e-3,
                cycle_time_us=1.0,
                factory=factory,
                data_block=data_block,
            )

            if float(qsvt_rt.duration_hr) > max_runtime_h:
                continue

            qd = qsvt_gc.asdict() if hasattr(qsvt_gc, "asdict") else vars(qsvt_gc)
            rows.append(
                Row(
                    d=d,
                    R_c=R_c,
                    L_c=L_c,
                    logical_qubits=lq,
                    ancilla_qubits=anc,
                    single_toffoli_like=_toffoli_like(sdict),
                    single_clifford=int(sdict.get("clifford", 0)),
                    single_rotation=int(sdict.get("rotation", 0)),
                    qsvt_toffoli_like=_toffoli_like(qd),
                    qsvt_clifford=int(qd.get("clifford", 0)),
                    qsvt_rotation=int(qd.get("rotation", 0)),
                    qsvt_runtime_hours=float(qsvt_rt.duration_hr),
                    single_runtime_hours=float(single_rt.duration_hr),
                )
            )

    rows.sort(key=lambda r: (r.qsvt_runtime_hours, -r.L_c, r.d, r.R_c))
    out_dir = REPO_ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    out_path.write_text(
        _latex_table(
            rows,
            m=m,
            D=D,
            r_loc=R_loc,
            max_logical=max_logical,
            max_runtime_h=max_runtime_h,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_path} with {len(rows)} rows.")
    print(f"Skipped {skipped_large_tensor} (d,R_c) points due to tensor-size guard ({max_entries}).")


if __name__ == "__main__":
    main()
