from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import argparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exciton.benchmark_tensors import generate_f_tensor, generate_v_tensor
from integrations.qualtran.block_encoding.exciton_hamiltonian_encoding import build_exciton_hamiltonian_block_encoding
from qualtran.resource_counting import QECGatesCost, get_cost_value
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code.gidney_fowler_model import get_ccz2t_costs


@dataclass
class Row:
    m: int
    D: int
    L_c: int
    R_c: int
    R_loc: int
    logical_qubits: int
    toffoli_like: int
    clifford: int
    rotation: int
    runtime_hours: float
    within_day: bool


def _logical_qubits(sig) -> int:
    return int(sum(int(r.total_bits()) for r in sig.lefts()))


def _max_entries_estimate(L_c: int, D: int, R_c: int, R_loc: int) -> int:
    # rough per-table size (entries count) for W/V oracle-convention tables.
    return (L_c ** (2 * D)) * ((2 * R_c + 1) ** D) * ((2 * R_loc + 1) ** D)


def _compute_row(
    *,
    m: int,
    D: int,
    L_c: int,
    R_c: int,
    R_loc: int,
    entry_bitsize: int,
    factory: MultiFactory,
    data_block: SimpleDataBlock,
) -> Row | None:
    shape = (L_c,) * D
    F = generate_f_tensor(
        shape=shape, metric="chebyshev", r_cut=R_loc, oracle_convention="row_oracle"
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

    # Rescale for entry-oracle range constraints [0,1].
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
    if lq > 200:
        return None

    qec = get_cost_value(bloq, QECGatesCost())
    qd = qec.asdict() if hasattr(qec, "asdict") else vars(qec)
    toff_like = int(qd.get("toffoli", 0) + qd.get("and_bloq", 0) + qd.get("cswap", 0))
    cliff = int(qd.get("clifford", 0))
    rot = int(qd.get("rotation", 0))

    sc = get_ccz2t_costs(
        n_logical_gates=qec,
        n_algo_qubits=lq,
        phys_err=1e-3,
        cycle_time_us=1.0,
        factory=factory,
        data_block=data_block,
    )
    runtime_hr = float(sc.duration_hr)
    if runtime_hr > 168.0:
        return None

    return Row(
        m=m,
        D=D,
        L_c=L_c,
        R_c=R_c,
        R_loc=R_loc,
        logical_qubits=lq,
        toffoli_like=toff_like,
        clifford=cliff,
        rotation=rot,
        runtime_hours=runtime_hr,
        within_day=runtime_hr <= 24.0,
    )


def _iter_rows(
    m: int,
    D: int,
    *,
    entry_bitsize: int,
    max_entries: int,
    max_lc: int,
    factory: MultiFactory,
    data_block: SimpleDataBlock,
) -> list[Row]:
    rows: list[Row] = []
    for R_c in range(1, 11):
        for R_loc in range(1, 4):
            for L_c in range(2, max_lc + 1):
                if _max_entries_estimate(L_c, D, R_c, R_loc) > max_entries:
                    break
                try:
                    r = _compute_row(
                        m=m,
                        D=D,
                        L_c=L_c,
                        R_c=R_c,
                        R_loc=R_loc,
                        entry_bitsize=entry_bitsize,
                        factory=factory,
                        data_block=data_block,
                    )
                except Exception:
                    continue
                if r is not None:
                    rows.append(r)
    rows.sort(key=lambda x: (x.runtime_hours, -x.L_c, x.R_c, x.R_loc))
    return rows


def _latex_table(rows: Iterable[Row], *, title: str) -> str:
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        f"\\caption{{{title}}}\n"
        "\\begin{tabularx}{\\linewidth}{rrrrrrrrrrr}\n"
        "\\toprule\n"
        "$L_c$ & $R_c$ & $R_{\\mathrm{loc}}$ & logical qubits & Toffoli-like & Clifford & Rotation & runtime [h] & $\\leq$ 24h \\\\\n"
        "\\midrule\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"{r.L_c} & {r.R_c} & {r.R_loc} & {r.logical_qubits} & {r.toffoli_like} & {r.clifford} & {r.rotation} & {r.runtime_hours:.3f} & {'Y' if r.within_day else 'N'} \\\\"
        )
    if not body_lines:
        body_lines = ["\\multicolumn{9}{c}{No feasible points under current constraints} \\\\"]
    footer = (
        "\n\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(body_lines) + footer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=None, help="Single m to run.")
    parser.add_argument("--D", type=int, default=None, help="Single D to run.")
    parser.add_argument("--max-lc", type=int, default=64, help="Maximum L_c to scan.")
    parser.add_argument(
        "--max-entries",
        type=int,
        default=2_000_000,
        help="Hard cutoff on estimated table entries for W/V tensor generation.",
    )
    parser.add_argument("--entry-bitsize", type=int, default=10)
    parser.add_argument(
        "--prefix",
        type=str,
        default="exciton_resource_table",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--keep-day",
        type=int,
        default=30,
        help="Rows <=24h to keep (after sorting by runtime).",
    )
    parser.add_argument(
        "--keep-week",
        type=int,
        default=10,
        help="Rows (24h,168h] to keep (after sorting by runtime).",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="runtime",
        choices=["runtime", "largest_lc"],
        help="How to prioritize rows in the output table.",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = MultiFactory(
        base_factory=CCZ2TFactory(distillation_l1_d=19, distillation_l2_d=31),
        n_factories=4,
    )
    data_block = SimpleDataBlock(data_d=31, routing_overhead=0.5)

    if args.m is not None and args.D is not None:
        configs = [(int(args.m), int(args.D))]
    elif args.m is None and args.D is None:
        configs = [(2, 2), (2, 3), (3, 3)]
    else:
        raise ValueError("Provide both --m and --D, or neither.")

    for m, D in configs:
        rows = _iter_rows(
            m,
            D,
            entry_bitsize=int(args.entry_bitsize),
            max_entries=int(args.max_entries),
            max_lc=int(args.max_lc),
            factory=factory,
            data_block=data_block,
        )

        # Keep the table practical: prioritize points that fit in a day, then fastest week-scale.
        day_rows = [r for r in rows if r.within_day]
        week_rows = [r for r in rows if not r.within_day]
        if args.selection_mode == "largest_lc":
            day_rows.sort(key=lambda x: (-x.L_c, x.runtime_hours, x.R_c, x.R_loc))
            week_rows.sort(key=lambda x: (-x.L_c, x.runtime_hours, x.R_c, x.R_loc))
        else:
            day_rows.sort(key=lambda x: (x.runtime_hours, -x.L_c, x.R_c, x.R_loc))
            week_rows.sort(key=lambda x: (x.runtime_hours, -x.L_c, x.R_c, x.R_loc))
        selected = (day_rows[: int(args.keep_day)] + week_rows[: int(args.keep_week)])[
            : int(args.keep_day) + int(args.keep_week)
        ]

        tex = _latex_table(
            selected,
            title=(
                f"Exciton block-encoding resource points for $m={m}$, $D={D}$ "
                "(filtered to logical qubits $\\leq 200$ and runtime $\\leq 168$ h)."
            ),
        )
        out_path = out_dir / f"{args.prefix}_m{m}_d{D}.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {out_path} with {len(selected)} rows (from {len(rows)} feasible points).")


if __name__ == "__main__":
    main()
