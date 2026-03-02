from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code.gidney_fowler_model import (
    get_ccz2t_costs,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
    iter_simple_data_blocks,
)

from block_encoding.qualtran_lattice_index_oracles import (
    SingleParticleSparseIndexOracle,
    TwoParticleSparseIndexOracle,
)


def _nbits_sites(linear_size: int, dim: int) -> tuple[int, int]:
    n_sites = linear_size**dim
    nbits = max(1, math.ceil(math.log2(n_sites)))
    return nbits, n_sites


def _span(linear_size: int, radius: int) -> int:
    return min(linear_size, 2 * radius + 1)


def _entry_f_counts(nbits: int, dim: int, entry_bits: int) -> tuple[int, int, int]:
    toff = 8 * dim * nbits + 4 * entry_bits
    cliff = 24 * dim * nbits + 12 * entry_bits
    rot = entry_bits
    return toff, cliff, rot


def _entry_v_counts(nbits: int, dim: int, entry_bits: int) -> tuple[int, int, int]:
    toff = 20 * dim * nbits + 6 * entry_bits
    cliff = 60 * dim * nbits + 18 * entry_bits
    rot = entry_bits
    return toff, cliff, rot


def compute_exciton_query_costs(
    *,
    sizes: list[int],
    m: int,
    dim: int,
    r_loc: int,
    r_c: int,
    entry_bits: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for L in sizes:
        nbits, n_sites = _nbits_sites(L, dim)
        s_or = SingleParticleSparseIndexOracle(lattice_shape=(L,) * dim, r_loc=r_loc)
        t_or = TwoParticleSparseIndexOracle(lattice_shape=(L,) * dim, r_loc=r_loc, r_c=r_c)

        s_gc = get_cost_value(s_or, QECGatesCost())
        t_gc = get_cost_value(t_or, QECGatesCost())
        s_toff = int(s_gc.toffoli + s_gc.and_bloq + s_gc.cswap)
        s_cliff = int(s_gc.clifford)
        t_toff = int(t_gc.toffoli + t_gc.and_bloq + t_gc.cswap)
        t_cliff = int(t_gc.clifford)

        span_loc = _span(L, r_loc)
        span_c = _span(L, r_c)
        log_s_f = max(1, math.ceil(math.log2(span_loc**dim)))
        log_s_v = max(1, math.ceil(math.log2((span_c**dim) * (span_loc**dim))))

        ef_toff, ef_cliff, ef_rot = _entry_f_counts(nbits, dim, entry_bits)
        ev_toff, ev_cliff, ev_rot = _entry_v_counts(nbits, dim, entry_bits)

        f_toff = 2 * s_toff + ef_toff + 2 * log_s_f
        f_cliff = 2 * s_cliff + ef_cliff + nbits + 6 * log_s_f

        v_toff = 2 * t_toff + ev_toff + 2 * log_s_v
        v_cliff = 2 * t_cliff + ev_cliff + (2 * nbits) + 6 * log_s_v

        h_toff = 4 * f_toff + 10 * v_toff + 80
        h_cliff = 4 * f_cliff + 10 * v_cliff + 250
        h_rot = 4 * ef_rot + 10 * ev_rot

        logical_qubits = (2 * m) * nbits + (2 * nbits + 1) + 8
        rows.append(
            {
                "L": L,
                "N": n_sites,
                "nbits": nbits,
                "logical_qubits": logical_qubits,
                "H_toff_per_query": h_toff,
                "H_cliff_per_query": h_cliff,
                "H_rot_per_query": h_rot,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surface-code style resource estimates for exciton block-encoding."
    )
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--R-loc", type=int, default=5)
    parser.add_argument("--R-c", type=int, default=30)
    parser.add_argument("--entry-bits", type=int, default=10)
    parser.add_argument("--sizes", default="2,4,8,16,32")
    parser.add_argument("--query-repetitions", type=int, default=10_000)
    parser.add_argument("--phys-err", type=float, default=1e-3)
    parser.add_argument("--error-budget", type=float, default=1e-2)
    parser.add_argument("--cycle-time-us", type=float, default=1.0)
    parser.add_argument("--manual-l1", type=int, default=19)
    parser.add_argument("--manual-l2", type=int, default=31)
    parser.add_argument("--manual-factories", type=int, default=4)
    parser.add_argument("--manual-data-d", type=int, default=31)
    parser.add_argument("--routing-overhead", type=float, default=0.5)
    parser.add_argument("--grid-l1-start", type=int, default=7)
    parser.add_argument("--grid-l1-stop", type=int, default=21)
    parser.add_argument("--grid-l2-stop", type=int, default=35)
    parser.add_argument("--grid-d-start", type=int, default=7)
    parser.add_argument("--grid-d-stop", type=int, default=35)
    parser.add_argument("--outdir", default="experiments/plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    rows = compute_exciton_query_costs(
        sizes=sizes,
        m=args.m,
        dim=args.D,
        r_loc=args.R_loc,
        r_c=args.R_c,
        entry_bits=args.entry_bits,
    )

    manual_factory = MultiFactory(
        base_factory=CCZ2TFactory(
            distillation_l1_d=args.manual_l1, distillation_l2_d=args.manual_l2
        ),
        n_factories=args.manual_factories,
    )
    manual_data = SimpleDataBlock(data_d=args.manual_data_d, routing_overhead=args.routing_overhead)

    print(
        "L,N,n_algo_qubits,toff_per_query,queries,total_toffoli,"
        "manual_fail,manual_days,manual_mqubits,grid_fail,grid_days,grid_mqubits"
    )

    manual_days: list[float] = []
    manual_mqubits: list[float] = []
    grid_days: list[float] = []
    grid_mqubits: list[float] = []
    n_values: list[int] = []

    for r in rows:
        total_toff = int(r["H_toff_per_query"] * args.query_repetitions)
        n_logical = GateCounts(toffoli=total_toff)
        n_algo_qubits = int(r["logical_qubits"])

        manual = get_ccz2t_costs(
            n_logical_gates=n_logical,
            n_algo_qubits=n_algo_qubits,
            phys_err=args.phys_err,
            cycle_time_us=args.cycle_time_us,
            factory=manual_factory,
            data_block=manual_data,
        )

        try:
            factory_space = tuple(
                iter_ccz2t_factories(
                    l1_start=args.grid_l1_start,
                    l1_stop=args.grid_l1_stop,
                    l2_stop=args.grid_l2_stop,
                    n_factories=args.manual_factories,
                )
            )
            data_block_space = tuple(
                iter_simple_data_blocks(d_start=args.grid_d_start, d_stop=args.grid_d_stop)
            )
            grid, _, _ = get_ccz2t_costs_from_grid_search(
                n_logical_gates=n_logical,
                n_algo_qubits=n_algo_qubits,
                phys_err=args.phys_err,
                error_budget=args.error_budget,
                cycle_time_us=args.cycle_time_us,
                factory_iter=factory_space,
                data_block_iter=data_block_space,
                cost_function=(lambda pc: pc.duration_hr),
            )
            g_fail = grid.failure_prob
            g_days = grid.duration_hr / 24.0
            g_mq = grid.footprint / 1e6
        except ValueError:
            g_fail = float("nan")
            g_days = float("nan")
            g_mq = float("nan")

        m_days = manual.duration_hr / 24.0
        m_mq = manual.footprint / 1e6

        print(
            f"{int(r['L'])},{int(r['N'])},{n_algo_qubits},{int(r['H_toff_per_query'])},"
            f"{args.query_repetitions},{total_toff},"
            f"{manual.failure_prob:.6f},{m_days:.6f},{m_mq:.6f},"
            f"{g_fail:.6f},{g_days:.6f},{g_mq:.6f}"
        )

        n_values.append(int(r["N"]))
        manual_days.append(m_days)
        manual_mqubits.append(m_mq)
        grid_days.append(g_days)
        grid_mqubits.append(g_mq)

    n_arr = np.array(n_values, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_arr, manual_days, marker="o", label="manual params")
    ax.plot(n_arr, grid_days, marker="o", label="grid-search best (time)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("System size N = L^D")
    ax.set_ylabel("Wall time (days)")
    ax.set_title("Exciton Surface-Code Runtime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = outdir / "exciton_surface_code_runtime_days.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_arr, manual_mqubits, marker="o", label="manual params")
    ax.plot(n_arr, grid_mqubits, marker="o", label="grid-search best (time)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("System size N = L^D")
    ax.set_ylabel("Footprint (million physical qubits)")
    ax.set_title("Exciton Surface-Code Footprint")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = outdir / "exciton_surface_code_footprint_mqubits.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    print("\nSaved plots:")
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
