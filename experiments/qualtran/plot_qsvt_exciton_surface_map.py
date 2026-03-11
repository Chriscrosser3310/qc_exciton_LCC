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
from qualtran.surface_code.gidney_fowler_model import get_ccz2t_costs

from integrations.qualtran.block_encoding.lattice_index_oracles import (
    SingleParticleSparseIndexOracle,
    TwoParticleSparseIndexOracle,
)


def _nbits_sites(linear_size: int, dim: int) -> tuple[int, int]:
    n_sites = linear_size**dim
    nbits = max(1, math.ceil(math.log2(n_sites)))
    return nbits, n_sites


def _span(linear_size: int, radius: int) -> int:
    return min(linear_size, 2 * radius + 1)


def _entry_f_counts(
    nbits: int,
    dim: int,
    entry_bits: int,
    *,
    n_sites: int,
    full_data_loading: bool,
) -> tuple[int, int, int]:
    if full_data_loading:
        # Full data-loading model:
        # QROM-style access over a dense table of F_{pq} entries (size n_sites^2).
        table_size = n_sites * n_sites
        toff = table_size * entry_bits
        cliff = 2 * table_size * entry_bits
        rot = entry_bits
        return toff, cliff, rot
    toff = 8 * dim * nbits + 4 * entry_bits
    cliff = 24 * dim * nbits + 12 * entry_bits
    rot = entry_bits
    return toff, cliff, rot


def _entry_v_counts(
    nbits: int,
    dim: int,
    entry_bits: int,
    *,
    n_sites: int,
    full_data_loading: bool,
) -> tuple[int, int, int]:
    if full_data_loading:
        # Keep same leading scaling as F entry model so asymptotics track N^2 data loading.
        table_size = n_sites * n_sites
        toff = table_size * entry_bits
        cliff = 2 * table_size * entry_bits
        rot = entry_bits
        return toff, cliff, rot
    toff = 20 * dim * nbits + 6 * entry_bits
    cliff = 60 * dim * nbits + 18 * entry_bits
    rot = entry_bits
    return toff, cliff, rot


def _exciton_block_toffoli_per_query(
    *,
    L: int,
    m: int,
    dim: int,
    r_loc: int,
    r_c: int,
    entry_bits: int,
    full_data_loading: bool,
) -> tuple[int, int]:
    nbits, n_sites = _nbits_sites(L, dim)

    s_or = SingleParticleSparseIndexOracle(lattice_shape=(L,) * dim, r_loc=r_loc)
    t_or = TwoParticleSparseIndexOracle(lattice_shape=(L,) * dim, r_loc=r_loc, r_c=r_c)
    s_gc = get_cost_value(s_or, QECGatesCost())
    t_gc = get_cost_value(t_or, QECGatesCost())

    s_toff = int(s_gc.toffoli + s_gc.and_bloq + s_gc.cswap)
    t_toff = int(t_gc.toffoli + t_gc.and_bloq + t_gc.cswap)

    span_loc = _span(L, r_loc)
    span_c = _span(L, r_c)
    log_s_f = max(1, math.ceil(math.log2(span_loc**dim)))
    log_s_v = max(1, math.ceil(math.log2((span_c**dim) * (span_loc**dim))))

    ef_toff, _, _ = _entry_f_counts(
        nbits, dim, entry_bits, n_sites=n_sites, full_data_loading=full_data_loading
    )
    ev_toff, _, _ = _entry_v_counts(
        nbits, dim, entry_bits, n_sites=n_sites, full_data_loading=full_data_loading
    )

    f_toff = 2 * s_toff + ef_toff + 2 * log_s_f
    v_toff = 2 * t_toff + ev_toff + 2 * log_s_v

    h_toff = 4 * f_toff + 10 * v_toff + 80
    logical_qubits = (2 * m) * nbits + (2 * nbits + 1) + 8
    return h_toff, logical_qubits


def _save_heatmap(
    data: np.ndarray,
    *,
    xvals: np.ndarray,
    yvals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    cbar_label: str,
    outpath: Path,
    log_color: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    z = np.log10(np.maximum(data, 1e-30)) if log_color else data
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(("log10 " + cbar_label) if log_color else cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D QSVT + exciton block-encoding cost maps.")
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--R-loc", type=int, default=5)
    parser.add_argument("--entry-bits", type=int, default=10)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--d-max", type=int, default=10)
    parser.add_argument("--Rc-max", type=int, default=30)
    parser.add_argument("--phys-err", type=float, default=1e-3)
    parser.add_argument("--cycle-time-us", type=float, default=1.0)
    parser.add_argument("--manual-l1", type=int, default=19)
    parser.add_argument("--manual-l2", type=int, default=31)
    parser.add_argument("--manual-factories", type=int, default=4)
    parser.add_argument("--manual-data-d", type=int, default=31)
    parser.add_argument("--routing-overhead", type=float, default=0.5)
    parser.add_argument("--outdir", default="experiments/plots")
    parser.add_argument(
        "--full-data-loading",
        action="store_true",
        default=True,
        help="Use full data-loading model for entry oracle costs.",
    )
    parser.add_argument(
        "--no-full-data-loading",
        dest="full_data_loading",
        action="store_false",
        help="Use lightweight proxy model for entry oracle costs.",
    )
    parser.add_argument(
        "--fit-asymptotic",
        action="store_true",
        help="Fit log(cost) ~ a + alpha*log(d) + beta*log(R_c) and report exponents.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d_vals = np.arange(1, args.d_max + 1, dtype=int)
    rc_vals = np.arange(1, args.Rc_max + 1, dtype=int)

    block_toff = np.zeros((len(d_vals), len(rc_vals)), dtype=float)
    qsvt_toff = np.zeros_like(block_toff)
    runtime_hours = np.zeros_like(block_toff)
    mqubits = np.zeros_like(block_toff)
    logical_qubits = np.zeros_like(block_toff)
    side_lengths = np.zeros_like(block_toff)

    manual_factory = MultiFactory(
        base_factory=CCZ2TFactory(
            distillation_l1_d=args.manual_l1, distillation_l2_d=args.manual_l2
        ),
        n_factories=args.manual_factories,
    )
    manual_data = SimpleDataBlock(data_d=args.manual_data_d, routing_overhead=args.routing_overhead)

    for i, d in enumerate(d_vals):
        for j, r_c in enumerate(rc_vals):
            L = max(2, int(round(args.C * d * r_c)))
            h_toff, n_algo_qubits = _exciton_block_toffoli_per_query(
                L=L,
                m=args.m,
                dim=args.D,
                r_loc=args.R_loc,
                r_c=int(r_c),
                entry_bits=args.entry_bits,
                full_data_loading=args.full_data_loading,
            )
            total_toff = int(d * h_toff)  # QSVT proxy: d queries to block-encoding.
            pcost = get_ccz2t_costs(
                n_logical_gates=GateCounts(toffoli=total_toff),
                n_algo_qubits=int(n_algo_qubits),
                phys_err=args.phys_err,
                cycle_time_us=args.cycle_time_us,
                factory=manual_factory,
                data_block=manual_data,
            )

            block_toff[i, j] = h_toff
            qsvt_toff[i, j] = total_toff
            runtime_hours[i, j] = pcost.duration_hr
            mqubits[i, j] = pcost.footprint / 1e6
            logical_qubits[i, j] = n_algo_qubits
            side_lengths[i, j] = L

    _save_heatmap(
        block_toff,
        xvals=rc_vals,
        yvals=d_vals,
        xlabel="R_c",
        ylabel="QSVT iteration d",
        title=f"Exciton Block-Encoding Toffoli/query (m={args.m}, D={args.D}, R_loc={args.R_loc})",
        cbar_label="Toffoli/query",
        outpath=outdir / "qsvt_block_toffoli_per_query_heatmap.png",
        log_color=True,
    )
    _save_heatmap(
        qsvt_toff,
        xvals=rc_vals,
        yvals=d_vals,
        xlabel="R_c",
        ylabel="QSVT iteration d",
        title=f"QSVT Total Toffoli (L = C*d*R_c, C={args.C:g})",
        cbar_label="Total Toffoli",
        outpath=outdir / "qsvt_total_toffoli_heatmap.png",
        log_color=True,
    )
    _save_heatmap(
        runtime_hours,
        xvals=rc_vals,
        yvals=d_vals,
        xlabel="R_c",
        ylabel="QSVT iteration d",
        title="QSVT Surface-Code Runtime (hours)",
        cbar_label="Runtime (hours)",
        outpath=outdir / "qsvt_runtime_hours_heatmap.png",
        log_color=True,
    )
    _save_heatmap(
        mqubits,
        xvals=rc_vals,
        yvals=d_vals,
        xlabel="R_c",
        ylabel="QSVT iteration d",
        title="QSVT Physical Footprint (million qubits)",
        cbar_label="Million physical qubits",
        outpath=outdir / "qsvt_footprint_mqubits_heatmap.png",
        log_color=False,
    )
    _save_heatmap(
        logical_qubits,
        xvals=rc_vals,
        yvals=d_vals,
        xlabel="R_c",
        ylabel="QSVT iteration d",
        title="Logical Qubits Proxy",
        cbar_label="Logical qubits",
        outpath=outdir / "qsvt_logical_qubits_heatmap.png",
        log_color=False,
    )
    _save_heatmap(
        side_lengths,
        xvals=rc_vals,
        yvals=d_vals,
        xlabel="R_c",
        ylabel="QSVT iteration d",
        title=f"Side Length L = round(C*d*R_c), C={args.C:g}",
        cbar_label="L",
        outpath=outdir / "qsvt_side_length_heatmap.png",
        log_color=False,
    )

    print("Saved plots:")
    print(outdir / "qsvt_block_toffoli_per_query_heatmap.png")
    print(outdir / "qsvt_total_toffoli_heatmap.png")
    print(outdir / "qsvt_runtime_hours_heatmap.png")
    print(outdir / "qsvt_footprint_mqubits_heatmap.png")
    print(outdir / "qsvt_logical_qubits_heatmap.png")
    print(outdir / "qsvt_side_length_heatmap.png")

    i_max, j_max = np.unravel_index(np.argmax(qsvt_toff), qsvt_toff.shape)
    print(
        "\nMax point on grid:",
        f"d={d_vals[i_max]}, R_c={rc_vals[j_max]},",
        f"L={int(side_lengths[i_max, j_max])},",
        f"QSVT_Toffoli={int(qsvt_toff[i_max, j_max])},",
        f"runtime_hours={runtime_hours[i_max, j_max]:.6f},",
        f"mqubits={mqubits[i_max, j_max]:.6f}",
    )

    if args.fit_asymptotic:
        dd, rr = np.meshgrid(d_vals.astype(float), rc_vals.astype(float), indexing="ij")
        x = np.column_stack([np.ones(dd.size), np.log(dd.ravel()), np.log(rr.ravel())])

        def _fit(z: np.ndarray) -> tuple[float, float]:
            y = np.log(np.maximum(z.ravel(), 1e-30))
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            return float(coef[1]), float(coef[2])

        a_qsvt, b_qsvt = _fit(qsvt_toff)
        a_time, b_time = _fit(runtime_hours)
        a_mq, b_mq = _fit(mqubits)
        print("\nScaling fit on full grid: cost ~ d^alpha * R_c^beta")
        print(f"QSVT total Toffoli: alpha={a_qsvt:.4f}, beta={b_qsvt:.4f}")
        print(f"Runtime hours:      alpha={a_time:.4f}, beta={b_time:.4f}")
        print(f"Mqubits:            alpha={a_mq:.4f}, beta={b_mq:.4f}")
        print(
            f"Target for D={args.D}: alpha=2D+1={2*args.D+1}, beta=2D={2*args.D}"
        )


if __name__ == "__main__":
    main()
