from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from block_encoding.qualtran_sparse_bench import build_all_sparse_oracles
from qualtran.resource_counting import QECGatesCost, get_cost_value


@dataclass(frozen=True)
class BenchPoint:
    x: float
    clifford: int
    toffoli: int
    t: int
    total_t_equiv: int
    build_time_s: float
    cost_time_s: float


def _gate_counts(bundle) -> tuple[int, int, int, int]:
    t0 = time.perf_counter()
    qec = get_cost_value(bundle.block_encoding, QECGatesCost())
    t1 = time.perf_counter()
    return (
        int(qec.clifford),
        int(qec.toffoli + qec.and_bloq + qec.cswap),
        int(qec.t),
        int(qec.total_t_count()),
        t1 - t0,
    )


def _bench_once(
    shape: tuple[int, ...],
    epsilon: float,
    bundle_key: str,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    metric: str = "euclidean",
) -> BenchPoint:
    t0 = time.perf_counter()
    bundles = build_all_sparse_oracles(
        shape=shape,
        epsilon=epsilon,
        a=a,
        b=b,
        c=c,
        metric=metric,
    )
    t1 = time.perf_counter()
    bundle = bundles[bundle_key]

    cliff, toff_like, tcount, t_equiv, cost_time = _gate_counts(bundle)
    return BenchPoint(
        x=float(shape[0] if len(shape) == 1 else np.prod(shape)),
        clifford=cliff,
        toffoli=toff_like,
        t=tcount,
        total_t_equiv=t_equiv,
        build_time_s=t1 - t0,
        cost_time_s=cost_time,
    )


def benchmark_size_sweep(
    sizes: list[int],
    epsilon: float,
    bundle_key: str,
    metric: str = "euclidean",
) -> list[BenchPoint]:
    points: list[BenchPoint] = []
    for n in sizes:
        p = _bench_once(shape=(n,), epsilon=epsilon, bundle_key=bundle_key, metric=metric)
        points.append(p)
    return points


def benchmark_epsilon_sweep(
    fixed_size: int,
    epsilons: list[float],
    bundle_key: str,
    metric: str = "euclidean",
) -> list[BenchPoint]:
    points: list[BenchPoint] = []
    for eps in epsilons:
        p = _bench_once(shape=(fixed_size,), epsilon=eps, bundle_key=bundle_key, metric=metric)
        points.append(
            BenchPoint(
                x=eps,
                clifford=p.clifford,
                toffoli=p.toffoli,
                t=p.t,
                total_t_equiv=p.total_t_equiv,
                build_time_s=p.build_time_s,
                cost_time_s=p.cost_time_s,
            )
        )
    return points


def _plot_points(
    points: list[BenchPoint],
    x_label: str,
    title: str,
    out_png: Path,
    x_log: bool = False,
    x_log_base: float = 10.0,
) -> None:
    x = np.array([p.x for p in points], dtype=float)
    cliff = np.array([p.clifford for p in points], dtype=float)
    toff = np.array([p.toffoli for p in points], dtype=float)
    teq = np.array([p.total_t_equiv for p in points], dtype=float)
    t_build = np.array([p.build_time_s for p in points], dtype=float)
    t_cost = np.array([p.cost_time_s for p in points], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, cliff, marker="o", label="clifford")
    axes[0].plot(x, toff, marker="o", label="toffoli-like")
    axes[0].plot(x, teq, marker="o", label="total T-equiv")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Gate count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, t_build, marker="o", label="build_oracles time")
    axes[1].plot(x, t_cost, marker="o", label="gate_count time")
    axes[1].plot(x, t_build + t_cost, marker="o", label="total time")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if x_log:
        axes[0].set_xscale("log", base=x_log_base)
        axes[1].set_xscale("log", base=x_log_base)

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gate count and runtime scaling.")
    parser.add_argument(
        "--bundle",
        choices=["F_pq", "FV_pq_rs", "FV_pr_qs"],
        default="F_pq",
        help="Which matrix layout to benchmark.",
    )
    parser.add_argument(
        "--size-list",
        default="4,8,16,32",
        help="Comma-separated 1D system sizes for the size sweep.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        help="Threshold epsilon for size sweep.",
    )
    parser.add_argument(
        "--fixed-size",
        type=int,
        default=16,
        help="Fixed 1D system size for epsilon sweep.",
    )
    parser.add_argument(
        "--eps-list",
        default="1e-1,5e-2,2e-2,1e-2,5e-3,1e-3",
        help="Comma-separated epsilons for epsilon sweep.",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "manhattan"],
        default="euclidean",
        help="Lattice distance metric.",
    )
    parser.add_argument(
        "--outdir",
        default="experiments/plots",
        help="Output directory for png plots.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    sizes = _parse_csv_ints(args.size_list)
    epsilons = _parse_csv_floats(args.eps_list)

    size_points = benchmark_size_sweep(
        sizes=sizes,
        epsilon=args.epsilon,
        bundle_key=args.bundle,
        metric=args.metric,
    )
    eps_points = benchmark_epsilon_sweep(
        fixed_size=args.fixed_size,
        epsilons=epsilons,
        bundle_key=args.bundle,
        metric=args.metric,
    )

    _plot_points(
        size_points,
        x_label="System size N (1D chain length)",
        title=f"{args.bundle}: Gate count and runtime vs system size (epsilon={args.epsilon:g})",
        out_png=outdir / f"{args.bundle}_size_sweep.png",
        x_log=True,
        x_log_base=2.0,
    )
    _plot_points(
        eps_points,
        x_label="Threshold epsilon",
        title=f"{args.bundle}: Gate count and runtime vs epsilon (N={args.fixed_size})",
        out_png=outdir / f"{args.bundle}_epsilon_sweep.png",
        x_log=True,
        x_log_base=10.0,
    )

    print("Generated plots:")
    print(outdir / f"{args.bundle}_size_sweep.png")
    print(outdir / f"{args.bundle}_epsilon_sweep.png")


if __name__ == "__main__":
    main()
