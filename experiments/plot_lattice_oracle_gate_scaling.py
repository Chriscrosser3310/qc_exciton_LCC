from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from integrations.qualtran.block_encoding.lattice_index_oracles import (
    SingleParticleSparseIndexOracle,
    TwoParticleSparseIndexOracle,
)
from qualtran.resource_counting import QECGatesCost, get_cost_value


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _make_shape(linear_size: int, dim: int) -> tuple[int, ...]:
    return tuple([linear_size] * dim)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gate-count scaling for lattice sparse index oracles."
    )
    parser.add_argument("--dim", type=int, default=1, help="Lattice dimension D.")
    parser.add_argument(
        "--linear-sizes",
        default="8,16,32,64,128,256",
        help="Comma-separated linear lattice sizes per dimension.",
    )
    parser.add_argument("--r-loc", type=int, default=1, help="Local radius R_loc.")
    parser.add_argument("--r-c", type=int, default=2, help="Coupling radius R_c.")
    parser.add_argument(
        "--out",
        default="experiments/plots/lattice_oracle_gate_scaling.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    linear_sizes = _parse_csv_ints(args.linear_sizes)
    n_sites = []
    single_toff = []
    single_cliff = []
    two_toff = []
    two_cliff = []

    for l in linear_sizes:
        shape = _make_shape(l, args.dim)
        n = int(np.prod(shape))
        n_sites.append(n)

        s_oracle = SingleParticleSparseIndexOracle(lattice_shape=shape, r_loc=args.r_loc)
        t_oracle = TwoParticleSparseIndexOracle(lattice_shape=shape, r_loc=args.r_loc, r_c=args.r_c)

        s_gc = get_cost_value(s_oracle, QECGatesCost())
        t_gc = get_cost_value(t_oracle, QECGatesCost())

        single_toff.append(int(s_gc.toffoli + s_gc.and_bloq + s_gc.cswap))
        single_cliff.append(int(s_gc.clifford))
        two_toff.append(int(t_gc.toffoli + t_gc.and_bloq + t_gc.cswap))
        two_cliff.append(int(t_gc.clifford))

    x = np.array(n_sites, dtype=float)
    logn = np.log2(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, single_toff, marker="o", label="single toffoli-like")
    axes[0].plot(x, single_cliff, marker="o", label="single clifford")
    axes[0].plot(x, two_toff, marker="o", label="two-particle toffoli-like")
    axes[0].plot(x, two_cliff, marker="o", label="two-particle clifford")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("System size N (number of lattice sites)")
    axes[0].set_ylabel("Gate count")
    axes[0].set_title("Gate count vs N")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Show near-linear-in-logN trend explicitly.
    axes[1].plot(logn, single_toff, marker="o", label="single toffoli-like")
    axes[1].plot(logn, two_toff, marker="o", label="two-particle toffoli-like")
    axes[1].plot(logn, single_cliff, marker="o", label="single clifford")
    axes[1].plot(logn, two_cliff, marker="o", label="two-particle clifford")
    axes[1].set_xlabel("log2(N)")
    axes[1].set_ylabel("Gate count")
    axes[1].set_title("Gate count vs log2(N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Lattice index oracle scaling (D={args.dim}, R_loc={args.r_loc}, R_c={args.r_c})"
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print("Saved plot:", out)
    print("N:", n_sites)
    print("single_toffoli_like:", single_toff)
    print("two_toffoli_like:", two_toff)


if __name__ == "__main__":
    main()
