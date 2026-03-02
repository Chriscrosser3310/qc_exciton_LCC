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

from qualtran.resource_counting import QECGatesCost, get_cost_value
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


def compute_rows(
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

        F_toff = 2 * s_toff + ef_toff + 2 * log_s_f
        F_cliff = 2 * s_cliff + ef_cliff + nbits + 6 * log_s_f
        F_rot = ef_rot

        V_toff = 2 * t_toff + ev_toff + 2 * log_s_v
        V_cliff = 2 * t_cliff + ev_cliff + (2 * nbits) + 6 * log_s_v
        V_rot = ev_rot

        H_toff = 4 * F_toff + 10 * V_toff + 80
        H_cliff = 4 * F_cliff + 10 * V_cliff + 250
        H_rot = 4 * F_rot + 10 * V_rot

        logical_qubits = (2 * m) * nbits + (2 * nbits + 1) + 8
        rows.append(
            {
                "L": L,
                "N": n_sites,
                "nbits": nbits,
                "logical_qubits": logical_qubits,
                "F_toff": F_toff,
                "F_cliff": F_cliff,
                "V_toff": V_toff,
                "V_cliff": V_cliff,
                "H_toff": H_toff,
                "H_cliff": H_cliff,
                "H_rot": H_rot,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exciton block-encoding complexity.")
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--R-loc", type=int, default=5)
    parser.add_argument("--R-c", type=int, default=30)
    parser.add_argument("--entry-bits", type=int, default=10)
    parser.add_argument("--sizes", default="2,4,8,16,32,64")
    parser.add_argument("--qubit-limit", type=int, default=100)
    parser.add_argument("--outdir", default="experiments/plots")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    rows = compute_rows(sizes, args.m, args.D, args.R_loc, args.R_c, args.entry_bits)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N = np.array([r["N"] for r in rows], dtype=float)
    logical = np.array([r["logical_qubits"] for r in rows], dtype=float)
    H_toff = np.array([r["H_toff"] for r in rows], dtype=float)
    H_cliff = np.array([r["H_cliff"] for r in rows], dtype=float)
    H_rot = np.array([r["H_rot"] for r in rows], dtype=float)
    F_toff = np.array([r["F_toff"] for r in rows], dtype=float)
    V_toff = np.array([r["V_toff"] for r in rows], dtype=float)

    # Plot 1: total Hamiltonian complexity.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(N, H_toff, marker="o", label="H toffoli-like")
    ax.plot(N, H_cliff, marker="o", label="H clifford")
    ax.plot(N, H_rot, marker="o", label="H rotations")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("System size N = L^D")
    ax.set_ylabel("Gate count")
    ax.set_title(f"Exciton Hamiltonian Encoding (m={args.m}, D={args.D})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = outdir / "exciton_hamiltonian_gate_counts.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Plot 2: F vs V building block toffoli-like.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(N, F_toff, marker="o", label="F block-encoding toffoli-like")
    ax.plot(N, V_toff, marker="o", label="V block-encoding toffoli-like")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("System size N = L^D")
    ax.set_ylabel("Toffoli-like count")
    ax.set_title("F vs V Block-Encoding Complexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = outdir / "exciton_F_vs_V_toffoli.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # Plot 3: logical qubits and 100-qubit line.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(N, logical, marker="o", label="logical qubits")
    ax.axhline(args.qubit_limit, color="red", linestyle="--", label=f"limit={args.qubit_limit}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("System size N = L^D")
    ax.set_ylabel("Logical qubits (proxy)")
    ax.set_title("Logical Qubit Budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p3 = outdir / "exciton_logical_qubits.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    print("Saved plots:")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
