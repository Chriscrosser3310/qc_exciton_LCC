from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qualtran.resource_counting import QECGatesCost, get_cost_value

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


def _entry_f_counts(nbits: int, dim: int, entry_bits: int) -> tuple[int, int, int]:
    # Proxy model for function-based QROM+rotation entry oracle for F.
    toff = 8 * dim * nbits + 4 * entry_bits
    cliff = 24 * dim * nbits + 12 * entry_bits
    rot = entry_bits
    return toff, cliff, rot


def _entry_v_counts(nbits: int, dim: int, entry_bits: int) -> tuple[int, int, int]:
    # Proxy model for function-based QROM+rotation entry oracle for V.
    toff = 20 * dim * nbits + 6 * entry_bits
    cliff = 60 * dim * nbits + 18 * entry_bits
    rot = entry_bits
    return toff, cliff, rot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count exciton block-encoding gate complexity with fixed m, D, R_loc, R_c."
    )
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--R-loc", type=int, default=5)
    parser.add_argument("--R-c", type=int, default=30)
    parser.add_argument("--entry-bits", type=int, default=10)
    parser.add_argument("--qubit-limit", type=int, default=100)
    parser.add_argument("--sizes", default="2,4,8,16,32,64")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    header = (
        "L,N,nbits,logical_qubits,"
        "F_toff,F_cliff,F_rot,"
        "V_toff,V_cliff,V_rot,"
        "H_toff,H_cliff,H_rot"
    )
    print(header)

    for L in sizes:
        nbits, n_sites = _nbits_sites(L, args.D)

        s_or = SingleParticleSparseIndexOracle(lattice_shape=(L,) * args.D, r_loc=args.R_loc)
        t_or = TwoParticleSparseIndexOracle(
            lattice_shape=(L,) * args.D, r_loc=args.R_loc, r_c=args.R_c
        )

        s_gc = get_cost_value(s_or, QECGatesCost())
        t_gc = get_cost_value(t_or, QECGatesCost())
        s_toff = int(s_gc.toffoli + s_gc.and_bloq + s_gc.cswap)
        s_cliff = int(s_gc.clifford)
        t_toff = int(t_gc.toffoli + t_gc.and_bloq + t_gc.cswap)
        t_cliff = int(t_gc.clifford)

        span_loc = _span(L, args.R_loc)
        span_c = _span(L, args.R_c)
        log_s_f = max(1, math.ceil(math.log2(span_loc**args.D)))
        log_s_v = max(1, math.ceil(math.log2((span_c**args.D) * (span_loc**args.D))))

        ef_toff, ef_cliff, ef_rot = _entry_f_counts(nbits, args.D, args.entry_bits)
        ev_toff, ev_cliff, ev_rot = _entry_v_counts(nbits, args.D, args.entry_bits)

        # SparseMatrix(F) proxy: row + col + entry + swap + diffusion + diffusion^\dagger
        F_toff = 2 * s_toff + ef_toff + 2 * log_s_f
        F_cliff = 2 * s_cliff + ef_cliff + nbits + 6 * log_s_f
        F_rot = ef_rot

        # SparseMatrix(V) proxy (direct/exchange same structural cost):
        V_toff = 2 * t_toff + ev_toff + 2 * log_s_v
        V_cliff = 2 * t_cliff + ev_cliff + (2 * nbits) + 6 * log_s_v
        V_rot = ev_rot

        # m=2 exciton Hamiltonian term multiplicities:
        # F terms: 2m = 4
        # V_direct terms: C(m,2)+C(m,2)+m^2 = 1+1+4 = 6
        # V_exchange terms: m^2 = 4
        H_toff = 4 * F_toff + 10 * V_toff
        H_cliff = 4 * F_cliff + 10 * V_cliff
        H_rot = 4 * F_rot + 10 * V_rot

        # Small LCU overhead proxy for 14 terms.
        H_toff += 80
        H_cliff += 250

        # Logical qubit proxy for exciton encoding:
        # system=2m*nbits, ancilla~2*nbits+1 (V term), LCU select/junk ~8.
        logical_qubits = (2 * args.m) * nbits + (2 * nbits + 1) + 8

        row = (
            L,
            n_sites,
            nbits,
            logical_qubits,
            F_toff,
            F_cliff,
            F_rot,
            V_toff,
            V_cliff,
            V_rot,
            H_toff,
            H_cliff,
            H_rot,
        )
        print(",".join(str(x) for x in row))

    print("\nFiltered (logical_qubits <= limit):")
    for L in sizes:
        nbits, n_sites = _nbits_sites(L, args.D)
        logical_qubits = (2 * args.m) * nbits + (2 * nbits + 1) + 8
        if logical_qubits <= args.qubit_limit:
            print(f"L={L}, N={n_sites}, nbits={nbits}, logical_qubits={logical_qubits}")


if __name__ == "__main__":
    main()
