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

from integrations.qualtran.block_encoding.exciton_hamiltonian_encoding import build_exciton_hamiltonian_block_encoding
from exciton.benchmark_tensors import generate_f_tensor
from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code.gidney_fowler_model import get_ccz2t_costs


@dataclass
class Row:
    L_c: int
    d: int
    logical_qubits: int
    ancilla_qubits: int
    single_toffoli_like: int
    single_clifford: int
    single_rotation: int
    qsvt_toffoli_like: int
    qsvt_clifford: int
    qsvt_rotation: int
    qsvt_runtime_hours: float


def _logical_qubits(sig) -> int:
    return int(sum(int(r.total_bits()) for r in sig.lefts()))


def _system_qubits(*, m: int, D: int, L_c: int) -> int:
    n = int(np.ceil(np.log2(L_c)))
    return 2 * m * D * n


def _toffoli_like(qd: dict) -> int:
    return int(qd.get("toffoli", 0) + qd.get("and_bloq", 0) + qd.get("cswap", 0))


def _mc_rz_cost(control_size: int) -> GateCounts:
    k = max(0, int(control_size))
    toff = max(0, 2 * (k - 1))
    return GateCounts(toffoli=toff, clifford=0, rotation=1)


def _random_dense_diag_table(*, L: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Return M[i,j,m,l] with R_c=R_loc=0 and dense values over (i,j).

    Shape: (L,)*D + (L,)*D + (1,)*D + (1,)*D
    """
    ij_shape = (L,) * D + (L,) * D
    dense = rng.random(size=ij_shape, dtype=np.float64)
    return dense.reshape(ij_shape + (1,) * D + (1,) * D)


def _embed_wv_to_global_rloc1(src: np.ndarray, *, D: int) -> np.ndarray:
    """Embed src with (R_c=0,R_loc=0) into expected global shape (R_c=0,R_loc=1)."""
    l_shape = src.shape[: 2 * D]
    out = np.zeros(l_shape + (1,) * D + (3,) * D, dtype=src.dtype)
    # Put singleton local index at center of size-3 local window.
    sl = [slice(None)] * (2 * D) + [slice(0, 1)] * D + [slice(1, 2)] * D
    out[tuple(sl)] = src
    return out


def _latex_table(rows: list[Row], *, m: int, D: int, max_runtime_h: float, seed: int) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        (
            "\\caption{QSVT-style exciton costs with independent $L_c$ and $d$, "
            f"$m={m}$, $D={D}$. "
            "Settings: $F$ unchanged with $R_{\\mathrm{loc}}^F=1$; "
            "$W$ and $V$ use dense-diagonal data with $R_c=R_{\\mathrm{loc}}=0$ "
            "(entry-oracle tensor is dense over $(i,j)$ of size $L_c^D\\times L_c^D$, "
            "with singleton locality axes). "
            f"Random seed={seed}. Filtered to logical qubits $\\leq 200$ and QSVT runtime $\\leq {max_runtime_h:g}$ h."
            "}"
        ),
        "\\begin{tabularx}{\\linewidth}{rrrrrrrrrrr}",
        "\\toprule",
        "$L_c$ & $d$ & logical & ancilla & single Toff-like & single Cliff & single Rot & QSVT Toff-like & QSVT Cliff & QSVT Rot & QSVT runtime [h] \\\\",
        "\\midrule",
    ]
    if not rows:
        lines.append("\\multicolumn{11}{c}{No feasible points under current constraints.} \\\\")
    else:
        for r in rows:
            lines.append(
                f"{r.L_c} & {r.d} & {r.logical_qubits} & {r.ancilla_qubits} & "
                f"{r.single_toffoli_like} & {r.single_clifford} & {r.single_rotation} & "
                f"{r.qsvt_toffoli_like} & {r.qsvt_clifford} & {r.qsvt_rotation} & {r.qsvt_runtime_hours:.3f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--entry-bitsize", type=int, default=10)
    parser.add_argument("--max-logical", type=int, default=200)
    parser.add_argument("--max-runtime-h", type=float, default=24.0)
    parser.add_argument("--max-entries", type=int, default=100_000_000)
    parser.add_argument("--d-max", type=int, default=10)
    parser.add_argument("--l-max", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--out-name",
        type=str,
        default="qsvt_exciton_table_m2_d2_dense_diag_wv.tex",
    )
    args = parser.parse_args()

    m = int(args.m)
    D = int(args.D)
    entry_bitsize = int(args.entry_bitsize)
    max_logical = int(args.max_logical)
    max_runtime_h = float(args.max_runtime_h)
    max_entries = int(args.max_entries)
    d_max = int(args.d_max)
    l_max = int(args.l_max)
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    factory = MultiFactory(
        base_factory=CCZ2TFactory(distillation_l1_d=19, distillation_l2_d=31),
        n_factories=4,
    )
    data_block = SimpleDataBlock(data_d=31, routing_overhead=0.5)

    rows: list[Row] = []
    skipped_large_tensor = 0
    skipped_non_pow2 = 0

    for L_c in range(2, l_max + 1):
        if (L_c & (L_c - 1)) != 0:
            skipped_non_pow2 += 1
            continue

        # Dense-diagonal W/V data size over (i,j): L^(2D)
        est_entries = L_c ** (2 * D)
        if est_entries > max_entries:
            skipped_large_tensor += 1
            continue

        shape = (L_c,) * D
        F = generate_f_tensor(
            shape=shape,
            metric="chebyshev",
            r_cut=1,
            oracle_convention="row_oracle",
        )

        W0 = _random_dense_diag_table(L=L_c, D=D, rng=rng)
        V0 = _random_dense_diag_table(L=L_c, D=D, rng=rng)
        # Builder currently uses shared R_loc=1; embed W/V accordingly.
        W = _embed_wv_to_global_rloc1(W0, D=D)
        V = _embed_wv_to_global_rloc1(V0, D=D)

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
            R_c=0,
            R_loc=1,
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
        single_toff = _toffoli_like(sdict)
        single_cliff = int(sdict.get("clifford", 0))
        single_rot = int(sdict.get("rotation", 0))

        mc_rz = _mc_rz_cost(anc)
        for d in range(1, d_max + 1):
            qsvt_gc = d * single_gc + d * mc_rz
            qd = qsvt_gc.asdict() if hasattr(qsvt_gc, "asdict") else vars(qsvt_gc)
            qsvt_rt = get_ccz2t_costs(
                n_logical_gates=qsvt_gc,
                n_algo_qubits=lq,
                phys_err=1e-3,
                cycle_time_us=1.0,
                factory=factory,
                data_block=data_block,
            )
            qsvt_h = float(qsvt_rt.duration_hr)
            if qsvt_h > max_runtime_h:
                continue

            rows.append(
                Row(
                    L_c=L_c,
                    d=d,
                    logical_qubits=lq,
                    ancilla_qubits=anc,
                    single_toffoli_like=single_toff,
                    single_clifford=single_cliff,
                    single_rotation=single_rot,
                    qsvt_toffoli_like=_toffoli_like(qd),
                    qsvt_clifford=int(qd.get("clifford", 0)),
                    qsvt_rotation=int(qd.get("rotation", 0)),
                    qsvt_runtime_hours=qsvt_h,
                )
            )

    rows.sort(key=lambda r: (r.L_c, r.d, r.qsvt_runtime_hours))
    out_dir = REPO_ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    out_path.write_text(
        _latex_table(rows, m=m, D=D, max_runtime_h=max_runtime_h, seed=seed),
        encoding="utf-8",
    )

    print(f"Wrote {out_path} with {len(rows)} rows.")
    print(f"Skipped {skipped_non_pow2} L_c values due to power-of-two requirement.")
    print(f"Skipped {skipped_large_tensor} L_c values due to dense-table guard ({max_entries}).")


if __name__ == "__main__":
    main()

