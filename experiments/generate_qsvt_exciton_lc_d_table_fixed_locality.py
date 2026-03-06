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

from block_encoding.exciton_hamiltonian_encoding import build_exciton_hamiltonian_block_encoding
from exciton.benchmark_tensors import generate_f_tensor, generate_v_tensor
from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code.gidney_fowler_model import get_ccz2t_costs


@dataclass
class Row:
    L_c: int
    d: int
    R_c: int
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


def _embed_two_particle_table(
    src: np.ndarray,
    *,
    D: int,
    src_r_c: int,
    src_r_loc: int,
    dst_r_c: int,
    dst_r_loc: int,
) -> np.ndarray:
    if src_r_c > dst_r_c or src_r_loc > dst_r_loc:
        raise ValueError("Cannot embed source table into a smaller destination locality window.")

    l_shape = src.shape[: 2 * D]
    dst_shape = (
        l_shape
        + (2 * dst_r_c + 1,) * D
        + (2 * dst_r_loc + 1,) * D
    )
    out = np.zeros(dst_shape, dtype=src.dtype)

    rc_start = dst_r_c - src_r_c
    rc_stop = rc_start + (2 * src_r_c + 1)
    rl_start = dst_r_loc - src_r_loc
    rl_stop = rl_start + (2 * src_r_loc + 1)

    sl = [slice(None)] * (2 * D)
    sl += [slice(rc_start, rc_stop)] * D
    sl += [slice(rl_start, rl_stop)] * D
    out[tuple(sl)] = src
    return out


def _randomize_nonzero_entries(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.array(x, copy=True)
    mask = y != 0
    if np.any(mask):
        y[mask] = rng.uniform(-1.0, 1.0, size=int(np.count_nonzero(mask)))
    return y


def _latex_table(rows: list[Row], *, m: int, D: int, f_r_loc: int, w_r_loc: int, v_r_loc: int, v_r_c: int, max_runtime_h: float, r_c: int) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        (
            "\\caption{QSVT-style exciton costs with independent $L_c$ and $d$, "
            f"$m={m}$, $D={D}$, fixed $R_c={r_c}$, "
            f"$R_{{\\mathrm{{loc}}}}^F={f_r_loc}$, $R_{{\\mathrm{{loc}}}}^W={w_r_loc}$, "
            f"$R_{{\\mathrm{{loc}}}}^V={v_r_loc}$, $R_c^V={v_r_c}$. "
            f"Filtered to logical qubits $\\leq 200$ and QSVT runtime $\\leq {max_runtime_h:g}$ h."
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--r-c", type=int, default=3, help="Global R_c used by W-oracle and block-encoding.")
    parser.add_argument("--f-r-loc", type=int, default=1)
    parser.add_argument("--w-r-loc", type=int, default=0)
    parser.add_argument("--v-r-loc", type=int, default=0)
    parser.add_argument("--v-r-c", type=int, default=0)
    parser.add_argument("--entry-bitsize", type=int, default=10)
    parser.add_argument("--max-logical", type=int, default=200)
    parser.add_argument("--max-runtime-h", type=float, default=24.0)
    parser.add_argument("--max-entries", type=int, default=50_000_000)
    parser.add_argument("--d-max", type=int, default=10)
    parser.add_argument("--l-max", type=int, default=256)
    parser.add_argument("--random-nonzero", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-name", type=str, default="qsvt_exciton_table_m2_d2_fixed_locality_lc_d.tex")
    args = parser.parse_args()

    m = int(args.m)
    D = int(args.D)
    r_c = int(args.r_c)
    f_r_loc = int(args.f_r_loc)
    w_r_loc = int(args.w_r_loc)
    v_r_loc = int(args.v_r_loc)
    v_r_c = int(args.v_r_c)
    entry_bitsize = int(args.entry_bitsize)
    max_logical = int(args.max_logical)
    max_runtime_h = float(args.max_runtime_h)
    max_entries = int(args.max_entries)
    d_max = int(args.d_max)
    l_max = int(args.l_max)
    random_nonzero = bool(args.random_nonzero)
    rng = np.random.default_rng(int(args.seed))

    global_r_loc = max(f_r_loc, w_r_loc, v_r_loc)
    if v_r_c > r_c:
        raise ValueError("v_r_c must be <= global r_c.")

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

        est_entries = (L_c ** (2 * D)) * ((2 * r_c + 1) ** D) * ((2 * global_r_loc + 1) ** D)
        if est_entries > max_entries:
            skipped_large_tensor += 1
            continue

        shape = (L_c,) * D
        F = generate_f_tensor(
            shape=shape,
            metric="chebyshev",
            r_cut=f_r_loc,
            oracle_convention="row_oracle",
        )
        if f_r_loc < global_r_loc:
            f_pad = np.zeros((shape + (2 * global_r_loc + 1,) * D), dtype=F.dtype)
            start = global_r_loc - f_r_loc
            stop = start + (2 * f_r_loc + 1)
            sl = [slice(None)] * D + [slice(start, stop)] * D
            f_pad[tuple(sl)] = F
            F = f_pad

        W_small = generate_v_tensor(
            shape=shape,
            metric="chebyshev",
            r_loc=w_r_loc,
            r_c=r_c,
            oracle_convention="exchange",
        )
        W = _embed_two_particle_table(
            W_small,
            D=D,
            src_r_c=r_c,
            src_r_loc=w_r_loc,
            dst_r_c=r_c,
            dst_r_loc=global_r_loc,
        )

        V_small = generate_v_tensor(
            shape=shape,
            metric="chebyshev",
            r_loc=v_r_loc,
            r_c=v_r_c,
            oracle_convention="direct",
        )
        V = _embed_two_particle_table(
            V_small,
            D=D,
            src_r_c=v_r_c,
            src_r_loc=v_r_loc,
            dst_r_c=r_c,
            dst_r_loc=global_r_loc,
        )

        if random_nonzero:
            F = _randomize_nonzero_entries(F, rng)
            W = _randomize_nonzero_entries(W, rng)
            V = _randomize_nonzero_entries(V, rng)
            # Current entry-oracle decomposition supports amplitudes in [0, 1] only.
            # Keep random sparsity/value spread while making values oracle-compatible.
            F = np.abs(F)
            W = np.abs(W)
            V = np.abs(V)

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
            R_c=r_c,
            R_loc=global_r_loc,
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
                    R_c=r_c,
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
        _latex_table(
            rows,
            m=m,
            D=D,
            f_r_loc=f_r_loc,
            w_r_loc=w_r_loc,
            v_r_loc=v_r_loc,
            v_r_c=v_r_c,
            max_runtime_h=max_runtime_h,
            r_c=r_c,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_path} with {len(rows)} rows.")
    print(f"Skipped {skipped_non_pow2} L_c values due to power-of-two requirement.")
    print(f"Skipped {skipped_large_tensor} L_c values due to tensor-size guard ({max_entries}).")


if __name__ == "__main__":
    main()
