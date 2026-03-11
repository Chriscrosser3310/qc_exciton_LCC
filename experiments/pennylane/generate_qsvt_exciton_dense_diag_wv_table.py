from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import comb
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pennylane as qml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from integrations.pennylane.block_encoding import exciton_block_encoding
from integrations.pennylane.resource_estimation import compiled_counts_from_qnode
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code.gidney_fowler_model import get_ccz2t_costs


@dataclass
class Row:
    L_c: int
    d: int
    logical_qubits: int
    ancilla_qubits: int
    compiled_t_count: int | None
    compiled_toffoli_count: int | None
    compiled_clifford_count: int | None
    compiled_rotation_count: int | None
    qsvt_t_count: int | None
    qsvt_toffoli_count: int | None
    qsvt_runtime_hours: float | None


def _allocate_wires(*, m: int, D: int, L: int, R_c: int, R_loc: int, angle_bits: int):
    n_idx = int(np.ceil(np.log2(L)))
    # Use index-width local registers for arithmetic compatibility in current PennyLane index oracles.
    n_fm = max(n_idx, max(1, int(np.ceil(np.log2(2 * R_loc + 1)))))
    n_tm = max(n_idx, max(1, int(np.ceil(np.log2(2 * R_c + 1)))))
    n_tl = max(n_idx, max(1, int(np.ceil(np.log2(2 * R_loc + 1)))))

    n_regs = 2 * m
    h_sel_bits = max(1, int(np.ceil(np.log2(3))))
    f_sel_bits = max(1, int(np.ceil(np.log2(n_regs))))
    w_sel_bits = max(1, int(np.ceil(np.log2(int(comb(n_regs, 2))))))
    v_sel_bits = max(1, int(np.ceil(np.log2(m * m))))

    w = 0
    particle = []
    for _ in range(n_regs):
        reg = []
        for _ in range(D):
            reg.append(list(range(w, w + n_idx)))
            w += n_idx
        particle.append(reg)

    h_sel = list(range(w, w + h_sel_bits)); w += h_sel_bits
    f_sel = list(range(w, w + f_sel_bits)); w += f_sel_bits
    w_sel = list(range(w, w + w_sel_bits)); w += w_sel_bits
    v_sel = list(range(w, w + v_sel_bits)); w += v_sel_bits

    def alloc_dim(bits: int):
        nonlocal w
        out = []
        for _ in range(D):
            out.append(list(range(w, w + bits)))
            w += bits
        return out

    f_m = alloc_dim(n_fm)
    w_m = alloc_dim(n_tm)
    w_l = alloc_dim(n_tl)
    v_m = alloc_dim(n_tm)
    v_l = alloc_dim(n_tl)

    anc = w; w += 1
    angle = list(range(w, w + angle_bits)); w += angle_bits

    work_bits = max(0, n_idx - 1)
    index_work = list(range(w, w + work_bits)); w += work_bits
    return {
        "particle_registers": particle,
        "h_sel_wires": h_sel,
        "f_sel_wires": f_sel,
        "w_sel_wires": w_sel,
        "v_sel_wires": v_sel,
        "f_m_wires": f_m,
        "w_m_wires": w_m,
        "w_l_wires": w_l,
        "v_m_wires": v_m,
        "v_l_wires": v_l,
        "ancilla_wire": anc,
        "angle_wires": angle,
        "index_work_wires": index_work,
        "n_wires": w,
    }


def _build_tables(*, L: int, D: int, R_c: int, R_loc: int, seed: int):
    rng = np.random.default_rng(seed + L)
    n_i = L**D
    n_m = (2 * R_loc + 1) ** D
    n_row = (L**D) * (L**D)
    n_col = ((2 * R_c + 1) ** D) * ((2 * R_loc + 1) ** D)

    # F unchanged-style (dense on local neighborhood column-space)
    F = rng.uniform(-1.0, 1.0, size=(n_i, n_m))
    F = np.abs(F)
    if np.max(F) > 0:
        F = F / np.max(F)

    # W/V dense-diagonal equivalent with effective Rc=Rloc=0 embedded in global Rc,Rloc.
    W = np.zeros((n_row, n_col), dtype=float)
    V = np.zeros((n_row, n_col), dtype=float)
    center_col = n_col // 2
    W[:, center_col] = rng.uniform(0.0, 1.0, size=n_row)
    V[:, center_col] = rng.uniform(0.0, 1.0, size=n_row)
    return F, W, V


def _build_qnode(*, m: int, D: int, L: int, R_c: int, R_loc: int, angle_bits: int, seed: int):
    wires = _allocate_wires(m=m, D=D, L=L, R_c=R_c, R_loc=R_loc, angle_bits=angle_bits)
    F, W, V = _build_tables(L=L, D=D, R_c=R_c, R_loc=R_loc, seed=seed)
    dev = qml.device("default.qubit", wires=wires["n_wires"])

    @qml.qnode(dev)
    def circuit():
        exciton_block_encoding(
            F_table=F,
            W_table=W,
            V_table=V,
            particle_registers=wires["particle_registers"],
            h_sel_wires=wires["h_sel_wires"],
            f_sel_wires=wires["f_sel_wires"],
            w_sel_wires=wires["w_sel_wires"],
            v_sel_wires=wires["v_sel_wires"],
            f_m_wires=wires["f_m_wires"],
            w_m_wires=wires["w_m_wires"],
            w_l_wires=wires["w_l_wires"],
            v_m_wires=wires["v_m_wires"],
            v_l_wires=wires["v_l_wires"],
            ancilla_wire=wires["ancilla_wire"],
            angle_wires=wires["angle_wires"],
            L=L,
            R_c=R_c,
            R_loc=R_loc,
            index_work_wires=wires["index_work_wires"],
            entry_work_wires=[],
        )
        return qml.probs(wires=[wires["ancilla_wire"]])

    return circuit, wires


def _compiled_counts_worker(queue: mp.Queue, kwargs: dict) -> None:
    try:
        circuit, _ = _build_qnode(**kwargs)
        compiled = compiled_counts_from_qnode(circuit)
        queue.put(compiled)
    except Exception as exc:  # pragma: no cover - subprocess error path
        queue.put(exc)


def _compiled_counts_with_timeout(timeout_s: float | None, **kwargs):
    if timeout_s is None or timeout_s <= 0:
        circuit, _ = _build_qnode(**kwargs)
        return compiled_counts_from_qnode(circuit)

    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_compiled_counts_worker, args=(queue, kwargs))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    if queue.empty():
        return None
    result = queue.get()
    if isinstance(result, Exception):
        return None
    return result


def _single_resources(
    *,
    m: int,
    D: int,
    L: int,
    R_c: int,
    R_loc: int,
    angle_bits: int,
    seed: int,
    compile_timeout_s: float | None,
):
    _, wires = _build_qnode(m=m, D=D, L=L, R_c=R_c, R_loc=R_loc, angle_bits=angle_bits, seed=seed)
    t0 = time.perf_counter()
    compiled = _compiled_counts_with_timeout(
        compile_timeout_s,
        m=m,
        D=D,
        L=L,
        R_c=R_c,
        R_loc=R_loc,
        angle_bits=angle_bits,
        seed=seed,
    )
    build_time = time.perf_counter() - t0
    return (
        int(wires["n_wires"]),
        compiled,
        float(build_time),
    )


def _system_qubits(*, m: int, D: int, L_c: int) -> int:
    n = int(np.ceil(np.log2(L_c)))
    return 2 * m * D * n


def _mc_rz_cost(control_size: int) -> GateCounts:
    k = max(0, int(control_size))
    toff = max(0, 2 * (k - 1))
    return GateCounts(toffoli=toff, clifford=0, rotation=1)


def _latex(rows: list[Row], *, max_runtime_h: float, d_max: int) -> str:
    def fmt(x):
        if x is None:
            return "--"
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        (
            "\\caption{QSVT-style exciton costs with independent $L_c$ and $d$, $m = 2$, $D = 2$. "
            "Settings: $F$ unchanged with $R_{\\mathrm{loc}}^F = 1$; $W$ and $V$ use dense-diagonal "
            "data with $R_c = R_{\\mathrm{loc}} = 0$ (entry-oracle tensor is dense over $(i, j)$ of size "
            "$L_c^D \\times L_c^D$, with singleton locality axes). Random seed=12345. "
            f"Filtered to logical qubits $\\leq 200$ and $d \\leq {d_max}$"
            + (
                f", with QSVT runtime $\\leq {max_runtime_h:g}$ h."
                if max_runtime_h < 1.0e8
                else ", with no QSVT runtime cutoff."
            )
            + "}"
        ),
        "\\begin{tabularx}{\\linewidth}{rrrrrrrrr}",
        "\\toprule",
        "$L_c$ & $d$ & logical & ancilla & single Toffoli & single $T$ & single Cliff & QSVT Toffoli & QSVT $T$ & QSVT runtime [h] \\\\",
        "\\midrule",
    ]
    if not rows:
        lines.append("\\multicolumn{10}{c}{No feasible points under current constraints.} \\\\")
    else:
        for r in rows:
            lines.append(
                f"{r.L_c} & {r.d} & {r.logical_qubits} & {r.ancilla_qubits} & "
                f"{fmt(r.compiled_toffoli_count)} & {fmt(r.compiled_t_count)} & {fmt(r.compiled_clifford_count)} & "
                f"{fmt(r.qsvt_toffoli_count)} & {fmt(r.qsvt_t_count)} & {fmt(r.qsvt_runtime_hours)} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l-max", type=int, default=64)
    parser.add_argument("--d-max", type=int, default=9)
    parser.add_argument("--max-qubits", type=int, default=200)
    parser.add_argument("--max-runtime-h", type=float, default=24.0)
    parser.add_argument("--max-build-time-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--angle-bits", type=int, default=10)
    parser.add_argument("--compile-timeout-s", type=float, default=60.0)
    parser.add_argument("--out-name", type=str, default="pennylane_qsvt_exciton_table_m2_d2_dense_diag_wv.tex")
    args = parser.parse_args()

    m, D, R_c, R_loc = 2, 2, 0, 1
    rows: list[Row] = []
    factory = MultiFactory(
        base_factory=CCZ2TFactory(distillation_l1_d=19, distillation_l2_d=31),
        n_factories=4,
    )
    data_block = SimpleDataBlock(data_d=31, routing_overhead=0.5)
    runtime_limit_hit = False

    for L in range(2, int(args.l_max) + 1):
        if runtime_limit_hit:
            break
        if (L & (L - 1)) != 0:
            continue
        try:
            lq, compiled, build_time = _single_resources(
                m=m,
                D=D,
                L=L,
                R_c=R_c,
                R_loc=R_loc,
                angle_bits=int(args.angle_bits),
                seed=int(args.seed),
                compile_timeout_s=float(args.compile_timeout_s),
            )
        except Exception:
            continue
        if lq > int(args.max_qubits):
            continue
        if build_time > float(args.max_build_time_s):
            # Larger L_c points are expected to be slower; stop the sweep.
            break
        anc = lq - _system_qubits(m=m, D=D, L_c=L)
        single_gc = None
        mc_rz = _mc_rz_cost(anc)
        if compiled is not None:
            single_gc = GateCounts(
                toffoli=int(compiled.toffoli_count),
                clifford=int(compiled.clifford_count),
                rotation=int(compiled.rotation_count),
            )

        for d in range(1, int(args.d_max) + 1):
            qsvt_toff = None
            qsvt_t = None
            qsvt_h = None
            if single_gc is not None:
                qsvt_gc = d * single_gc + d * mc_rz
                qsvt_rt = get_ccz2t_costs(
                    n_logical_gates=qsvt_gc,
                    n_algo_qubits=lq,
                    phys_err=1e-3,
                    cycle_time_us=1.0,
                    factory=factory,
                    data_block=data_block,
                )
                qsvt_h = float(qsvt_rt.duration_hr)
                qsvt_toff = int(qsvt_gc.toffoli)
                qsvt_t = int(d * compiled.total_t)
                if qsvt_h > float(args.max_runtime_h):
                    if d == 1:
                        runtime_limit_hit = True
                    continue

            rows.append(
                Row(
                    L_c=L,
                    d=d,
                    logical_qubits=lq,
                    ancilla_qubits=anc,
                    compiled_t_count=None if compiled is None else int(compiled.total_t),
                    compiled_toffoli_count=None if compiled is None else int(compiled.toffoli_count),
                    compiled_clifford_count=None if compiled is None else int(compiled.clifford_count),
                    compiled_rotation_count=None if compiled is None else int(compiled.rotation_count),
                    qsvt_t_count=qsvt_t,
                    qsvt_toffoli_count=qsvt_toff,
                    qsvt_runtime_hours=qsvt_h,
                )
            )

    rows.sort(key=lambda r: (r.L_c, r.d))
    out_dir = REPO_ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    out_path.write_text(
        _latex(rows, max_runtime_h=float(args.max_runtime_h), d_max=int(args.d_max)),
        encoding="utf-8",
    )
    print(f"Wrote {out_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
