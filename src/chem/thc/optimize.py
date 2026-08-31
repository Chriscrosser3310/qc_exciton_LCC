#!/usr/bin/env python
"""Optimize THC collocation matrices for the ov block, with a selectable norm penalty.

A faithful port of jsun3's ``optimize_X_ov.py``. The fitting algorithm is unchanged --
it calls the vendored ``utils``/``optimize_X_common`` directly rather than
reimplementing anything. Three things differ:

1. **The penalty norms are selectable** (``--xo-norm``, ``--xv-norm``, ``--w-norm``).
   Upstream hardcodes the operator norm for both collocation factors and offers only
   ``--use_Fnorm`` for the central tensor. With the defaults this reproduces upstream's
   penalty exactly; ``--w-norm fro`` is upstream's ``--use_Fnorm``.
2. **Input paths are explicit**, instead of being resolved by naming convention through
   ``system_common.get_data_dir``. Defaults point at ``./data``.
3. **Device selection has a CPU fallback.** Upstream calls
   ``optimize_X_common.get_device()``, which imports a ``gpu_register`` module absent
   from the source tree.

The interpolation-point grid metadata (``mesh``, ``ix_sel``, ``group_sel``) is carried
from the init checkpoint into the output whether or not symmetrization is requested.
Upstream only propagates it under ``--symm``, so its optimized checkpoints lack it.

Every run writes a JSON sidecar next to the checkpoint recording the settings and the
final metrics, so a results table can be rebuilt without re-reading logs.

Example
-------
    python optimize.py --kmesh 2 --c-isdf 5 --c-ref 20 \
        --norm-const 0.1 --power 2 --base-lr 1e-2 --nsteps-factor 1 \
        --xv-norm 2inf --save
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chem.thc import penalties                                  # noqa: E402
from chem.thc.vendor import optimize_X_common as oc             # noqa: E402
from chem.thc.vendor import utils                               # noqa: E402


def format_number(value: float) -> str:
    return "%g" % value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kmesh", type=int, required=True, help="linear k-mesh dimension, e.g. 2 for 2x2x2")
    p.add_argument("--c-isdf", type=int, required=True, help="THC rank label of the initial guess")
    p.add_argument("--c-ref", type=int, required=True, help="THC rank label of the reference")
    p.add_argument("--data-dir", default=os.path.join(_HERE, "data"))
    p.add_argument("--out-dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--family", default="ISDFov_bareGDF_symm",
                   help="checkpoint family prefix used to build init/ref paths by convention")
    p.add_argument("--dft-pkl", default=None, help="override the mean-field pickle path")
    p.add_argument("--init-chk", default=None, help="override the initial-guess checkpoint path")
    p.add_argument("--ref-chk", default=None, help="override the reference checkpoint path")
    p.add_argument("--norm-const", type=float, default=0.1)
    p.add_argument("--power", type=int, default=2)
    p.add_argument("--base-lr", type=float, default=1e-2)
    p.add_argument("--nsteps-factor", type=int, default=1,
                   help="schedule multiplier; 0 runs the 100-step warmup only (smoke test)")
    p.add_argument("--init-reg", type=float, default=1e-2)
    p.add_argument("--xo-norm", default="op", help="op | 2inf | fro | p<N>")
    p.add_argument("--xv-norm", default="op", help="op | 2inf | fro | p<N>")
    p.add_argument("--w-norm", default="op", help="op | fro | absmax | l1  ('fro' == upstream --use_Fnorm)")
    p.add_argument("--device", default="cpu", help="cpu | cuda:N")
    p.add_argument("--float64", action="store_true", help="run in complex128 (upstream uses complex64)")
    p.add_argument("--symm", action="store_true",
                   help="alias for --symm-parts xo,xv,w (upstream behaviour)")
    p.add_argument("--symm-parts", default="",
                   help="comma-separated subset of xo,xv,w to symmetrise. The occupied and "
                        "virtual spaces are separately invariant under the crystal group "
                        "(verified to 2e-14), so xo and xv can be selected independently.")
    p.add_argument("--real", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--tag", default="", help="extra suffix appended to output names")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    parts = {p.strip() for p in args.symm_parts.split(',') if p.strip()}
    if args.symm:
        parts |= {'xo', 'xv', 'w'}
    unknown = parts - {'xo', 'xv', 'w'}
    if unknown:
        raise SystemExit(f"unknown --symm-parts entries: {sorted(unknown)}")
    need_symm = bool(parts)

    nk = args.kmesh
    kmesh = (nk, nk, nk)
    klabel = f"{nk}x{nk}x{nk}"
    complex_dtype = torch.complex128 if args.float64 else torch.complex64
    real_dtype = torch.float64 if args.float64 else torch.float32
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"requested {args.device} but CUDA is unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")

    dft_pkl = args.dft_pkl or os.path.join(args.data_dir, f"DFT_{klabel}_symm.pkl")
    init_chk = args.init_chk or os.path.join(args.data_dir, f"{args.family}_{klabel}_c{args.c_isdf}.chk")
    ref_chk = args.ref_chk or os.path.join(args.data_dir, f"{args.family}_{klabel}_c{args.c_ref}.chk")
    for path in (dft_pkl, init_chk, ref_chk):
        if not os.path.exists(path):
            raise SystemExit(f"missing input: {path}")

    norm_label = "norm%sp%s" % (format_number(args.norm_const), format_number(args.power))
    tag = penalties.norm_tag(args.xo_norm, args.xv_norm, args.w_norm) + args.tag
    symm_tag = ("_symm" + ("".join(sorted(parts)) if parts != {"xo", "xv", "w"} else "")) if parts else ""
    real_tag = "_real" if args.real else ""
    stem = (f"ISDFov_opt_bareGDF{symm_tag}{real_tag}_{klabel}"
            f"_c{args.c_isdf}_cref{args.c_ref}_{norm_label}{tag}")
    os.makedirs(args.out_dir, exist_ok=True)
    chk_save_path = os.path.join(args.out_dir, stem + ".chk")
    opt_save_path = os.path.join(args.out_dir, stem + ".pt")
    json_save_path = os.path.join(args.out_dir, stem + ".json")

    # norm_ratio follows upstream: the LINEAR mesh dimension raised to `power`, not n_k.
    norm_ratio = args.norm_const * nk ** args.power

    with open(dft_pkl, "rb") as handle:
        mean_field = pickle.load(handle)
    cell = mean_field.cell
    kpts = cell.make_kpts(kmesh)
    kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % np.asarray(kmesh)
    assert utils.is_k_ordered(kpts_int, kmesh), "k-point ordering does not match the assumed convention"

    C_np = np.asarray(mean_field.mo_coeff)
    C = oc.to_tensor(C_np, device, complex_dtype)
    C128 = oc.to_tensor128(C_np, device)
    nkpts, nao, nmo = C.shape
    nocc = cell.nelectron // 2
    Cocc, Cvir = C[:, :, :nocc], C[:, :, nocc:]
    Cocc128, Cvir128 = C128[:, :, :nocc], C128[:, :, nocc:]
    nvir = nmo - nocc

    X_ref_np, W_ref_np = oc.load_isdf(ref_chk)
    X_init_np, _ = oc.load_isdf(init_chk)
    X_ref_ao = oc.to_tensor(X_ref_np, device, complex_dtype)
    X_ref_ao128 = oc.to_tensor128(X_ref_np, device)
    W_ref = oc.to_tensor(W_ref_np, device, complex_dtype)
    W_ref128 = oc.to_tensor128(W_ref_np, device)
    X_init_ao = oc.to_tensor(X_init_np, device, complex_dtype)

    # grid metadata: propagate whenever the init checkpoint carries it
    try:
        mesh, ix_sel, group_sel = oc.load_isdf_grid(init_chk)
    except (KeyError, OSError):
        mesh = ix_sel = group_sel = None

    state_AO = False
    if need_symm:
        from chem.thc.vendor import libsymm
        if mesh is None:
            raise SystemExit("symmetrisation needs grid metadata (mesh/ix_sel) in the init checkpoint")
        cell_isdf = cell.copy()
        cell_isdf.mesh = mesh
        coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)
        symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=complex_dtype, device=device)
        perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)
        symm128 = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=torch.complex128, device=device)
        perm_check, phase128 = libsymm.build_isdf_grid_transform(symm128, coords, ix_sel, mesh=cell_isdf.mesh)
        assert torch.all(perm_check == perm)
    else:
        libsymm = None
        symm = perm = phase = symm128 = phase128 = None

    negative = oc.negative_k_indices(kmesh, device)

    Xo_ref = X_ref_ao @ Cocc
    Xv_ref = X_ref_ao @ Cvir
    Xo_ref128 = X_ref_ao128 @ Cocc128
    Xv_ref128 = X_ref_ao128 @ Cvir128
    ref_norm2 = utils.thc_ovvo_inner_from_mo(Xo_ref, Xv_ref, W_ref, Xo_ref, Xv_ref, W_ref, kmesh).real
    ref_norm2_128 = utils.thc_ovvo_inner_from_mo(
        Xo_ref128, Xv_ref128, W_ref128, Xo_ref128, Xv_ref128, W_ref128, kmesh
    ).real

    def split_mo(X, force_complex128=False):
        """Return (Xo, Xv), symmetrising each independently as requested.

        Upstream symmetrises the shared X_ao once, before the occ/vir split, so its
        `--symm` necessarily constrains both factors. Selecting only one is legitimate
        because the occupied and virtual spaces are separately invariant under the
        crystal group -- projecting X_ao and then taking the occupied columns gives the
        same result as projecting the occupied factor alone (checked to 2e-14). So the
        two halves genuinely decouple.
        """
        X_ao = oc.ao_from_state(X, C, C128, state_AO, force_complex128=force_complex128)
        if args.real:
            X_ao = oc.symmetrize_real_gauge(X_ao, negative)
        X_ao_sym = X_ao
        if parts & {"xo", "xv"}:
            symm_use = symm128 if force_complex128 else symm
            phase_use = phase128 if force_complex128 else phase
            X_ao_sym = libsymm.symmetrize_isdf_X_fast(symm_use, X_ao, perm, phase_use)
        occ_src = X_ao_sym if "xo" in parts else X_ao
        vir_src = X_ao_sym if "xv" in parts else X_ao
        C_use = C128 if force_complex128 else C
        return occ_src @ C_use[:, :, :nocc], vir_src @ C_use[:, :, nocc:]

    def get_W_error2_from_X(X, reg, ref_norm2_use, force_complex128=False):
        Xo, Xv = split_mo(X, force_complex128=force_complex128)
        W, L, rhs = utils.thc_ovvo_solve_w_intermediate_from_mo(
            Xo_ref128 if force_complex128 else Xo_ref,
            Xv_ref128 if force_complex128 else Xv_ref,
            W_ref128 if force_complex128 else W_ref,
            Xo, Xv, kmesh, reg=reg,
        )
        if "w" in parts:
            symm_use = symm128 if force_complex128 else symm
            phase_use = phase128 if force_complex128 else phase
            W = libsymm.symmetrize_isdf_W_fast(symm_use, W, perm, phase_use)
        if args.real:
            W = oc.symmetrize_real_gauge(W, negative)
        return W, utils.thc_solve_w_error2_from_intermediate(W, L, rhs, ref_norm2_use)

    def get_abs_norm_loss(X, W, force_complex128=False):
        Xo, Xv = split_mo(X, force_complex128=force_complex128)
        return penalties.abs_norm_loss(
            Xo, Xv, W, nkpts,
            xo_kind=args.xo_norm, xv_kind=args.xv_norm, w_kind=args.w_norm,
        )

    X_init = oc.state_from_ao(X_init_ao, C, state_AO)
    project_unproject_error = oc.state_unproject_error(X_init, X_init_ao, C, C128, state_AO)
    X_opt = X_init.detach().clone().requires_grad_(True)
    log_reg = torch.tensor(np.log(args.init_reg), dtype=real_dtype, device=device, requires_grad=True)
    adam_schedule = oc.make_schedule(args.base_lr, args.nsteps_factor)

    print("kmesh          =", kmesh)
    print("nao / nocc / nvir =", nao, nocc, nvir)
    print("dft pkl        =", dft_pkl)
    print("init chk       =", init_chk)
    print("ref chk        =", ref_chk)
    print("M (init)       =", X_init_np.shape[1])
    print("M (ref)        =", X_ref_np.shape[1])
    print("xo / xv / w norm =", args.xo_norm, "/", args.xv_norm, "/", args.w_norm)
    print("norm_const     =", args.norm_const)
    print("power          =", args.power)
    print("norm_ratio     =", norm_ratio)
    print("base_lr        =", args.base_lr)
    print("nsteps_factor  =", args.nsteps_factor)
    print("adam steps     =", sum(n for n, _, _ in adam_schedule))
    print("device / dtype =", device, "/", complex_dtype)
    print("symmetrised    =", sorted(parts) or "nothing")
    print("grid metadata  =", "present" if mesh is not None else "absent")
    print("project-unproject relerr = %.6e" % project_unproject_error.item())
    print("ref_norm2      = %.16e" % ref_norm2.item())
    print("", flush=True)

    loss_args = (
        get_W_error2_from_X,
        get_abs_norm_loss,
        ref_norm2,
        ref_norm2_128,
        norm_ratio,
        1e-2,   # rel_error_scale, as upstream
        1e-1,   # cosh_scale, as upstream
        False,  # cosh_sqrt, as upstream
    )
    (loss_opt, error2_opt, rel_error_opt, norm_loss_opt, W_opt, reg_opt), opt = oc.run_adam(
        X_opt, log_reg, oc.optimization_loss_from_X_log_reg, loss_args,
        adam_schedule, args.base_lr, device, complex_dtype,
        "Running Adam over X and log_reg ...", "ov",
    )

    X_save_state = X_opt.detach()
    X_ao_opt = oc.ao_from_state(X_save_state, C, C128, state_AO)
    W_save = W_opt.detach()
    if parts & {"xo", "xv"} and parts >= {"xo", "xv"}:
        # only project the saved X when BOTH halves were constrained; projecting a
        # deliberately-free half at save time would discard what was optimised
        X_ao_opt = libsymm.symmetrize_isdf_X_fast(symm, X_ao_opt, perm, phase)
    if "w" in parts:
        W_save = libsymm.symmetrize_isdf_W_fast(symm, W_save, perm, phase)
    if args.real:
        X_ao_opt = oc.symmetrize_real_gauge(X_ao_opt, negative)
        W_save = oc.symmetrize_real_gauge(W_save, negative)

    print("final loss      = %.16e" % loss_opt.item())
    print("final error2    = %.16e" % error2_opt.item())
    print("final rel_error = %.16e" % rel_error_opt.item())
    print("final norm_loss = %.16e" % norm_loss_opt.item())
    print("final reg       = %.16e" % reg_opt.item())

    record = {
        "kmesh": klabel, "c_isdf": args.c_isdf, "c_ref": args.c_ref,
        "M_init": int(X_init_np.shape[1]), "M_ref": int(X_ref_np.shape[1]),
        "n_k": int(nkpts), "n_occ": int(nocc), "n_vir": int(nvir),
        "xo_norm": args.xo_norm, "xv_norm": args.xv_norm, "w_norm": args.w_norm,
        "norm_const": args.norm_const, "power": args.power, "norm_ratio": norm_ratio,
        "base_lr": args.base_lr, "nsteps_factor": args.nsteps_factor,
        "adam_steps": int(sum(n for n, _, _ in adam_schedule)),
        "init_reg": args.init_reg, "symm_parts": sorted(parts), "real": bool(args.real),
        "dtype": str(complex_dtype), "device": str(device),
        "dft_pkl": os.path.basename(dft_pkl),
        "init_chk": os.path.basename(init_chk), "ref_chk": os.path.basename(ref_chk),
        "final_loss": float(loss_opt.item()),
        "final_error2": float(error2_opt.item()),
        "final_rel_error": float(rel_error_opt.item()),
        "final_norm_loss": float(norm_loss_opt.item()),
        "final_reg": float(reg_opt.item()),
        "ref_norm2": float(ref_norm2.item()),
        "chk": os.path.basename(chk_save_path) if args.save else None,
    }
    with open(json_save_path, "w") as handle:
        json.dump(record, handle, indent=1)
    print("Wrote run record to", json_save_path)

    if args.save:
        torch.save(
            {"X_opt": X_save_state.cpu(), "log_reg": log_reg.detach().cpu(),
             "opt_state": opt.state_dict()},
            opt_save_path,
        )
        oc.save_isdf(chk_save_path, X_ao_opt.cpu().numpy(), W_save.cpu().numpy(),
                     mesh=mesh, ix_sel=ix_sel, group_sel=group_sel)
        print("Saved optimizer state to", opt_save_path)
        print("Saved ISDF checkpoint to", chk_save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
