#!/usr/bin/env python
"""Survey the three embed norms over every THC checkpoint, all materials.

For each checkpoint three scalars are recorded, differing only in which collocation
factor is measured with the operator norm and which with the 2->inf norm::

    alpha_op       = (max_k ||X^{o,k}||_op  )^2 (max_k ||X^{v,k}||_op  )^2 max_Q ||W^Q||_op
    alpha_mod_v    = (max_k ||X^{o,k}||_op  )^2 (max_{k,I} ||X^{v,k}_{I,:}||_2)^2 max_Q ||W^Q||_op
    alpha_mod_o    = (max_{k,I} ||X^{o,k}_{I,:}||_2)^2 (max_k ||X^{v,k}||_op  )^2 max_Q ||W^Q||_op

The ``2 -> inf`` norm is the largest row 2-norm, ``max_I ||A_{I,:}||_2``, maximised over
``k`` as well. It is evaluated in the ``(k, I, p)`` basis: ``p`` is the orbital index, so
it is invariant under MO gauge rotations but *not* under mixing interpolation points --
which is the right way round, since ``I`` is what an LCU select oracle enumerates.

Since ``||A||_{2->inf} <= ||A||_op``, both modified norms are bounded above by
``alpha_op``. Neither modified norm bounds the other in general.

``X^o`` and ``X^v`` are not stored in the checkpoints; they are rebuilt as
``X^{ao} C^{occ/vir}`` from ``inpv_kpt`` and the mean field, using the same pairing rule
the generators use (``_symm`` checkpoints take ``DFT_<mesh>_symm.pkl``). See
``chem.lcu_norms.loaders``.

``W`` is streamed one momentum transfer at a time, so peak memory is one ``M x M`` block
rather than the whole tensor -- the largest checkpoint here would otherwise need 6 GB.

Writes an incremental JSONL (resumable), a CSV, and a Markdown summary.

    python survey_norms.py                       # full survey, ~25 min
    python survey_norms.py --materials diamond   # one material
    python survey_norms.py --pattern opt         # optimised family only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chem.lcu_norms import dft_checkpoint_for, kmesh_of, leading_singular_value  # noqa: E402

DEFAULT_ROOT = "/resnick/groups/changroup/members/jsun3/xprize/THC_general"

PRETTY = {
    "diamond": "Diamond (C)", "BN": "BN (zinc blende)", "MgO": "MgO (rocksalt)",
    "AlN": "AlN (wurtzite)", "blackPbulk": "Black P (bulk A17)",
    "CdSebulk": "CdSe (bulk wurtzite)", "MoSe2bulk": "MoSe2 (bulk 2H)",
}
FAMILY_ORDER = ["ISDFfull_bareGDF", "ISDFfull_screenGDF", "ISDFov_bareGDF_symm",
                "ISDFov_bareGDF", "ISDFov_opt_bareGDF"]


def parse_path(path: str) -> dict:
    directory = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)[:-4]
    match = re.match(r"^data_(.+?)_(gth-[a-z0-9\-]+)(_final|_ewald|)?$", directory)
    material, basis = (match.group(1), match.group(2)) if match else (directory, "?")
    shape = re.search(r"_(\d+)x(\d+)x(\d+)_c(\d+)", name)
    mesh = f"{shape.group(1)}x{shape.group(2)}x{shape.group(3)}" if shape else "?"
    return dict(
        file=path, checkpoint=os.path.basename(path), material=material, basis=basis,
        family=name[:shape.start()] if shape else name, mesh=mesh,
        c=int(shape.group(4)) if shape else -1,
    )


def survey_one(path: str) -> dict:
    import h5py

    pkl = dft_checkpoint_for(path)
    with open(pkl, "rb") as handle:
        mean_field = pickle.load(handle)
    cell = mean_field.cell
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_k, n_ao, _ = mo_coeff.shape
    n_occ = cell.nelectron // 2

    with h5py.File(path, "r") as handle:
        x_ao = np.asarray(handle["inpv_kpt"])
        n_aux = x_ao.shape[1]
        if x_ao.shape[0] != n_k or x_ao.shape[2] != n_ao:
            raise ValueError(f"shape mismatch vs {os.path.basename(pkl)}: {x_ao.shape}")
        x_occ = x_ao @ mo_coeff[:, :, :n_occ]
        x_vir = x_ao @ mo_coeff[:, :, n_occ:]
        xo_op = max(leading_singular_value(x_occ[k]) for k in range(n_k))
        xv_op = max(leading_singular_value(x_vir[k]) for k in range(n_k))
        xo_2inf = float(np.linalg.norm(x_occ, axis=2).max())
        xv_2inf = float(np.linalg.norm(x_vir, axis=2).max())
        # stream W: peak memory is one M x M block
        coul = handle["coul_kpt"]
        w_op_max = 0.0
        for q in range(coul.shape[0]):
            w_op_max = max(w_op_max, leading_singular_value(np.asarray(coul[q]).astype(np.complex128)))

    out = dict(M=int(n_aux), n_k=int(n_k), n_occ=int(n_occ), n_vir=int(mo_coeff.shape[2] - n_occ),
               xo_op=float(xo_op), xv_op=float(xv_op), xo_2inf=xo_2inf, xv_2inf=xv_2inf,
               w_op_max=float(w_op_max), dft_pkl=os.path.basename(pkl))
    out["alpha_op"] = out["xo_op"] ** 2 * out["xv_op"] ** 2 * out["w_op_max"]
    out["alpha_mod_v"] = out["xo_op"] ** 2 * out["xv_2inf"] ** 2 * out["w_op_max"]
    out["alpha_mod_o"] = out["xo_2inf"] ** 2 * out["xv_op"] ** 2 * out["w_op_max"]
    return out


CSV_FIELDS = ["material", "basis", "family", "mesh", "c", "M", "n_k", "n_occ", "n_vir",
              "xo_op", "xv_op", "xo_2inf", "xv_2inf", "w_op_max",
              "alpha_op", "alpha_mod_v", "alpha_mod_o", "checkpoint", "dft_pkl"]


def write_outputs(records, csv_path, md_path, root):
    ok = [r for r in records if r.get("ok")]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(ok, key=lambda x: (x["material"], FAMILY_ORDER.index(x["family"])
                                           if x["family"] in FAMILY_ORDER else 9,
                                           int(x["mesh"].split("x")[0]), x["c"])):
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()
                             if k in CSV_FIELDS})
    _write_markdown(ok, md_path, csv_path, root, len(records) - len(ok))


def _write_markdown(ok, md_path, csv_path, root, n_failed):
    import statistics as st
    L = []
    W = L.append
    W("# THC embed norms: all materials, all checkpoints\n")
    W(f"Generated by `survey_norms.py` from `{root}`. "
      f"{len(ok)} checkpoints" + (f", {n_failed} failed" if n_failed else "") + ".")
    W(f"Machine-readable companion: `{os.path.basename(csv_path)}`.\n")
    W("## The three norms\n")
    W("All share `max_Q ||W^Q||_op` and differ only in how the two collocation factors")
    W("are measured. `||A||_(2->inf) = max_I ||A_(I,:)||_2`, the largest row 2-norm,")
    W("maximised over `k` as well, evaluated in the `(k, I, p)` basis.\n")
    W("| column | formula |")
    W("|---|---|")
    W("| `alpha_op` | `(max_k \\|\\|X^{o,k}\\|\\|_op)^2 (max_k \\|\\|X^{v,k}\\|\\|_op)^2 max_Q \\|\\|W^Q\\|\\|_op` |")
    W("| `alpha_mod_v` | `(max_k \\|\\|X^{o,k}\\|\\|_op)^2 (max_{k,I} \\|\\|X^{v,k}_{I,:}\\|\\|_2)^2 max_Q \\|\\|W^Q\\|\\|_op` |")
    W("| `alpha_mod_o` | `(max_{k,I} \\|\\|X^{o,k}_{I,:}\\|\\|_2)^2 (max_k \\|\\|X^{v,k}\\|\\|_op)^2 max_Q \\|\\|W^Q\\|\\|_op` |")
    W("")
    W("Since `||A||_(2->inf) <= ||A||_op`, both modified norms are `<= alpha_op`.")
    W("Neither modified norm bounds the other in general.\n")

    W("## Systems\n")
    W("| material | basis | n_ao | n_occ | n_vir | checkpoints |")
    W("|---|---|---|---|---|---|")
    for m in sorted({r["material"] for r in ok}):
        g = [r for r in ok if r["material"] == m]
        W(f"| {PRETTY.get(m, m)} | `{g[0]['basis']}` | {g[0]['n_occ'] + g[0]['n_vir']} | "
          f"{g[0]['n_occ']} | {g[0]['n_vir']} | {len(g)} |")
    W("")

    W("## Summary of the ratios\n")
    W("| material | family | n | alpha_op/alpha_mod_v | alpha_op/alpha_mod_o | alpha_mod_o/alpha_mod_v |")
    W("|---|---|---|---|---|---|")
    rng = lambda v: f"{min(v):.2f}-{max(v):.2f} (med {st.median(v):.2f})"
    for m in sorted({r["material"] for r in ok}):
        for fam in FAMILY_ORDER:
            g = [r for r in ok if r["material"] == m and r["family"] == fam]
            if not g:
                continue
            W(f"| {PRETTY.get(m, m)} | `{fam}` | {len(g)} | "
              f"{rng([r['alpha_op']/r['alpha_mod_v'] for r in g])} | "
              f"{rng([r['alpha_op']/r['alpha_mod_o'] for r in g])} | "
              f"{rng([r['alpha_mod_o']/r['alpha_mod_v'] for r in g])} |")
    n_o_lower = sum(1 for r in ok if r["alpha_mod_o"] < r["alpha_mod_v"])
    W("")
    W(f"`alpha_mod_o < alpha_mod_v` in **{n_o_lower} of {len(ok)}** checkpoints "
      f"({100*n_o_lower/len(ok):.0f}%).\n")

    W("## Full data\n")
    for m in sorted({r["material"] for r in ok}):
        W(f"\n### {PRETTY.get(m, m)}\n")
        for fam in FAMILY_ORDER:
            g = [r for r in ok if r["material"] == m and r["family"] == fam]
            if not g:
                continue
            W(f"\n**`{fam}`**\n")
            W("| mesh | c | M | alpha_op | alpha_mod_v | alpha_mod_o |")
            W("|---|---|---|---|---|---|")
            for r in sorted(g, key=lambda x: (int(x["mesh"].split("x")[0]), x["c"])):
                W(f"| {r['mesh']} | {r['c']} | {r['M']} | {r['alpha_op']:.4f} | "
                  f"{r['alpha_mod_v']:.4f} | {r['alpha_mod_o']:.4f} |")
    W("")
    W("## Caveats\n")
    W("- Absolute values are in the checkpoints' internal units; only ratios are dimensionless.")
    W("- Families are not comparable to each other. Only `ISDFov_opt_*` was optimised against")
    W("  a norm at all (and against `alpha_op` specifically), so its values are far smaller.")
    W("  The `ISDFfull_*` families were never fitted for the ov block, so evaluating an")
    W("  ov-block quantity on them is valid arithmetic but not what that fit optimised.")
    W("- Neither column is monotonic in `c`. `generate_isdf_gdf*.py` selects its")
    W("  regularisation independently per run by a 1.5x-error stopping rule, which breaks the")
    W("  underlying trend. Do not read rank scaling off these columns.")
    W("- None of these is a validated LCU lambda for a circuit; the `4/n_k` prefactor in")
    W("  `bse_thc.py:apply_V_thc` is not folded in.")
    W("- jsun3's data was read only; nothing under the source tree was modified.")
    open(md_path, "w").write("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--out-dir", default=os.path.join(_HERE, "survey"))
    p.add_argument("--materials", nargs="*", default=None)
    p.add_argument("--pattern", default=None,
                   help="only survey checkpoints whose filename contains this substring, "
                        "e.g. --pattern opt for the optimised family")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl = os.path.join(args.out_dir, "all_materials_norms.jsonl")
    files = sorted(glob.glob(os.path.join(args.root, "data_*", "ISDF*.chk")))
    if args.materials:
        files = [f for f in files if parse_path(f)["material"] in args.materials]
    if args.pattern:
        files = [f for f in files if args.pattern in os.path.basename(f)]

    done = {}
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                r = json.loads(line); done[r["file"]] = r
            except Exception:
                pass

    handle = open(jsonl, "a")
    t0 = time.time()
    for i, path in enumerate(files):
        if path in done:
            continue
        rec = parse_path(path)
        try:
            rec.update(survey_one(path)); rec["ok"] = True
        except Exception as exc:
            rec.update(ok=False, error=repr(exc))
        handle.write(json.dumps(rec) + "\n"); handle.flush()
        done[path] = rec
        status = (f"op={rec['alpha_op']:.4g} v={rec['alpha_mod_v']:.4g} o={rec['alpha_mod_o']:.4g}"
                  if rec.get("ok") else "FAIL " + str(rec.get("error"))[:60])
        print(f"[{i+1}/{len(files)}] {rec['material']:11s} {rec['family']:20s} "
              f"{rec['mesh']:>7s} c={rec['c']:<3d} {status}  {time.time()-t0:.0f}s", flush=True)

    records = [done[f] for f in files if f in done]
    write_outputs(records, os.path.join(args.out_dir, "all_materials_norms.csv"),
                  os.path.join(args.out_dir, "NORMS.md"), args.root)
    n_ok = sum(1 for r in records if r.get("ok"))
    print(f"\nDONE {time.time()-t0:.0f}s  {n_ok}/{len(records)} ok -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
