#!/usr/bin/env python
"""Collect the symmetry / irreducible-representation structure of the THC factors.

jsun3's `analyze_isdf_irreps.py` decomposes the symmetry representations carried by the
THC objects and writes the block structure to `log_irreps/`. That analysis is *not* used
anywhere in the fitting or optimisation pipeline -- it is a standalone measurement of how
much structure the crystal symmetry implies. This script parses it, records it locally,
and derives the compression factors.

Three representations are decomposed:

- **grid-grid** -- the selected-grid representation, dimension `n_k * M`. This is the one
  `W^Q` lives on.
- **ao-ao** -- the AO/orbital representation, dimension `n_k * n_ao`.
- **grid-ao** -- the allowed blocks of `X`, pairing grid irreps with AO irreps.

For a representation decomposing as `sum_i (irrep of dim d_i) x C^{m_i}`, an operator that
commutes with the group action is block diagonal: one independent `m_i x m_i` block per
irrep, repeated `d_i` times. So

    dense parameters       = (sum_i d_i m_i)^2
    independent parameters = sum_i m_i^2

and the ratio is what block-diagonalising by irrep would save in storage. For `X`, which
maps the AO rep to the grid rep, the independent count is `sum_i grid_mult_i * ao_mult_i`.

Also copies `W_cost_grid-grid.txt`, jsun3's estimate of the *circuit* cost saving from the
same block structure (a different quantity from the storage ratio -- see THC_REFERENCE.md).

    python collect_irreps.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = "/resnick/groups/changroup/members/jsun3/xprize/THC_general/log_irreps"

SCALARS = [
    ("nkpts", r"^nkpts = (\d+)", int),
    ("M", r"^n selected points = (\d+)", int),
    ("n_groups", r"^n selected groups = (\d+)", int),
    ("n_ao", r"^nao = (\d+)", int),
    ("n_ops", r"^n symmetry operations = (\d+)", int),
    ("grid_rep_size", r"^grid representation size = (\d+)", int),
    ("ao_rep_size", r"^AO representation size = (\d+)", int),
    ("grid_invariance_relfro", r"^grid max invariance rel_fro = (\S+)", float),
    ("ao_commutator_relfro", r"^AO max commutator rel_fro = (\S+)", float),
    ("X_symmetry_relfro", r"^loaded X symmetry rel_fro = (\S+)", float),
    ("W_symmetry_relfro", r"^loaded W symmetry rel_fro = (\S+)", float),
]


def parse_log(path: str) -> dict:
    text = open(path).read()
    out = {}
    for key, pattern, cast in SCALARS:
        m = re.search(pattern, text, re.M)
        out[key] = cast(m.group(1)) if m else None
    return out


def load_table(path: str) -> np.ndarray:
    """Rows of (irrep_dim, multiplicity, total_size)."""
    return np.atleast_2d(np.loadtxt(path, dtype=int))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out-dir", default=os.path.join(_HERE, "survey"))
    args = p.parse_args(argv)

    raw = os.path.join(args.out_dir, "irreps_raw")
    os.makedirs(raw, exist_ok=True)
    for f in glob.glob(os.path.join(args.src, "*.txt")) + glob.glob(os.path.join(args.src, "*.log")):
        shutil.copy2(f, raw)
    print(f"copied {len(os.listdir(raw))} raw irrep files -> {raw}")

    rows = []
    for logfile in sorted(glob.glob(os.path.join(args.src, "irreps_ov_*_final.log"))):
        m = re.search(r"irreps_ov_(.+?)_(\d+x\d+x\d+)_c(\d+)_final\.log", os.path.basename(logfile))
        rec = dict(material=m.group(1), mesh=m.group(2), c=int(m.group(3)))
        rec.update(parse_log(logfile))
        for tag, key in [("grid-grid", "W"), ("ao-ao", "AO"), ("grid-ao", "X")]:
            t = os.path.join(args.src,
                             f"data_ov_{rec['material']}_{rec['mesh']}_c{rec['c']}_{tag}.txt")
            if not os.path.exists(t):
                continue
            tab = load_table(t)
            dim, mult = tab[:, 0], tab[:, 1]
            rec[f"{key}_nblocks"] = int(len(mult))
            rec[f"{key}_max_mult"] = int(mult.max())
            if tag == "grid-ao":
                # columns are (irrep_dim, grid_mult, ao_mult) for the allowed X blocks
                rec["X_independent"] = int((tab[:, 1] * tab[:, 2]).sum())
            else:
                rec[f"{key}_dense"] = int((dim * mult).sum() ** 2)
                rec[f"{key}_independent"] = int((mult ** 2).sum())
                rec[f"{key}_compression"] = rec[f"{key}_dense"] / rec[f"{key}_independent"]
        if "X_independent" in rec:
            rec["X_dense"] = rec["nkpts"] * rec["M"] * rec["n_ao"]
            rec["X_compression"] = rec["X_dense"] / rec["X_independent"]
        rows.append(rec)

    cost_src = os.path.join(args.src, "W_cost_grid-grid.txt")
    cost = {}
    if os.path.exists(cost_src):
        shutil.copy2(cost_src, args.out_dir)
        for line in open(cost_src):
            if line.startswith("#"):
                continue
            f = line.split()
            cost[f[0]] = dict(nblocks=int(f[4]), max_mult=int(f[5]),
                              original_cost=float(f[6]), new_cost=float(f[7]),
                              ratio=float(f[8]), opt_cost=float(f[9]), opt_ratio=float(f[10]),
                              exact_opt_cost=float(f[11]), exact_opt_ratio=float(f[12]))
    for r in rows:
        c = cost.get(f"{r['material']}_{r['mesh']}")
        if c:
            r.update({f"Wcost_{k}": v for k, v in c.items()})

    fields = sorted({k for r in rows for k in r})
    fields = ([f for f in ["material", "mesh", "c", "nkpts", "M", "n_groups", "n_ao", "n_ops"] if f in fields]
              + [f for f in fields if f not in
                 ("material", "mesh", "c", "nkpts", "M", "n_groups", "n_ao", "n_ops")])
    out_csv = os.path.join(args.out_dir, "irrep_structure.csv")
    with open(out_csv, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["material"], int(x["mesh"].split("x")[0]))):
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(rows, open(os.path.join(args.out_dir, "irrep_structure.json"), "w"), indent=1)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    for r in sorted(rows, key=lambda x: (x["material"], int(x["mesh"].split("x")[0]))):
        print(f"  {r['material']:8s} {r['mesh']:>7s}  ops={r['n_ops']:3d}  "
              f"W {r['W_dense']:>10d}->{r['W_independent']:<7d} ({r['W_compression']:7.1f}x)  "
              f"X {r['X_dense']:>8d}->{r['X_independent']:<6d} ({r['X_compression']:6.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
