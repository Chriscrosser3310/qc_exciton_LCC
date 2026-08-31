#!/usr/bin/env python
"""Evaluate the LCU and embed norms on THC checkpoints and write a CSV.

Reuses ``chem.lcu_norms`` -- the same code that produced the diamond tables in
``chem/lcu_norms/diamond_thc_norms.csv`` -- so numbers from an optimized checkpoint
are directly comparable to those.

The mean-field pickle must be given explicitly, because optimized checkpoints are
written to ``results/`` while the pickle lives in ``data/``, so the filename
convention that ``chem.lcu_norms.loaders`` normally uses cannot resolve it.

Example
-------
    python evaluate.py --data-dir data --out results/norms.csv results/*.chk
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chem.lcu_norms import check_relations, compute_norms, kmesh_of, load_thc_factors  # noqa: E402

FIELDS = [
    "checkpoint", "kmesh", "M", "n_k", "n_occ", "n_vir",
    "xo_op", "xv_op", "xo_2inf", "xv_2inf",
    "w_op_max", "w_fro_max", "w_l1_mean",
    "alpha_lcu_dir", "alpha_lcu_exch", "alpha_lcu_exch_max",
    "alpha_embed_op", "alpha_embed_fro", "alpha_embed_l1",
    "alpha_embed_op_mod", "alpha_embed_op_2inf", "alpha_embed_l1_2inf",
    "k_participation", "q_concentration", "dft_pkl",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data-dir", default=os.path.join(_HERE, "data"),
                        help="directory holding DFT_<mesh>_symm.pkl")
    parser.add_argument("--pkl-pattern", default="DFT_{klabel}_symm.pkl")
    parser.add_argument("--out", default=None, help="CSV output path; stdout table if omitted")
    args = parser.parse_args(argv)

    rows = []
    for chk in sorted(args.checkpoints):
        km = kmesh_of(chk)
        klabel = "x".join(str(v) for v in km)
        pkl = os.path.join(args.data_dir, args.pkl_pattern.format(klabel=klabel))
        if not os.path.exists(pkl):
            raise SystemExit(f"missing mean-field pickle {pkl} for {chk}")
        x_occ, x_vir, coul, kmesh = load_thc_factors(chk, dft_pkl=pkl)
        norms = compute_norms(x_occ, x_vir, coul, kmesh)
        check_relations(norms)
        record = norms.as_dict()
        record["checkpoint"] = os.path.basename(chk)
        record["kmesh"] = klabel
        record["M"] = norms.n_aux
        record["dft_pkl"] = os.path.basename(pkl)
        rows.append(record)
        print(f"{os.path.basename(chk):68s} "
              f"dir={norms.alpha_lcu_dir:11.5g} "
              f"emb_op={norms.alpha_embed_op:11.5g} "
              f"emb_op_mod={norms.alpha_embed_op_mod:11.5g}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for record in rows:
                writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                                 for k, v in record.items()})
        print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
