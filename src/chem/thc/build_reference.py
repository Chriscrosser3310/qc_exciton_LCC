#!/usr/bin/env python
"""Assemble context/thc-data.md, the complete record of the THC dataset.

Tables come from survey/all_materials_norms.csv and survey/irrep_structure.csv, so they
cannot drift from the measurements. Prose sections record conventions read out of jsun3's
source and traps that yield silently wrong results; the checkpoints alone do not supply
those. Prose follows context/writing-guides.md.

    python build_reference.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

PRETTY = {"diamond": "Diamond (C)", "BN": "BN (zinc blende)", "MgO": "MgO (rocksalt)",
          "AlN": "AlN (wurtzite)", "blackPbulk": "Black P (bulk A17)",
          "CdSebulk": "CdSe (bulk wurtzite)", "MoSe2bulk": "MoSe2 (bulk 2H)"}
FAMILIES = ["ISDFfull_bareGDF", "ISDFfull_screenGDF", "ISDFov_bareGDF_symm",
            "ISDFov_bareGDF", "ISDFov_opt_bareGDF"]
FAM_DESC = {
    "ISDFfull_bareGDF": "all-orbital block, bare Coulomb",
    "ISDFfull_screenGDF": "all-orbital block, screened interaction",
    "ISDFov_bareGDF_symm": "occupied-virtual block, symmetry-adapted point selection",
    "ISDFov_bareGDF": "occupied-virtual block, plain point selection",
    "ISDFov_opt_bareGDF": "occupied-virtual block, collocation matrix re-optimised",
}


def slug(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", "-", s.strip())


class Doc:
    def __init__(self):
        self.lines = []
        self.toc = []

    def w(self, text=""):
        self.lines.append(text)

    def h(self, level: int, title: str):
        self.toc.append((level, title, slug(title)))
        self.w(f"\n{'#' * level} {title}\n")

    def render(self) -> str:
        toc = ["## Contents\n"]
        for level, title, anchor in self.toc:
            if level == 1:
                continue
            toc.append(f"{'  ' * (level - 2)}- [{title}](#{anchor})")
        body = "\n".join(self.lines)
        head, rest = body.split("<!--TOC-->", 1)
        return head + "\n".join(toc) + rest


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as h:
        return list(csv.DictReader(h))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--survey-dir", default=os.path.join(_HERE, "survey"))
    # _HERE is <root>/qc_exciton_LCC/src/chem/thc; the coordination layer sits four up
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
    p.add_argument("--out", default=os.path.join(_ROOT, "context", "thc-data.md"))
    args = p.parse_args(argv)

    norms = read_csv(os.path.join(args.survey_dir, "all_materials_norms.csv"))
    irreps = read_csv(os.path.join(args.survey_dir, "irrep_structure.csv"))
    for r in norms:
        for k in ("alpha_op", "alpha_mod_v", "alpha_mod_o", "xo_op", "xv_op",
                  "xo_2inf", "xv_2inf", "w_op_max"):
            r[k] = float(r[k])
        for k in ("M", "n_k", "n_occ", "n_vir", "c"):
            r[k] = int(r[k])
    mats = sorted({r["material"] for r in norms})
    # full inventory from disk, independent of what was measured
    import collections
    ondisk = collections.Counter()
    for f in glob.glob(os.path.join(
            "/resnick/groups/changroup/members/jsun3/xprize/THC_general",
            "data_*", "ISDF*.chk")):
        n = os.path.basename(f)
        m = re.search(r"_(\d+)x\1x\1_c(\d+)", n)
        fam = n[:m.start()] if m else n
        ondisk[fam] += 1
    measured = collections.Counter(r["family"] for r in norms)
    mesh_key = lambda m: int(m.split("x")[0])

    d = Doc()
    d.w("# THC data reference\n")
    d.w("Six quantum-chemistry systems carry tensor-hypercontraction factorisations in")
    d.w("jsun3's tree, and costing a block encoding from them requires norms that nobody")
    d.w("recorded. This file supplies those norms for every checkpoint, the symmetry")
    d.w("structure the factors carry, the conventions the generating code follows, and the")
    d.w("failure modes that yield plausible wrong answers. Reading it should remove the need")
    d.w("to re-derive any of that.\n")
    d.w("Two audiences share the file. An agent may quote the tables directly, because a")
    d.w("script regenerates them from the CSV files; the prose sections, by contrast, record")
    d.w("facts that the checkpoints alone do not supply. A human seeking the findings should")
    d.w("read Sections 2 and 3, and anyone about to compute from this data should read")
    d.w("Section 5 first.\n")
    d.w("<!--TOC-->\n")

    # ---------------- 1
    d.h(2, "1. Source data and provenance")
    d.w("All measurements below read one directory tree, and nothing in it was modified.")
    d.w("This section states where the data lives, which files this directory adds, and what")
    d.w("the dataset contains. Section 1.1 gives the locations; Section 1.2 inventories the")
    d.w("systems and checkpoint families.\n")
    d.h(3, "1.1 Locations")
    d.w("The measured checkpoints live under")
    d.w("`/resnick/groups/changroup/members/jsun3/xprize/THC_general/`, in one directory per")
    d.w("system. Group membership in `hpc_changroup` grants read access. The survey scripts")
    d.w("opened those files read-only and wrote every output into this directory.\n")
    d.w("| path | contents |")
    d.w("|---|---|")
    d.w("| `qc_exciton_LCC/src/chem/thc/survey/all_materials_norms.csv` | three norms plus every ingredient, optimised family |")
    d.w("| `qc_exciton_LCC/src/chem/thc/survey/all_materials_norms.jsonl` | the same plus a partial sample of other families |")
    d.w("| `qc_exciton_LCC/src/chem/thc/survey/irrep_structure.csv` | symmetry block structure and compression factors |")
    d.w("| `qc_exciton_LCC/src/chem/thc/survey/irreps_raw/` | verbatim copies of jsun3's `log_irreps/`, 41 files |")
    d.w("| `qc_exciton_LCC/src/chem/thc/survey/W_cost_grid-grid.txt` | jsun3's circuit-cost estimate from the block structure |")
    d.w("| `qc_exciton_LCC/src/chem/thc/data/recovered/` | two checkpoints regenerated here, absent from the source tree |")
    d.w("| `qc_exciton_LCC/src/chem/thc/survey_norms.py` and companions | regenerate everything above |")

    d.h(3, "1.2 Inventory")
    d.w("Seven systems carry checkpoints, and five families partition them. The families")
    d.w("differ in which orbital block they factorise, whether the Coulomb interaction is")
    d.w("screened, and whether the collocation matrix was re-optimised afterwards.\n")
    if norms:
        d.w("| material | basis | n_ao | n_occ | n_vir | checkpoints |")
        d.w("|---|---|---|---|---|---|")
        for m in mats:
            g = [r for r in norms if r["material"] == m]
            d.w(f"| {PRETTY.get(m, m)} | `{g[0]['basis']}` | {g[0]['n_occ']+g[0]['n_vir']} | "
                f"{g[0]['n_occ']} | {g[0]['n_vir']} | {len(g)} |")
        d.w("")
        d.w("| family | block | on disk | measured here |")
        d.w("|---|---|---|---|")
        for fam in FAMILIES:
            if not ondisk.get(fam):
                continue
            d.w(f"| `{fam}` | {FAM_DESC[fam]} | {ondisk[fam]} | {measured.get(fam, 0)} |")
        d.w("")
        d.w("This survey measured the optimised family completely and left the other families")
        d.w("partly measured, by design: only the optimised checkpoints carried a norm in")
        d.w("their objective, so only their values bear on a cost estimate. Sections 2.3 and")
        d.w("2.4 therefore report the optimised family alone.")
        d.w("`qc_exciton_LCC/src/chem/thc/survey/all_materials_norms.csv` carries the optimised family alone, matching")
        d.w("the tables here. `qc_exciton_LCC/src/chem/thc/survey/all_materials_norms.jsonl` additionally retains the")
        d.w("other families the survey reached before it stopped; treat those as an")
        d.w("incomplete sample. Rerunning `survey_norms.py` without `--pattern` completes them.\n")
    d.w("The symmetry-adapted family and the optimised family never overlap. A search of the")
    d.w("whole tree for an optimiser output tagged `_symm` or `_real` returns nothing, so no")
    d.w("checkpoint combines the two. `optimize_X_ov.py --symm` supports the combination, and")
    d.w("nobody ran it.\n")
    d.w("Two further collections sit outside `THC_general`. `THC_diamond/data_GDF/` holds 30")
    d.w("`ISDFfull_opt_*` checkpoints and 12 saved optimiser states, all diamond, basis")
    d.w("`gth-dzvp`, all-orbital block. Those differ from everything measured here in both")
    d.w("basis and block, so the tables below exclude them.\n")
    d.w("The optimised family used a reference rank that varies per material: diamond and BN")
    d.w("used `cref12`, AlN used `cref16`, and MgO, black P, CdSe and MoSe2 used `cref20`.")
    d.w("Each checkpoint's fit error refers to its own reference. Comparing fit quality across")
    d.w("materials therefore compares against different targets.\n")

    # ---------------- 2
    d.h(2, "2. Embed norms")
    d.w("Three scalars bound a block encoding built from the THC factors, and they differ")
    d.w("only in which matrix norm measures each collocation factor. Section 2.1 defines")
    d.w("them, Section 2.2 establishes which basis the 2-to-infinity norm depends on,")
    d.w("Section 2.3 summarises the ratios, and Section 2.4 lists every checkpoint.\n")
    d.h(3, "2.1 Definitions")
    d.w("Each norm multiplies a squared occupied factor, a squared virtual factor, and one")
    d.w("central-tensor factor. Write `X^(o,k)` and `X^(v,k)` for the occupied and virtual")
    d.w("collocation matrices at k-point `k`, each of shape `M x n_orb`, and write `W^Q` for")
    d.w("the central tensor at momentum transfer `Q`. The 2-to-infinity norm of a matrix")
    d.w("equals its largest row 2-norm, `max_I ‖A_(I,:)‖_2`, maximised here over `k` as")
    d.w("well.\n")
    d.w("| name | formula |")
    d.w("|---|---|")
    d.w("| `alpha_op` | `(max_k ‖X^(o,k)‖_op)^2 (max_k ‖X^(v,k)‖_op)^2 max_Q ‖W^Q‖_op` |")
    d.w("| `alpha_mod_v` | `(max_k ‖X^(o,k)‖_op)^2 (max_(k,I) ‖X^(v,k)_(I,:)‖_2)^2 max_Q ‖W^Q‖_op` |")
    d.w("| `alpha_mod_o` | `(max_(k,I) ‖X^(o,k)_(I,:)‖_2)^2 (max_k ‖X^(v,k)‖_op)^2 max_Q ‖W^Q‖_op` |")
    d.w("")
    d.w("Every row norm of a matrix is at most its operator norm, so both modified norms")
    d.w("upper-bound nothing larger than `alpha_op`. Neither modified norm bounds the other.")
    d.w("The occupied and virtual factors carry different column counts, which is what makes")
    d.w("the two substitutions differ in strength.\n")
    d.w("The checkpoints store neither `X^o` nor `X^v`. Both follow from the stored AO-basis")
    d.w("collocation matrix `inpv_kpt` and the mean-field coefficients, as")
    d.w("`X^(o,k) = X^(ao,k) C^(occ,k)`. Section 5 states which mean field pairs with which")
    d.w("checkpoint.\n")
    d.h(3, "2.2 Basis dependence")
    d.w("The 2-to-infinity norm depends on the row basis and ignores the column basis, which")
    d.w("suits the quantity it bounds. Rotating the orbital index `p` changes it by")
    d.w("1.3e-16, whereas mixing the interpolation index `I` changes it by 1.9e-01. The")
    d.w("operator norm, by contrast, ignores both, changing by 1.6e-16 and 3.2e-16")
    d.w("respectively.\n")
    d.w("This asymmetry lands the right way round. An LCU select oracle enumerates")
    d.w("interpolation points, so sensitivity to `I` reflects a real cost, while the MO gauge")
    d.w("carries no physical content. One practical consequence follows: evaluating the same")
    d.w("checkpoint against two different mean fields moves `alpha_mod_v` by 1.8e-6 even")
    d.w("though the factors themselves differ by 139 percent.\n")
    d.w("The measurements below evaluate the norm in the `(k, I, p)` basis throughout. No")
    d.w("irrep transformation enters. Section 3 quantifies what an irrep basis would change.\n")

    if norms:
        optn = [r for r in norms if "opt" in r["family"]]
        d.h(3, "2.3 Ratios across the optimised family")
        d.w("The ratios below compare the three norms within each material, over the")
        d.w("optimised checkpoints. Ratios travel across materials; the absolute values do")
        d.w("not, for the reason Section 5 gives. Each entry states the minimum, the maximum,")
        d.w("and the median over that material's meshes.\n")
        d.w("| material | n | alpha_op/alpha_mod_v | alpha_op/alpha_mod_o | alpha_mod_o/alpha_mod_v |")
        d.w("|---|---|---|---|---|")
        rng = lambda v: f"{min(v):.2f}-{max(v):.2f} (med {st.median(v):.2f})"
        for m in sorted({r["material"] for r in optn}):
            g = [r for r in optn if r["material"] == m]
            d.w(f"| {PRETTY.get(m, m)} | {len(g)} | "
                f"{rng([r['alpha_op']/r['alpha_mod_v'] for r in g])} | "
                f"{rng([r['alpha_op']/r['alpha_mod_o'] for r in g])} | "
                f"{rng([r['alpha_mod_o']/r['alpha_mod_v'] for r in g])} |")
        norms_for_count = optn
        n_lo = sum(1 for r in optn if r["alpha_mod_o"] < r["alpha_mod_v"])
        d.w("")
        d.w(f"Swapping the occupied factor lowers the norm further than swapping the virtual")
        d.w(f"factor, in {n_lo} of {len(optn)} optimised checkpoints "
            f"({100*n_lo/len(optn):.0f} percent).")
        d.w("The occupied factor carries fewer columns, so its operator norm exceeds its")
        d.w("largest row norm by more, and that excess enters squared.\n")

        opt = [r for r in norms if "opt" in r["family"]]
        d.h(3, "2.4 Values for the optimised checkpoints")
        d.w("The optimised family is the one whose norms carry meaning for a cost estimate,")
        d.w("because only those checkpoints had a norm in their objective at all. Every such")
        d.w("checkpoint appears below, across all materials and meshes, with the three norms")
        d.w(f"and the five ingredients that build them. The set contains {len(opt)} checkpoints,")
        d.w("which is every optimised checkpoint in the tree.\n")
        d.w("| material | mesh | c | M | alpha_op | alpha_mod_v | alpha_mod_o | "
            "‖X^o‖_op | ‖X^v‖_op | ‖X^o‖_2inf | ‖X^v‖_2inf | max_Q‖W^Q‖_op |")
        d.w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(opt, key=lambda x: (x["material"], mesh_key(x["mesh"]), x["c"])):
            d.w(f"| {PRETTY.get(r['material'], r['material'])} | {r['mesh']} | {r['c']} | "
                f"{r['M']} | {r['alpha_op']:.4g} | {r['alpha_mod_v']:.4g} | "
                f"{r['alpha_mod_o']:.4g} | {r['xo_op']:.4g} | {r['xv_op']:.4g} | "
                f"{r['xo_2inf']:.4g} | {r['xv_2inf']:.4g} | {r['w_op_max']:.4g} |")
        d.w("")
        d.w("Every row shares `c=5`, because the optimiser was run at that rank alone. The")
        d.w("reference rank behind each row varies by material, as Section 1.2 records.\n")

    # ---------------- 3
    d.h(2, "3. Symmetry structure")
    d.w("The crystal symmetry implies a block structure that the fitting pipeline never")
    d.w("exploits. jsun3's `analyze_isdf_irreps.py` measured that structure and wrote it to")
    d.w("`log_irreps/`; no other script reads those files. Section 3.1 reports the")
    d.w("compression the block structure implies, Section 3.2 lists the blocks themselves,")
    d.w("and Section 3.3 distinguishes storage compression from circuit cost.\n")
    d.h(3, "3.1 Representation sizes and compression")
    d.w("A representation decomposing as a sum of irreps of dimension `d_i` with")
    d.w("multiplicity `m_i` constrains any operator that commutes with the group. Such an")
    d.w("operator becomes block diagonal, carrying one independent `m_i x m_i` block per")
    d.w("irrep and repeating it `d_i` times. Counting parameters therefore gives")
    d.w("`(sum_i d_i m_i)^2` dense against `sum_i m_i^2` independent.\n")
    if irreps:
        d.w("| material | mesh | ops | W dense | W independent | W compression | X dense | X independent | X compression |")
        d.w("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(irreps, key=lambda x: (x["material"], mesh_key(x["mesh"]))):
            d.w(f"| {PRETTY.get(r['material'], r['material'])} | {r['mesh']} | {r['n_ops']} | "
                f"{int(r['W_dense']):,} | {int(r['W_independent']):,} | "
                f"{float(r['W_compression']):.0f}x | {int(r['X_dense']):,} | "
                f"{int(r['X_independent']):,} | {float(r['X_compression']):.1f}x |")
        d.w("")
        d.w("Only diamond and MgO, and only rank `c=5`, carry this analysis. The full point")
        d.w("group survives in both systems, giving 48 operations. The symmetry checks inside")
        d.w("those logs confirm the loaded factors sit on the symmetric subspace to machine")
        d.w("precision.\n")
        d.w("| material | mesh | grid invariance | AO commutator | X symmetry | W symmetry |")
        d.w("|---|---|---|---|---|---|")
        for r in sorted(irreps, key=lambda x: (x["material"], mesh_key(x["mesh"]))):
            d.w(f"| {PRETTY.get(r['material'], r['material'])} | {r['mesh']} | "
                f"{float(r['grid_invariance_relfro']):.2e} | {float(r['ao_commutator_relfro']):.2e} | "
                f"{float(r['X_symmetry_relfro']):.2e} | {float(r['W_symmetry_relfro']):.2e} |")
        d.w("")
        d.w("An independent measurement confirms the `X` count. For diamond at 2x2x2 the irrep")
        d.w("decomposition gives exactly 714 independent parameters, and a Hutchinson trace")
        d.w("estimate of the symmetrisation projector used by `optimize.py --symm` gives")
        d.w("713.5 plus or minus 1.6. The projector acts within the `(k, I, p)`")
        d.w("parametrisation rather than changing basis, so an optimiser running under")
        d.w("`--symm` carries roughly 40 times more coordinates than the problem has degrees")
        d.w("of freedom.\n")

    logs = sorted(glob.glob(os.path.join(args.survey_dir, "irreps_raw", "irreps_ov_*_final.log")))
    if logs:
        d.h(3, "3.2 Block structure, as the logs report it")
        d.w("Three decompositions appear in each log, and the blocks below reproduce them")
        d.w("verbatim. Reading them requires one convention: a row states an irrep dimension,")
        d.w("how many times that irrep occurs in the representation, and the product of the")
        d.w("two. The product column sums to the representation size, which the dimension")
        d.w("check confirms.\n")
        d.w("The three decompositions describe different objects. The first covers the")
        d.w("selected-grid representation, of size `n_k * M`, on which `W^Q` acts; an operator")
        d.w("commuting with the group carries one independent block of size `multiplicity x")
        d.w("multiplicity` per row, repeated `irrep_dim` times. The second covers the")
        d.w("AO representation, of size `n_k * n_ao`. The third pairs grid irreps with AO")
        d.w("irreps and states, per matching irrep, the dense shape of the block that `X` may")
        d.w("occupy; summing `grid_mult * ao_mult` over rows counts the free parameters of a")
        d.w("symmetry-respecting `X`.\n")
        for logfile in logs:
            m = re.match(r"irreps_ov_(.+?)_(\d+x\d+x\d+)_c(\d+)_final\.log",
                         os.path.basename(logfile))
            text = open(logfile).read()
            d.w(f"\n**{PRETTY.get(m.group(1), m.group(1))}, {m.group(2)}, c={m.group(3)}**\n")
            scal = [ln for ln in text.splitlines()
                    if re.match(r"^(nkpts|n selected|nao|n symmetry|grid representation"
                                r"|AO representation|grid max|AO max|loaded)", ln)]
            d.w("```")
            for ln in scal:
                d.w(ln)
            d.w("```\n")
            for start, stop in [(r"^W / selected-grid block structure", r"^W dimension check.*"),
                                (r"^AO/orbital block structure", r"^AO dimension check.*"),
                                (r"^X allowed blocks.*", None)]:
                lines = text.splitlines()
                try:
                    i = next(j for j, ln in enumerate(lines) if re.match(start, ln))
                except StopIteration:
                    continue
                if stop is None:
                    block = lines[i:]
                else:
                    k = next(j for j, ln in enumerate(lines) if j > i and re.match(stop, ln))
                    block = lines[i:k + 1]
                block = [ln for ln in block if ln.strip() != ""]
                d.w("```")
                for ln in block:
                    d.w(ln)
                d.w("```\n")

    d.h(3, "3.3 Circuit cost against storage compression")
    d.w("Storage compression and circuit cost respond differently to the same block")
    d.w("structure, and conflating them overstates the benefit by two orders of magnitude.")
    d.w("Storage scales as the square of the block sizes, whereas the cost model in")
    d.w("`get_W_cost.py` scales as `sqrt(N_k) * n_thc^1.5`, a root of them. The table below")
    d.w("reproduces jsun3's estimate.\n")
    cost_path = os.path.join(args.survey_dir, "W_cost_grid-grid.txt")
    if os.path.exists(cost_path):
        d.w("| case | c | N_k | n_thc | blocks | max mult | dense cost | blocked cost | ratio | grouped cost | ratio | optimal cost | ratio |")
        d.w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for line in open(cost_path):
            if line.startswith("#"):
                continue
            f = line.split()
            d.w("| " + " | ".join(f) + " |")
        d.w("")
        d.w("The blocked column groups nothing, the grouped column packs multiplicities")
        d.w("greedily, and the optimal column sums over multiplicity thresholds. Ratios reach")
        d.w("8.2, 11.6 and 15.8 respectively at the smallest mesh and shrink as the mesh")
        d.w("grows. Storage compression over the same cases spans 350 to 10050.\n")

    # ---------------- 4
    d.h(2, "4. Conventions in the generating code")
    d.w("Four conventions govern how the checkpoints came about, and none appears inside the")
    d.w("files. Reading them out of jsun3's source settled several questions that the data")
    d.w("alone left open. Each entry below cites the line that fixes it.\n")
    d.w("1. **The optimiser minimises `alpha_op / n_k` plus a fit term.**")
    d.w("   `optimize_X_ov.py:50` forms the product of operator norms, and")
    d.w("   `optimize_X_common.py:147` adds it to the fit error with weight")
    d.w("   `norm_const * nk**power`, where `nk` denotes the linear mesh dimension. A")
    d.w("   `norm0.1p2` run at 6x6x6 therefore weights the norm by 3.6.")
    d.w("2. **The mean-field pickle follows the `_symm` tag.** Checkpoints whose names carry")
    d.w("   `_symm` pair with `DFT_<mesh>_symm.pkl`; all others pair with `DFT_<mesh>.pkl`")
    d.w("   (`generate_isdf_gdf_symm.py:180,182`, `optimize_X_ov.py:164`).")
    d.w("3. **Every family stores the AO-basis collocation matrix.**")
    d.w("   `generate_isdf_gdf.py:241` and `optimize_X_ov.py:286` both write `X_ao`, so the")
    d.w("   same reconstruction applies throughout.")
    d.w("4. **Symmetry-adapted selection takes whole orbits.**")
    d.w("   `generate_isdf_gdf_symm.py:93` extends the selected set by a complete symmetry")
    d.w("   orbit at each step, which is why `M` exceeds `c * n_ao` and lands on orbit")
    d.w("   boundaries.\n")

    # ---------------- 5
    d.h(2, "5. Traps")
    d.w("Each item below produces a plausible wrong answer rather than an error, and several")
    d.w("cost real time during this survey. Read them before computing anything from the")
    d.w("dataset. The first three concern reproducing the data; the rest concern")
    d.w("interpreting it.\n")
    d.w("1. **Pairing the wrong mean field corrupts the collocation factors silently.** The")
    d.w("   two mean fields differ by unitary mixing inside degenerate bands, so a mismatch")
    d.w("   yields a valid tensor in the wrong MO gauge. Element-wise discrepancies exceeding")
    d.w("   100 percent follow, with no error raised.")
    d.w("   `chem.lcu_norms.loaders.dft_checkpoint_for` encodes the rule.")
    d.w("2. **Two script versions produced the shipped `c5` and `c12` ov references.** Their")
    d.w("   logs record `reg0 = 1e-9` for `c5` but `reg0 = 1e-10` for `c12`, and the current")
    d.w("   `generate_isdf_gdf.py` hardcodes `1e-9` without a flag. Regenerating both today")
    d.w("   reproduces `c5` exactly and gives `c12` a regularisation of `5.12e-7` instead of")
    d.w("   `4.096e-7`. `qc_exciton_LCC/src/chem/thc/vendor/generate_isdf_gdf_reg0.py` exposes `-reg0`.")
    d.w("3. **`optimize_X_ov.py` cannot run as shipped.** `optimize_X_common.get_device()`")
    d.w("   imports a `gpu_register` module absent from the tree. `qc_exciton_LCC/src/chem/thc/reproduce_jsun3.sh` writes")
    d.w("   a CPU stub.")
    d.w("4. **The optimiser's own inputs no longer exist.** It reads")
    d.w("   `ISDFov_bareGDF_<mesh>_c5.chk` and `_c12.chk`; neither survives for diamond, and")
    d.w("   no `opt_X_ov_*.pt` state survives either. `qc_exciton_LCC/src/chem/thc/data/recovered/` holds regenerated")
    d.w("   copies for 2x2x2, which exist nowhere else.")
    d.w("5. **No artifact records `use_Fnorm`, `base_lr` or `nsteps_factor`.** No")
    d.w("   `optimize_X` invocation was captured anywhere in the tree, so those three")
    d.w("   settings are unrecoverable. The `_symm` tag does record `use_symm`.")
    d.w("6. **Norms vary non-monotonically with rank.** `generate_isdf_gdf*.py` doubles its")
    d.w("   regularisation until the fit error reaches 1.5 times its best value and keeps that")
    d.w("   solution, so the selected regularisation jumps between rank values. Holding it")
    d.w("   fixed restores monotonicity: the `c=6` and `c=8` runs happen to draw the same")
    d.w("   regularisation at every mesh, and `‖W‖` falls with rank in all five cases.")
    d.w("7. **The stored fit is deliberately about 1.5 times worse than achievable**, by that")
    d.w("   same stopping rule.")
    d.w("8. **A scale gauge leaves every norm invariant.** Rescaling `X -> sX` with")
    d.w("   `W -> W/s^4` preserves both the represented tensor and all three norms, so an")
    d.w("   optimiser drifts freely along it. Two runs of one configuration differed by 28")
    d.w("   percent in `‖X‖` while agreeing to 0.1 percent on the gauge-invariant spectrum.")
    d.w("9. **Families measure different objects.** The `ISDFfull_*` families never fitted the")
    d.w("   occupied-virtual block, so an ov-block norm evaluated on them is valid arithmetic")
    d.w("   applied to the wrong target.")
    d.w("10. **The optimised family was trained on `alpha_op`.** Its small values reflect that")
    d.w("    objective, and its `alpha_mod_*` values come from factors never optimised for them.")
    d.w("11. **None of these three scalars is a validated LCU lambda.** The `4/n_k` prefactor")
    d.w("    in `bse_thc.py:apply_V_thc` stays outside them, and no encoding convention is")
    d.w("    assumed.\n")

    # ---------------- 6
    d.h(2, "6. Regeneration")
    d.w("Three scripts rebuild everything in this directory, and each accepts a restricted")
    d.w("scope for quick checks. Running them in order reproduces the CSV files and this")
    d.w("document. The full survey takes roughly 40 minutes on CPU.\n")
    d.w("```")
    d.w("cd qc_exciton_LCC/src/chem/thc")
    d.w("python survey_norms.py                    # all materials -> survey/*.csv, *.jsonl")
    d.w("python survey_norms.py --pattern opt      # optimised family only")
    d.w("python collect_irreps.py                  # copy and parse log_irreps/")
    d.w("python build_reference.py                 # rebuild this file into context/")
    d.w("```\n")
    d.w("`chem.lcu_norms` supplies the norm definitions that `survey_norms.py` calls, and")
    d.w("`check_relations` asserts every inequality among them. Editing a definition in one")
    d.w("place therefore propagates to the survey, the optimiser and this document.\n")

    open(args.out, "w").write(d.render())
    print(f"wrote {args.out}: {len(norms)} norm rows, {len(irreps)} irrep rows, "
          f"{len(d.toc)} TOC entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
