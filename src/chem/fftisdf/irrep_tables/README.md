# Space-group irrep block tables — diamond, gth-cc-pvdz

Block structures for the symmetry-adapted ISDF–THC BSE block encoding. Consumed by
`src/integrations/qualtran/symmetry_adapted_bse.py` (`load_irrep_data`).

## Provenance

Produced by jsun3's pipeline. The `c5` tables and logs are **copied verbatim** from
`/central/groups/changroup/members/jsun3/xprize/THC_general/log_irreps/`.

The `c8` tables were generated here on 2026-08-20 by running jsun3's analysis on the
`c8` ISDF checkpoints, which already existed:

```bash
cd /central/groups/changroup/members/jsun3/xprize/THC_general
export OMP_NUM_THREADS=8
python -u analyze_isdf_irreps.py diamond <n> <n> <n> gth-cc-pvdz 8 -suffix final --ov
```

Inputs: `data_diamond_gth-cc-pvdz_final/ISDFov_bareGDF_symm_<mesh>_c8.chk` and
`DFT_<mesh>_symm.pkl`. `analyze_isdf_irreps.py` is read-only — it prints to stdout and
writes nothing, so nothing in jsun3's tree was modified. Logs are in `logs/`.

`extract_irrep_tables.py` here is jsun3's parser, pointed at those logs.

## The three tables

All are whitespace-separated integers, **one row per irrep**, 3 columns.

**`grid-grid`** — log section *"W / selected-grid block structure"*, header
`irrep_dim multiplicity total_size`:

| col 1 | col 2 | col 3 |
|---|---|---|
| `m_i` (irrep dimension) | `n_i` (copies in the selected-grid space) | `m_i · n_i` |

`W`'s block structure: block `w_i` is `n_i × n_i`, carried `m_i` times. Checked in the log
as `W dimension check = n_IP·N_k / n_IP·N_k`.

**`ao-ao`** — *"AO/orbital block structure"*, same header, with `n^ao_i` in col 2.
Checked as `AO dimension check = n_orb·N_k / n_orb·N_k`. This is the block structure the
Fock term (`C_0`) needs.

**`grid-ao`** — *"X allowed blocks: selected-grid irreps against AO irreps"*, header
`irrep_dim  grid_mult  ao_mult  dense_shape`:

| col 1 | col 2 | col 3 | (col 4) |
|---|---|---|---|
| `m_i` | `n^grid_i` | `n^ao_i` | `dense_shape`, e.g. `7x2` — **dropped by the extractor** |

`X`'s block structure: block `x_i` is `n^grid_i × n^ao_i`, carried `m_i` times.
**Note col 3 means something different here** — `ao_mult`, not `total_size`.

## Two things that trip you up

**`grid-ao` is shorter than `grid-grid`.** It lists only irreps where *both* sides are
nonzero — that is what "allowed blocks" means. At 6×6×6 `c5` it is 49 rows vs 57. The
missing rows are grid irreps with no AO content; they exist in `W` and not in `X`, which
is correct, not a data error. Consequence: the tables are **not row-aligned** — join on
the irrep label, never on position.

**The occ/virt split is not in the data.** `ao-ao` and `grid-ao` give `n^ao_i` per irrep
but not how it divides into occupied and virtual. `load_irrep_data` currently distributes
it proportionally and corrects to satisfy `Σ_i m_i n^occ_i = N_o·N_k` exactly. That is a
plausible split, not the measured one, and it is the last estimated input in the cost
model. Getting the true split needs the occ/virt classification of the symmetrized MOs,
downstream of jsun3's `symmetrize_mf.py`.

## Contents

| c | mesh | N_k | n_IP | groups | grid-grid rows | ao-ao rows | grid-ao rows | max m_i | max n_i | max n_ao_i |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 2x2x2 | 8 | 136 | 7 | 20 | 14 | 14 | 8 | 24 | 6 |
| 5 | 3x3x3 | 27 | 136 | 7 | 22 | 17 | 17 | 16 | 44 | 10 |
| 5 | 4x4x4 | 64 | 136 | 7 | 36 | 29 | 29 | 24 | 80 | 18 |
| 5 | 5x5x5 | 125 | 136 | 7 | 40 | 34 | 34 | 24 | 80 | 18 |
| 5 | 6x6x6 | 216 | 136 | 7 | 57 | 49 | 49 | 24 | 80 | 18 |
| 8 | 2x2x2 | 8 | 208 | 9 | 20 | 14 | 14 | 8 | 34 | 6 |
| 8 | 3x3x3 | 27 | 208 | 9 | 22 | 17 | 17 | 16 | 68 | 10 |
| 8 | 4x4x4 | 64 | 208 | 9 | 36 | 29 | 29 | 24 | 118 | 18 |
| 8 | 5x5x5 | 125 | 208 | 9 | 40 | 34 | 34 | 24 | 118 | 18 |

`n_IP` is per unit cell and independent of `N_k`. At `c=5` the target `5·26 = 130` rounds
up to **136** because the selected set must be a union of whole point-group orbits
(7 of them: `48+24+24+12+12+8+8`). At `c=8` the target `8·26 = 208` is hit **exactly**,
with 9 groups.

The irrep **counts** are identical at `c=5` and `c=8` for every mesh — irrep content is
fixed by the space group and the k-mesh; only the multiplicities scale with `n_IP`.

`ISDFov_bareGDF_symm_*` checkpoints also exist for `c ∈ (6, 10, 16, 20)`, so further
points on the `n_IP` curve are cheap to add with the command above.
