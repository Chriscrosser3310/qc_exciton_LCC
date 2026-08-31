# Symmetry-adapted ISDF–THC BSE in Qualtran — implementation spec

**Purpose.** Everything a fresh agent needs to implement the *fully symmetry-adapted* ISDF–THC
block encoding for the multi-exciton BSE effective Hamiltonian in Qualtran: the block structures of
`X` (=χ) and `W` (=ζ/central), both legs in the irrep basis; the `Q` transform that produces them; the
occ/virt split; the full tensor network; and — crucially — **which pieces are group-determined
(material-free) vs. material-dependent, and where the material data lives**.

Repo: `/resnick/home/jchen9/XPRIZE/qc_exciton_LCC`. Running example: **diamond** (space group `Fd-3m`,
point group `O_h`, order `|P|=48`, nonsymmorphic).

---

## 0. Naming
- **`X` = χ** = ISDF interpolation vectors: `X[k, I, p] = φ_{p,k}(r_I)` — value of orbital/band `p`
  (momentum `k`) at interpolation point `I`. Legs: **grid `(k,I)`** ↔ **orbital `(k,p)`**.
- **`W` = ζ-metric** = central Coulomb tensor between interpolation points: `W[q, I, J]`
  (`q` = momentum transfer). Legs: **grid** ↔ **grid**.
- ov ERI (THC form): `(i k_i, a k_a | j k_j, b k_b) = Σ_{IJ} [X^o*_{k_i,I,i} X^v_{k_a,I,a}] W[q]_{IJ}
  [X^o*_{k_j,J,j} X^v_{k_b,J,b}]`, momentum transfer `q = k_i − k_a`. The pointwise product
  `P_I = X^o* X^v` is the **co-density / fusion vertex** (a COPY tensor on `I`).
- BSE effective Hamiltonian = **direct** term (bare Coulomb, `W` used as a *diagonal* central tensor)
  + **exchange** term (screened Coulomb, `W` used as an *eigendecomposed* operator = "diagram rotated
  90°").

---

## 1. Symmetry labels and the `Q` transform

**Q maps the grid basis to the irrep basis:** `Q : |k⟩|I⟩ → |i⟩|a⟩|c⟩`, block-diagonal in `k`.
Label bookkeeping (a *regrouping*, not a tensor factorization — ranges depend on family):
```
k = (f, κ)             f = family/star,  κ = star-member index
I → (μ, ν, c)          μ = little-group irrep, ν = partner (1..d_μ), c = copy (multiplicity)
i = (f, μ)             full space-group irrep
a = (κ, ν)             full partner index,  m_i = |star_f|·d_μ
c = c                  copy (the ONLY material-data slot)
```

**Factorization (block-diagonal in k):** `Q^{(k)} = P(g_k) · Λ(g_k, k) · Q^{(f(k))}`
- `f(k)` = family; `g_k` = coset op (`g_k·k_0 = k`); `k_0` = family representative.
- `Q^{(f)}` = little-group reduction on the `n_IP`-point fiber at `k_0`, itself **block-diagonal over
  little-group orbits of interpolation points**: `Q^{(f)} = ⊕_o U_o`, each `U_o` an `|o|×|o|` unitary,
  `|o| ≤ |L_{k_0}| ≤ 48`.
- `Λ(g,k)` = diagonal Bloch/glide phase `e^{-i k·L_I(g)}`; `P(g)` = fiber permutation `I ↦ perm[g,I]`.
- `L_I(g)` = fold-back lattice vector from `g·r_I = r_{perm[g,I]} + L_I(g)` (k-independent).

**How `Q^{(f)}` is built (classical, group-determined):** for each `g ∈ L_{k_0}` build the
phased-permutation `D(g)` on the fiber; split by orbit; reduce each orbit into little-group irreps via
**character projection** `P^μ = (d_μ/|L|)Σ_g χ_μ(g)* D(g)` OR **symmetrize-a-random-Hermitian + eigh**
(`H = Σ_g D_o(g) H_0 D_o(g)†`, diagonalize, label by characters). Fix the copy-space gauge (one global
phase per irrep block). Induce to members via the phased permutation. See `cg_transform.py`
(verifies `Q† D(g) Q = ⊕_i D_i(g)⊗𝟙_{m_i}`, leak ~2e-13).

**Real-space variant:** `Q' : |R⟩|I⟩ → |i⟩|a⟩|c⟩` on real-space cells `R` equals `Q' = Q·(F⊗𝟙)`,
`F` = cell DFT on `R↔k`, `𝟙` on the interpolation register.

**`Q` is the point-group half of the space-group generalized Fourier; the copy tensor (fusion vertex),
Fourier-transformed by `Q`, becomes the Clebsch–Gordan/fusion tensor** (momentum-δ × point-group CG on
partners × reduced tensor on copies). Abelian shadow = momentum conservation.

---

## 2. `W` block structure (both legs irrep basis)

`W` commutes with the space group ⇒ in the irrep basis it is
```
Q† W Q = ⊕_i  𝟙_{m_i} ⊗ w_i
```
- **block-diagonal by `i=(f,μ)`**, **identity on the partner `a=(κ,ν)`**, a **free `n_i×n_i` matrix
  `w_i` on the copies `c`** (`n_i` = multiplicity of irrep `i`).
- Coupling of two grid legs (Schur): `W_{(iac),(i'a'c')} = δ_{ii'} δ_{aa'} (w_i)_{cc'}`. The `(i,a)`
  match is momentum conservation (`κ`) + point-group selection (`μ,ν`); `w_i` acts only on copies.
- **Unique data = Σ_i n_i²** (the `w_i` entries). This is MATERIAL.
- **Direct term:** apply `W` as the diagonal central coefficient → diagonal BE over the reduced values.
- **Exchange term:** eigendecompose `w_i = u_i D_i u_i†` → `W = U D U†`, `U = ⊕_i 𝟙_{m_i}⊗u_i`.
  **The eigenvector unitary `U` (and `U†`) MUST be synthesized — it is not free.** `D` = diagonal
  (`Σn_i` eigenvalues). (Hermitian ⇒ ordinary eig; complex-symmetric ⇒ Takagi.)

---

## 3. `X` (=χ) block structure (both legs irrep basis)

`X` intertwines the **grid rep** and the **orbital rep**. Adapt *both* legs (grid via `Q_grid`,
orbital via `Q_orb` — see §4) ⇒
```
Q_grid† X Q_orb = ⊕_i  𝟙_{m_i} ⊗ x_i
```
- **block-diagonal by `i`**, **identity on partner `a`**, `x_i` a **rectangular `n_grid_i × n_orb_i`**
  block (tall-thin: `n_grid_i ≫ n_orb_i`).
- Present only for irreps in BOTH grid and orbital reps.
- **Unique data = Σ_i n_grid_i · n_orb_i.** MATERIAL.
- Dimension checks (per cell): `Σ_i m_i n_grid_i = n_IP·N_k`(grid dim), `Σ_i m_i n_orb_i = n_orb·N_k`.

---

## 4. Orbital leg & the occ/virt split (`X_o`, `X_v`)

**Bands are automatically symmetry-adapted** (the Hamiltonian commutes with the space group), so the
orbital-leg transform `Q_orb` = the band symmetry (no separate SALC step, unlike raw AOs). `X_o` (occ)
and `X_v` (virt) **share the grid side** and split the orbital copies:
```
n_orb_i = n_occ_i + n_virt_i        X_o block: n_grid_i × n_occ_i,   X_v block: n_grid_i × n_virt_i
```
**occ/virt projector** (in the irrep basis): `P_o = ⊕_i 𝟙_{m_i} ⊗ p_i^o`
- block-diagonal by `i`, **identity on partner `a`**, a copy-space projector `p_i^o` on `c`.
- With **energy-ordered copies** (diagonalize the per-irrep Hamiltonian block `h_i`; natural if `X` is
  built from bands), `p_i^o = diag(1..1,0..0)` ⇒ `P_o` is a **diagonal mask "keep `c < n_occ_i`"**.
- **Circuit:** load `n_occ_i` from a tiny QROM indexed by `i`, compare `c < n_occ_i` (`LessThanConstant`).
  `O(log)` + small table. `X_v` = complement.
- **The only material input for the split is the small integer table `{n_occ_i}`** (per-irrep occ
  counts = the band symmetry). Everything else (block-diagonal, partner-identity) is symmetry-fixed.
- Global constraints: `Σ_i m_i n_occ_i = n_occ·N_k`, `Σ_i m_i n_virt_i = n_virt·N_k`.

For diamond the occupied manifold is the **sp³ valence** = `A₁ ⊕ T₂` at Γ (group theory of the 4
tetrahedral bonds); insulator ⇒ complete multiplets; a shape generally has BOTH occ and virt copies, so
`{n_occ_i}` genuinely needs the band symmetry (not just the count `n_occ`).

---

## 5. Full ERI tensor network (what the block encoding assembles)

```
ERI = (X ⊗ X)  (Q ⊗ Q)†  [W in irrep basis]  (Q ⊗ Q)  (X ⊗ X)
```
Practically, per BSE term:
- **6 `Q`'s** (reused): one per grid leg — 4 X-grid-legs + 2 W-grid-legs — convert irrep-stored tensors
  to the grid basis at the two fusion (co-density) vertices, because the fusion is a pointwise product
  (sparse only in the grid basis; densifies into CG in the irrep basis).
- **Register-recording** = compute the canonical space-group orbit representative of the grid-point pair
  (the general-symmetry analog of `R₁−R₂`): subtract cell difference + canonicalize over the 48 point
  ops (perm + fold-back + argmin). Addresses the reduced `W`.
- **Direct `W`** = diagonal BE over reduced values; **exchange `W`** = `U D U†` (synthesize `U`, `U†`).
- **X** = block isometries per §3–4.

Momentum handled as QROM address prefix (construction-A equal blocks): cost enters via `⌈·/λ⌉`, NOT as
a naive `×N_k`. **Result: cost is `N_k`-independent** in the dominant terms.

---

## 6. Material dependence — what is data, and where it lives

**Group-determined (MATERIAL-FREE, hard-codeable, no stored numbers):**
- `Q`'s orbit blocks `U_o` (little-group Fourier/CG), the perm/fold-back tables `perm[g,I]`, `L_I(g)`,
  the point ops `{A_g, τ_g}`, the CG/fusion coefficients, the family/coset structure. All derivable from
  the ~`(n_IP + |P|)` geometry kernel.

**Material-dependent (must be supplied/derived):**
- `w_i` — the `W` copy-blocks (`Σn_i²` complex numbers).
- `x_i` — the `X` copy-blocks (`Σ n_grid_i·n_orb_i`), split into occ/virt.
- `{n_occ_i}` — per-irrep occupancy thresholds (band symmetry).
- Irrep block dimensions `{(m_i, n_grid_i, n_orb_i, n_i)}` (the block table).

**Where to get it — jsun3's data:**
`/central/groups/changroup/members/jsun3/xprize/THC_general/`
- `data_diamond_gth-cc-pvdz_final/ISDFov_*_symm_{mesh}_c{c}.chk` — HDF5 with
  `inpv_kpt` = `X` `(N_k, n_IP, n_orb)` and `coul_kpt` = `W` `(N_k, n_IP, n_IP)` (raw, momentum basis;
  n_orb=26, occ 4 / virt 22 for diamond). A pickled `mf` (pyscf mean-field) sits beside it with
  `mo_coeff`, `mo_occ` (per k) → the band energies/occupations for `{n_occ_i}`.
- `log_irreps/data_ov_diamond_{mesh}_c5_grid-grid.txt` — `W` irrep block table `(m_i, n_i, size)`.
- `log_irreps/data_ov_diamond_{mesh}_c5_grid-ao.txt` — `X` (grid↔AO) block table `(m_i, n_grid, n_ao)`.
- `analyze_isdf_irreps.py` — computes these irrep decompositions (builds the AO symmetry ops `U`,
  `kmap`, decomposes reps). **NOTE it does grid↔AO, not occ/virt** — to get `{n_occ_i}` you must
  project the occ/virt band subspaces (load `mf.mo_coeff`/`mo_occ`, project, decompose) — a modification
  of that script.

**Derived-locally alternatives (this repo):**
- `src/chem/fftisdf/cg_transform.py` — builds `Q_grid`/`Q_AO`, verifies block-diagonalization,
  `--save-transform` (`cg_transform_diamond_2x2x2_c5.npz`).
- `src/integrations/qualtran/diamond_symm_orbit_data.json` — orbit-size distribution per family
  (drives the `Q` orbit synthesis).

**Diamond numbers (for validation):**
- 2×2×2 c5: `N_k=8`, `n_IP=136`, `n_orb=26`(4occ/22virt). `W`: 20 irreps, `Σn_i=220`, `Σn_i²=3384`.
  `X`: 14 grid-ao blocks, `Σn_grid·n_ao=714`. Orbits: `Σ|o|²=6800` (3 families).
- 6×6×6 c5: `N_k=216`. `W`: 57 irreps, `Σn_i=1768`, `Σn_i²=85864`. `X`: 49 grid-ao blocks,
  `Σn_grid·n_ao=17006`. Orbits: `Σ|o|²=12912` (16 families, 730 orbit-blocks).

---

## 7. Qualtran implementation plan

> **File-map re-check (2026-08-14).** The tree was cleaned up: the walk operator is now
> `BSEBlockEncoding` in `bse_block_encoding.py` (composing `FockTemplate` / `ExchangeTemplate` /
> `ExchangeDensityFittingTemplate` / `DirectTemplate`); the central kernel is
> `DiagonalCoulombKernelBlockEncoding` in **`diagonal_kernel_block_encoding.py`**. The current code is
> **translation-only** (no point-group `Q`): `symmetry_adaptation_QROAM.py` and
> `direct_Coulomb_block_encoding.py` are gone. So the point-group-`Q` route (§1–5, §9) is **not yet
> wired** — the circuit `Q` would need (re)building; the classical `Q` construction is in
> `src/chem/fftisdf/cg_transform.py`.

**Reusable bloqs (`src/integrations/qualtran/`):** the templates + `BSEBlockEncoding` above; the
column-isometry, interferometer-unitary, eigendecomposition, and Frobenius/diagonal-kernel bloqs; the
range-safe / block-indexed QROAM helpers. (The prior `Q` bloq shape — orbit-unitary synthesis addressed
by `(family,orbit)`, `|P|`-op reconstruction, `(k,I')→(i,a,c)` reindex — is described in §9.8 and is the
pattern to rebuild if/when the `Q` route is wired.)

**Cost model / primitives (read for costs & conventions):**
- `XPRIZE_paper/script/_resource_dispatch.py` — `unitary_synthesis` (P-06),
  `forward_isometry_synthesis` (P-07), `diagonal_matrix_block_encoding` (P-09),
  `eigendecomposition_block_encoding` (P-12); `mode="T-opt"`, `b=32`. Handles the `N_k` block prefix.
- `XPRIZE/context/primitives.md` — the P-01..P-16 cost primitives (spec + provenance).

**Bloqs to build / compose (per BSE term):**
1. `Q` (have it) — reuse ×6.
2. **Register-recording** bloq — canonicalize the grid-pair to its orbit rep (subtract + 48-op
   argmin using perm/`L` QROMs). This is essentially stage-B machinery of `SymmetryAdaptationQROAM`.
3. **`W_exchange` = `2·U + D`**: `U` = block `unitary_synthesis` over the copy blocks `{n_i}`
   (group by size); `D` = diagonal BE over `Σn_i` eigenvalues. **Do not drop `U`.**
4. **`W_direct`** = diagonal BE over the reduced `W` data (`Σn_i²` or the diagonalized form).
5. **`X_o`, `X_v`** = block isometries `forward_isometry_synthesis` over `{(n_grid_i, n_occ_i)}` and
   `{(n_grid_i, n_virt_i)}`. occ/virt selected by the `c < n_occ_i` mask (QROM `{n_occ_i}` + compare).
6. Compose with a controlled-SWAP routing network; count via `get_cost_value(bloq, QECGatesCost(),
   generalizer=generalize_cswap_approx).total_t_and_ccz_count(ts_per_rotation=0)['n_ccz']`.

**Validation target (diamond 6×6×6, b=32, T-opt):** raw (k-only, no point-group) ≈ **2.46M** Toffoli
@n_IP=208; fully symmetry-adapted ≈ **0.6–0.85M** (≈ 2.8–2.9× reduction). Reduction hierarchy:
`W` collapses (diagonal); `X` (isometry) reduces only ~3.4× and dominates; `U` (eigenvector) is the
2nd term; the 6 `Q`'s are cheap.

**Key implementation rule:** the orbit blocks `U_o`, perm/`L` tables, and CG coefficients are
**group-determined** — build them once from geometry (hard-code / data-free `Shaped` QROAM for cost
modeling). Only `w_i`, `x_i`, `{n_occ_i}` come from the material (jsun3 chks / band symmetry).

---

## 8. Current implementation status (post-cleanup 2026-08-13) — TRACK THE RIGHT BLOQ

The walk operator now lives in the **template-based `bse_block_encoding.py`** (the old monolith was
archived to `archive/bse_block_encoding_v1.py`). Terms: `FockTemplate` (C₀), `ExchangeTemplate`
(C_ov^ex), `ExchangeDensityFittingTemplate`, `DirectTemplate` (C_ov^dir + oo/vv via `same_spin`).

**CORRECT direct/oo/vv implementation = `DirectTemplate` in `bse_block_encoding.py`.** It is
**k-points-only (translation) + real-space**, NOT the point-group `Q` adaptation of §1–5 (that is the
separate, more aggressive route). Structure (verified call graph, ov-direct):
```
2× ColumnIsometryRectangularBlockEncoding (electron X, X†)
2× ColumnIsometryRectangularBlockEncoding (hole X, X†)
1× DiagonalCoulombKernelBlockEncoding      (Δ_ζ: real ζ̃^W(R₁⊖R₂)_{μν}, over (R,μ,ν))
4× QFTTextBook                             (both momentum regs, fwd+inv → real space)
1× Subtract                                (R = R₁ − R₂)
```
- **`alpha = 1.0` — no residual `N_k`**: the unitary DFT on both momentum registers makes the operator
  block-diagonal in `(R₁,R₂)` with entry `ζ̃^W_{R₁⊖R₂}`, consuming the THC `1/N_k`. (This is the
  real-space win; the old momentum-`Q`-LCU form carried `α ∝ N_k`.)
- `ζ̃^W(R)` is **real** ⇒ one rotation, `b`-bit word (`DiagonalCoulombKernelBlockEncoding.complex_data
  =False`; data-verified `W[-Q]=conj(W[Q])` with 3-D-mesh arithmetic — beware the 1-D-DFT trap).
- `χ` (=X) is **not** an isometry (`‖X†X−I‖≈31` on ISDF data) ⇒ `chi_embedding="dilation"` completes it
  to `V=[A; √(I−A†A)]` (or `"svd"`). oo/vv (`same_spin`) reuse `ζ^W` + two `χ` → 2 incremental isometries.
- **Costs** (diamond `N_o=4,N_v=22,N_IP=208,N_k=216,b=32,optimal_T`): ov-direct **554,407**; oo **87,150**;
  vv **426,096** Toffoli; all `α=1.0`.

**SUPERSEDED / removed in the cleanup:** the standalone `DirectCoulombBlockEncoding` (and the older file
`direct_Coulomb_block_encoding.py`) — the momentum-`Q`-LCU direct term (`PrepareUniformSuperposition(Q)` +
modular-add + `W(Q)` + unprepare), `α ∝ N_k`. `DirectTemplate` replaced it. The kernel bloq survives as
`DiagonalCoulombKernelBlockEncoding` in **`diagonal_kernel_block_encoding.py`**, still used by
`DirectTemplate` as the central diagonal.

So: **direct/oo/vv = translation-only, real-space, `α=1.0` (`DirectTemplate`)**; the point-group `Q`
symmetry-adaptation (§1–5) is the *separate, not-yet-wired* route that would further compress via
`(f,μ)`/copy blocks.

---

## 9. Authoritative symmetry-adapted term structures & implementation (author, 2026-08-14)

This section is the **precise, authoritative** prescription for the point-group-`Q` symmetry-adapted
route; it supersedes the sketch in §5. `Q` maps `|k⟩|I⟩ → |i⟩|a⟩|c⟩`.

### 9.1 `X_o` / `X_v` = column selection *within the copy index* (NOT a separate projector)
`X_o` and `X_v` are the **same full `X`**, with columns **selected inside the copy index `c`** of each
irrep block: `X_o` keeps the occupied copies, `X_v` the virtual ones. There is **no separate projector
bloq** — you absorb the occ/virt choice into the isometry by synthesizing only those columns. With
**energy-ordered copies** (canonical bands), occ = a prefix `c < n_occ_i`, so `x_i^o = x_i[:, :n_occ_i]`,
`x_i^v = x_i[:, n_occ_i:]`. Grid side and partners `a` untouched. (This is the absorbed `P_o`; see §4.)

### 9.2 `n_occ_i` for resource estimation
Use the **most plausible** `{n_occ_i}` (from the sp³/bond-centered valence band rep, or any reasonable
per-irrep distribution) subject to the hard constraint
```
Σ_i m_i · n_occ_i = 4 · N_k .
```
The exact distribution barely moves the cost (the isometry cost is set by the column count + QROAM,
~the same for any split summing to `4·N_k`), so pick a plausible one and enforce the sum.

### 9.3 `ov` EXCHANGE term — precise sequence (6 `Q`'s)
```
X_o ⊗ X_v
   → Q† ⊗ Q†
   → [momentum-transfer op]            (NON-unitary, correctly post-selected)
   → Q W Q†                            (W in the irrep basis = ⊕_i 𝟙_{m_i}⊗w_i; exchange central = U D U†, synthesize U,U†)
   → [reversed momentum-transfer op]
   → Q ⊗ Q
   → X_o† ⊗ X_v†
```
- **Six `Q`'s**: `Q†⊗Q†` (2) + `Q…Q†` around `W` (2) + `Q⊗Q` (2).
- **Momentum-transfer op**: a non-unitary, **post-selected** block that shifts the transfer momentum —
  the existing implementation (prepare-uniform over the transfer + modular subtract into a momentum
  register + unprepare/post-select; cf. `ExchangeTemplate.momentum_prep` / `momentum_sub` in
  `bse_block_encoding.py`). Applied forward, then reversed, around the central `Q W Q†`.

### 9.4 `ov` DIRECT term — precise sequence (4 `Q`'s, reused)
```
X_o ⊗ X_v
   → Q† ⊗ Q†
   → F† ⊗ F†                           (Fourier: |k⟩|I⟩→|R_1⟩|I⟩ and |k'⟩|I⟩→|R_2⟩|I⟩; F or F† per convention)
   → [ compute |R⟩=|R_1 − R_2⟩ (modular subtract);
       apply diagonal W = Σ_{R,μ,ν} W_{R,μ,ν} |R,μ,ν⟩⟨R,μ,ν|;
       uncompute |R⟩ ]
   → F ⊗ F
   → Q ⊗ Q
   → X_o† ⊗ X_v†
```
- **Four `Q`'s** (the ones attached to the `X`'s). No separate `W`-`Q`'s: after `F`, `W` is a **real
  diagonal** `W_{R,μ,ν}` over `(R=R_1−R_2, μ, ν)` (compute `R`, apply, uncompute `R`), `α=1` (the DFT
  consumes the `1/N_k`; see §8). This is the point-group-`Q` refinement of the `DirectTemplate` idea.

### 9.5 `Q` counting and reuse across the five templates
- **`ov` exchange**: 6 `Q`'s.
- **`ov` direct**: 4 `Q`'s — **reuse** the `X`-attached `Q`'s (`Q†⊗Q†`, `Q⊗Q`) via a **re-routing / SWAP
  network**; nothing extra.
- **`oo`, `vv`**: add only **2 `X_o` + 2 `X_v`** each; **everything else — `Q`, `W` — is reusable**
  (routed in by the SWAP network).

### 9.6 Padding for variable dimensions (do it with a block-encoded projector, keep 0)
Irreps have **different partner dims `m_i` and copy dims `n_i`**. At the implementation level, allocate
the max widths and **pad the extra dimensions with zeros**, enforced by a **block-encoded diagonal
projector** with diagonal `[1,…,1,0,…,0]` (`1` on the valid range, `0` on the padding) — the same
`P_o`-style diagonal mask (QROAM threshold `< valid_size` + `Z R_y` / range-safe flag). Do **not** pad
with data. Two consequences:
- The **block-diagonal isometry / unitary synthesis must accept VARIABLE block sizes** (and, for the
  isometries, a **variable number of columns** per block). Use the **range-safe + block-indexed**
  synthesis (P-13 / P-15 / P-16) so ragged `{m_i}`, `{n_i}`, `{n_occ_i}` cost their true `Σ`, not
  `#irreps·max²`. Never uniform-pad the *data* (§8-adjacent note: ~2–4× waste on diamond).

### 9.7 `Q`'s block structure does NOT automatically match `X`'s or `W`'s — CAREFUL
- **`Q`'s blocks are ORBITS** (on the input/grid side): `Q^{(f)}=⊕_o U_o`, one block per little-group
  orbit of interpolation points. Its **output** is labelled `(i,a,c)`.
- **`X` and `W`'s blocks are IRREPS** `i=(f,μ)` (block-diagonal by irrep).
- These are **transverse partitions**: one orbit contains **several** irreps `μ`; one irrep gets copies
  from **several** orbits (the copy index `c` pools across orbits). So `Q`'s orbit-blocks **do not line
  up** with `X`/`W`'s irrep-blocks. When composing `Q` with `X`/`W` you must insert the **reindex**
  (the "stage-C" regrouping `(orbit-organized μ,ν,c) → (irrep-organized i,a,c)`) — do **not** assume the
  block boundaries coincide. (This is why `Q` is built orbit-by-orbit but hands `X`/`W` irrep-organized
  registers.)

### 9.8 `Q` construction — `N_k`-independent, optimized
Build once from geometry (group-determined, material-free):
1. **Families**: get the family of `k` by **canonicalizing over the 48 point ops** — `k* = min_g A_g·k`
   (`A_g·k` = signed coordinate permutation, cheap) + argmin. `O(|P|·log N_k)`, **no `N_k` table**;
   `k*` = family, `g*` = coset op / star member `κ`.
2. **Little group** `L_{k_0}` at each family rep; **orbits** of interpolation points under `L_{k_0}`.
3. **Orbit reduction** → orbit blocks `U_o` (character projection or symmetrize-random-Hermitian + `eigh`;
   `cg_transform.py`, leak ~2e-13). Group-determined (could be hard-coded, no data).
4. **Induction** to members: `Q^{(k)} = P(g_k) Λ(g_k,k) Q^{(f)}` (phased permutation + Bloch/glide phase).

**`N_k`-independence**: stage-A orbit synthesis is addressed by `(family, orbit)`, **never by `k`**;
members are rebuilt by a **select over the 48 ops** + arithmetic phase `e^{-ik·L}` from fixed `|P|×n_IP`
geometry tables; the only `k`-touching widths are `~log₂ N_k`. **Optimize**: interferometer orbit
synthesis + select-swap `λ` chosen by exact Toffoli argmin (as in `SymmetryAdaptationQROAM`). Verified
`N_k`-flat: dominant Toffoli grows ~1.2× while `N_k` grows 27× (2×2×2 → 6×6×6).
