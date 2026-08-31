# LCU and block-encoding norms for THC factorizations

Five scalars for the occupied--virtual ERI block of a THC/ISDF factorization, the
formulas that define them, the exact relations among them, and measured values for
diamond across all 75 THC checkpoints that exist for it.

`norms.py` is the definition of record: every formula below is implemented there and
`check_relations` asserts every inequality. `diamond_thc_norms.csv` is the measured
table. This README is commentary on both.

## Notation

`I, J` index THC interpolation points (`M` of them); `p, q` index orbitals; `k` and `Q`
index the k-point mesh (`N_k` points). `W^Q` is the `M x M` central tensor at momentum
transfer `Q`, stored as `coul_kpt[Q]`. `X^{o,k}` is `M x n_occ` and `X^{v,k}` is
`M x n_vir`.

Two bracket choices, differing only in the norm taken over `k`:

```
A^Q_I (l_inf) = max_k   ||X^{o,k}_{I,:}||_2 ||X^{v,k-Q}_{I,:}||_2
A^Q_I (l_2)   = sqrt( sum_k ( ||X^{o,k}_{I,:}||_2 ||X^{v,k-Q}_{I,:}||_2 )^2 )
```

with `B^Q_J` the same expression with occupied and virtual exchanged, and

```
S_Q = sum_{I,J} |W^Q_IJ| A^Q_I B^Q_J
```

## The five norms

### LCU family

These weight each entry `|W^Q_IJ|` by per-`(I, J)` row-norm factors.

```
alpha_LCU^dir       = (1 / N_k) sum_Q S_Q   with l_inf brackets
alpha_LCU^exch      = (1 / N_k) sum_Q S_Q   with l_2   brackets
alpha_LCU^exch-max  = (1 / N_k) max_Q S_Q   with l_2   brackets
```

`alpha_LCU^exch-max` retains the `1 / N_k` prefactor when the sum becomes a maximum.
That choice matters: it makes this norm the **smallest** of the three, not the largest,
since replacing a sum of `N_k` non-negative terms by their maximum can only shrink it.
The prefactor-free `max_Q S_Q` is also tabulated in the CSV.

### embed family

A product of five extremal scalars, with
`P = (max_k ||X^{o,k}||_op)^2 (max_k ||X^{v,k}||_op)^2`:

```
alpha_embed^op  = P * max_Q ||W^Q||_op
alpha_embed^F   = P * max_Q ||W^Q||_F
alpha_embed^L1  = P * (1 / N_k) sum_{Q,I,J} |W^Q_IJ|
```

The LCU and embed families are **different constructions, not variants of each other**.
Comparisons between them are not tighter-versus-looser comparisons of one bound.

### The `2 -> inf` substitution

The `2 -> inf` norm of a matrix is its largest row 2-norm, maximized here over `k` too:

```
xo_2inf = max_{k,I} ||X^{o,k}_{I,:}||_2      xv_2inf = max_{k,I} ||X^{v,k}_{I,:}||_2
```

Since `||A||_{2->inf} <= ||A||_op` always, swapping it in shrinks any embed norm:

```
alpha_embed^op-mod   = (max_k ||X^{o,k}||_op)^2 * xv_2inf^2 * max_Q ||W^Q||_op
alpha_embed^op-2inf  = xo_2inf^2 * xv_2inf^2   * max_Q ||W^Q||_op
alpha_embed^L1-2inf  = xo_2inf^2 * xv_2inf^2   * (1 / N_k) sum_{Q,I,J} |W^Q_IJ|
```

`op-mod` replaces only the `X^v` factor; `op-2inf` replaces both.

**`2 -> inf` is the natural norm here.** The LCU brackets are built from row 2-norms, so
`A^Q_I <= xo_2inf * xv_2inf` holds directly, whereas routing through the operator norm
overshoots. That makes `alpha_embed^L1-2inf` a rigorous upper bound on `alpha_LCU^dir`
about two orders of magnitude tighter than `alpha_embed^L1`:

| bound on `alpha_LCU^dir` | tightness (min / median / max over 75 files) |
|---|---|
| `alpha_embed^L1` (operator norms) | 5.7e-05 / 2.7e-04 / 5.8e-03 |
| `alpha_embed^L1-2inf` | 1.6e-02 / 2.4e-02 / 1.4e-01 |

Measured shrinkage of the operator-norm variant, over all 75 checkpoints:
`op-mod / op` is 0.082 / 0.194 / 0.399 and `op-2inf / op` is 0.0026 / 0.0047 / 0.054
(min / median / max).

## Two things that look like new definitions but are not

**The explicit double orbital sum is the same as the `l_2` bracket.** The form

```
sqrt( sum_{k,p,q} | X^{o,k}_{Ip} X^{v,k-Q}_{Iq} |^2 )
```

factorizes, because `|X^o_{Ip} X^v_{Iq}|^2 = |X^o_{Ip}|^2 |X^v_{Iq}|^2` and so
`sum_{p,q} = (sum_p |X^o_{Ip}|^2)(sum_q |X^v_{Iq}|^2) = ||X^{o,k}_{I,:}||_2^2
||X^{v,k-Q}_{I,:}||_2^2`. It equals `A^Q_I (l_2)` identically -- verified numerically to
relative difference 0. `norms.py` never forms `sum_{k,p,q}` explicitly.

**`B^Q = A^{-Q}`.** Substituting `k -> k + Q` in the `B` bracket turns it into the `A`
bracket at negated momentum transfer. `norms.py` computes both anyway rather than relying
on it.

## Exact relations

All asserted by `check_relations`, all verified with zero violations across the 75
diamond checkpoints.

| relation | why |
|---|---|
| `alpha_LCU^dir <= alpha_LCU^exch <= N_k * alpha_LCU^dir` | `\|\|v\|\|_inf <= \|\|v\|\|_2 <= sqrt(N_k) \|\|v\|\|_inf`, two brackets |
| `alpha_LCU^exch / N_k <= alpha_LCU^exch-max <= alpha_LCU^exch` | a max is at least the mean and never exceeds the sum |
| `alpha_LCU^dir <= alpha_embed^L1-2inf <= alpha_embed^L1` | `A^Q_I B^Q_J <= (xo_2inf * xv_2inf)^2`, and `2->inf <= op` |
| `alpha_embed^op-2inf <= alpha_embed^op-mod <= alpha_embed^op` | `2->inf <= op`, applied to one or both factors |
| `max_Q \|\|W^Q\|\|_op <= max_Q \|\|W^Q\|\|_F` | pointwise in `Q` |

The third row is the only relation tying the LCU family to the embed family. Even in its
tight `2 -> inf` form it still overshoots by a median factor of about 40, so it remains a
unit test rather than a cost estimate -- but the operator-norm form overshoots by a median
factor of about 3700, so the improvement is real.

`max_Q ||W^Q||_F <= (1 / N_k) sum_{Q,I,J} |W^Q_IJ|` holds on all 363 checkpoints in the
dataset but is **not** guaranteed in general, since it compares a `Q`-maximum against a
`Q`-average. Do not assume it elsewhere.

## Two derived diagnostics

Both are ratios in which the overall scale of `W` cancels, which makes them insensitive
to the regularization confound described below.

- **k-participation** `= (alpha_LCU^exch / alpha_LCU^dir) / N_k`. The effective fraction
  of k-points carrying weight; 1 if the row-norm products are flat in `k`. Measured
  0.66--0.80 for diamond, drifting down slowly with mesh, and **independent of THC rank**
  (165.8, 165.1, 164.6, 164.7 for `M` = 130, 156, 208, 624 at 6x6x6).
- **Q-concentration** `= max_Q S_Q / mean_Q S_Q`. 1 if flat over `Q`, `N_k` if all weight
  on one `Q`. Measured 1.04--5.49 (median 1.53), so the `Q`-weight is nearly flat.

## Measured: diamond

`diamond_thc_norms.csv`, 75 rows, one per checkpoint. Four families:
`ISDFfull_bareGDF` and `ISDFfull_screenGDF` (`c` = 5, 6, 8, 24), `ISDFov_bareGDF_symm`
(`c` = 5, 6, 8, 10, 16, 20), `ISDFov_opt_bareGDF` (`c` = 5 only), each over k-meshes
2x2x2 to 6x6x6.

The re-optimized family, which is the one with the smallest norms by a wide margin:

| mesh | N_k | alpha_LCU^dir | alpha_LCU^exch | alpha_LCU^exch-max | alpha_embed^op | alpha_embed^F | alpha_embed^op-mod | alpha_embed^op-2inf |
|---|---|---|---|---|---|---|---|---|
| 2x2x2 | 8 | 29.604 | 183.10 | 25.412 | 15.338 | 78.424 | 6.1259 | 0.83426 |
| 3x3x3 | 27 | 33.108 | 680.29 | 29.652 | 25.342 | 100.87 | 8.6084 | 1.3249 |
| 4x4x4 | 64 | 32.278 | 1482.8 | 27.596 | 37.517 | 118.48 | 9.7336 | 1.2810 |
| 5x5x5 | 125 | 31.853 | 2748.8 | 24.931 | 48.818 | 126.94 | 10.478 | 1.3548 |
| 6x6x6 | 216 | 32.213 | 4613.7 | 23.121 | 54.215 | 147.21 | 10.469 | 1.2757 |

Note the ordering flip: `alpha_LCU^dir` sits *below* `alpha_embed^op` from 4x4x4 upward
(32.2 against 54.2 at 6x6x6) but *above* `alpha_embed^op-mod` at every mesh (32.2 against
10.5, a factor of 3.1). Tightening the embed norm moves it below the direct LCU norm.

Scaling with `N_k` (log-log slopes at fixed `c` = 5):

| family | dir | exch | exch-max | embed op | embed F |
|---|---|---|---|---|---|
| full / bare | +0.184 | +1.174 | +0.561 | +0.591 | +0.533 |
| full / screened | +0.090 | +1.079 | +0.288 | +0.381 | +0.267 |
| ov / bare / symm | +0.055 | +1.041 | +0.170 | +0.414 | +0.316 |
| ov / bare / optimised | +0.019 | +0.971 | -0.035 | +0.396 | +0.184 |

**Only `alpha_LCU^exch` is extensive in `N_k`**, with a slope of essentially 1. `dir` is
flat (+0.02 to +0.18). `exch-max` sits in between (-0.04 to +0.56): the `1 / N_k` removes
most, though not all, of the extensivity that the `l_2`-over-`k` introduces, the residual
being the slow drift of the Q-concentration. The two embed norms grow at intermediate
rates (+0.18 to +0.59).

Note these are slopes in `N_k` at fixed `c` = 5. Slopes in rank `M` at fixed mesh are a
different and much less stable quantity -- see caveat 3.

## Provenance

Source data, read only, nothing modified:

```
/resnick/groups/changroup/members/jsun3/xprize/THC_general/data_diamond_gth-cc-pvdz_final/
```

Each `ISDF*.chk` supplies `inpv_kpt` (`X^{ao}`, shape `(N_k, M, n_ao)`) and `coul_kpt`
(`W`, shape `(N_k, M, M)`). `X^o` and `X^v` are **not stored** and are rebuilt against the
matching `DFT_<mesh>[_symm].pkl`; see `loaders.py`, which reproduces the pairing rule and
asserts `N_k` and `n_ao` against the checkpoint's own shapes.

Conventions were taken from the generator source, not guessed:
`generate_isdf_gdf.py:92,158,241`, `generate_isdf_gdf_symm.py:180,182,271`,
`optimize_X_ov.py:164,286`, `utils.py:600-613`, and `utils.add_k` / `utils.negative_k`
for the k-index arithmetic (`momentum_difference_table` was checked to reproduce them
exactly for 2^3, 3^3, 4^3, 6^3 and 2x2x4).

## Caveats

1. **None of the five is a validated lambda for a circuit.** The `4 / N_k` prefactor in
   `bse_thc.py:apply_V_thc` is not folded in and no encoding convention is assumed.
2. **`dir` and `exch` most likely cost different terms** -- the direct and exchange
   contributions of the BSE kernel -- so they belong side by side. The inequality chains
   are statements about the expressions, useful as implementation checks and for reading
   `N_k` scaling, not evidence that one bounds the other physically.
3. **The regularization confound.** `generate_isdf_gdf*.py` picks `reg` independently per
   run, doubling it until the fit error reaches 1.5x its best value and keeping *that*
   solution. So `||W||` tracks `reg` as much as it tracks `c`, absolute norms are not
   monotonic in `c`, and rank-scaling exponents fitted to them are unreliable. Visible
   directly in the data: at 6x6x6 `ov/symm`, `c` = 8 is cheaper on every measure than
   `c` = 5 and `c` = 6 while also being more accurate, purely because it drew a smaller
   `reg`. The k-participation and Q-concentration diagnostics are immune to this; nothing
   else here is.
4. **The stored factorization is deliberately about 1.5x worse than achievable**, by the
   same stopping rule. Any accuracy quoted from these checkpoints is the 1.5x number.
5. **`reg` and `rel_error` are unavailable for the `full` and optimised families** -- those
   generation logs are group `postdoc`. They are readable only for `ov/symm`.
6. **The `full` families were not fitted for the ov block.** Evaluating an ov-block
   quantity on them is valid arithmetic but not what that fit optimized.
7. **Coverage limits.** The optimised family exists only at `c` = 5, so it probes `N_k`
   only. Five `N_k` points from 8 to 216 is thin for a power law; the slopes describe that
   range, not asymptotics. The `full` families have four `c` points with a large 8 -> 24 gap.

## Usage

```python
from chem.lcu_norms import load_thc_factors, compute_norms, check_relations

chk = (
    "/resnick/groups/changroup/members/jsun3/xprize/THC_general/"
    "data_diamond_gth-cc-pvdz_final/ISDFov_opt_bareGDF_6x6x6_c5_cref12_norm0.1p2.chk"
)
x_occ, x_vir, coul, kmesh = load_thc_factors(chk)   # pairs the DFT pickle for you
norms = compute_norms(x_occ, x_vir, coul, kmesh)
check_relations(norms)

print(norms.alpha_lcu_dir, norms.alpha_lcu_exch, norms.alpha_lcu_exch_max)
print(norms.k_participation, norms.q_concentration)
```
