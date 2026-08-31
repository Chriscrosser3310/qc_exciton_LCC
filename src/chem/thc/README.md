# THC collocation-matrix optimization with selectable norm penalties

Upstream (`jsun3/xprize/THC_general/optimize_X_ov.py`) optimizes the ISDF collocation
matrices against a penalty that hardcodes the **operator norm** for both collocation
factors. This package makes that norm a parameter, so the `2 -> inf` norm can be used
for `X^v` instead, and runs the comparison on diamond.

## Why `2 -> inf`

The LCU norms are built from *row* 2-norms of the collocation matrices:

    A^Q_I = max_k ||X^{o,k}_{I,:}||_2 ||X^{v,k-Q}_{I,:}||_2

so `||A||_{2->inf} = max_I ||A_{I,:}||_2` is the *tight* bound on them. Routing through
the operator norm overshoots. Measured on diamond, substituting it for `X^v` alone
shrinks `alpha_embed^op` by 2.5-12x (median 5.2x), and using it for both factors makes
the rigorous bound on `alpha_LCU^dir` about 90x tighter.

Two things temper this. The gap cannot be closed: with `M > n_vir` the rows are
overcomplete, `rank(X X^H) <= n_vir`, and `lambda_max >= tr / n_vir` forces
`||X||_op > ||X||_{2->inf}` strictly. And the gradient is sparser -- see `penalties.py`.

## What upstream's penalty actually is

`optimize_X_ov.py:50` computes

    norm_Xo**2 * norm_Xv**2 * norm_W        # = alpha_embed / n_k

and `optimize_X_common.py:147` folds it into

    loss = log10(rel_error/1e-2 + 1) + cosh(rel_error/1e-1) - 1 + norm_loss * norm_ratio

with `norm_ratio = norm_const * nk**power` (`nk` is the **linear** mesh dimension, so
`0.1 * 6**2 = 3.6` for a `norm0.1p2` run at 6x6x6). So the shipped `ISDFov_opt_*`
checkpoints were **trained to minimize `alpha_embed`**. Their small `alpha_embed` is the
objective, not an emergent property.

## Layout

    penalties.py      the norms: x_norm, w_norm, abs_norm_loss, norm_tag
    optimize.py       runnable optimizer (port of optimize_X_ov.py)
    evaluate.py       runnable norm evaluation, reuses chem.lcu_norms
    run_diamond.sh    the runs reported below
    vendor/           jsun3's fitting code, verbatim -- see PROVENANCE.md
    data/             mean fields + init/reference checkpoints -- see MANIFEST.md
    results/          optimized checkpoints, JSON run records, norms CSV
    logs/             full optimizer logs (one line per Adam step)

## Running

    ./run_diamond.sh          # nsteps_factor=1: ~5 min per 2x2x2 run, ~14 min per 3x3x3
    ./run_diamond.sh 0        # 100-step warmup only, a smoke test

or directly:

    python optimize.py --kmesh 2 --c-isdf 5 --c-ref 20 \
        --norm-const 0.1 --power 2 --base-lr 1e-2 --nsteps-factor 1 \
        --xv-norm 2inf --save

    python evaluate.py results/*.chk --out results/optimized_norms.csv

Available norms. `--xo-norm` / `--xv-norm`: `op` (default, upstream), `2inf`, `fro`,
`p<N>` (a smooth upper bound on `2inf`, tending to it as N grows -- use if plain `2inf`
stalls on its sparse gradient). `--w-norm`: `op` (default, upstream), `fro` (upstream's
`--use_Fnorm`), `absmax` (what `optimize_X_full.py` uses), `l1`.

Defaults reproduce upstream's penalty exactly -- verified to a relative difference of 0
against a literal transcription of `get_abs_norm_loss`.

## Differences from upstream

1. Penalty norms are selectable; upstream hardcodes `op` for both X factors.
2. Input paths are explicit; upstream resolves them through
   `system_common.get_data_dir` by naming convention.
3. Device selection has a CPU fallback. Upstream calls
   `optimize_X_common.get_device()`, which imports a `gpu_register` module that is not
   in the source tree.
4. Grid metadata (`mesh`, `ix_sel`, `group_sel`) is carried from the init checkpoint into
   the output regardless of `--symm`. Upstream propagates it only under `--symm`, which
   is why its optimized checkpoints carry only `inpv_kpt` and `coul_kpt`.
5. Each run writes a JSON sidecar with settings and final metrics.

The fit itself is untouched -- `optimize.py` calls the vendored
`utils.thc_ovvo_solve_w_intermediate_from_mo` and `optimize_X_common.run_adam` directly.

## Results: does optimizing the 2->inf norm help?

Diamond 2x2x2, init c=5 (M=136), reference c=20, 6100 Adam steps, CPU, complex64.
Each penalty was swept over `--norm-const` to trace its accuracy/norm trade-off, because
comparing the two at a single equal `norm-const` is invalid: the `2->inf` norm is
numerically smaller, so the same weight applies less pressure and the optimizer spends
the slack on fit accuracy instead. Only a curve-vs-curve reading at matched `rel_error`
means anything.

| X^v penalty | norm_const | rel_error | alpha_embed^op | alpha_embed^op-mod | alpha_LCU^dir |
|---|---|---|---|---|---|
| 2inf | 0.1 | 0.00446 | 42.356 | 5.368 | 29.52 |
| 2inf | 0.4 | 0.02121 | 29.643 | 3.131 | 26.36 |
| 2inf | 1.6 | 0.04550 | 26.810 | 2.386 | 27.99 |
| 2inf | 6.4 | 0.10262 | 23.714 | 2.054 | 27.28 |
| op | 0.1 | 0.02339 | 15.124 | 6.290 | 29.22 |
| op | 0.4 | 0.05885 | 11.278 | 5.142 | 27.24 |
| op | 1.6 | 0.13709 | 7.995 | 4.483 | 24.64 |
| op | 6.4 | 0.23656 | 5.761 | 3.818 | 20.19 |

Read at matched `rel_error`, the `2->inf` penalty is consistently about **2x better** on
the norm it targets:

| rel_error | alpha_embed^op-mod, 2inf | alpha_embed^op-mod, op | ratio |
|---|---|---|---|
| ~0.022 | 3.131 (@0.0212) | 6.290 (@0.0234) | 2.01x |
| ~0.05 | 2.386 (@0.0455) | 5.142 (@0.0589) | 2.15x |
| ~0.12 | 2.054 (@0.1026) | 4.483 (@0.1371) | 2.18x |

The picture is symmetric, which is the reassuring part: at matched accuracy the `op`
penalty is likewise about 2x better on `alpha_embed^op` (15.1 vs 29.6 at rel_error
~0.022). Each penalty wins on its own metric by roughly the same factor. Nothing is
free -- optimizing the tighter norm simply stops spending effort on a quantity that
overshoots.

`alpha_LCU^dir` is largely unmoved by either (20-30 across every run). No LCU norm was
ever an optimization target here; all three are evaluated only.

### Reproduction of jsun3's shipped checkpoint

`./reproduce_jsun3.sh` regenerates the deleted `c5`/`c12` inputs and runs their
`optimize_X_ov.py` unmodified. Against their
`ISDFov_opt_bareGDF_2x2x2_c5_cref12_norm0.1p2.chk`:

| quantity | jsun3 | reproduction | rel diff |
|---|---|---|---|
| alpha_embed^op | 15.338 | 15.789 | 2.9% |
| alpha_LCU^dir | 29.604 | 28.581 | 3.5% |
| Q=0 gauge-invariant spectrum | -- | -- | **1.0e-3** |

The individual factors differ by ~28% (`xo_op` 9.07 vs 6.53, `w_op` 1.49e-3 vs 5.62e-3).
That is the **scale gauge**, not error: `X -> s X` with `W -> W / s^4` leaves the tensor
and every alpha invariant, so the loss is flat along it and Adam drifts freely. Measured
`s = 0.720` from `||X^o||`, and the `||W||` ratio 3.57 matches the predicted
`1/(s_o^2 s_v^2) = 3.69`. `base_lr` and `nsteps_factor` are recorded nowhere, so the
residual ~3% is most likely those; they were guessed once (`1e-2`, `1`) and not tuned.

## Caveats

- **Not comparable to jsun3's `ISDFov_opt_*` files.** Those used the non-symmetry-adapted
  `ISDFov_bareGDF_<mesh>_c5` / `_c12` pair, which no longer exists. See `data/MANIFEST.md`.
  Runs here use the `_symm` family with `c_ref = 20`, so the target is harder and
  `rel_error` reads higher.
- **`use_Fnorm` for the shipped files is unrecoverable** -- not printed, not in the
  filename.
- These runs are CPU, `complex64` (upstream's default), `nsteps_factor = 1`. They are a
  comparison of two penalties under identical settings, not converged production runs.
- None of the alpha norms is a validated lambda for a circuit; the `4/n_k` prefactor in
  `bse_thc.py:apply_V_thc` is not folded in.
