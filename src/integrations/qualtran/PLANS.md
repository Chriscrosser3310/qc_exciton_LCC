1. [DONE] Construct a block-encoding named "exchange_Coulomb_block_encoding", which only supports data-free version, the input is N_up, N_down, N_IP, N_k, and it constructs a block-encoding of the following: first, construct a block-encoding of matrix $\sum_{k=0}^{N_k} |k><k| \otimes A_k$ where each A_k is of size N_up * N_IP, and similarly another one except each block is of dimension N_down * N_IP, and do a tensor product of these two block-encoding. Then consider the two registers encoding the k indices of each of them, say they are |k1> and |k2>, then we compute a modular subtraction |k1> |k2> -> |k1> |k1 - k2 mod N_k>, and then post select |k1> by a superposition of uniform state (only N_k of them). Rename the |k2> register to be |Q>. Call this entire thing B, which is also a block-encoding. Then it constructs a block-encoding of another block-matrix where block-index is on |Q>, and it implements $\sum_{Q=0}^{N_k} |Q><Q| \otimes A_k'$ with A_k' acts on the register A_k acts on. Then we apply the entire conjugate transpose of B. This gives the full block-encoding. Similarly, this entire thing should also allow control.
   - Implemented in `src/integrations/qualtran/exchange_Coulomb_block_encoding.py` as `ExchangeCoulombBlockEncoding` (uncontrolled) and `ControlledExchangeCoulombBlockEncoding` (sandwich-structure controlled, ~0% Toffoli overhead).

2. [DONE] For N_up = 4, N_down = 22, N_k = 1^3, 2^3, ..., 6^3, N_IP = 26 * 8, give a report of Toffoli and qubit count, for Toffoli-optimal and qubit-optimal implementation of the above. Send me a repor.t
   - Implemented in `scripts/exchange_coulomb_report.py`; report at `docs/exchange_coulomb_report_Nup4_Ndown22_NIP208_b32.pdf` emailed to jchen9@caltech.edu.  T-opt / Q-opt λ choices use the analytic QROAM optimum (`λ* ≈ √(M·b)` forward, `√M` adjoint) — no data is loaded, the lbs are picked by closed form.
## Isometry synthesis: why the inverse costs more (2026-08-23)

**Rule.** In the forward direction the multiplex *grows*: qubit 1 is rotated unmultiplexed,
qubit 2 is multiplexed by qubit 1, qubit 3 by qubits 1-2, and the last layer is a phase
layer multiplexed by all `n` qubits. So the `±1` phase left by layer `t`'s X-basis
measurement is diagonal on a **subset of the qubits layer `t+1` multiplexes on**. A diagonal
on control qubits commutes through a gate controlled by those qubits, so every measurement
phase commutes trivially forward and is absorbed into the final all-qubit phase layer.
Erasure is free.

In the inverse the multiplex *shrinks*, so that phase now sits on qubits a later layer
**rotates**. A diagonal does not commute through a rotation on the same qubit, so nothing
can be deferred and every layer pays its own explicit measurement-based uncomputation. The
same argument kills the `absorb_mcg_erasure` premise, so the inverse charges those too.

**Implemented** as `BlockIsometryColumnSynthesisQROAM.inverse` (and on the
`ColumnIsometryRectangularBlockEncoding` wrapper), charging one `QROAMCleanAdjoint` per phase
layer over that layer's own `(n_blocks, n_s)` table. Measurement-based, so width-independent
-- the penalty is a `b`-free term.

**Measured** (`n_blocks=216`, `b=20`):

| | forward | inverse |
|---|---|---|
| Q-optimal | `1.03-1.25 MNN_k` | `1.94-2.21 MNN_k` |
| T-optimal | `7.09 sqrt(b) + 0.03` | `7.09 sqrt(b) + 4.83` |

**SUPERSEDED** -- that `2MNN_k` came from leaving `Lambda'=1` on the erasure, which is
strictly dominated. With the erasure batched inside the workspace the forward lookup already
holds, the inverse is only ~4% above the forward. See the phase-first section below. The `4.83` is exactly the
closed form `sum_s 2 sqrt(n_s) ~ 4.83 sqrt(d)` over the P-07 tree levels. Ratio at `b=20`
is 1.15.

**Before this there was no inverse isometry bloq at all.** `absorb_mcg_erasure=False` was
sometimes used as a stand-in; it is not one -- it only governs the multi-controlled fix-up's
angle register, a ~2% effect on a subleading term.


## Phase-first isometry synthesis (author's scheme, 2026-08-23) -- IMPLEMENTED

`BlockIsometryColumnSynthesisQROAM.scheme = 'phase-first'` (now the default; `'iten'` keeps
the old path).

**Construction.** Think of the circuit that maps `V -> identity`; the INVERSE isometry *is*
that circuit, the forward is its inverse. Per column `c`:

1. a diagonal phase layer over `N` entries (`N_k` blocks), making the column real -- phases on
   indices `< c` set to zero, which also mops up residual signs from earlier columns;
2. real `R_y` sublayers `s = 1..n`. Sublayer `s` is **multiplex-controlled on the leading
   `n-s` qubits** and **multi-controlled on the trailing `s-1` qubits held at
   `f = c mod 2^(s-1)`**, targeting bit `s-1`. It zeroes the entries
   `= 2^(s-1) (mod 2^s)`: `1,3,5,...` then `2,6,10,...` then `4,12,20,...`.

`f` is `c`'s already-fixed low bits. Entering sublayer `s` the live entries are exactly those
`= f (mod 2^(s-1))`, so `f` tells you what is already zero. `f` never enters the address --
tables are indexed contiguously from 0 -- it lives only in the multi-control, and it is
classical, so that control is X gates plus an AND ladder, no lookup, no `N_k` dependence.

**No fix-up gates.** Whenever a pair contains an index below `c` the entry to be zeroed is
already zero, since column `c` is orthogonal to `e_0..e_{c-1}`: if `kill < c` directly, or if
`keep < c` then `keep <= c - 2^s` forces `kill <= c - 2^(s-1) < c`. The Givens angle is 0 and
the gate is the identity. This is where the scheme beats Iten, whose general two-level gates
need Lemma-11 fix-ups.

**Table sizes.** Sublayer `s` holds `ceil(N/2^s)` single-`b`-bit angles. Sum is bracketed by
`N-1 <= sum <= N-1+n`, exactly `N-1` for a power of two: 135 at `N=130`, 209 at `N=208`, 255
at 256. Column-independent, since the criterion is `base < N`.

**Transient leaks are benign.** For non-power-of-two `N` a sublayer's surviving side can sit
above `N`, so amplitude does leave the physical range. But the destination is the *keep* side,
hence inside the next sublayer's congruence class, so a later sublayer always has an address
for it. `ceil(N/2^s)` covers it with no range gymnastics. An earlier `min(pair) < N` criterion
FAILED for exactly this reason -- at `N=21, s=2` the pair `(21,23)` is entirely above `N` yet
carries the leak.

**Uncomputation.** The disentangler (= inverse) must erase every sublayer and every diagonal
immediately: its staircase decreases, so a measurement sign lands on a qubit a later sublayer
*rotates*. The forward's staircase increases, so each sign is diagonal on a subset of the next
layer's *controls*, commutes through, and all of them collapse into one final `(N_k, N)` phase
layer. The erasure batching cap is the workspace the forward lookup already holds,
`Lambda_fwd * b` -- so `b` at Q-optimal, `~sqrt(Lb)` at T-optimal.

**Verified.** 39/39 random complex isometries reduced to the identity embedding at residual
~1e-16, `n_phys` in {21,26,30,37,45,50,60,128,130,208}, no fix-ups. Per-sublayer counts match
`ceil(N/2^s)` in every layer of every case.

**Measured cost** (`N_k=216`, `b=20`):

| | forward | inverse |
|---|---|---|
| Q-optimal | `2MNN_k` (fitted 1.06-1.12) | x1.04 |
| T-optimal | `6..7 M sqrt(bNN_k)` (fitted 6.10-7.35) | x1.12-1.21 |

**Against the old padded Iten path** -- and this is the part worth knowing:

| | `N=130` | `N=208` |
|---|---|---|
| Q-optimal | phase-first **1.06x dearer** | phase-first **1.64x dearer** |
| T-optimal | phase-first **1.41x cheaper** | phase-first **1.28x cheaper** |

It loses at Q-optimal because it pays `M` per-column diagonals (`M*N` addresses) where Iten
pays one final diagonal, and because Iten packs two angles per address, halving its rotation
address count. It wins at T-optimal because its tables scale with `N` rather than `2^n`.


## Corrected uncomputation accounting for phase-first isometry (2026-08-23)

Two errors in the first cost model, both found by the author.

**1. There is no separate final sign layer.** The accumulated `+-1` erasure signs and the
per-column diagonal are both diagonal over the *same* full `n`-qubit address (plus block), and
a `+-1` is just a `pi` offset in the `b`-bit phase word, so they merge into one table at zero
extra cost. The `+2*L_d` term in the first model is spurious -- 56,160 Toffolis at
`N=130, M=22`, about 4% of the Q-optimal total. The implementation still pays it.

**2. Sublayer signs commute only to the end of their OWN column, not globally.** A sublayer
sign is diagonal on qubits `1..n-s`; the *next column's* sublayers rotate every qubit,
including some of those. What it does commute past is the rest of its own column (later
sublayers there rotate only `n-s+1..n`) and any intervening diagonal (diagonals commute).
So each column's `n` sublayer signs merge into **that column's** diagonal. Per-column, not
global. Each diagonal therefore still owes its own erasure, and there are `M` of them, not
`M-1`: the last is the final operation, with nothing to fold into, so it pays a proper adjoint.

Corrected uncompute:

```
forward:   M * L_d/b                      (Q-opt)     M * 2 sqrt(L_d)              (T-opt)
inverse:   M (L_d + sum_s L_s)/b                      M [2 sqrt(L_d) + sum_s 2 sqrt(L_s)]
```

At `N=130`: forward 30,888 not 85,644. Recomputed total 1,294,194 against the code's
1,396,770 -- the code over-charges Q-optimal forward by ~8%. T-optimal barely moves, since
`(M-1) 2 sqrt(L_d) + 4 sqrt(L_d) ~ M 2 sqrt(L_d)`: 7,374 against 7,709.

| | Q-optimal | T-optimal |
|---|---|---|
| forward | `2MNN_k` | `7M sqrt(bNN_k) + 2M sqrt(NN_k)` |
| inverse | `2MNN_k` | `7M sqrt(bNN_k) + 7M sqrt(NN_k)` |

Forward/inverse differ by `x1.025` at Q-optimal and `x1.14` at T-optimal, the difference being
the inverse's `M*n` sublayer erasures that the forward merges away for free.

**Also fixed earlier in the same pass:** the per-column diagonal must be
`BlockInterferometerFinalPhasesQROAM`, `target_bitsizes=(b,)` -- ALL ENTRIES ARE COMPLEX, so
it carries one full `b`-bit phase per entry. `RealSignLayerQROAM` is the 1-bit `+-1` sign
table and is only right for a real matrix; using it understated the diagonal by `sqrt(b)` at
T-optimal. And the final layer's `Lambda` ignored `optimal_T`, leaving it at 1 and making that
one layer 31% of the T-optimal total, immune to the operating point.

**FIXED in the code, 2026-08-23.** Both defects. The separate final sign layer is gone, and
the diagonal erasure is now charged for all `K` columns in both directions. Effect at
`N=130, M=22, N_k=216`: Q-optimal forward 1,396,770 -> 1,342,374 (-3.9%); T-optimal forward
131,814 -> 131,450 (unchanged, as predicted). The model-to-code ratio is now UNIFORM --
1.037 Q-forward, 1.043 Q-inverse, 1.066 T-forward, 1.055 T-inverse across
`N` in {130, 208, 256} and `M` in {8, 22} -- which is the signature that the accounting is
structurally right; the residual few percent is Qualtran's clean-ancilla bookkeeping above the
textbook QROAM form. Measured `inv/fwd` 1.030 (Q) and 1.132 (T), against the derived 1.025
and 1.14.

### Structural constants

`A = 2(1 + sum_{s=1}^{n} 2^{-s/2})`, the `s->inf` limit being 6.828. Finite-`n` truncation
makes it *decrease* toward that as `N` approaches `2^n`: 6.791 at `N=130`, 6.600 at 208,
6.527 at 256 -- which is exactly the drift seen in the fitted values, not noise. The `B`
constants are not structural: they bundle the diagonal erasures, a term **linear in b**
(`M(n+1)(b-2)`, the phase-gradient rotations, under 3%), and the range-safety ANDs. A
two-point `A sqrt(b) + B` fit biases `A` up by ~0.6 and `B` down by ~1.8.


## State preparation is the M=1 isometry (2026-08-23)

Layer-by-layer state preparation is the `M=1` case read in the preparation direction. Layers
run `s = n..1`, so the address GROWS `1, 2, 4, ..., ceil(N/2)` -- the standard rotation tree --
and `sum_s ceil(N/2^s) ~ N`, exactly `N-1` for a power of two, which is the textbook `d-1`
rotations for a `d`-dimensional state. Independent confirmation of the `ceil(N/2^s)` counts.

Two simplifications relative to the isometry:

- **No multi-control.** With `M=1` there are no earlier columns to protect, and in the
  preparation direction the trailing qubits are provably still `|0>` when layer `s` fires,
  since the state is built up from `|0...0>`. So no trailing control and no AND ladders.
- **Uncomputation is free** in the preparation direction: growing address, so every sign
  commutes forward into the final diagonal. The disentangling direction pays per layer,
  `x(1 + 1/b)`.

The complex phases do NOT simplify away: still one `b`-bit diagonal over `N` entries.

Derived cost: **Q-optimal `2NN'`** (one `N` for the phase diagonal, one for the rotation
tree); **T-optimal `~7 sqrt(bNN')`** layer-by-layer, which is why load-all
(`3 sqrt(bNN') + 2bN`) wins at T-optimal -- matching the measured ~7 against ~3.

**RESOLVED -- it was never a discrepancy.** `ThreePhaseLayerStatePreparation` is a DIFFERENT
ansatz: `D3 H^n D2 H^n D1 |+^n>` from arXiv:2409.11748, three *full* length-`N` diagonal phase
layers interleaved with Walsh-Hadamard transforms. Its `3NN'` is literally its three layers.
Comparing it to the `M=1` derivation was comparing two different circuits.

**We use layer-by-layer (author, 2026-08-23), i.e. exactly `M=1` of the phase-first isometry.**
Verified by evaluating the isometry bloq at `n_reflections=1`: Q-optimal `1.05-1.10 x 2NN'`,
T-optimal `6.95-7.97 x sqrt(bNN')`, `inv/fwd` 1.029-1.036 (Q) and 1.122-1.134 (T), over `N'` in
{216, 44928} and `N` in {22, 32, 130}. Derivation check at `N'=44928, N=22`: predicted
2,071,288 against 2,145,475 measured, ratio 1.036 -- the same uniform overhead as the isometry,
which is the consistency signature.

| | Q-optimal | T-optimal |
|---|---|---|
| forward | `2NN'` | `7 sqrt(bNN') + 2 sqrt(NN')` |
| inverse | `2NN'` | `7 sqrt(bNN') + 7 sqrt(NN')` |

**The two columns use DIFFERENT constructions (author, 2026-08-23):**

| | construction | Toffolis | Ancillas |
|---|---|---|---|
| Q-optimal | layer-by-layer (= `M=1` isometry) | `2NN'` | `2b + log2(NN')` |
| T-optimal | load-all | `3 sqrt(bNN') + 2bN` | `sqrt(2bNN')` |

Layer-by-layer at Q-optimal because load-all needs a `(2N-1)b`-wide register, so it is not the
qubit-minimizing choice. Load-all at T-optimal because it is genuinely cheaper in Toffolis
there, ~3 against ~7 -- so the `M=1` formulas apply to the Q-optimal column ONLY.

Only the Q-optimal rows change from the old table: `3NN'` -> `2NN'` forward and `4NN'` ->
`2NN'` inverse. The T-optimal rows and their `sqrt(2bNN')` ancillas stay as they were, and
were already verified against the load-all bloq at ratio 1.00-1.07.

One small over-charge if state prep is costed by calling the isometry bloq at `M=1`: it charges
the trailing-qubit AND ladders, `(n-2)(n-1)/2`, which state preparation does not need (no
earlier columns to protect, and the trailing qubits are provably `|0>`). ~21 Toffolis.
