# Conversation Log - 2026-03-03

This file records key requests and completed work from this collaboration session.

## Repository sync and baseline
- Located repository at `/home/jielunchen/Documents/Github/qc_exciton_LCC`.
- Synced local `main` to `origin/main` with local override as requested.
- Confirmed clean branch alignment before subsequent edits.

## Notebook updates (`experiments/qrom_cost.ipynb`)
- Added Qualtran-focused examples for chaining two oracles and chaining several Bloqs in sequence.
- Added an end-to-end resource-estimation pipeline cell showing:
  - Bloq composition via `BloqBuilder`,
  - QEC gate-count extraction,
  - Toffoli/T-style scaling by query repetitions,
  - surface-code style runtime/footprint estimation.
- Added a dedicated helper/demo cell for combining multiple Bloqs into a single `CompositeBloq`.
- Added a subset-register application demo (applying a Bloq to only part of the state dictionary).
- Fixed finalize-state issue in subset demo by preserving untouched registers (e.g., `flag`) with `state.update(...)`.

## New tensor entry-oracle implementation (Qualtran style)
- Replaced initial non-Bloq tensor oracle sketch with a proper decomposable Qualtran `Bloq` in notebook form.
- Implemented `TensorEntryOracleBloq` with:
  - explicit multi-register signature for `q, m, l, i, j`,
  - tensor shape validation for `M` under `(i,j,m,l)` indexing,
  - `build_composite_bloq(...)` decomposition using QROM load -> controlled-`Ry` ladder -> QROM uncompute,
  - factory helper `build_tensor_entry_oracle_bloq(...)` and small runnable demo.
- Verified decomposition path to `CompositeBloq` in local environment.

## Clarifications discussed
- `CompositeBloq` vs `Bloq` and recursive composition.
- Signature matching rules when chaining.
- Subset-register application pattern in `BloqBuilder`.
- Cost intuition: naive per-entry controlled rotations vs QROM-style loading.

## Git workflow and authentication
- Committed notebook updates.
- Initial HTTPS push failed due to missing credentials.
- Switched `origin` to SSH and configured local SSH-agent usage.
- Verified GitHub SSH authentication and successfully pushed commit `6c6a94c` to `main`.

## Note
- This log is intended as a practical record of the session’s delivered artifacts and decisions.
