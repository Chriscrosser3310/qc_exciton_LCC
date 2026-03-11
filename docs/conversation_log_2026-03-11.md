# Conversation Log - 2026-03-11

This file records key requests, decisions, and delivered work from this session.

## Initial repository operations
- Synced local repository to latest `origin/main` with local override/discard, as requested.
- Read prior session context from `docs/conversation_log_2026-03-02.md`.

## Notebook updates and Qualtran experiments
- Added/updated cells in `experiments/qrom_cost.ipynb` for:
  - chaining multiple Bloqs,
  - combining multiple Bloqs into one `CompositeBloq`,
  - subset-register Bloq application,
  - end-to-end resource-estimation flow.
- Implemented a Qualtran-style tensor entry oracle as a decomposable Bloq in notebook form
  (QROM load -> controlled-Ry ladder -> QROM uncompute).
- Added new experiment notebook:
  - `experiments/two_particle_controlled_vsum_call_graph.ipynb`
  - builds and visualizes call graph for `TwoParticleControlledVSumBlockEncoding`.
- Updated call-graph visualization behavior to show full graph while keeping QROM collapsed as unit leaf nodes.

## Git and auth workflow
- Configured SSH-based GitHub auth for the local environment.
- Pushed earlier notebook updates after SSH auth setup.

## Major structural refactor
- Per user request, moved all Qualtran-dependent implementation code under:
  - `src/integrations/qualtran/`
- Created PennyLane-specific integration scaffold:
  - `src/integrations/pennylane/`
- Added provider-specific backend adapters under integration folders.

## Cleanup and archival
- Archived old adapter-agnostic/legacy layers and removed them from active `src`:
  - archived to `archive/legacy_adapter_layer/` and `archive/non_qualtran_core/`.
- Removed old top-level packages/layers now considered out-of-scope for active code:
  - `src/algorithms/`, `src/backends/`, `src/block_encoding/`, `src/oracles/`.
- Updated tests and experiment/script imports to new integration-first paths.
- Archived tests tied to discarded legacy abstraction layers.
- Archived outdated example `examples/exciton_workflow.py`.
- Updated `docs/WORKFLOW.md` and script import paths to reflect new structure.

## Final structure direction
- Active architecture is integration-first:
  - `src/integrations/qualtran/...`
  - `src/integrations/pennylane/...`
  - plus domain/data modules (e.g., `src/exciton`, `src/chem`).

## Verification
- Ran full test suite after structural changes.
- Result at handoff: all active tests pass (`51 passed`).

## Note
- This log is a practical summary of outcomes and rationale for future continuity.
