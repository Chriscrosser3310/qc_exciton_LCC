# Conversation Log - 2026-03-02

This file records the key user requests and resulting work items from the current collaboration thread.

## Project direction and architecture
- Planned project for exciton and linearized coupled-cluster simulation pipelines with PySCF integration.
- Focused initial implementation on exciton simulation.
- Designed for multiple block-encoding strategies and algorithm layers that can query varying block-encodings over time (including QSVT-like workflows).
- Emphasized interoperability with Qiskit first, with forward compatibility for Qualtran and resource-estimation tooling.

## Implemented modules and capabilities
- Sparse block-encoding scaffolding with separate row/column access and entry/amplitude oracles.
- Function-based reversible-oracle compilation flow and gate-cost reporting hooks.
- Lattice-aware single-particle and two-particle sparse index oracles.
- Exciton Hamiltonian structured encoding with direct and exchange terms.
- Tensor generation utilities for benchmark F and V tensors on D-dimensional lattices.
- Screening and chemistry workflow support (PySCF-side integration path).
- Sparse-QROM helper utility for COO-style sparse tensors and conversion to Qualtran-compatible QROM construction.

## Experiments and analysis artifacts
- Added multiple experiment scripts and notebooks for:
  - sparse oracle scaling,
  - lattice oracle gate scaling,
  - exciton block-encoding complexity,
  - surface-code style runtime/footprint estimates,
  - QSVT scaling maps over (d, R_c).
- Generated plot sets in `experiments/plots/` for gate-count, logical-qubit, runtime, and footprint trends.

## QSVT and scaling updates
- Added QSVT-focused experiment script with configurable relation:
  - side length per dimension: `L = round(C * d * R_c)`.
- Added asymptotic exponent fitting on computed grids.
- Updated entry-oracle model to full data loading for scaling studies.
- Verified fitted scaling (D=3 case) is consistent with target behavior near `d^(2D+1) R_c^(2D)` under full data-loading assumptions.
- Updated runtime visualization/reporting from days to hours where requested.

## Notebook workflow updates
- Updated notebook imports and path bootstrapping for Jupyter (`__file__`-free path resolution).
- Added cells to reshape `V_{pqrs}` into matrix forms:
  - `V_direct`: `(p, q) -> (r, s)`
  - `V_exchange`: `(p, r) -> (q, s)`
- Added plotting cells to visualize `|V_direct|` and `|V_exchange|` as heatmaps.

## Formula correction applied
- Corrected benchmark `V` tensor exponent from:
  - `(3 - delta_pr - delta_qs)`
  to:
  - `-(3 - delta_pr - delta_qs)`.
- Added safe handling for singular/self-distance terms in the benchmark tensor generation path.

## Environment and workflow support
- Set up and documented WSL/Ubuntu-compatible workflows.
- Added guidance and scripts for virtual environment usage and repeatable runs.

## Git activity in this thread
- Created and pushed commits for core modules/tests and for experiment assets/notebooks.
- Latest requested action: save this conversation log, commit, and push.

## Note
- This is a practical log of the collaboration trajectory and delivered artifacts, intended for future reference in the repository.
