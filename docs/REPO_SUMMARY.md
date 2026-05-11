# Current Repository Summary

Last updated: 2026-05-11

## Purpose

This repository supports localized-orbital quantum chemistry workflows and
quantum resource-estimation experiments for exciton-related Hamiltonians. The
current active work is narrower than some older documentation suggested:

- `src/chem` produces localized molecular orbital data from PySCF.
- `src/integrations` contains provider-specific resource-estimation code.
- `src/exciton` contains early model scaffolding, but is not yet the reliable
  integration point for current Hamiltonian/block-encoding work.

## Active Source Areas

### `src/chem`

`src/chem/pyscf_adapter.py` is the main chemistry module. It can:

- build a small molecular MoS2 geometry for examples
- run PySCF SCF calculations
- localize occupied and virtual molecular orbitals separately
- transform one- and two-electron integrals into an LMO basis
- estimate orbital centers
- compute a simple statically screened Coulomb tensor
- package the results as `LMOData`

This is the most coherent active domain layer.

### `src/integrations`

The active integration work is Qualtran-focused and currently lives directly in
`src/integrations/qualtran`.

Important files:

- `state_prep_QROAM.py`
- `block_state_preparation_QROAM.py`
- `unitary_synthesis_QROAM.py`
- `block_unitary_synthesis_QROAM.py`
- `block_unitary_interferometer_QROAM.py`
- `data_loading_comparison.py`
- `data_loading_comparison.ipynb`
- `utils.py`

The code focuses on QROAM-backed state preparation, block-indexed state
preparation, block-unitary synthesis, data-loading comparisons, and Qualtran
resource-count helper routines.

The previous active `src/integrations/pennylane` package and older
`src/integrations/qualtran/block_encoding` package have been removed from the
source tree and archived as tarballs under `archive/`.

### `src/exciton`

This directory currently provides lightweight dataclasses for exciton models,
screening providers, benchmark tensors, and a minimal builder. It is useful as
scaffolding, but it is not yet consistent with the active effective-Hamiltonian
and integration work. Avoid treating it as the source of truth for current
block-encoding resource estimates.

## Older and Transitional Areas

- `experiments/`: older notebooks and scripts. Some still point at modules that
  have since been archived or deleted.
- `archive/`: preserved legacy adapter and integration code. Use this for
  reference, not as active package code.
- `tests/`: currently transitional. Several tests still import deleted
  integration modules and need cleanup.
- `logs/` and `.ipynb_checkpoints/`: local notebook/job artifacts.

## Reports and Notes

Recent report outputs are under `docs/`, especially:

- Qualtran block-preparation reports
- block-unitary resource reports
- generated LaTeX tables from older benchmark sweeps
- PDFs from resource-estimation runs

`docs/codex-handoff.md` is a local handoff note for the recent data-loading
comparison notebook. It is not a complete project overview; use this file and
`docs/WORKFLOW.md` for the current high-level map.

## Immediate Cleanup Opportunities

- Remove or rewrite tests that reference deleted PennyLane and old Qualtran
  block-encoding modules.
- Decide whether generated notebook checkpoints, logs, and local job artifacts
  should be tracked or ignored.
- Reconcile `src/exciton` with the current chemistry output and the desired
  effective-Hamiltonian block-encoding interface.
- Promote stable Qualtran scripts into a package-style API once the current
  resource-estimation workflow settles.
