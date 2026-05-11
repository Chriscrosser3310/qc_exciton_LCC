# qc_exciton_lcc

Quantum chemistry and quantum resource-estimation tooling for localized-orbital
exciton workflows.

The repository is currently organized around three active areas:

- `src/chem`: PySCF-based molecular calculations and localized molecular orbital
  (LMO) data generation.
- `src/integrations`: provider-specific quantum resource-estimation experiments,
  currently centered on Qualtran QROAM/data-loading workflows.
- `src/exciton`: early exciton model containers and builders. This layer is not
  yet consistent with the active chemistry and integration work, so treat it as
  experimental scaffolding for now.

Older adapter and integration paths have either been removed from the active
source tree or preserved under `archive/`.

## Current Status

The active code no longer exposes the previous adapter-agnostic
`src/block_encoding`, `src/backends`, `src/algorithms`, or `src/oracles` layout.
PennyLane and older Qualtran block-encoding packages are archived as tarballs in
`archive/`; the active `src/integrations/qualtran` tree contains newer QROAM
state-preparation, unitary-synthesis, and data-loading comparison code.

Some tests still reference deleted integration modules and should be expected to
fail until they are retired or rewritten against the current source layout.

## Environment

Install the base package:

```bash
python -m pip install -e .
```

Optional dependency groups:

```bash
python -m pip install -e '.[chem]'
python -m pip install -e '.[qualtran]'
python -m pip install -e '.[dev]'
```

For chemistry work, Linux or WSL is recommended because PySCF is the main
optional dependency.

## Chemistry Workflow

The chemistry entry point is `src/chem/pyscf_adapter.py`.

Implemented pieces:

- build a small molecular MoS2 geometry with `build_mos2_molecule`
- run PySCF SCF with RHF, UHF, RKS, or UKS
- localize occupied and virtual spaces separately with Boys or Pipek-Mezey
- transform one-electron and two-electron integrals into the LMO basis
- compute simple static screened Coulomb tensors from bare LMO integrals
- bundle the workflow with `PySCFExcitonDataBuilder`

Example:

```bash
bash ./scripts/linux/bootstrap.sh
bash ./scripts/linux/run.sh mos2
```

On Windows, use the WSL wrapper:

```powershell
.\scripts\run_in_wsl.ps1 bootstrap
.\scripts\run_in_wsl.ps1 mos2
```

## Qualtran Resource Estimation

The active Qualtran work is under `src/integrations/qualtran`.

Current focus:

- QROAM-backed state preparation with rotation tables
- block-indexed state preparation for block-diagonal unitary synthesis
- Householder-reflection based block-unitary synthesis experiments
- shape-only data-loading comparisons across QROM, SelectSwapQROM, and QROAMClean
- helper utilities for Toffoli-style and qubit/ancilla counts

Useful files:

- `src/integrations/qualtran/state_prep_QROAM.py`
- `src/integrations/qualtran/block_state_preparation_QROAM.py`
- `src/integrations/qualtran/unitary_synthesis_QROAM.py`
- `src/integrations/qualtran/block_unitary_synthesis_QROAM.py`
- `src/integrations/qualtran/data_loading_comparison.py`
- `src/integrations/qualtran/data_loading_comparison.ipynb`
- `src/integrations/qualtran/utils.py`

Generated reports are kept in `docs/`.

## Documentation

Start with:

- `docs/REPO_SUMMARY.md` for a current map of the repository
- `docs/WORKFLOW.md` for practical run and development workflows
- `docs/qualtran_block_prep_report.md` and
  `docs/qualtran_block_prep_operations_report.md` for recent Qualtran reports
