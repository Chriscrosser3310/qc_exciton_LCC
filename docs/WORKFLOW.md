# Project Workflow Guide

This guide describes the workflow for the repository as it currently stands.
The active work is concentrated in chemistry data generation and Qualtran
resource-estimation experiments.

## 1. Repository Map

- `src/chem`: PySCF-backed quantum chemistry utilities.
- `src/integrations/qualtran`: active Qualtran QROAM, state-preparation,
  block-unitary, and data-loading resource-estimation code.
- `src/integrations/pennylane`: removed from the active tree; archived in
  `archive/src_integrations_pennylane_20260505_1600.tar.gz`.
- `src/exciton`: early exciton containers and a minimal builder. This area is
  not yet aligned with the current effective-Hamiltonian/block-encoding work.
- `archive`: previous integration and adapter layers retained for reference.
- `experiments`: older notebooks and scripts, some of which reference archived
  modules.
- `docs`: current summaries, generated reports, tables, and PDFs.

## 2. Environment Setup

Base install:

```bash
python -m pip install -e .
```

Install optional groups as needed:

```bash
python -m pip install -e '.[chem]'
python -m pip install -e '.[qualtran]'
python -m pip install -e '.[dev]'
```

For PySCF work, prefer Linux or WSL. Existing helper scripts are still useful:

```bash
bash ./scripts/linux/bootstrap.sh
bash ./scripts/linux/run.sh test
bash ./scripts/linux/run.sh mos2
```

From Windows PowerShell:

```powershell
.\scripts\run_in_wsl.ps1 bootstrap
.\scripts\run_in_wsl.ps1 test
.\scripts\run_in_wsl.ps1 mos2
```

## 3. Chemistry -> LMO Data

Main module: `src/chem/pyscf_adapter.py`

The intended flow is:

1. Build or provide a PySCF molecule.
2. Run SCF with `run_scf`.
3. Localize occupied and virtual orbital spaces separately with
   `localize_orbitals`.
4. Transform one-electron and two-electron quantities into the LMO basis with
   `compute_one_electron_integrals_lmo` and
   `compute_two_electron_integrals_lmo`.
5. Estimate LMO centers with `compute_orbital_centers`.
6. Build a simple screened Coulomb tensor with
   `compute_static_screened_coulomb_lmo`.

The one-shot helper `PySCFExcitonDataBuilder.build(...)` returns `LMOData` with:

- molecular orbital coefficients
- localized orbital coefficients
- hcore and Fock matrices in the LMO basis
- four-index LMO electron-repulsion integrals
- orbital centers
- occupied and virtual index partitions in the reordered LMO basis

Example script:

```bash
bash ./scripts/linux/run.sh mos2
```

## 4. Qualtran Resource Estimation

Main directory: `src/integrations/qualtran`

The current Qualtran code is integration-specific. It is not a stable package
API yet, but the active files are:

- `state_prep_QROAM.py`: QROAM-backed controlled dense state preparation using
  phase-gradient rotations.
- `block_state_preparation_QROAM.py`: block-indexed extension that loads
  rotation data conditioned on a block register.
- `unitary_synthesis_QROAM.py`: Householder-reflection unitary synthesis based
  on QROAM-backed state preparation.
- `block_unitary_synthesis_QROAM.py`: block-diagonal unitary synthesis from
  block-indexed Householder reflections.
- `data_loading_comparison.py` and `data_loading_comparison.ipynb`: shape-only
  comparison of QROM, SelectSwapQROM, and QROAMClean for data loading plus
  data-register-controlled rotations.
- `utils.py`: local helper functions for Qualtran Toffoli-style, qubit, and
  ancilla counts.

For notebook or plotting validation on headless systems, set a writable
Matplotlib config directory:

```bash
MPLCONFIGDIR=/tmp/matplotlib python src/integrations/qualtran/data_loading_comparison.py
```

## 5. Reports and Generated Outputs

Recent generated report material lives in `docs/`, including:

- `docs/qualtran_block_prep_report.md`
- `docs/qualtran_block_prep_operations_report.md`
- `docs/block_unitary_resource_report.pdf`
- `docs/block_unitary_resource_report_b32.pdf`
- `docs/block_unitary_interferometer_resource_report_b32.pdf`

The script `scripts/block_unitary_resource_report.py` is used for block-unitary
resource-report generation.

## 6. Testing Notes

The test suite is not currently consistent with the active source layout. Many
tests still import modules from the deleted `src/integrations/pennylane` and
older `src/integrations/qualtran/block_encoding` trees.

Use focused validation while the layout is in transition:

```bash
python -m compileall src/chem src/exciton src/integrations/qualtran
python -m pytest tests/test_pyscf_adapter_screening.py tests/test_screening.py
```

Qualtran validation usually requires the `qualtran` optional dependency and may
also require matching the local Qualtran API version used when the notebooks
were written.

## 7. Development Conventions

- Keep active chemistry work in `src/chem`.
- Keep provider-specific resource-estimation code under `src/integrations`.
- Treat `src/exciton` as experimental until its effective-Hamiltonian interface
  is reconciled with the chemistry and block-encoding work.
- Prefer shape-only Qualtran objects such as `qualtran.symbolics.Shaped` when
  resource estimates do not require concrete ROM data.
- Do not reintroduce archived adapter-agnostic paths unless there is a clear
  migration plan and matching tests.
