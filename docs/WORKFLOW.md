# Project Workflow Guide

This document records the practical workflow for this repository so you can run and extend it later.

## 1. Environment Workflow (Windows + WSL)

PySCF support is handled in Linux (WSL), while editing can stay in Windows.

### One-time setup

From PowerShell in the repo root:

```powershell
.\scripts\run_in_wsl.ps1 bootstrap
```

This does:
- create `.venv-wsl` in the repo
- install editable package and dependencies: `.[chem,dev,qiskit]`
- install PySCF in WSL

### Daily commands

```powershell
.\scripts\run_in_wsl.ps1 test
.\scripts\run_in_wsl.ps1 example
.\scripts\run_in_wsl.ps1 mos2
.\scripts\run_in_wsl.ps1 cmd python -m pytest -q
```

## 2. Chemistry -> LMO -> Integrals Workflow

Main module: [src/chem/pyscf_adapter.py](../src/chem/pyscf_adapter.py)

### Step A: Build molecule (MoS2)

Use:
- `build_mos2_molecule(...)`

Current default is a triatomic molecular MoS2 geometry.

### Step B: SCF

Use:
- `run_scf(mol, method="RHF"|"UHF"|"RKS"|"UKS", xc="PBE")`

Output is a converged PySCF mean-field object (`mf`).

### Step C: Localize orbitals

Use:
- `localize_orbitals(mol, mo_coeff, mo_occ, scheme="boys"|"pipek-mezey")`

Design choice:
- occupied and virtual spaces are localized separately
- final LMO coefficient matrix is `[C_occ_loc | C_vir_loc]`
- returned `occupied` / `virtual` indices are in this reordered LMO basis

### Step D: LMO integrals

Use:
- `compute_one_electron_integrals_lmo(mf, lmo_coeff)` -> `(hcore_lmo, fock_lmo)`
- `compute_two_electron_integrals_lmo(mol, lmo_coeff)` -> `eri_lmo[p,q,r,s]`

### Step E: Static screening

Use:
- `compute_static_screened_coulomb_lmo(eri_lmo, epsilon_r, orbital_centers=None, kappa=None)`

Model:
- base: `W = eri / epsilon_r`
- optional damping: `exp(-kappa * (d_pr + d_qs)/2)` using LMO centers

Helper:
- `compute_orbital_centers(mol, lmo_coeff)` for center coordinates.

### Step F: One-shot builder

Use:
- `PySCFExcitonDataBuilder.build(...)`

This returns `LMOData` with:
- `mo_coeff`, `lmo_coeff`
- `hcore_lmo`, `fock_lmo`, `eri_lmo`
- `orbital_centers`
- `occupied`, `virtual`

## 3. Example You Can Re-run

Example script: [examples/mos2_lmo_workflow.py](../examples/mos2_lmo_workflow.py)

Run from PowerShell:

```powershell
.\scripts\run_in_wsl.ps1 mos2
```

Expected outputs include:
- number of LMOs
- shapes of `hcore_lmo` and `eri_lmo`
- sample screened element `W(0,0,0,0)`

## 4. Sparse Block-Encoding Workflow

Main module: [src/block_encoding/sparse_matrix.py](../src/block_encoding/sparse_matrix.py)

### Current structure

- `RowAccessOracle.from_function(...)`
- `ColAccessOracle.from_function(...)`
- `EntryBinaryOracle.from_function(...)` / `from_dense(...)`
- `FullDataLoadingAmplitudeOracle(...)`
- `SparseOracleBundle.from_functions(...)`
- `SparseMatrixBlockEncoding(...)`

### Reversible compilation + costs

Oracle synthesis modules:
- [src/oracles/function_ir.py](../src/oracles/function_ir.py)
- [src/oracles/reversible_synth.py](../src/oracles/reversible_synth.py)

For row/col oracles:
- call `compile_reversible_circuit()`
- then call `.estimate_cost()`

Cost fields:
- `t_count`
- `toffoli_count`
- `cnot_count`
- `x_count`
- `t_depth_estimate`
- `ancilla_peak_estimate`

## 5. Recommended Iteration Loop

1. Edit code in Windows.
2. Run tests in WSL:
   - `.\scripts\run_in_wsl.ps1 test`
3. Run MoS2 chemistry flow:
   - `.\scripts\run_in_wsl.ps1 mos2`
4. Add/adjust block-encoding logic and estimate costs.
5. Commit once tests and examples are stable.

## 6. Known Conventions

- Source package is flat under `src/` (e.g., `from chem import ...`, `from block_encoding import ...`).
- Do not reintroduce old `qc_exciton_lcc.*` import paths.
- Use WSL runner scripts for anything requiring PySCF.
