# qc_exciton_lcc

Exciton-focused quantum simulation tooling with:
- block-encoding abstractions and sparse-oracle variants
- query-scheduled algorithm interfaces
- backend adapters (Qiskit/Qualtran/resource-estimation)
- PySCF chemistry utilities for LMO + screened Coulomb workflows

## WSL Quick Start (Recommended on Windows for PySCF)

1. Install WSL + Ubuntu and open a WSL terminal once.
2. From PowerShell in this repo:

```powershell
.\scripts\run_in_wsl.ps1 bootstrap
```

Then run common tasks:

```powershell
.\scripts\run_in_wsl.ps1 test
.\scripts\run_in_wsl.ps1 example
.\scripts\run_in_wsl.ps1 mos2
```

Run custom commands inside the same WSL virtualenv:

```powershell
.\scripts\run_in_wsl.ps1 cmd python -m pytest -q
```

## Linux/WSL Direct Usage

```bash
bash ./scripts/wsl/bootstrap.sh
bash ./scripts/wsl/run.sh test
bash ./scripts/wsl/run.sh mos2
```

## Package Layout

- `src/exciton`: exciton model, builder, screening interfaces
- `src/block_encoding`: block-encoding contracts and sparse matrix scheme
- `src/algorithms`: query schedule and algorithm skeletons
- `src/backends`: backend/export adapters
- `src/chem`: PySCF integration (LMO/integrals/screening)
- `src/oracles`: function IR and reversible synthesis with cost estimates

## Detailed Workflow

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the full step-by-step workflow.
