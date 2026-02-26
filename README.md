# qc_exciton_lcc

Initial scaffold for exciton-focused quantum simulation workflows with:
- exciton model and screening abstractions in one module boundary
- block-encoding interfaces and query schedules
- backend adapters for Qiskit, Qualtran, and resource-estimation exporters
- PySCF adapter stubs for chemistry/LMO data ingestion

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev,chem,qiskit]
```

## Layout

- `src/qc_exciton_lcc/exciton`: exciton domain model + screening providers
- `src/qc_exciton_lcc/block_encoding`: backend-agnostic block-encoding contracts
- `src/qc_exciton_lcc/algorithms`: query schedule and generalized query runner skeleton
- `src/qc_exciton_lcc/backends`: interfaces to map abstractions to Qiskit/Qualtran/estimators
- `src/qc_exciton_lcc/chem`: PySCF integration stubs

## Next build targets

1. Implement LMO transformation pipeline in `chem/pyscf_adapter.py`.
2. Build exciton Hamiltonian assembly in `exciton/builder.py`.
3. Add one concrete block-encoding strategy and wire to backend translators.
4. Add logical and physical resource estimator adapters.