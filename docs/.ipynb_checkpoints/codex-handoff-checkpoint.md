# Codex Handoff

## Current Goal

Build a Qualtran notebook at `src/integrations/qualtran/data_loading_comparison.ipynb` that compares three data-loading schemes across data sizes and block-size settings:

- `QROM`
- `SelectSwapQROM`, imported as `SelectSwap`
- `QROAMClean`

For each configuration, the notebook constructs a shape-only bloq that performs:

1. data loading
2. symbolic data-register-controlled `CRy(theta_i)` rotations
3. inverse data loading

The notebook then computes and plots Toffoli counts and ancilla counts for the schemes.

## Files Changed

- `src/integrations/qualtran/data_loading_comparison.ipynb`
  - New notebook.
  - Defines `DataLoadingControlledRotation`, a `GateWithRegisters` wrapper.
  - Uses `Shaped((num_entries,))` so no concrete ROM data is supplied.
  - Sweeps:
    - `DATA_SIZES = [{2**8, 8 bits}, {2**10, 8 bits}, {2**12, 12 bits}, {2**14, 16 bits}]`
    - `LOG_BLOCK_SIZES = [0, 2, 4, 6]` for `SelectSwap` and `QROAMClean`
    - `QROM` as a fixed baseline, since it has no `log_block_sizes` parameter.
  - Computes `records`, a list of resource-count dictionaries.
  - Creates `toffoli_fig` and `ancilla_fig`.

- `docs/codex-handoff.md`
  - This handoff document.

- `AGENTS.md`
  - Added stable project rules for future agents.

## Important Decisions

- Shape-only data is used via `qualtran.symbolics.Shaped`; the notebook intentionally does not allocate or input actual data values.
- `QROM` is plotted once per data size as a fixed baseline because its constructor does not expose a block-size tradeoff.
- `SelectSwapQROM` and `QROAMClean` are swept over `log_block_sizes`.
- The inverse operations are:
  - `QROM(...).adjoint()`
  - `SelectSwapQROM(...).adjoint()`
  - explicit `QROAMCleanAdjoint(...)`
- `QROAMClean` may produce junk registers for nonzero `log_block_sizes`; the wrapper applies `QROAMCleanAdjoint` to `target0_` and then frees the junk registers so the composite bloq closes cleanly.
- Toffoli counts are computed from `QECGatesCost().total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"]`, using `generalize_cswap_approx`.
- Ancilla counts are computed as peak logical qubits from `QubitCount()` minus external signature qubits.
- Symbolic `CRy(theta_i)` rotations are included structurally, but `ts_per_rotation=0` means rotation-synthesis T cost is not included in the reported Toffoli count.
- `MPLCONFIGDIR=/tmp/matplotlib` was used for command-line validation because `/resnick/home/jchen9/.config/matplotlib` was not writable in the sandbox.

## Commands Run

Repository inspection:

```bash
rg --files src/integrations/qualtran
find src/integrations/qualtran -maxdepth 2 -type f
git status --short
sed -n '1,220p' src/integrations/qualtran/state_prep_QROAM.py
sed -n '1,220p' src/integrations/qualtran/unitary_synthesis_QROAM.py
sed -n '1,220p' src/integrations/qualtran/utils.py
```

Qualtran API checks:

```bash
python - <<'PY'
import qualtran, inspect
print('qualtran', getattr(qualtran, '__version__', 'no version'))
from qualtran.bloqs.data_loading.qrom import QROM
print('QROM', inspect.signature(QROM))
try:
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
    print('SelectSwapQROM', inspect.signature(SelectSwapQROM))
except Exception as e:
    print('SelectSwapQROM err', type(e).__name__, e)
try:
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    print('QROAMClean qroam_clean', inspect.signature(QROAMClean))
except Exception as e:
    print('QROAMClean qroam_clean err', type(e).__name__, e)
try:
    from qualtran.bloqs.data_loading.qroam_clean import QROAMCleanAdjoint
    print('QROAMCleanAdjoint', inspect.signature(QROAMCleanAdjoint))
except Exception as e:
    print('QROAMCleanAdjoint err', type(e).__name__, e)
PY
```

Additional API and signature checks:

```bash
python - <<'PY'
from qualtran.bloqs.basic_gates import *
import qualtran.bloqs.basic_gates as bg
print([x for x in dir(bg) if 'Rot' in x or 'Ry' in x or 'Rz' in x or 'ZPow' in x or 'Y' == x])
try:
 from qualtran.bloqs.basic_gates.rotation import Ry, Rz, ZPowGate
 import inspect
 print('Ry', inspect.signature(Ry))
 print('Rz', inspect.signature(Rz))
 print('ZPowGate', inspect.signature(ZPowGate))
except Exception as e: print('rotation imports', type(e).__name__, e)
PY

python - <<'PY'
from qualtran import BloqBuilder, Signature
import inspect
print('BloqBuilder.add', inspect.signature(BloqBuilder.add))
print('BloqBuilder.finalize', inspect.signature(BloqBuilder.finalize))
PY

python - <<'PY'
from qualtran.symbolics import Shaped, HasLength
import inspect
print('Shaped', inspect.signature(Shaped))
print('HasLength', inspect.signature(HasLength))
PY

python - <<'PY'
from qualtran.bloqs.basic_gates import CRy, CRz
import inspect
print('CRy', inspect.signature(CRy))
print('CRz', inspect.signature(CRz))
PY

python - <<'PY'
from qualtran.bloqs.basic_gates import CRy
import sympy
print(CRy(sympy.Symbol('theta')).signature)
PY
```

Shape-only loader experiments:

```bash
python - <<'PY'
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.symbolics import Shaped
for cls in [QROM, SelectSwapQROM, QROAMClean, QROAMCleanAdjoint]:
 b=cls(data_or_shape=(Shaped((8,)),), selection_bitsizes=(3,), target_bitsizes=(8,))
 print('\n',cls.__name__, b)
 print(b.signature)
 try: print('adj type', type(b.adjoint()).__name__)
 except Exception as e: print('adj err', e)
PY

python - <<'PY'
from qualtran import BloqBuilder
print([m for m in dir(BloqBuilder) if not m.startswith('_')])
PY
```

Wrapper/decomposition validation:

```bash
MPLCONFIGDIR=/tmp/matplotlib python - <<'PY'
# Built a temporary DataLoadingRotationComparisonBloq and decomposed
# qrom, select_swap, and qroam_clean small instances.
PY
```

Resource-counting and plotting validation:

```bash
MPLCONFIGDIR=/tmp/matplotlib python - <<'PY'
import json
from pathlib import Path
ns={}
nb=json.loads(Path('src/integrations/qualtran/data_loading_comparison.ipynb').read_text())
code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
exec(code, ns)
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting.generalizers import generalize_cswap_approx
D=ns['DataLoadingControlledRotation']
for loader in ns['LOADER_NAMES']:
  for lbs in [None,0,1,2]:
    try:
      b=D(loader, 8, 4, 3, lbs)
      gc=get_cost_value(b,QECGatesCost(),generalizer=generalize_cswap_approx)
      tq=gc.total_t_and_ccz_count(ts_per_rotation=0)['n_ccz']
      qc=get_cost_value(b,QubitCount())
      print(loader,lbs,'toff',tq,'qubits',qc)
    except Exception as e:
      print('ERR',loader,lbs,type(e).__name__,e)
PY

MPLCONFIGDIR=/tmp/matplotlib python - <<'PY'
import json
from pathlib import Path
path=Path('src/integrations/qualtran/data_loading_comparison.ipynb')
nb=json.loads(path.read_text())
print('cells', len(nb['cells']))
code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
compile(code, str(path), 'exec')
print('compiled')
PY

MPLCONFIGDIR=/tmp/matplotlib python - <<'PY'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
path=Path('src/integrations/qualtran/data_loading_comparison.ipynb')
nb=json.loads(path.read_text())
code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
ns={}
exec(code, ns)
print('records', len(ns['records']))
print('figures', ns['toffoli_fig'] is not None, ns['ancilla_fig'] is not None)
print('sample', ns['records'][0]['loader'], ns['records'][0]['toffoli_count'], ns['records'][0]['ancilla_count'])
PY
```

Final status checks:

```bash
git status --short src/integrations/qualtran/data_loading_comparison.ipynb
git diff --stat -- src/integrations/qualtran/data_loading_comparison.ipynb
pwd
rg --files -g 'AGENTS.md' -g 'docs/**'
ls -la
ls -la docs
find .. -name AGENTS.md -print
ps -o pid,ppid,stat,cmd -u jchen9
```

## Tests Passing/Failing

Passing:

- Notebook JSON parsed successfully.
- Notebook code cell compiled successfully.
- Full notebook code executed successfully under `matplotlib.use("Agg")`.
- The full sweep produced:
  - `records 36`
  - `figures True True`
  - sample output: `QROM 508 15`
- Small decomposition checks passed for all three schemes after handling `QROAMClean` junk registers.

Failing or expected intermediate failures:

- Initial shape-only `QROM(data_or_shape=Shaped((8,)), ...)` failed because Qualtran expects `data_or_shape` as a tuple of target arrays/shapes; fixed by using `(Shaped((8,)),)`.
- Initial QROAMClean nonzero block-size wrapper failed because `QROAMClean` emitted junk registers; fixed by collecting and freeing those registers.
- No full repository test suite was run.

## Unresolved Issues

- `src/integrations/qualtran/data_loading_comparison.ipynb` is still untracked.
- The repository had many pre-existing deleted files under `src/integrations/pennylane` and `src/integrations/qualtran` before this handoff work. These were not touched.
- There are pre-existing untracked archives, logs, `.ipynb_checkpoints`, and QROAM-related files. These were not touched except for the new comparison notebook.
- Notebook outputs are not saved in the file; the notebook is intended to be executed to render plots.
- The plotted Toffoli counts use `ts_per_rotation=0`, so they omit synthesis cost for symbolic controlled rotations.

## Exact Next Steps

1. Open and run `src/integrations/qualtran/data_loading_comparison.ipynb`.
2. Inspect the `records` list for raw numbers and the `toffoli_fig` / `ancilla_fig` plots.
3. Adjust `DATA_SIZES` or `LOG_BLOCK_SIZES` in the notebook if a denser or larger sweep is needed.
4. Decide whether rotation synthesis should be included; if yes, change `ts_per_rotation=0` in `resource_counts`.
5. Decide whether notebook outputs should be saved in git or kept output-free.
6. Review `git status --short` and separate this notebook/handoff work from the pre-existing deleted and untracked files before committing.
