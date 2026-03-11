# Legacy Adapter Layer Archive

This folder stores code removed from `src/` during cleanup to keep the active source tree
focused and provider-specific.

Archived on this pass:
- `backends/` (cross-SDK adapter abstraction layer)
- `algorithms/` adapter-oriented abstractions and old Qualtran shim
- root-level `block_encoding` compatibility shims that mirrored Qualtran modules

Active provider-specific code now lives in:
- `src/integrations/qualtran/`
- `src/integrations/pennylane/`

Core SDK-neutral code remains in:
- `src/block_encoding/base.py`
- `src/block_encoding/sparse_matrix.py`
- `src/exciton/*`, `src/oracles/*`, etc.
