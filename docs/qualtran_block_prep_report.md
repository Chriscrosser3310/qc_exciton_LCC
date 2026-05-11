---
title: "Qualtran Block State Preparation and Block Unitary Preparation"
author: "Generated code report"
date: "2026-05-10"
geometry: margin=0.8in
fontsize: 10pt
---

# Scope

This report explains the Qualtran code in this repository that prepares block-selected states and block-selected unitaries.

The relevant files are:

- `src/integrations/qualtran/block_state_preparation_QROAM.py`
- `src/integrations/qualtran/block_unitary_synthesis_QROAM.py`

The code is built around one idea: keep a `block` register unchanged, and use it as an extra address into QROAM tables. If the block register stores `j`, then the circuit prepares or applies the data for block `j`.

# Block State Preparation

## Helper conversion functions

Location: `block_state_preparation_QROAM.py`, lines 35-44.

`_to_block_array_or_shape(x)` accepts either concrete coefficient data or a Qualtran `Shaped` object. Concrete data is converted to a complex NumPy array. `Shaped` is kept as-is so the code can do resource estimates without storing real coefficient data.

`_to_int_table_or_shape(x)` does the same job for integer ROM tables. Concrete data is converted into tuples of Python integers. Symbolic shape data is kept as a `Shaped` object.

## `BlockPRGAViaPhaseGradientQROAM`

Location: `block_state_preparation_QROAM.py`, lines 47-229.

This class is the low-level controlled rotation loader.

Inputs:

- `block`: selects which block of data to use.
- `selection`: selects which entry inside that block to use.
- `control`: controls whether the rotation happens.
- `phase_gradient`: receives the loaded phase as an addition into a phase-gradient state.

What it does:

1. It builds a `QROAMClean` lookup.
2. The lookup address is either `(block, selection)` or only `selection` when there is exactly one block.
3. It loads an integer rotation angle from `rom_values`.
4. It adds that angle into the phase-gradient register using `AddIntoPhaseGrad`.
5. It uncomputes the QROAM lookup with `QROAMCleanAdjoint`.

Plain-language summary: this bloq looks up the rotation angle for the selected block and selected state node, applies that angle through phase kickback, and cleans up the lookup junk.

Important methods and properties:

- `qroam_log_block_sizes` and `qroam_adjoint_log_block_sizes`: choose valid QROAM blocking parameters for forward and inverse table lookups.
- `n_blocks`: returns how many block rows are stored.
- `n_values`: returns how many values exist inside each block.
- `qroam_selection_sources`: decides which public registers are actually needed as QROAM selection registers.
- `signature`: declares the public registers.
- `qroam_bloq`: builds the forward QROAM lookup.
- `qroam_bloq_for_cost`: builds a data-free version for resource counting.
- `qroam_adj_bloq`: builds the inverse QROAM cleanup.
- `build_composite_bloq`: performs lookup, phase addition, and cleanup.
- `build_call_graph`: reports one forward QROAM, one inverse QROAM, and one phase-gradient addition.

## `BlockStatePreparationViaQROAMRotations`

Location: `block_state_preparation_QROAM.py`, lines 232-443.

This class prepares one normalized state from a list of states. The `block` register selects which state is prepared. The `block` register is not changed.

Data shape:

- `state_coefficients` has shape `(n_blocks, n_coeff)`.
- Each row is one state vector.
- `n_coeff` must be a power of two.
- Concrete state rows must have norm 1.

Registers:

- `prepare_control`: optional external control bits.
- `block`: chooses the state row.
- `target_state`: the qubits that receive the prepared state.
- `phase_gradient`: workspace for phase-gradient rotations.

What it does:

1. It checks the input shape and normalization.
2. It builds one rotation tree per block.
3. It creates QROAM-backed amplitude rotations.
4. It creates QROAM-backed phase rotations.
5. It applies amplitude preparation and phase preparation in the correct order.
6. If `uncompute=True`, it runs the inverse order.

Plain-language summary: this bloq prepares `target_state` into the state stored in row `block` of `state_coefficients`. It uses QROAM tables so the same circuit structure can select many states from one block index.

Important methods and properties:

- `from_bitsize`: builds a resource-estimate object from only dimensions.
- `__attrs_post_init__`: validates control size, phase precision, state shape, power-of-two dimension, and normalization.
- `n_blocks`, `n_coeff`, `block_bitsize`, `state_bitsize`: compute sizes used by the Qualtran signature.
- `signature`: declares `prepare_control`, `block`, `target_state`, and `phase_gradient`.
- `rotation_trees`: converts each block's coefficient vector into a rotation tree.
- `prga_prepare_amplitude`: builds one `BlockPRGAViaPhaseGradientQROAM` per amplitude-preparation layer.
- `prga_prepare_phases`: builds one `BlockPRGAViaPhaseGradientQROAM` for the final phase layer.
- `build_composite_bloq`: decomposes the state preparation into amplitude and phase steps. It refuses to decompose when only symbolic shapes are present.
- `build_call_graph`: reports the rotation gates, X gates, phase PRGA, and amplitude PRGAs needed for cost estimation.
- `_prepare_amplitudes`: walks through the target-state qubits, applies basis changes, applies QROAM-controlled rotations, and restores the qubits.
- `_prepare_phases`: allocates one rotation ancilla, turns it on, applies the phase-rotation PRGA, then frees the ancilla.

# Block Unitary Preparation

## Helper conversion function

Location: `block_unitary_synthesis_QROAM.py`, lines 27-32.

`_to_block_unitaries_or_shape(x)` accepts either concrete block-unitary data or a Qualtran `Shaped` object. Concrete data is converted to a complex NumPy array. Symbolic shape data is kept as `Shaped`.

## `BlockPrepareHouseholderStateQROAM`

Location: `block_unitary_synthesis_QROAM.py`, lines 35-193.

This class prepares a block-selected Householder state. For block `j` and basis column `k`, it prepares

`(|1>|k> - |0>|u_{j,k}>)/sqrt(2)`.

The `block` register is used only as a selector and is left unchanged.

Registers:

- `block`: selects which block's vector to use.
- `reflection_ancilla`: one ancilla used in the Householder construction.
- `system`: the system register.
- `phase_gradient`: workspace for the QROAM state preparation.

What it does:

1. It validates that each block-selected vector is normalized.
2. It builds a controlled `BlockStatePreparationViaQROAMRotations`.
3. It prepares the basis part `|k>` using CNOTs controlled by the reflection ancilla.
4. It prepares the data-vector part `|u_{j,k}>` when the ancilla selects that branch.
5. It supports `adjoint()` by toggling `uncompute`.

Plain-language summary: this bloq builds the special state needed for a Householder reflection, and it chooses the correct vector by reading the `block` register.

Important methods and properties:

- `state_prep`: creates the controlled block state-preparation bloq used inside the Householder state.
- `adjoint`: returns the inverse preparation.
- `_basis_one_positions`: identifies which bits are 1 in the selected basis index.
- `_apply_basis_cnot_ladder`: writes the basis index into the system register where needed.
- `_apply_controlled_state_prep`: applies block-selected state preparation controlled by the reflection ancilla.
- `build_composite_bloq`: orders Hadamard, Z, CNOT ladder, and controlled state preparation. It reverses the order when uncomputing.
- `build_call_graph`: reports the gate and sub-bloq counts.

## `BlockHouseholderReflectionQROAM`

Location: `block_unitary_synthesis_QROAM.py`, lines 196-291.

This class reflects about the block-selected Householder state.

What it does:

1. It unprepares the Householder state.
2. It reflects around the all-zero computational basis state.
3. It prepares the Householder state again.

Plain-language summary: this bloq applies one reflection associated with one column of the block-selected unitary.

Important methods and properties:

- `prepare_w`: builds the `BlockPrepareHouseholderStateQROAM` object.
- `_reflect_around_zero`: applies a Z or multi-controlled Z around the zero state.
- `build_composite_bloq`: applies `prepare_w.adjoint()`, the zero reflection, and `prepare_w`.
- `build_call_graph`: reports two Householder preparations, two X gates, and either one Z or one multi-controlled Z.

## `BlockUnitarySynthesisQROAM`

Location: `block_unitary_synthesis_QROAM.py`, lines 294-404.

This is the top-level block-unitary synthesis bloq.

It synthesizes block-diagonal data of the form

`sum_j |j><j| tensor U_j`.

Data shape:

- `block_unitaries` has shape `(n_blocks, N, n_reflections)`.
- `n_blocks` is the number of block-selected unitaries.
- `N` is the system dimension and must be a power of two.
- `n_reflections` is the number of columns/reflections to synthesize.
- Concrete columns must be orthonormal inside each block.

Registers:

- `block`: selects which `U_j` is applied.
- `reflection_ancilla`: workspace for Householder reflections.
- `system`: target system register.
- `phase_gradient`: workspace for QROAM rotations.

What it does:

1. It validates the block-unitary tensor shape.
2. It validates that each block's columns are orthonormal.
3. For each requested basis index, it extracts the corresponding block-column state data.
4. It builds a `BlockHouseholderReflectionQROAM`.
5. It applies all requested reflections in sequence.

Plain-language summary: this bloq applies a unitary chosen by the `block` register. It does that by breaking each unitary into Householder reflections, and each reflection uses block-selected QROAM state preparation.

Important methods and properties:

- `from_shape`: builds a data-free object for resource estimates.
- `__attrs_post_init__`: validates shape, system dimension, number of columns, and orthonormality.
- `n_blocks`, `n_rows`, `n_reflections`, `block_bitsize`, `system_bitsize`: compute dimensions and register sizes.
- `signature`: declares the public Qualtran registers.
- `reflection(basis_index)`: creates one block-selected Householder reflection for one basis column.
- `build_composite_bloq`: applies each reflection. It requires concrete data.
- `build_call_graph`: reports one reflection per requested basis index. For symbolic `n_reflections`, it reports a symbolic multiplier.

# Execution Chain

The full block-unitary path is:

1. `BlockUnitarySynthesisQROAM`
2. `BlockHouseholderReflectionQROAM`
3. `BlockPrepareHouseholderStateQROAM`
4. `BlockStatePreparationViaQROAMRotations`
5. `BlockPRGAViaPhaseGradientQROAM`
6. `QROAMClean`, `AddIntoPhaseGrad`, and `QROAMCleanAdjoint`

In direct terms: the top-level unitary code loops over reflections. Each reflection prepares a Householder state. Each Householder state calls block-selected state preparation. State preparation calls QROAM-backed rotation lookups. The block register is threaded through these calls as an address bit string and is preserved.

# Resource-Estimate Behavior

Both modules support two modes:

- Concrete mode: real NumPy arrays are supplied. The code can decompose into actual Qualtran bloqs.
- Shape-only mode: `Shaped(...)` objects are supplied. The code can build call graphs for resource estimates, but data-dependent decomposition is blocked with `DecomposeTypeError`.

This split is intentional. It lets large experiments estimate resources without storing every coefficient, while still allowing small concrete examples to decompose exactly.

# Main Takeaways

- Block state preparation is implemented by `BlockStatePreparationViaQROAMRotations`.
- Block unitary preparation is implemented by `BlockUnitarySynthesisQROAM`.
- The shared mechanism is block-indexed QROAM lookup.
- The `block` register selects data but is not modified.
- State preparation uses rotation trees and phase-gradient rotations.
- Unitary synthesis uses Householder reflections, and each reflection uses the block state-preparation code.
- Concrete data enables decomposition. Symbolic shape data enables resource counting.
