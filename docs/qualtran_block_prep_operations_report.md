---
title: "Operations Implemented by Qualtran Block State and Block Unitary Preparation"
author: "Generated code report"
date: "2026-05-10"
geometry: margin=0.85in
fontsize: 10pt
---

# Summary

This report summarizes the mathematical operations implemented by:

- `src/integrations/qualtran/block_state_preparation_QROAM.py`
- `src/integrations/qualtran/block_unitary_synthesis_QROAM.py`

The shared pattern is that a block register selects which data table, state, or unitary is used. The block register is treated as an address and is preserved by the operation.

# Block-Indexed QROAM Rotation Lookup

The low-level operation is implemented by `BlockPRGAViaPhaseGradientQROAM`.

Let `j` be the block index, `x` be the selection index, and `a_{j,x}` be an integer rotation-table entry. The QROAM lookup implements

$$
|j\rangle |x\rangle |0\rangle
\longmapsto
|j\rangle |x\rangle |a_{j,x}\rangle .
$$

When there is only one block, the block address is omitted internally:

$$
|x\rangle |0\rangle
\longmapsto
|x\rangle |a_x\rangle .
$$

The loaded value is added into a phase-gradient register. In direct terms, the code performs

$$
|a_{j,x}\rangle |\mathrm{grad}\rangle
\longmapsto
|a_{j,x}\rangle e^{i \theta(a_{j,x})} |\mathrm{grad}\rangle ,
$$

where the integer table value encodes the rotation angle at the chosen phase precision.

After the phase is applied, the QROAM lookup is uncomputed:

$$
|j\rangle |x\rangle |a_{j,x}\rangle
\longmapsto
|j\rangle |x\rangle |0\rangle .
$$

So the net effect is a clean controlled phase rotation:

$$
|j\rangle |x\rangle |\mathrm{grad}\rangle
\longmapsto
e^{i \theta(a_{j,x})}
|j\rangle |x\rangle |\mathrm{grad}\rangle .
$$

# Block State Preparation

The block state-preparation operation is implemented by `BlockStatePreparationViaQROAMRotations`.

The input data is a matrix of normalized state coefficients:

$$
C =
\begin{bmatrix}
c_{0,0} & c_{0,1} & \cdots & c_{0,N-1} \\
c_{1,0} & c_{1,1} & \cdots & c_{1,N-1} \\
\vdots & \vdots & \ddots & \vdots \\
c_{B-1,0} & c_{B-1,1} & \cdots & c_{B-1,N-1}
\end{bmatrix},
$$

with one normalized state per block:

$$
\sum_{x=0}^{N-1} |c_{j,x}|^2 = 1
\quad \text{for each } j .
$$

For a selected block `j`, the target operation is

$$
|j\rangle |0\rangle
\longmapsto
|j\rangle
\sum_{x=0}^{N-1} c_{j,x} |x\rangle .
$$

More generally, with optional preparation controls, the intended controlled operation is

$$
|p\rangle |j\rangle |0\rangle
\longmapsto
\begin{cases}
|p\rangle |j\rangle \sum_x c_{j,x}|x\rangle, & p = 1, \\
|p\rangle |j\rangle |0\rangle, & p = 0 .
\end{cases}
$$

The code builds this operation from a binary rotation tree. The amplitude part prepares the magnitudes:

$$
|j\rangle |0\rangle
\longmapsto
|j\rangle
\sum_x |c_{j,x}| |x\rangle .
$$

The phase part then adds the complex phases:

$$
|j\rangle
\sum_x |c_{j,x}| |x\rangle
\longmapsto
|j\rangle
\sum_x |c_{j,x}| e^{i \phi_{j,x}} |x\rangle ,
$$

where

$$
c_{j,x} = |c_{j,x}| e^{i \phi_{j,x}} .
$$

Thus the final prepared state is

$$
|j\rangle
\sum_x c_{j,x}|x\rangle .
$$

If `uncompute=True`, the operation is inverted:

$$
|j\rangle
\sum_x c_{j,x}|x\rangle
\longmapsto
|j\rangle |0\rangle .
$$

# Block-Indexed Householder State Preparation

The Householder state preparation is implemented by `BlockPrepareHouseholderStateQROAM`.

For each block `j` and basis index `k`, let

$$
|u_{j,k}\rangle
= \sum_x u_{j,k,x} |x\rangle
$$

be the column vector selected from the block-unitary data. The code prepares the Householder state

$$
|w_{j,k}\rangle
=
\frac{|1\rangle |k\rangle - |0\rangle |u_{j,k}\rangle}{\sqrt{2}} .
$$

In operation form, with a clean reflection ancilla and system register, the preparation is

$$
|j\rangle |0\rangle |0\rangle
\longmapsto
|j\rangle |w_{j,k}\rangle .
$$

The block register is unchanged:

$$
|j\rangle \longmapsto |j\rangle .
$$

The code builds the two branches as follows:

1. It creates an ancilla superposition and phase convention.
2. It writes the computational basis state `|k>` into the system branch.
3. It applies block-selected state preparation to create `|u_{j,k}>` in the other branch.

The inverse operation is

$$
|j\rangle |w_{j,k}\rangle
\longmapsto
|j\rangle |0\rangle |0\rangle .
$$

# Block Householder Reflection

The reflection is implemented by `BlockHouseholderReflectionQROAM`.

For each selected block `j` and basis index `k`, the code implements a reflection about the prepared Householder state:

$$
R_{j,k}
=
I - 2 |w_{j,k}\rangle \langle w_{j,k}| .
$$

Because the block register selects the vector, the full block-indexed reflection is

$$
R_k
=
\sum_j |j\rangle \langle j| \otimes
\left(I - 2 |w_{j,k}\rangle \langle w_{j,k}|\right).
$$

The decomposition used in code is:

$$
R_k
=
W_k
\left(I - 2 |0\rangle \langle 0|\right)
W_k^\dagger ,
$$

where `W_k` is the block-indexed Householder state-preparation operation:

$$
W_k |j\rangle |0\rangle
=
|j\rangle |w_{j,k}\rangle .
$$

In words: unprepare the Householder state, apply a zero-state phase flip, then prepare the Householder state again.

# Block-Diagonal Unitary Synthesis

The top-level block-unitary operation is implemented by `BlockUnitarySynthesisQROAM`.

The target unitary is block diagonal:

$$
U_{\mathrm{block}}
=
\sum_{j=0}^{B-1}
|j\rangle \langle j| \otimes U_j .
$$

For an input superposition over block indices and system states,

$$
\sum_j \alpha_j |j\rangle |\psi_j\rangle ,
$$

the intended operation is

$$
\sum_j \alpha_j |j\rangle |\psi_j\rangle
\longmapsto
\sum_j \alpha_j |j\rangle U_j |\psi_j\rangle .
$$

The block register is not mixed between blocks. There is no transition of the form

$$
|j\rangle \to |j'\rangle
\quad \text{for } j \ne j' .
$$

The data tensor has shape

$$
\texttt{block\_unitaries} \in \mathbb{C}^{B \times N \times r},
$$

where:

- `B` is the number of blocks,
- `N` is the system dimension,
- `r` is the number of Householder reflections used.

For each block, the columns must be orthonormal:

$$
U_j^\dagger U_j = I
$$

on the supplied columns.

The code applies one reflection per supplied basis index:

$$
U_{\mathrm{block}}
\approx
R_{r-1} R_{r-2} \cdots R_1 R_0 .
$$

With all required columns/reflections supplied, this realizes the intended block-selected unitary synthesis:

$$
|j\rangle |\psi\rangle
\longmapsto
|j\rangle U_j |\psi\rangle .
$$

# End-to-End Operation

Putting the pieces together, the implementation chain is:

$$
\boxed{
\text{QROAM lookup}
\rightarrow
\text{phase-gradient rotation}
\rightarrow
\text{block state preparation}
\rightarrow
\text{Householder state}
\rightarrow
\text{Householder reflection}
\rightarrow
\text{block-diagonal unitary}
}
$$

The key preserved-register property is:

$$
|j\rangle \text{ is used as an address, and exits as } |j\rangle .
$$

The key selected-operation property is:

$$
\boxed{
|j\rangle |\psi\rangle
\longmapsto
|j\rangle U_j |\psi\rangle
}
$$

For state preparation alone, the corresponding operation is:

$$
\boxed{
|j\rangle |0\rangle
\longmapsto
|j\rangle \sum_x c_{j,x}|x\rangle
}
$$

# Practical Notes

The code supports two usage modes.

Concrete-data mode uses actual NumPy arrays. In this mode, the code can decompose the bloqs into lower-level Qualtran operations.

Shape-only mode uses `Shaped(...)` objects. In this mode, the code can build resource call graphs without storing all numerical data, but it intentionally refuses exact decomposition because the actual rotation tables are missing.
