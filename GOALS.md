# Goals

The major goal of the project is to a concrete extension of https://arxiv.org/abs/2508.15765. Currently, we do not consider locality yet, but otherwise almost similar. Also, we focus on exciton, not LCC yet. What we have now is, to decompose the Coulomb operator using the tensor hypercontraction (THC), and account for translational symmetry. Our block-encoding scheme is the following: each tensor (matrix) we use direct unitary synthesis, and use space-time trade-off schemes such as https://arxiv.org/abs/1812.00954 and improved version (constant factor) from Sec III.A of https://arxiv.org/abs/2409.11748. For a system of N_k unit cells/k-points and N orbitals per cell, we can have either a O(N_k N^2/\epsilon) scaling and O(\log(N_k N^2)) qubits, or a O(\sqrt{N_k} N^{3/2}/\epsilon) scaling and O(\sqrt{N_k N}) qubits. You could also check XPRIZE_notes.pdf in /notes folder, but be careful it can contain error. Another key thing we consider is, the effective Hamilotnian we consider has much smaller norm, in particular in the case of BSE, the norm should be O(1), while the true many-body Hamiltonian has quite bad norm (scaling with system size). Currently we have the following goals:

- Use QTT to further compress data (e.g. $Q$ indices in THC), or more generally consider decompositions that can further save data.
- Improve constant factors in quantum algorithms, or even scaling if possible.
- Consider where can we apply sparse block-encoding schemes.
- Compare density fitting with THC (worse scaling but maybe smaller constants/subnormalization)
- Incorporate non-abelian symmetries
- Any potential improvements on the final Toffoli complexity/qubit counts/scaling.
