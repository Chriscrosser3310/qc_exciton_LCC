"""Fixed, rounded isometry gates with measurement-driven diagonal sign tracking.

This is an executable dense reference, not a decomposition into hardware gates.
The resource bloqs use the same erasure policy. Classical table updates and
measurement latency are outside their Toffoli metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np


def _capacity(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _rotation(theta: float) -> np.ndarray:
    """G(theta) = Ry(-2 theta); a half-turn here must retain the pair's minus sign."""
    return np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])


@dataclass(frozen=True)
class ReferenceLayer:
    """One diagonal or one matching of disjoint row pairs and its fixed ROM.

    ``records`` contains (ROM address, first row, second row); diagonal records
    use the same row twice. ``addresses`` maps *every* basis state to a ROM
    address, including rows outside the matching and padding. Zero words on
    those addresses do not license dropping their QROAM junk measurements.
    """

    kind: str
    column: int
    level: int
    words: tuple[int, ...]
    addresses: tuple[int, ...]
    records: tuple[tuple[int, int, int], ...]


def qroam_payload(words, addresses, block_size: int) -> np.ndarray:
    """All clean-QROAM output words after its swap-with-zero tree.

    Slot 0 is the used angle; the other slots are the unused words. Invalid
    records are zero padded. This retains their address-dependent signs even
    when the selected word is zero.
    """
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    table = tuple(int(w) for w in words)
    selection = np.asarray(addresses, dtype=int)
    if np.any(selection < 0) or np.any(selection >= len(table)):
        raise ValueError("lookup address is outside the ROM")
    if block_size > _capacity(len(table)):
        raise ValueError("block_size exceeds the ROM capacity")
    padded = np.zeros(((len(table) + block_size - 1) // block_size) * block_size,
                      dtype=object)
    padded[:len(table)] = table
    result = padded[(selection // block_size)[:, None] * block_size
                    + np.arange(block_size)[None, :]].copy()
    # Qualtran's SwapWithZero uses a binary tree, not just one transposition
    # of slots 0 and low. The distinction matters for every unused-word sign.
    for row, low in enumerate(selection % block_size):
        half = 1
        while half < block_size:
            if low & half:
                for first in range(0, block_size, 2 * half):
                    second = first + half
                    result[row, first], result[row, second] = result[row, second], result[row, first]
            half *= 2
    return result


def measurement_signs(payload: np.ndarray, outcomes: tuple[int, ...], bits: int):
    """X-measure every output qubit; return its diagonal byproduct.

    Each outcome is a b-bit mask for one entire word. A measured |+> is reset
    by H; a measured |-> by Z then H. Thus all these qubits are recyclable.
    The caller retains only the classical masks and the basis-address signs.
    """
    if len(outcomes) != payload.shape[1]:
        raise ValueError("one measurement mask is required for every output word")
    if any(mask < 0 or mask >= 1 << bits for mask in outcomes):
        raise ValueError("measurement mask exceeds the word width")
    return np.array([
        -1 if sum((int(word) & int(mask)).bit_count()
                  for word, mask in zip(row, outcomes)) % 2 else 1
        for row in payload
    ], dtype=np.int8)


@dataclass(frozen=True)
class SignFrameRun:
    unitary: np.ndarray
    pending_signs: np.ndarray
    outcomes: tuple[tuple[int, ...], ...]
    measured_qubits: int
    peak_lookup_qubits: int
    final_correction: np.ndarray


@dataclass(frozen=True)
class IsometryReference:
    """An annihilator rounded once, whose adjoint defines forward synthesis."""

    bits: int
    dimension: int
    layers: tuple[ReferenceLayer, ...]
    input_addresses: tuple[int, ...]
    embedded_isometry: np.ndarray

    def directed_layers(self, *, inverse: bool):
        return self.layers if inverse else tuple(reversed(self.layers))

    def _words(self, layer, inverse: bool, signs=None):
        modulus = 1 << self.bits
        direction = 1 if inverse else -1
        words = [(direction * w) % modulus for w in layer.words]
        if signs is not None and layer.kind == "rotation":
            for address, x, y in layer.records:
                words[address] = (int(signs[x]) * int(signs[y]) * words[address]) % modulus
        return words

    def _apply(self, matrix, layer, words):
        for address, x, y in layer.records:
            # Center the modular integer before conversion to avoid subtractive
            # numerical error when forming the adjoint of a small angle.
            word = words[address]
            if word >= (1 << (self.bits - 1)):
                word -= 1 << self.bits
            theta = 2 * pi * word / (1 << self.bits)
            if layer.kind == "phase":
                matrix[x, :] *= np.exp(1j * theta)
            else:
                matrix[[x, y], :] = _rotation(theta) @ matrix[[x, y], :]

    def unitary(self, *, inverse: bool = False) -> np.ndarray:
        result = np.eye(self.dimension, dtype=complex)
        for layer in self.directed_layers(inverse=inverse):
            self._apply(result, layer, self._words(layer, inverse))
        return result

    def run(self, *, inverse: bool = False, block_size: int = 1,
            block_sizes=None, outcomes=None, seed: int = 0, correct: bool = True) -> SignFrameRun:
        """Execute a normalized measurement branch on the entire input space.

        At every boundary actual = D @ reference. A pair uses d_x*d_y*theta;
        phases commute with D. All selected and unused words are X-measured
        after their angle has been used. The final exact diagonal D cancels
        the frame. Independent outcome streams in the two directions still
        give adjoints of the same rounded reference, on *all* columns.

        ``block_sizes`` optionally declares one swap-bank size per stored
        annihilator layer. Forward execution reverses this schedule with the
        gates. Otherwise ``block_size`` applies uniformly to every lookup.
        """
        layers = self.directed_layers(inverse=inverse)
        sizes = (tuple(block_size for _ in self.layers) if block_sizes is None
                 else tuple(block_sizes))
        if len(sizes) != len(self.layers):
            raise ValueError("one block size is required per stored layer")
        if block_sizes is not None and block_size != 1:
            raise ValueError("choose either a uniform or a per-layer block size")
        if not inverse:
            sizes = tuple(reversed(sizes))
        if outcomes is not None and len(outcomes) != len(layers):
            raise ValueError("one outcome tuple is required per lookup")
        rng = np.random.default_rng(seed)
        result = np.eye(self.dimension, dtype=complex)
        signs = np.ones(self.dimension, dtype=np.int8)
        masks, measured, peak = [], 0, 0
        for index, (layer, size) in enumerate(zip(layers, sizes)):
            words = self._words(layer, inverse, signs)
            self._apply(result, layer, words)
            payload = qroam_payload(words, layer.addresses, size)
            outcome = (tuple(int(x) for x in rng.integers(1 << self.bits, size=size))
                       if outcomes is None else tuple(outcomes[index]))
            byproduct = measurement_signs(payload, outcome, self.bits)
            result *= byproduct[:, None]
            signs *= byproduct
            masks.append(outcome)
            measured += self.bits * size
            peak = max(peak, self.bits * size)
        correction = signs.copy()
        if correct:
            result *= correction[:, None]
            signs[:] = 1
        return SignFrameRun(result, signs, tuple(masks), measured, peak, correction)


def compile_isometry_reference(isometries, *, bits: int, layout: str = "fused",
                               row_capacities=None, tree: str = "aligned") -> IsometryReference:
    """Precompute the reference from exact column annihilation, then round once.

    Fused blocks are sorted by decreasing (capacity, column count). Indexed
    blocks keep their order in equal local row registers. ``tree='compact'``
    gives the P-07 live-row tree, with an abstract pair permutation (its gate
    cost remains open). The aligned tree fixes every lower bit, including c=0.
    """
    arrays = [np.asarray(v, dtype=complex) for v in isometries]
    if not arrays or bits < 2 or bits > 62:
        raise ValueError("need isometries and 2 <= bits <= 62")
    if layout not in ("fused", "indexed") or tree not in ("aligned", "compact"):
        raise ValueError("invalid layout or tree")
    for v in arrays:
        if v.ndim != 2 or not 1 <= v.shape[1] <= v.shape[0]:
            raise ValueError("each matrix must have 1 <= columns <= rows")
        if not np.all(np.isfinite(v)) or not np.allclose(v.conj().T @ v, np.eye(v.shape[1]),
                                                      atol=1e-11, rtol=1e-11):
            raise ValueError("columns must be orthonormal")
    capacities = ([_capacity(v.shape[0]) for v in arrays] if row_capacities is None
                  else list(row_capacities))
    if len(capacities) != len(arrays) or any(
            p < v.shape[0] or p & (p - 1) for p, v in zip(capacities, arrays)):
        raise ValueError("one valid binary row capacity is required per matrix")
    if layout == "fused":
        order = sorted(range(len(arrays)), key=lambda r: (capacities[r], arrays[r].shape[1]),
                       reverse=True)
        arrays, capacities = [arrays[r] for r in order], [capacities[r] for r in order]
    else:
        capacities = [max(capacities)] * len(arrays)
    offsets = np.cumsum([0] + capacities[:-1]).tolist()
    dimension = _capacity(sum(capacities))
    work = np.zeros((dimension, sum(v.shape[1] for v in arrays)), dtype=complex)
    inputs, column_offsets, next_column = [], [], 0
    for v, start in zip(arrays, offsets):
        column_offsets.append(next_column)
        work[start:start + v.shape[0], next_column:next_column + v.shape[1]] = v
        inputs.extend(range(start, start + v.shape[1]))
        next_column += v.shape[1]
    embedded = work.copy()
    layers = []
    modulus = 1 << bits

    def append(kind, column, level, angles, records, addresses):
        words = tuple(int(np.rint(angle * modulus / (2 * pi))) % modulus for angle in angles)
        layers.append(ReferenceLayer(kind, column, level, words, tuple(addresses), tuple(records)))

    for column in range(max(v.shape[1] for v in arrays)):
        phase = np.zeros(dimension)
        for r, (v, start, capacity) in enumerate(zip(arrays, offsets, capacities)):
            if column >= v.shape[1]:
                continue
            stop = v.shape[0] if tree == "compact" else capacity
            for row in range(start + column, start + stop):
                value = work[row, column_offsets[r] + column]
                if abs(value) > 1e-14:
                    phase[row] = -np.angle(value)
        work *= np.exp(1j * phase[:, None])
        append("phase", column, -1, phase, [(j, j, j) for j in range(dimension)],
               range(dimension))

        matchings = {}
        for r, (v, start, capacity) in enumerate(zip(arrays, offsets, capacities)):
            if column >= v.shape[1]:
                continue
            if tree == "aligned":
                for q in range(capacity.bit_length() - 1):
                    low = column % (1 << q)
                    beta = (column >> q) & 1
                    first = (column >> (q + 1)) + beta
                    for h in range(first, capacity >> (q + 1)):
                        x = start + (h << (q + 1)) + low
                        matchings.setdefault(q, []).append((x, x + (1 << q), r, beta))
            else:
                # Heap nodes 1..d-1 are internal; splitting in breadth-first
                # order gives the left-complete tree. Label leaves in DFS order.
                d = v.shape[0] - column
                representatives, next_leaf = {}, [start + column]
                def visit(node):
                    if node >= d:
                        representatives[node] = next_leaf[0]
                        next_leaf[0] += 1
                        return
                    visit(2 * node)
                    visit(2 * node + 1)
                    representatives[node] = representatives[2 * node]
                visit(1)
                for node in range(1, d):
                    depth = node.bit_length() - 1
                    matchings.setdefault(-depth, []).append(
                        (representatives[2 * node], representatives[2 * node + 1], r, 0))

        for level in sorted(matchings):
            pairs = matchings[level]
            angles = np.zeros(max(1, dimension // 2))
            if tree == "aligned":
                q = level
                high_bits = (dimension.bit_length() - 1) - q - 1
                addresses = [((j % (1 << q)) << high_bits) + (j >> (q + 1))
                             for j in range(dimension)]
                records = [(addresses[x], x, y) for x, y, _, _ in pairs]
            else:
                # Abstract reversible layout: map each pair to (2a,2a+1),
                # then pair the unused addresses. This has no implicit cost claim.
                unused = sorted(set(range(dimension)) - {j for x, y, _, _ in pairs for j in (x, y)})
                matching = [(x, y) for x, y, _, _ in pairs] + list(zip(unused[::2], unused[1::2]))
                addresses = [0] * dimension
                for address, (x, y) in enumerate(matching):
                    addresses[x] = addresses[y] = address
                records = [(a, x, y) for a, (x, y, _, _) in enumerate(pairs)]
            for (address, x, y), (_, _, r, beta) in zip(records, pairs):
                a, b = work[[x, y], column_offsets[r] + column]
                if max(abs(a.imag), abs(b.imag)) > 1e-10:
                    raise ArithmeticError("reference tree did not remain real")
                theta = (atan2(-a.real, b.real) if beta else atan2(b.real, a.real))
                if abs(a) + abs(b) < 1e-14:
                    theta = 0.0
                angles[address] = theta
                work[[x, y], :] = _rotation(theta) @ work[[x, y], :]
            append("rotation", column, level, angles, records, addresses)
    target = np.eye(dimension, dtype=complex)[:, inputs]
    if not np.allclose(work, target, atol=1e-10, rtol=1e-10):
        raise ArithmeticError("reference annihilation did not produce the identity embedding")
    embedded.setflags(write=False)
    return IsometryReference(bits, dimension, tuple(layers), tuple(inputs), embedded)
