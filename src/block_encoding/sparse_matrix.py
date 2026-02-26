from __future__ import annotations

from dataclasses import dataclass
from math import asin, ceil, log2, pi, sqrt
from typing import Callable

from oracles.function_ir import CompilableFunctionForm
from oracles.reversible_synth import ReversibleCircuit, compile_function_form

from .base import BlockEncoding, BlockEncodingMetadata, BlockEncodingQuery


def _bits_for_range(size: int) -> int:
    if size <= 1:
        return 1
    return ceil(log2(size))


def _encode_uint(value: int, n_bits: int) -> str:
    if value < 0 or value >= (1 << n_bits):
        raise ValueError(f"Value {value} cannot fit in {n_bits} bits.")
    return format(value, f"0{n_bits}b")


def _decode_uint(bits: str) -> int:
    return int(bits, 2)

def _encode_signed_fixed(value: float, total_bits: int, frac_bits: int) -> str:
    scale = 1 << frac_bits
    scaled = int(round(value * scale))
    min_int = -(1 << (total_bits - 1))
    max_int = (1 << (total_bits - 1)) - 1
    if scaled < min_int or scaled > max_int:
        raise ValueError(
            f"Value {value} overflows signed fixed-point [{min_int/scale}, {max_int/scale}] "
            f"for total_bits={total_bits}, frac_bits={frac_bits}."
        )
    if scaled < 0:
        scaled = (1 << total_bits) + scaled
    return format(scaled, f"0{total_bits}b")


def _decode_signed_fixed(bits: str, frac_bits: int) -> float:
    raw = int(bits, 2)
    total_bits = len(bits)
    if bits[0] == "1":
        raw -= 1 << total_bits
    return raw / float(1 << frac_bits)


@dataclass(frozen=True)
class RowAccessOracle:
    """Classical row-access oracle O_r: (row, l) -> col(row, l)."""

    n_rows: int
    n_cols: int
    max_row_nnz: int
    table: dict[tuple[int, int], int]

    @classmethod
    def from_function(
        cls,
        n_rows: int,
        n_cols: int,
        max_row_nnz: int,
        row_to_col_fn: Callable[[int, int], int],
    ) -> RowAccessOracle:
        table: dict[tuple[int, int], int] = {}
        for row in range(n_rows):
            for l_pos in range(max_row_nnz):
                col = int(row_to_col_fn(row, l_pos))
                if col < 0 or col >= n_cols:
                    raise ValueError(f"Row oracle produced invalid column index {col}.")
                table[(row, l_pos)] = col
        return cls(n_rows=n_rows, n_cols=n_cols, max_row_nnz=max_row_nnz, table=table)

    def lookup(self, row: int, l_pos: int) -> int:
        return self.table[(row, l_pos)]

    def compile_truth_table(self) -> dict[str, str]:
        row_bits = _bits_for_range(self.n_rows)
        l_bits = _bits_for_range(self.max_row_nnz)
        col_bits = _bits_for_range(self.n_cols)
        compiled: dict[str, str] = {}
        for (row, l_pos), col in self.table.items():
            key = _encode_uint(row, row_bits) + _encode_uint(l_pos, l_bits)
            compiled[key] = _encode_uint(col, col_bits)
        return compiled

    def compile_reversible_circuit(self) -> ReversibleCircuit:
        row_bits = _bits_for_range(self.n_rows)
        l_bits = _bits_for_range(self.max_row_nnz)
        in_bits = row_bits + l_bits
        out_bits = _bits_for_range(self.n_cols)

        def packed_fn(x: int) -> int:
            l_mask = (1 << l_bits) - 1
            l_pos = x & l_mask
            row = x >> l_bits
            return self.lookup(row=row, l_pos=l_pos)

        form = CompilableFunctionForm(
            n_input_bits=in_bits,
            n_output_bits=out_bits,
            fn=packed_fn,
            name="row_access_oracle",
        )
        return compile_function_form(form)


@dataclass(frozen=True)
class ColAccessOracle:
    """Classical column-access oracle O_c: (col, l) -> row(col, l)."""

    n_rows: int
    n_cols: int
    max_col_nnz: int
    table: dict[tuple[int, int], int]

    @classmethod
    def from_function(
        cls,
        n_rows: int,
        n_cols: int,
        max_col_nnz: int,
        col_to_row_fn: Callable[[int, int], int],
    ) -> ColAccessOracle:
        table: dict[tuple[int, int], int] = {}
        for col in range(n_cols):
            for l_pos in range(max_col_nnz):
                row = int(col_to_row_fn(col, l_pos))
                if row < 0 or row >= n_rows:
                    raise ValueError(f"Col oracle produced invalid row index {row}.")
                table[(col, l_pos)] = row
        return cls(n_rows=n_rows, n_cols=n_cols, max_col_nnz=max_col_nnz, table=table)

    def lookup(self, col: int, l_pos: int) -> int:
        return self.table[(col, l_pos)]

    def compile_truth_table(self) -> dict[str, str]:
        col_bits = _bits_for_range(self.n_cols)
        l_bits = _bits_for_range(self.max_col_nnz)
        row_bits = _bits_for_range(self.n_rows)
        compiled: dict[str, str] = {}
        for (col, l_pos), row in self.table.items():
            key = _encode_uint(col, col_bits) + _encode_uint(l_pos, l_bits)
            compiled[key] = _encode_uint(row, row_bits)
        return compiled

    def compile_reversible_circuit(self) -> ReversibleCircuit:
        col_bits = _bits_for_range(self.n_cols)
        l_bits = _bits_for_range(self.max_col_nnz)
        in_bits = col_bits + l_bits
        out_bits = _bits_for_range(self.n_rows)

        def packed_fn(x: int) -> int:
            l_mask = (1 << l_bits) - 1
            l_pos = x & l_mask
            col = x >> l_bits
            return self.lookup(col=col, l_pos=l_pos)

        form = CompilableFunctionForm(
            n_input_bits=in_bits,
            n_output_bits=out_bits,
            fn=packed_fn,
            name="col_access_oracle",
        )
        return compile_function_form(form)


@dataclass(frozen=True)
class EntryBinaryOracle:
    """Classical entry oracle O_A: (row, col) -> fixed-point bitstring for A[row, col]."""

    n_rows: int
    n_cols: int
    value_bits: int
    frac_bits: int
    table: dict[tuple[int, int], str]

    @classmethod
    def from_function(
        cls,
        n_rows: int,
        n_cols: int,
        value_bits: int,
        frac_bits: int,
        entry_fn: Callable[[int, int], float],
    ) -> EntryBinaryOracle:
        table: dict[tuple[int, int], str] = {}
        for row in range(n_rows):
            for col in range(n_cols):
                value = float(entry_fn(row, col))
                table[(row, col)] = _encode_signed_fixed(
                    value=value, total_bits=value_bits, frac_bits=frac_bits
                )
        return cls(
            n_rows=n_rows, n_cols=n_cols, value_bits=value_bits, frac_bits=frac_bits, table=table
        )

    @classmethod
    def from_dense(
        cls, matrix: list[list[float]], value_bits: int, frac_bits: int
    ) -> EntryBinaryOracle:
        if not matrix or not matrix[0]:
            raise ValueError("Matrix must be non-empty.")
        n_rows = len(matrix)
        n_cols = len(matrix[0])
        for row in matrix:
            if len(row) != n_cols:
                raise ValueError("Matrix must be rectangular.")
        return cls.from_function(
            n_rows=n_rows,
            n_cols=n_cols,
            value_bits=value_bits,
            frac_bits=frac_bits,
            entry_fn=lambda i, j: matrix[i][j],
        )

    def lookup_bits(self, row: int, col: int) -> str:
        return self.table[(row, col)]

    def lookup_value(self, row: int, col: int) -> float:
        bits = self.lookup_bits(row, col)
        return _decode_signed_fixed(bits, frac_bits=self.frac_bits)

    def compile_truth_table(self) -> dict[str, str]:
        row_bits = _bits_for_range(self.n_rows)
        col_bits = _bits_for_range(self.n_cols)
        compiled: dict[str, str] = {}
        for (row, col), out_bits in self.table.items():
            key = _encode_uint(row, row_bits) + _encode_uint(col, col_bits)
            compiled[key] = out_bits
        return compiled


@dataclass(frozen=True)
class AmplitudeEncoding:
    value: float
    normalized_abs: float
    theta: float
    phase: float


@dataclass(frozen=True)
class FullDataLoadingAmplitudeOracle:
    """Amplitude oracle built from binary entry loading.

    This is the "full data-loading" version: all entries are loaded classically
    as fixed-point bitstrings, then mapped to rotation/phase data.
    """

    entry_oracle: EntryBinaryOracle
    alpha: float

    def encode(self, row: int, col: int) -> AmplitudeEncoding:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        value = self.entry_oracle.lookup_value(row, col)
        normalized_abs = min(abs(value) / self.alpha, 1.0)
        theta = 2.0 * asin(sqrt(normalized_abs))
        phase = 0.0 if value >= 0 else pi
        return AmplitudeEncoding(
            value=value,
            normalized_abs=normalized_abs,
            theta=theta,
            phase=phase,
        )


@dataclass(frozen=True)
class SparseOracleBundle:
    row_oracle: RowAccessOracle
    col_oracle: ColAccessOracle
    amplitude_oracle: FullDataLoadingAmplitudeOracle

    @classmethod
    def from_functions(
        cls,
        n_rows: int,
        n_cols: int,
        max_row_nnz: int,
        max_col_nnz: int,
        row_to_col_fn: Callable[[int, int], int],
        col_to_row_fn: Callable[[int, int], int],
        entry_fn: Callable[[int, int], float],
        value_bits: int,
        frac_bits: int,
        alpha: float,
    ) -> SparseOracleBundle:
        row_oracle = RowAccessOracle.from_function(
            n_rows=n_rows, n_cols=n_cols, max_row_nnz=max_row_nnz, row_to_col_fn=row_to_col_fn
        )
        col_oracle = ColAccessOracle.from_function(
            n_rows=n_rows, n_cols=n_cols, max_col_nnz=max_col_nnz, col_to_row_fn=col_to_row_fn
        )
        entry_oracle = EntryBinaryOracle.from_function(
            n_rows=n_rows,
            n_cols=n_cols,
            value_bits=value_bits,
            frac_bits=frac_bits,
            entry_fn=entry_fn,
        )
        amplitude_oracle = FullDataLoadingAmplitudeOracle(entry_oracle=entry_oracle, alpha=alpha)
        return cls(
            row_oracle=row_oracle, col_oracle=col_oracle, amplitude_oracle=amplitude_oracle
        )


class SparseMatrixBlockEncoding(BlockEncoding):
    """Sparse block-encoding skeleton with separate row/col/entry-amplitude oracles.

    This class returns SDK-neutral operation dictionaries. Backend adapters can
    lower these dictionaries to Qiskit, Qualtran, or estimator IR later.
    """

    def __init__(self, bundle: SparseOracleBundle, name: str = "sparse_full_load") -> None:
        self.bundle = bundle
        self._meta = BlockEncodingMetadata(
            name=name,
            alpha=bundle.amplitude_oracle.alpha,
            ancilla_qubits=1,
            logical_cost_hint={"query_oracles": 3.0},
        )

    def metadata(self) -> BlockEncodingMetadata:
        return self._meta

    def query(self, request: BlockEncodingQuery) -> object:
        row = int(request.parameters["row"])
        l_pos = int(request.parameters["l_pos"])
        col = self.bundle.row_oracle.lookup(row, l_pos)
        amp = self.bundle.amplitude_oracle.encode(row, col)
        return {
            "op": "sparse_block_encoding_query",
            "step": request.step,
            "row": row,
            "l_pos": l_pos,
            "col": col,
            "theta": amp.theta,
            "phase": amp.phase,
            "value": amp.value,
            "normalized_abs": amp.normalized_abs,
        }

    def adjoint_query(self, request: BlockEncodingQuery) -> object:
        payload = self.query(request)
        payload["op"] = "sparse_block_encoding_query_dagger"
        return payload
