from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SynthConfig:
    """Compilation settings for reversible-synthesis routines."""

    max_truth_table_input_bits: int = 12
    allow_callable_enumeration: bool = True


@dataclass(frozen=True)
class LookupTableForm:
    """Explicit truth table mapping x -> f(x) as packed integers."""

    n_input_bits: int
    n_output_bits: int
    table: dict[int, int]
    name: str = "lut"

    def validate(self) -> None:
        n_entries = 1 << self.n_input_bits
        if len(self.table) != n_entries:
            raise ValueError(
                f"LookupTableForm requires full table of size {n_entries}, got {len(self.table)}."
            )
        max_in = 1 << self.n_input_bits
        max_out = 1 << self.n_output_bits
        for in_key, out_val in self.table.items():
            if in_key < 0 or in_key >= max_in:
                raise ValueError(f"Input key {in_key} outside n_input_bits={self.n_input_bits}.")
            if out_val < 0 or out_val >= max_out:
                raise ValueError(
                    f"Output value {out_val} outside n_output_bits={self.n_output_bits}."
                )


@dataclass(frozen=True)
class AffineXorForm:
    """Bit-affine mapping y = A x xor b over GF(2)."""

    n_input_bits: int
    n_output_bits: int
    matrix: tuple[tuple[int, ...], ...]
    offset_bits: tuple[int, ...]
    name: str = "affine_xor"

    def validate(self) -> None:
        if len(self.matrix) != self.n_output_bits:
            raise ValueError("Matrix row count must match n_output_bits.")
        if len(self.offset_bits) != self.n_output_bits:
            raise ValueError("offset_bits length must match n_output_bits.")
        for row in self.matrix:
            if len(row) != self.n_input_bits:
                raise ValueError("Each matrix row length must match n_input_bits.")
            if any(bit not in (0, 1) for bit in row):
                raise ValueError("Affine matrix entries must be 0/1.")
        if any(bit not in (0, 1) for bit in self.offset_bits):
            raise ValueError("offset_bits entries must be 0/1.")

    def evaluate(self, x: int) -> int:
        max_x = 1 << self.n_input_bits
        if x < 0 or x >= max_x:
            raise ValueError(f"x={x} outside n_input_bits={self.n_input_bits}.")
        out = 0
        for out_bit, row in enumerate(self.matrix):
            bit_val = self.offset_bits[out_bit]
            for in_bit, coeff in enumerate(row):
                if coeff == 1:
                    bit_val ^= (x >> in_bit) & 1
            if bit_val:
                out |= 1 << out_bit
        return out

    def to_lookup_table(self) -> LookupTableForm:
        table = {x: self.evaluate(x) for x in range(1 << self.n_input_bits)}
        return LookupTableForm(
            n_input_bits=self.n_input_bits,
            n_output_bits=self.n_output_bits,
            table=table,
            name=f"{self.name}_as_lut",
        )


@dataclass(frozen=True)
class CompilableFunctionForm:
    """Callable mapping that can be auto-enumerated into a lookup table."""

    n_input_bits: int
    n_output_bits: int
    fn: Callable[[int], int]
    name: str = "callable_fn"

    def to_lookup_table(self, config: SynthConfig) -> LookupTableForm:
        if not config.allow_callable_enumeration:
            raise ValueError("Callable enumeration disabled in SynthConfig.")
        if self.n_input_bits > config.max_truth_table_input_bits:
            raise ValueError(
                f"n_input_bits={self.n_input_bits} exceeds configured maximum "
                f"{config.max_truth_table_input_bits}."
            )
        max_out = 1 << self.n_output_bits
        table: dict[int, int] = {}
        for x in range(1 << self.n_input_bits):
            y = int(self.fn(x))
            if y < 0 or y >= max_out:
                raise ValueError(
                    f"Callable produced y={y}, outside n_output_bits={self.n_output_bits}."
                )
            table[x] = y
        return LookupTableForm(
            n_input_bits=self.n_input_bits,
            n_output_bits=self.n_output_bits,
            table=table,
            name=f"{self.name}_enumerated",
        )
