from __future__ import annotations

from dataclasses import dataclass, field

from .function_ir import AffineXorForm, CompilableFunctionForm, LookupTableForm, SynthConfig


@dataclass(frozen=True)
class ReversibleOp:
    gate: str
    controls: tuple[int, ...] = ()
    control_values: tuple[int, ...] = ()
    target: int = 0


@dataclass(frozen=True)
class GateCost:
    x_count: int = 0
    cnot_count: int = 0
    toffoli_count: int = 0
    t_count: int = 0
    t_depth_estimate: int = 0
    ancilla_peak_estimate: int = 0

    def __add__(self, other: GateCost) -> GateCost:
        return GateCost(
            x_count=self.x_count + other.x_count,
            cnot_count=self.cnot_count + other.cnot_count,
            toffoli_count=self.toffoli_count + other.toffoli_count,
            t_count=self.t_count + other.t_count,
            t_depth_estimate=self.t_depth_estimate + other.t_depth_estimate,
            ancilla_peak_estimate=max(self.ancilla_peak_estimate, other.ancilla_peak_estimate),
        )


@dataclass
class ReversibleCircuit:
    """Reversible map implementing |x>|0> -> |x>|f(x)| over packed bit registers."""

    n_input_bits: int
    n_output_bits: int
    operations: list[ReversibleOp] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def append(self, op: ReversibleOp) -> None:
        self.operations.append(op)

    @property
    def n_qubits(self) -> int:
        return self.n_input_bits + self.n_output_bits

    @property
    def output_offset(self) -> int:
        return self.n_input_bits

    def estimate_cost(self) -> GateCost:
        total = GateCost()
        for op in self.operations:
            if op.gate == "x":
                total = total + GateCost(x_count=1)
            elif op.gate == "cx":
                total = total + GateCost(cnot_count=1)
            elif op.gate == "mcx":
                k = len(op.controls)
                toffoli = _estimate_mcx_toffoli(k)
                total = total + GateCost(
                    toffoli_count=toffoli,
                    t_count=7 * toffoli,
                    t_depth_estimate=max(1, 3 * toffoli),
                    ancilla_peak_estimate=max(0, k - 2),
                )
            else:
                raise ValueError(f"Unsupported gate type in estimator: {op.gate}")
        return total


def _estimate_mcx_toffoli(n_controls: int) -> int:
    if n_controls <= 1:
        return 0
    if n_controls == 2:
        return 1
    return 2 * n_controls - 3


def compile_lookup_table(form: LookupTableForm) -> ReversibleCircuit:
    """Compile a full lookup table into a baseline reversible implementation.

    Strategy:
    - Output register starts in |0...0>.
    - For each output bit and each minterm with bit=1, apply an MCX controlled on
      all input bits (with negative controls handled by X conjugation).
    """

    form.validate()
    circ = ReversibleCircuit(n_input_bits=form.n_input_bits, n_output_bits=form.n_output_bits)
    n_inputs = form.n_input_bits
    output_offset = circ.output_offset

    for out_bit in range(form.n_output_bits):
        target = output_offset + out_bit
        minterms = [x for x, y in form.table.items() if ((y >> out_bit) & 1) == 1]
        for x in minterms:
            zero_control_wires: list[int] = []
            controls: list[int] = []
            control_values: list[int] = []
            for in_bit in range(n_inputs):
                bit_val = (x >> in_bit) & 1
                controls.append(in_bit)
                control_values.append(1)
                if bit_val == 0:
                    zero_control_wires.append(in_bit)

            for wire in zero_control_wires:
                circ.append(ReversibleOp(gate="x", target=wire))
            if len(controls) == 0:
                circ.append(ReversibleOp(gate="x", target=target))
            elif len(controls) == 1:
                circ.append(
                    ReversibleOp(
                        gate="cx",
                        controls=(controls[0],),
                        control_values=(1,),
                        target=target,
                    )
                )
            else:
                circ.append(
                    ReversibleOp(
                        gate="mcx",
                        controls=tuple(controls),
                        control_values=tuple(control_values),
                        target=target,
                    )
                )
            for wire in reversed(zero_control_wires):
                circ.append(ReversibleOp(gate="x", target=wire))

    circ.metadata["source"] = form.name
    circ.metadata["method"] = "sum_of_minterms"
    return circ


def compile_affine_xor(form: AffineXorForm) -> ReversibleCircuit:
    """Compile affine XOR maps to CNOT/X network without T gates."""

    form.validate()
    circ = ReversibleCircuit(n_input_bits=form.n_input_bits, n_output_bits=form.n_output_bits)
    output_offset = circ.output_offset

    for out_bit, row in enumerate(form.matrix):
        target = output_offset + out_bit
        if form.offset_bits[out_bit] == 1:
            circ.append(ReversibleOp(gate="x", target=target))
        for in_bit, coeff in enumerate(row):
            if coeff == 1:
                circ.append(
                    ReversibleOp(
                        gate="cx",
                        controls=(in_bit,),
                        control_values=(1,),
                        target=target,
                    )
                )
    circ.metadata["source"] = form.name
    circ.metadata["method"] = "affine_xor"
    return circ


def compile_function_form(
    form: LookupTableForm | AffineXorForm | CompilableFunctionForm,
    config: SynthConfig | None = None,
) -> ReversibleCircuit:
    cfg = config or SynthConfig()
    if isinstance(form, AffineXorForm):
        return compile_affine_xor(form)
    if isinstance(form, LookupTableForm):
        return compile_lookup_table(form)
    if isinstance(form, CompilableFunctionForm):
        lut = form.to_lookup_table(cfg)
        return compile_lookup_table(lut)
    raise TypeError(f"Unsupported form type: {type(form)}")
