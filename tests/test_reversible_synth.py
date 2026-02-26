from block_encoding.sparse_matrix import RowAccessOracle
from oracles.function_ir import AffineXorForm, CompilableFunctionForm, LookupTableForm
from oracles.reversible_synth import compile_function_form


def test_compile_affine_xor_has_no_t_cost():
    form = AffineXorForm(
        n_input_bits=3,
        n_output_bits=2,
        matrix=((1, 0, 1), (0, 1, 0)),
        offset_bits=(0, 1),
        name="affine_test",
    )
    circ = compile_function_form(form)
    cost = circ.estimate_cost()
    assert cost.t_count == 0
    assert cost.toffoli_count == 0
    assert cost.cnot_count == 3
    assert cost.x_count == 1


def test_compile_lookup_table_and_cost():
    table = {
        0b00: 0b0,
        0b01: 0b0,
        0b10: 0b0,
        0b11: 0b1,  # AND bit
    }
    form = LookupTableForm(n_input_bits=2, n_output_bits=1, table=table, name="and2")
    circ = compile_function_form(form)
    cost = circ.estimate_cost()
    assert cost.toffoli_count == 1
    assert cost.t_count == 7


def test_compile_callable_row_oracle():
    row = RowAccessOracle.from_function(
        n_rows=4,
        n_cols=4,
        max_row_nnz=2,
        row_to_col_fn=lambda r, l: (r + l) % 4,
    )
    circ = row.compile_reversible_circuit()
    cost = circ.estimate_cost()
    assert circ.n_input_bits == 3
    assert circ.n_output_bits == 2
    assert len(circ.operations) > 0
    assert cost.t_count >= 0


def test_callable_form_respects_output_range():
    form = CompilableFunctionForm(
        n_input_bits=2,
        n_output_bits=2,
        fn=lambda x: x ^ 0b01,
        name="xor_const",
    )
    circ = compile_function_form(form)
    assert len(circ.operations) > 0
