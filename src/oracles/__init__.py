"""Function-form IR and reversible synthesis utilities."""

from .function_ir import (
    AffineXorForm,
    CompilableFunctionForm,
    LookupTableForm,
    SynthConfig,
)
from .reversible_synth import (
    GateCost,
    ReversibleCircuit,
    ReversibleOp,
    compile_function_form,
    compile_lookup_table,
)

__all__ = [
    "AffineXorForm",
    "CompilableFunctionForm",
    "LookupTableForm",
    "SynthConfig",
    "GateCost",
    "ReversibleCircuit",
    "ReversibleOp",
    "compile_lookup_table",
    "compile_function_form",
]
