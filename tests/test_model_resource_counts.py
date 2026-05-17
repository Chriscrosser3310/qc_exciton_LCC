"""Tests for the analytic interferometer resource model.

These tests cover the pure-Python helpers in
``integrations.qualtran.model_resource_counts`` and check that the
shape-only Bloq estimator
``estimate_interferometer_resources`` agrees with the closed-form
``block_unitary_interferometer_count`` for matched parameters.
"""

from __future__ import annotations

import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without pytest

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__}, got nothing")
            return issubclass(exc_type, self.exc)

    class _PytestShim:
        @staticmethod
        def raises(exc):
            return _Raises(exc)

        @staticmethod
        def importorskip(name):
            return __import__(name)

    pytest = _PytestShim()  # type: ignore[assignment]
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

from integrations.qualtran.model_resource_counts import (
    SYNTHESIS_PANEL_N_BLOCKS,
    SYNTHESIS_PER_REFLECTION_INTERCEPT,
    SYNTHESIS_WORKSPACE_QUBITS,
    SynthesisResourceCount,
    _synthesis_panel_records,
    assert_power_of_two,
    block_unitary_interferometer_count,
    block_unitary_interferometer_qubits,
    block_unitary_interferometer_toffoli,
    block_unitary_synthesis_count,
    block_unitary_synthesis_signature_qubits,
    block_unitary_synthesis_toffoli,
    block_unitary_synthesis_workspace_qubits,
    ceil_log2,
    optimize_block_unitary_interferometer,
)


def test_ceil_log2_small_values():
    assert ceil_log2(1) == 0
    assert ceil_log2(2) == 1
    assert ceil_log2(3) == 2
    assert ceil_log2(4) == 2
    assert ceil_log2(5) == 3
    assert ceil_log2(8) == 3
    assert ceil_log2(9) == 4


def test_ceil_log2_rejects_non_positive():
    with pytest.raises(ValueError):
        ceil_log2(0)
    with pytest.raises(ValueError):
        ceil_log2(-1)


def test_assert_power_of_two():
    for x in (1, 2, 4, 8, 16, 1024):
        assert_power_of_two(x, "x")  # should not raise
    for bad in (0, 3, 5, 6, 7, 9, 12, -2):
        with pytest.raises(ValueError):
            assert_power_of_two(bad, "x")


def test_block_unitary_interferometer_toffoli_minimal_case():
    # K=1, N=2 (n=1), b=2, lambda_1=lambda_2=1:
    #   layer_cost = ceil(2/2) + 2*1*2 - 5 = 0
    #   shift_cost = max(0, -1)*(N-1) = 0
    #   final_cost = ceil(2/1) + 1*2 + ceil(2/1) + 1 - 6 = 1
    #   total      = N*layer + shift + final = 2*0 + 0 + 1 = 1
    assert block_unitary_interferometer_toffoli(1, 2, 2, 1, 1) == 1


def test_block_unitary_interferometer_toffoli_two_blocks():
    # K=2, N=2, b=2, lambda_1=lambda_2=1:
    #   layer_cost = ceil(4/2) + 4 - 5 = 1
    #   final_cost = 4 + 2 + 4 + 1 - 6 = 5
    #   total      = 2*1 + 0 + 5 = 7
    assert block_unitary_interferometer_toffoli(2, 2, 2, 1, 1) == 7


def test_block_unitary_interferometer_toffoli_n4():
    # K=1, N=4 (n=2), b=2, lambda_1=lambda_2=1:
    #   layer_cost = ceil(4/2) + 4 - 5 = 1
    #   shift_cost = max(0, 0)*(N-1) = 0
    #   final_cost = 4 + 2 + 4 + 1 - 6 = 5
    #   total      = 4*1 + 0 + 5 = 9
    assert block_unitary_interferometer_toffoli(1, 4, 2, 1, 1) == 9


def test_block_unitary_interferometer_qubits_minimal_case():
    # base = ceil_log2(1) + log2(2) + 2 = 0 + 1 + 2 = 3
    # workspace = max(2*2*1, 2*1, 1) = 4
    # total = 7
    assert block_unitary_interferometer_qubits(1, 2, 2, 1, 1) == 7


def test_block_unitary_interferometer_qubits_two_blocks():
    # base = ceil_log2(2)+1+2 = 4
    # workspace = max(4, 2, 1) = 4
    assert block_unitary_interferometer_qubits(2, 2, 2, 1, 1) == 8


def test_block_unitary_interferometer_count_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        block_unitary_interferometer_count(1, 3, 2, 1, 1)  # block_dim
    with pytest.raises(ValueError):
        block_unitary_interferometer_count(1, 4, 2, 3, 1)  # lambda_1
    with pytest.raises(ValueError):
        block_unitary_interferometer_count(1, 4, 2, 1, 5)  # lambda_2


def test_block_unitary_interferometer_count_returns_consistent_fields():
    c = block_unitary_interferometer_count(8, 256, 32, 16, 16)
    assert c.lambda_1 == 16
    assert c.lambda_2 == 16
    assert c.log_lambda_1 == 4
    assert c.log_lambda_2 == 4
    assert c.toffoli == block_unitary_interferometer_toffoli(8, 256, 32, 16, 16)
    assert c.qubits == block_unitary_interferometer_qubits(8, 256, 32, 16, 16)


def test_optimize_objective_orderings():
    t_opt = optimize_block_unitary_interferometer(8, 256, 32, objective="toffoli")
    q_opt = optimize_block_unitary_interferometer(8, 256, 32, objective="qubits")
    # toffoli-opt must not be beaten on toffoli, qubit-opt must not be beaten on qubits
    assert t_opt.toffoli <= q_opt.toffoli
    assert q_opt.qubits <= t_opt.qubits


def test_optimize_rejects_unknown_objective():
    with pytest.raises(ValueError):
        optimize_block_unitary_interferometer(1, 4, 2, objective="bogus")


def test_optimize_chooses_powers_of_two():
    res = optimize_block_unitary_interferometer(4, 16, 8, objective="toffoli")
    assert res.lambda_1 == 1 << res.log_lambda_1
    assert res.lambda_2 == 1 << res.log_lambda_2


def test_estimator_matches_closed_form():
    """The shape-only estimator and the closed-form count must agree."""
    qualtran = pytest.importorskip("qualtran")
    _ = qualtran  # silence linter
    from integrations.qualtran.block_unitary_interferometer_QROAM import (
        estimate_interferometer_resources,
    )

    # In the closed-form model lambda_1 = layer-load lambda = final-load lambda,
    # and lambda_2 = final-erasure lambda. So we only compare with matched
    # layer/final loads (l_layer == l_final) — the estimator splits them.
    cases = [
        (1, 32, 8, 0, 0),
        (1, 32, 8, 3, 3),
        (8, 256, 32, 4, 5),
        (27, 256, 32, 5, 5),
    ]
    for n_blocks, n_rows, b, l_load, l_final_adj in cases:
        est = estimate_interferometer_resources(
            n_blocks,
            n_rows,
            b,
            layer_log_block_size=l_load,
            final_log_block_size=l_load,
            final_adjoint_log_block_size=l_final_adj,
        )
        ref = block_unitary_interferometer_count(
            n_blocks, n_rows, b, 1 << l_load, 1 << l_final_adj
        )
        assert est.toffoli == ref.toffoli, (n_blocks, n_rows, b, l_load, l_final_adj)
        assert est.qubits == ref.qubits, (n_blocks, n_rows, b, l_load, l_final_adj)


def test_block_unitary_synthesis_toffoli_validates_inputs():
    with pytest.raises(ValueError):
        block_unitary_synthesis_toffoli(1, 3, 4, 1)  # n_rows not power of two
    with pytest.raises(ValueError):
        block_unitary_synthesis_toffoli(3, 4, 4, 1)  # n_blocks not power of two
    with pytest.raises(ValueError):
        block_unitary_synthesis_toffoli(1, 4, 0, 1)  # bitsize must be positive
    with pytest.raises(ValueError):
        block_unitary_synthesis_toffoli(1, 4, 4, 0)  # n_reflections must be positive
    with pytest.raises(ValueError):
        block_unitary_synthesis_toffoli(1, 4, 4, 5)  # n_reflections > n_rows
    with pytest.raises(KeyError):
        block_unitary_synthesis_toffoli(128, 4, 4, 1)  # missing intercept entry


def test_block_unitary_synthesis_toffoli_decomposition_identity():
    """Analytic count must match ``K * (slope*b + I_1)`` exactly."""
    import math as _math

    for (n_blocks, N), I_1 in SYNTHESIS_PER_REFLECTION_INTERCEPT.items():
        slope = 2 * (int(_math.log2(N)) + 1)
        for b in (2, 4, 8, 12):
            for K in (1, max(1, N // 2), N):
                assert block_unitary_synthesis_toffoli(
                    n_blocks, N, b, K
                ) == K * (slope * b + I_1)


def test_block_unitary_synthesis_toffoli_matches_bloq():
    """Analytic estimator and the Bloq's QECGatesCost agree exactly on the grid."""
    qualtran = pytest.importorskip("qualtran")
    _ = qualtran
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from integrations.qualtran.block_unitary_synthesis_QROAM import (
        BlockUnitarySynthesisQROAM,
    )

    for (n_blocks, N) in SYNTHESIS_PER_REFLECTION_INTERCEPT:
        for b in (2, 4, 8):
            for K in (1, max(1, N // 2), N):
                bloq = BlockUnitarySynthesisQROAM.from_shape(
                    n_blocks=n_blocks, n_rows=N, phase_bitsize=b, n_reflections=K
                )
                bloq_t = get_cost_value(bloq, QECGatesCost()).toffoli
                analytic_t = block_unitary_synthesis_toffoli(n_blocks, N, b, K)
                assert bloq_t == analytic_t, (n_blocks, N, b, K, bloq_t, analytic_t)


def test_block_unitary_synthesis_signature_qubits_validates_inputs():
    with pytest.raises(ValueError):
        block_unitary_synthesis_signature_qubits(1, 3, 4)  # n_rows not power of two
    with pytest.raises(ValueError):
        block_unitary_synthesis_signature_qubits(0, 4, 4)  # n_blocks must be positive
    with pytest.raises(ValueError):
        block_unitary_synthesis_signature_qubits(1, 4, 0)  # bitsize must be positive


def test_block_unitary_synthesis_signature_qubits_formula():
    # ceil_log2(1) + 1 + log2(4) + 2 = 0 + 1 + 2 + 2 = 5
    assert block_unitary_synthesis_signature_qubits(1, 4, 2) == 5
    # ceil_log2(64) + 1 + log2(256) + 32 = 6 + 1 + 8 + 32 = 47
    assert block_unitary_synthesis_signature_qubits(64, 256, 32) == 47
    # n_blocks=3 (not a power of two) is allowed; ceil_log2(3) = 2
    assert block_unitary_synthesis_signature_qubits(3, 8, 4) == 2 + 1 + 3 + 4


def test_block_unitary_synthesis_signature_qubits_matches_bloq():
    """Analytic signature qubit count must equal ``bloq.signature.n_qubits()``."""
    qualtran = pytest.importorskip("qualtran")
    _ = qualtran
    from integrations.qualtran.block_unitary_synthesis_QROAM import (
        BlockUnitarySynthesisQROAM,
    )

    for (n_blocks, N) in SYNTHESIS_PER_REFLECTION_INTERCEPT:
        for b in (2, 8, 32):
            bloq = BlockUnitarySynthesisQROAM.from_shape(
                n_blocks=n_blocks, n_rows=N, phase_bitsize=b, n_reflections=1
            )
            assert (
                block_unitary_synthesis_signature_qubits(n_blocks, N, b)
                == bloq.signature.n_qubits()
            ), (n_blocks, N, b)


def test_block_unitary_synthesis_count_composes_existing_estimators():
    """``block_unitary_synthesis_count`` must compose the three analytic helpers."""
    for n_blocks, n_rows, b, K in [
        (1, 4, 2, 1),
        (8, 16, 4, 4),
        (64, 256, 32, 128),
    ]:
        rec = block_unitary_synthesis_count(n_blocks, n_rows, b, K)
        assert isinstance(rec, SynthesisResourceCount)
        assert rec.toffoli == block_unitary_synthesis_toffoli(n_blocks, n_rows, b, K)
        assert rec.signature_qubits == block_unitary_synthesis_signature_qubits(
            n_blocks, n_rows, b
        )
        assert rec.workspace_qubits == block_unitary_synthesis_workspace_qubits(
            n_blocks, n_rows, b
        )
        assert rec.total_qubits == rec.signature_qubits + rec.workspace_qubits
        assert (rec.n_blocks, rec.n_rows, rec.bitsize, rec.n_reflections) == (
            n_blocks,
            n_rows,
            b,
            K,
        )


def test_block_unitary_synthesis_count_propagates_validation():
    """The wrapper must inherit input validation from its underlying helpers."""
    with pytest.raises(ValueError):
        block_unitary_synthesis_count(1, 3, 4, 1)  # n_rows not power of two
    with pytest.raises(ValueError):
        block_unitary_synthesis_count(1, 4, 0, 1)  # bitsize must be positive
    with pytest.raises(KeyError):
        block_unitary_synthesis_count(128, 4, 4, 1)  # off-grid intercept


def test_block_unitary_synthesis_workspace_qubits_validates_inputs():
    """``block_unitary_synthesis_workspace_qubits`` rejects bad inputs and off-grid points."""
    with pytest.raises(ValueError):
        block_unitary_synthesis_workspace_qubits(1, 3, 4)  # n_rows not power of two
    with pytest.raises(ValueError):
        block_unitary_synthesis_workspace_qubits(3, 4, 4)  # n_blocks not power of two
    with pytest.raises(ValueError):
        block_unitary_synthesis_workspace_qubits(1, 4, 0)  # bitsize must be positive
    with pytest.raises(KeyError):
        block_unitary_synthesis_workspace_qubits(1, 4, 7)  # bitsize off-grid
    with pytest.raises(KeyError):
        block_unitary_synthesis_workspace_qubits(128, 4, 4)  # n_blocks off-grid


def test_block_unitary_synthesis_workspace_table_consistent():
    """Pure-Python invariants on ``SYNTHESIS_WORKSPACE_QUBITS``.

    The table is parameterized over the same ``(n_blocks, n_rows)`` 49-point
    grid as ``SYNTHESIS_PER_REFLECTION_INTERCEPT`` and ``bitsize`` in
    ``{2, 4, 8, 16, 32}``; every cell must be present and positive.
    """
    bitsizes = (2, 4, 8, 16, 32)
    for (nb, N) in SYNTHESIS_PER_REFLECTION_INTERCEPT.keys():
        for b in bitsizes:
            assert (nb, N, b) in SYNTHESIS_WORKSPACE_QUBITS, (nb, N, b)
            assert SYNTHESIS_WORKSPACE_QUBITS[(nb, N, b)] > 0


def test_block_unitary_synthesis_count_total_qubits_matches_bloq():
    """``total_qubits = signature + workspace`` must equal the Bloq's ``QubitCount`` exactly.

    Cross-checks across the same 49-point grid x bitsizes the table covers.
    This is the qubit-side analog of
    ``test_block_unitary_synthesis_toffoli_matches_bloq``.
    """
    qualtran = pytest.importorskip("qualtran")
    _ = qualtran
    from qualtran.resource_counting import QubitCount, get_cost_value
    from integrations.qualtran.block_unitary_synthesis_QROAM import (
        BlockUnitarySynthesisQROAM,
    )

    for (nb, N) in SYNTHESIS_PER_REFLECTION_INTERCEPT.keys():
        for b in (2, 4, 8, 16, 32):
            rec = block_unitary_synthesis_count(nb, N, b, 1)
            bloq = BlockUnitarySynthesisQROAM.from_shape(
                n_blocks=nb, n_rows=N, phase_bitsize=b, n_reflections=1
            )
            qc = int(get_cost_value(bloq, QubitCount()))
            assert rec.total_qubits == qc, (nb, N, b, rec.total_qubits, qc)


def test_synthesis_panel_records_match_block_unitary_synthesis_count():
    """The panel helper must be a thin sweep over ``block_unitary_synthesis_count``."""
    recs = _synthesis_panel_records(n_rows=256, bitsize=32, n_reflections=256)
    assert [r.n_blocks for r in recs] == list(SYNTHESIS_PANEL_N_BLOCKS)
    for r in recs:
        ref = block_unitary_synthesis_count(r.n_blocks, 256, 32, 256)
        assert r == ref


def test_synthesis_panel_records_use_tabulated_grid():
    """Default panel ``n_blocks`` values are all present in the workspace table."""
    for nb in SYNTHESIS_PANEL_N_BLOCKS:
        for n_rows in (4, 8, 16, 32, 64, 128, 256):
            assert (nb, n_rows, 32) in SYNTHESIS_WORKSPACE_QUBITS
            assert (nb, n_rows) in SYNTHESIS_PER_REFLECTION_INTERCEPT


def test_plot_report_generates_pdf(tmp_path=None):
    """Smoke test: ``_plot_report`` writes a non-empty PDF with the new synthesis pages."""
    pytest.importorskip("matplotlib")
    import os as _os
    import tempfile

    from integrations.qualtran.model_resource_counts import _plot_report

    out_dir = tempfile.mkdtemp(prefix="model_resource_counts_test_") if tmp_path is None else str(tmp_path)
    out_pdf = _os.path.join(out_dir, "report.pdf")
    t_opt, q_opt = _plot_report(
        block_dim=256,
        bitsize=32,
        k_values=range(1, 3),
        out_pdf=out_pdf,
    )
    assert _os.path.exists(out_pdf)
    assert _os.path.getsize(out_pdf) > 0
    assert len(t_opt) == 2 and len(q_opt) == 2


def test_optimal_log_block_sizes_returns_non_negative():
    qualtran = pytest.importorskip("qualtran")
    _ = qualtran
    from integrations.qualtran.block_unitary_interferometer_QROAM import (
        optimal_interferometer_log_block_sizes,
    )

    for n_blocks in (1, 8, 27):
        l_layer, l_final = optimal_interferometer_log_block_sizes(n_blocks, 256, 32)
        assert l_layer >= 0
        assert l_final >= 0
        assert l_layer == int(l_layer)
        assert l_final == int(l_final)


if __name__ == "__main__":
    # Allow running without pytest installed: invoke every test_* function.
    failed = 0
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
