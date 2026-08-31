"""Tests for the symmetry-adaptation (Clebsch--Gordan) transform bloq.

Pins the *resource model* of ``integrations.qualtran.symmetry_adaptation_QROAM``:

  * The bloq builds and ``QECGatesCost`` / ``QubitCount`` evaluate for the diamond 2x2x2 and 6x6x6
    meshes.
  * The **hard requirement**: the dominant Toffoli term (the orbit-unitary synthesis (A)) does NOT
    scale with ``n_k``.  Between 2x2x2 and 6x6x6, ``n_k`` grows 27x, ``Sum|o|^2`` grows ~1.9x, yet the
    dominant Toffoli grows well under 3x -- while a naive per-``k`` dense model would grow 27x.
  * Qubit counts are finite and reasonable.
"""

from __future__ import annotations

import sys

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for envs without pytest

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

qualtran = pytest.importorskip("qualtran")

from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting.generalizers import generalize_cswap_approx

from integrations.qualtran.block_unitary_interferometer_QROAM import (
    BlockUnitaryInterferometerSynthesisQROAM,
)
from integrations.qualtran.symmetry_adaptation_QROAM import (
    SymmetryAdaptationQROAM,
    bloq_for_mesh,
    estimate_symmetry_adaptation_resources,
    group_orbit_sizes,
    load_material,
    optimal_interferometer_log_block_sizes_by_toffoli,
    orbit_one_qubit_gates,
)


def _cost(bloq):
    return get_cost_value(bloq, QECGatesCost(), generalizer=generalize_cswap_approx)


def _toffoli(bloq):
    return int(_cost(bloq).total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"])


# ---------------------------------------------------------------------------
# construction + cost evaluation
# ---------------------------------------------------------------------------


def test_bloq_builds_and_costs_for_both_meshes():
    for mesh in ("2x2x2", "6x6x6"):
        bloq = bloq_for_mesh(mesh)
        cost = _cost(bloq)  # QECGatesCost must evaluate
        assert int(cost.total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"]) > 0
        q = int(get_cost_value(bloq, QubitCount()))
        assert 0 < q < 10_000, q  # finite and reasonable


def test_signature_layout():
    bloq = bloq_for_mesh("6x6x6")
    names = [r.name for r in bloq.signature]
    assert names == ["k", "I", "i", "a", "c", "phase_gradient"]
    # k register width is the only thing that grows with n_k.
    assert bloq.k_bitsize == 8  # ceil(log2(216))
    assert bloq.I_bitsize == 8  # ceil(log2(136))


# ---------------------------------------------------------------------------
# the hard requirement: dominant cost is n_k-independent
# ---------------------------------------------------------------------------


def test_dominant_toffoli_is_nk_independent():
    data = load_material()
    e0 = bloq_for_mesh("2x2x2", data).resource_estimate()
    e1 = bloq_for_mesh("6x6x6", data).resource_estimate()

    # n_k grows 27x between the two meshes.
    nk_ratio = e1.n_k / e0.n_k
    assert abs(nk_ratio - 27.0) < 0.5, nk_ratio

    # Sum|o|^2 (the unique data) grows ~1.9x.
    data_ratio = e1.unique_data_sum_o2 / e0.unique_data_sum_o2
    assert 1.5 < data_ratio < 2.5, data_ratio

    # DOMINANT term (orbit synthesis) grows far less than n_k -- bounded by the Sum|o|^2 regime.
    dom_ratio = e1.orbit_synthesis_toffoli / e0.orbit_synthesis_toffoli
    assert dom_ratio < 3.0, dom_ratio  # << 27x
    assert dom_ratio < 0.3 * nk_ratio, (dom_ratio, nk_ratio)

    # orbit synthesis is genuinely the dominant Toffoli contribution.
    assert e0.orbit_synthesis_toffoli > 0.8 * e0.toffoli
    assert e1.orbit_synthesis_toffoli > 0.8 * e1.toffoli

    # total Toffoli also stays flat; a naive per-k dense model would grow 27x.
    total_ratio = e1.toffoli / e0.toffoli
    assert total_ratio < 3.0, total_ratio
    naive_ratio = (e1.n_k * e1.n_IP**2) / (e0.n_k * e0.n_IP**2)
    assert naive_ratio > 20.0  # the model it avoids
    assert total_ratio < 0.2 * naive_ratio


def test_only_k_touching_widths_grow():
    """The only registers/quantities that differ with n_k are the log2(n_k)-wide k register and the
    k.L / k-map arithmetic widths -- never a table dimension."""
    b0 = bloq_for_mesh("2x2x2")
    b1 = bloq_for_mesh("6x6x6")
    # QROAM table dimensions are n_k-independent.
    assert b0.perm_qroam.data_shape == b1.perm_qroam.data_shape
    assert b0.geom_qroam.data_shape == b1.geom_qroam.data_shape
    # k register width grows only logarithmically.
    assert b1.k_bitsize > b0.k_bitsize
    assert b1.k_bitsize <= b0.k_bitsize + 6


# ---------------------------------------------------------------------------
# grouping / data proxies
# ---------------------------------------------------------------------------


def test_orbit_grouping_by_size():
    data = load_material()
    m = data["6x6x6"]
    sizes = tuple(o for fam in m["families"] for o in fam["orbit_sizes"])
    groups = group_orbit_sizes(sizes)
    # every orbit is accounted for exactly once.
    assert sum(groups.values()) == len(sizes) == m["N_orbit_blocks"]
    # rounded sizes are powers of two >= 2.
    for N in groups:
        assert N >= 2 and (N & (N - 1)) == 0
    # the one-qubit-gate proxy tracks Sum|o|^2, not n_k.
    assert orbit_one_qubit_gates(sizes) > 0


def test_estimate_matches_bloq_total():
    """The closed-form estimate's total Toffoli equals the bloq's aggregated QECGatesCost."""
    for mesh in ("2x2x2", "6x6x6"):
        bloq = bloq_for_mesh(mesh)
        est = bloq.resource_estimate()
        assert est.toffoli == _toffoli(bloq), mesh


# ---------------------------------------------------------------------------
# interferometer backend + select-swap Toffoli minimization
# ---------------------------------------------------------------------------


def test_stage_A_uses_interferometer_backend():
    bloq = bloq_for_mesh("6x6x6")
    subs = [sub for sub, _ in bloq.orbit_synthesizers()]
    assert subs, "expected at least one orbit synthesizer"
    for sub in subs:
        assert isinstance(sub, BlockUnitaryInterferometerSynthesisQROAM)
        # one block-diagonal synthesis per distinct rounded orbit size N = 2^ceil(log2 d) (>= 4).
        assert sub.n_rows in (4, 8, 16, 32, 64)


def test_select_swap_lowers_toffoli():
    """The Toffoli-minimizing QROAM split is never worse than the default (0,0), and strictly better
    on the dominant size group -- proving the select-swap tradeoff was actually exploited."""
    plan = bloq_for_mesh("6x6x6").orbit_synthesis_plan()
    assert plan
    # never worse than default for any group.
    for row in plan:
        assert row["toffoli"] <= row["toffoli_default"], row
    # the dominant group (largest default Toffoli) is strictly improved.
    dominant = max(plan, key=lambda r: r["toffoli_default"])
    assert dominant["toffoli"] < dominant["toffoli_default"], dominant
    # overall stage (A) strictly cheaper with the tradeoff on.
    tot_opt = sum(r["toffoli"] for r in plan)
    tot_def = sum(r["toffoli_default"] for r in plan)
    assert tot_opt < tot_def


def test_minimize_toffoli_flag_is_default_and_helps():
    """The bloq defaults to the Toffoli-minimizing tradeoff; turning it off never lowers cost."""
    data = load_material()
    opt = bloq_for_mesh("6x6x6", data)  # minimize_toffoli=True by default
    assert opt.minimize_toffoli is True
    default_split = bloq_for_mesh("6x6x6", data, minimize_toffoli=False)
    assert _toffoli(opt) <= _toffoli(default_split)


def test_argmin_helper_returns_valid_split():
    lbs = optimal_interferometer_log_block_sizes_by_toffoli(648, 4, 24)
    assert isinstance(lbs, tuple) and len(lbs) == 2
    assert all(int(x) >= 0 for x in lbs)
