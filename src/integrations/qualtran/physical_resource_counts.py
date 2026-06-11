# Physical resource estimation following the style of
# https://qualtran.readthedocs.io/en/latest/surface_code/thc_compilation.html

import sympy

from qualtran.resource_counting import (
    get_cost_value,
    QubitCount,
    QECGatesCost,
    GateCounts,
)
from qualtran.surface_code import (
    CCZ2TFactory,
    MultiFactory,
    SimpleDataBlock,
    LogicalErrorModel,
    QECScheme,
)
from qualtran.surface_code.gidney_fowler_model import (
    get_ccz2t_costs,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
)


def to_t_and_clifford(gate_counts: GateCounts, rotation_eps: float = 1e-11) -> GateCounts:
    """Reduce a mixed GateCounts to a (T, Clifford)-only GateCounts.

    Uses Qualtran's legacy T-complexity decomposition:
      - Each Toffoli/AND/CSWAP is unrolled into its T and Clifford cost.
      - Each rotation is synthesized via Ross-Selinger with target eps,
        adding ~1.15 * log2(1/eps) T gates per rotation.
      - Measurements are dropped (they are free in Clifford+T accounting).

    The returned GateCounts has only `t` and `clifford` populated, so it can
    still be fed to `get_ccz2t_costs` / `get_ccz2t_costs_from_grid_search`.
    """
    tc = gate_counts.to_legacy_t_complexity()
    return GateCounts(t=tc.t_incl_rotations(eps=rotation_eps), clifford=tc.clifford)


def _extract_logical_costs(bloq, substitutions=None, rotation_eps: float = 1e-11):
    """Pull (n_algo_qubits, GateCounts) out of a Bloq, reduced to T+Clifford."""
    n_algo_qubits = get_cost_value(bloq, QubitCount())
    gate_counts = get_cost_value(bloq, QECGatesCost())

    if substitutions:
        n_algo_qubits = sympy.sympify(n_algo_qubits).subs(substitutions)
        gate_counts = gate_counts.subs(substitutions)

    gate_counts = to_t_and_clifford(gate_counts, rotation_eps=rotation_eps)
    return int(n_algo_qubits), gate_counts


def physical_resources_manual(
    bloq,
    *,
    phys_err: float = 1e-3,
    cycle_time_us: float = 1.0,
    distillation_l1_d: int = 19,
    distillation_l2_d: int = 31,
    data_d: int = 31,
    n_factories: int = 4,
    routing_overhead: float = 0.5,
    substitutions: dict | None = None,
):
    """Approach 1 (THC docs): manually picked factory/data-block parameters."""
    n_algo_qubits, gate_counts = _extract_logical_costs(bloq, substitutions)

    factory = MultiFactory(
        base_factory=CCZ2TFactory(
            distillation_l1_d=distillation_l1_d,
            distillation_l2_d=distillation_l2_d,
        ),
        n_factories=n_factories,
    )
    data_block = SimpleDataBlock(data_d=data_d, routing_overhead=routing_overhead)

    cost = get_ccz2t_costs(
        n_logical_gates=gate_counts,
        n_algo_qubits=n_algo_qubits,
        phys_err=phys_err,
        cycle_time_us=cycle_time_us,
        factory=factory,
        data_block=data_block,
    )

    return {
        "n_algo_qubits": n_algo_qubits,
        "logical_gate_counts": gate_counts,
        "factory": factory,
        "data_block": data_block,
        "failure_prob": cost.failure_prob,
        "duration_hr": cost.duration_hr,
        "footprint": cost.footprint,
    }


def physical_resources_grid_search(
    bloq,
    *,
    phys_err: float = 1e-3,
    error_budget: float = 1e-2,
    n_factories: int = 4,
    cost_function=lambda pc: pc.duration_hr,
    substitutions: dict | None = None,
):
    """Approach 2 (THC docs): grid search over factory/data-block choices."""
    n_algo_qubits, gate_counts = _extract_logical_costs(bloq, substitutions)

    err_model = LogicalErrorModel(
        qec_scheme=QECScheme.make_gidney_fowler(),
        physical_error=phys_err,
    )

    best_cost, best_factory, best_data_block = get_ccz2t_costs_from_grid_search(
        n_logical_gates=gate_counts,
        n_algo_qubits=n_algo_qubits,
        error_budget=error_budget,
        phys_err=phys_err,
        factory_iter=iter_ccz2t_factories(n_factories=n_factories),
        cost_function=cost_function,
    )

    distillation_error = best_factory.factory_error(
        gate_counts, logical_error_model=err_model
    )
    data_error = best_data_block.data_error(
        n_algo_qubits=n_algo_qubits,
        n_cycles=best_factory.n_cycles(gate_counts, logical_error_model=err_model),
        logical_error_model=err_model,
    )

    return {
        "n_algo_qubits": n_algo_qubits,
        "logical_gate_counts": gate_counts,
        "best_factory": best_factory,
        "best_data_block": best_data_block,
        "distillation_error": distillation_error,
        "data_error": data_error,
        "failure_prob": best_cost.failure_prob,
        "duration_hr": best_cost.duration_hr,
        "footprint": best_cost.footprint,
    }


# ---- Use it ----

if __name__ == "__main__":

    from block_unitary_interferometer_QROAM import (
        BlockUnitaryInterferometerSynthesisQROAM,
        optimal_interferometer_log_block_sizes,
    )
    from block_unitary_reflection_QROAM import BlockUnitaryReflectionQROAM

    from qualtran.bloqs.swap_network import SwapWithZero
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    from exchange_Coulomb_block_encoding import ExchangeCoulombBlockEncoding
    from svd_block_encoding_interferometer import SVDBlockEncodingInterferometer
    from rectangular_block_encoding_reflection import ReflectionRectangularBlockEncoding

    from classical_matrix_block_encoding_QROAM import ClassicalMatrixBlockEncoding

    '''
    n_blocks = 216
    n_rows = 256
    phase_bitsize = 32
    svd_bloq = SVDBlockEncodingInterferometer(
        n_blocks, n_rows, phase_bitsize,
         n_layers=256,
        optimal_T=True
    )
    # Estimate both the bare block encoding and its single-qubit-controlled version
    # (controlled via get_ctrl_system: only the AddIntoPhaseGrad ops gain a control,
    # so the controlled cost is essentially the uncontrolled cost plus a small overhead).
    bloqs = {
        "uncontrolled": svd_bloq,
        "controlled": svd_bloq.controlled(),
    }
    #'''

    '''
    # Rectangular (Householder-reflection) block encoding of sum_k |k><k| (x) A_k,
    # with A_k an (n_rows x n_reflections) isometry synthesized by n_reflections
    # block Householder reflections.  As with the SVD case, .controlled() (via
    # get_ctrl_system) only appends one extra control to the reflect-about-|0...0>
    # MultiControlZ per reflection, leaving the QROAM prepare/uncompute uncontrolled,
    # so the controlled cost ~ uncontrolled + n_reflections control bits.
    n_blocks = 216
    n_rows = 256
    phase_bitsize = 32
    n_reflections = 22
    refl_bloq = ReflectionRectangularBlockEncoding(
        n_blocks=n_blocks,
        n_rows=n_rows,
        phase_bitsize=phase_bitsize,
        n_reflections=n_reflections,
        optimal_T=True,
    )
    bloqs = {
        "uncontrolled": refl_bloq,
        "controlled": refl_bloq.controlled(),
    }
    #'''

    '''
    # Exchange-Coulomb block encoding  B . C . B^dagger  (the full data-free
    # construction).  Its .controlled() (via get_ctrl_system) exploits the sandwich:
    # only the middle SVD block encoding C is controlled (itself cheaply, via
    # SVDBlockEncodingInterferometer.get_ctrl_system); the outer reflection pair
    # B / B^dagger and the mod-sub / uniform-prep bookkeeping cancel pairwise when
    # ctrl = 0, so the controlled cost is the uncontrolled cost plus only the small
    # controlled-C overhead.
    exch_bloq = ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=26*8, N_k=216, phase_bitsize=32,
        optimal_T=True,
    )
    bloqs = {
        "uncontrolled": exch_bloq,
        "controlled": exch_bloq.controlled(),
    }
    #'''

    #'''
    # Exchange-Coulomb block encoding: COMPARE two central-block choices, both controlled.
    #   * central = SVD interferometer  (default: central_via_reflection=False)
    #   * central = Householder reflection isometry (central_via_reflection=True)
    # Outer (B_up, B_down) and bookkeeping are identical in the two cases; the difference
    # is purely in C.  .controlled() exploits the B . C . B^dagger sandwich so only C
    # gains the external control -- and via its own get_ctrl_system the control reaches
    # only the cheap parts inside (AddIntoPhaseGrad in the SVD case, the
    # multi-controlled-Z in each reflection in the reflection case).
    exch_svd = ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=26*8, N_k=216, phase_bitsize=32,
        optimal_T=True, central_via_reflection=False,
    )
    exch_refl = ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=26*8, N_k=216, phase_bitsize=32,
        optimal_T=True, central_via_reflection=True,
    )
    exch_svd_fro = ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=26*8, N_k=216, phase_bitsize=32,
        optimal_T=True, central_via_reflection=False, use_fro_BE=True
    )
    exch_refl_fro = ExchangeCoulombBlockEncoding(
        N_up=4, N_down=22, N_IP=26*8, N_k=216, phase_bitsize=32,
        optimal_T=True, central_via_reflection=True, use_fro_BE=True
    )
    bloqs = {
        "exchange, central=SVD, fro=False":        exch_svd.controlled(),
        "exchange, central=reflection, fro=False": exch_refl.controlled(),
        "exchange, central=SVD, fro=True":        exch_svd_fro.controlled(),
        "exchange, central=reflection, fro=True": exch_refl_fro.controlled(),
    }
    #'''

    '''
    n_blocks = 216 
    n_rows = 256
    phase_bitsize = 32
    bloq1 = BlockUnitaryInterferometerSynthesisQROAM.from_shape(
        n_blocks, n_rows, phase_bitsize, 
         n_layers=256,
        optimal_T=True
    )
    n_blocks = 216
    n_rows = 256
    phase_bitsize = 32
    bloq2 = BlockUnitaryReflectionQROAM.from_shape(
        n_blocks, n_rows, phase_bitsize,
        n_reflections=256,
        optimal_T=True,
    )

    bloqs = {"Interferometer": bloq1, "Reflection":bloq2}
    #'''

    '''
    # Per-staircase-layer lambdas inside one Householder reflection's state preparation.
    # Each amp and phase layer does forward QROAMClean + QROAMCleanAdjoint with its own
    # per-layer-capped log_block_sizes driven by ``amp_log_block_sizes`` /
    # ``amp_adjoint_log_block_sizes`` / ``phase_log_block_sizes`` /
    # ``phase_adjoint_log_block_sizes``.
    state_prep = bloq.reflection(0).prepare_w.state_prep
    print("\n--- Reflection state-prep staircase: per-layer lambdas ---")
    print(f"  amp_log_block_sizes (global)              = {bloq.amp_log_block_sizes}")
    print(f"  amp_adjoint_log_block_sizes (global)      = {bloq.amp_adjoint_log_block_sizes}")
    print(f"  phase_log_block_sizes                     = {bloq.phase_log_block_sizes}")
    print(f"  phase_adjoint_log_block_sizes             = {bloq.phase_adjoint_log_block_sizes}")
    print(f"  {'layer':10s} {'shape':>14s} {'fwd lbs':>10s} {'fwd lam':>8s} {'adj lbs':>10s} {'adj lam':>8s}")
    for qi, prga in enumerate(state_prep.prga_prepare_amplitude):
        shape = (int(bloq.n_blocks), 1 << prga.selection_bitsize)
        fwd_lbs = prga.qroam_log_block_sizes
        adj_lbs = prga.qroam_adjoint_log_block_sizes
        fwd_lam = 1 << (sum(fwd_lbs) if fwd_lbs else 0)
        adj_lam = 1 << (sum(adj_lbs) if adj_lbs else 0)
        print(f"  amp qi={qi:<5d} {str(shape):>14s} {str(fwd_lbs):>10s} {fwd_lam:>8d} {str(adj_lbs):>10s} {adj_lam:>8d}")
    phases = state_prep.prga_prepare_phases
    shape = (int(bloq.n_blocks), 1 << phases.selection_bitsize)
    fwd_lbs = phases.qroam_log_block_sizes
    adj_lbs = phases.qroam_adjoint_log_block_sizes
    fwd_lam = 1 << (sum(fwd_lbs) if fwd_lbs else 0)
    adj_lam = 1 << (sum(adj_lbs) if adj_lbs else 0)
    print(f"  {'phases':10s} {str(shape):>14s} {str(fwd_lbs):>10s} {fwd_lam:>8d} {str(adj_lbs):>10s} {adj_lam:>8d}")
    #'''
    
    n_blocks = 216 
    n_rows = 256
    phase_bitsize = 32
    bloq1 = SVDBlockEncodingInterferometer(
        n_blocks, n_rows, phase_bitsize, 
        n_layers=256,
        optimal_T=True
    ).controlled()
    n_blocks = 216
    n_rows = 512
    phase_bitsize = 32
    bloq2 = ReflectionRectangularBlockEncoding(
        n_blocks, n_rows, phase_bitsize,
        n_reflections=256,
        optimal_T=True,
    ).controlled()

    bloqs |= {"Interferometer": bloq1, "Reflection":bloq2}

    def _print_gc(gc: GateCounts):
        print(f"  T gates:   {int(gc.t):,}")
        print(f"  Cliffords: {int(gc.clifford):,}")

    for label, bloq in bloqs.items():
        print()
        print("#" * 60)
        print(f"#{label}")
        print("#" * 60)

        #print("=" * 60)
        #print("Approach 1: manual factory / data-block selection")
        #print("=" * 60)
        r1 = physical_resources_manual(bloq)
        print(f"Logical qubits: {r1['n_algo_qubits']}")
        print("Logical gate counts (T + Clifford only):")
        _print_gc(r1["logical_gate_counts"])
        print(f"Failure probability: {r1['failure_prob']:.3%}")
        print(f"Wall time: {r1['duration_hr'] / 24:.3g} days "
              f"({r1['duration_hr']:.3g} hr)")
        print(f"Footprint: {r1['footprint'] * 1e-6:.2f} million physical qubits")

        '''
        print()
        print("=" * 60)
        print("Approach 2: grid-search optimal factory / data-block")
        print("=" * 60)
        r2 = physical_resources_grid_search(bloq)
        print(f"Logical qubits: {r2['n_algo_qubits']}")
        print("Logical gate counts (T + Clifford only):")
        _print_gc(r2["logical_gate_counts"])
        print(f"Best factory: {r2['best_factory']}")
        print(f"Best data block: {r2['best_data_block']}")
        print(f"Distillation error: {r2['distillation_error']:.3%}")
        print(f"Data error: {r2['data_error']:.3%}")
        print(f"Total failure probability: {r2['failure_prob']:.3%}")
        print(f"Wall time: {r2['duration_hr'] / 24:.3g} days "
              f"({r2['duration_hr']:.3g} hr)")
        print(f"Footprint: {r2['footprint'] * 1e-6:.2f} million physical qubits")
        '''
