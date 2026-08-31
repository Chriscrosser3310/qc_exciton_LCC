# Generated from data_loading_comparison.ipynb
# Edit the notebook if you want to preserve notebook/script parity manually.

# %% cell 0
import math
from typing import Optional

import attrs
import matplotlib.pyplot as plt
import numpy as np
import sympy

from qualtran import BloqBuilder, GateWithRegisters, Signature, SoquetT
from qualtran.bloqs.basic_gates import CRy
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM as SelectSwap
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting.generalizers import generalize_cswap_approx
from qualtran.symbolics import Shaped


@attrs.frozen
class DataLoadingControlledRotation(GateWithRegisters):
    """Load shape-only data, rotate from the loaded data register, then uncompute."""

    loader_name: str
    num_entries: int
    data_bitsize: int
    selection_bitsize: int
    log_block_size: Optional[int] = None

    @property
    def signature(self) -> Signature:
        return Signature.build(selection=self.selection_bitsize, rotation_target=1)

    @property
    def data_or_shape(self):
        # Shape-only ROM data: no concrete data values are supplied.
        return (Shaped((self.num_entries,)),)

    def _loader_kwargs(self):
        return dict(
            data_or_shape=self.data_or_shape,
            selection_bitsizes=(self.selection_bitsize,),
            target_bitsizes=(self.data_bitsize,),
        )

    def _with_optional_log_block_size(self, loader_cls):
        kwargs = self._loader_kwargs()
        if self.log_block_size is not None:
            kwargs["log_block_sizes"] = (self.log_block_size,)
        return loader_cls(**kwargs)

    @property
    def data_loading(self):
        if self.loader_name == "QROM":
            return QROM(**self._loader_kwargs())
        if self.loader_name == "SelectSwap":
            return self._with_optional_log_block_size(SelectSwap)
        if self.loader_name == "QROAMClean":
            return self._with_optional_log_block_size(QROAMClean)
        raise ValueError(f"Unknown loader_name={self.loader_name!r}")

    @property
    def inverse_data_loading(self):
        if self.loader_name == "QROM":
            return QROM(**self._loader_kwargs()).adjoint()
        if self.loader_name == "SelectSwap":
            return self._with_optional_log_block_size(SelectSwap).adjoint()
        if self.loader_name == "QROAMClean":
            return self._with_optional_log_block_size(QROAMCleanAdjoint)
        raise ValueError(f"Unknown loader_name={self.loader_name!r}")

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT):
        selection = soqs["selection"]
        rotation_target = soqs["rotation_target"]

        if self.loader_name in ("QROM", "SelectSwap"):
            data = bb.allocate(self.data_bitsize)
            selection, data = bb.add(
                self.data_loading, selection=selection, target0_=data
            )
        else:
            selection, data, *junk_registers = bb.add_t(
                self.data_loading, selection=selection
            )

        data_bits = list(bb.split(data))
        for bit_index, data_bit in enumerate(data_bits):
            data_bits[bit_index], rotation_target = bb.add(
                CRy(sympy.Symbol(f"theta_{bit_index}")),
                ctrl=data_bit,
                q=rotation_target,
            )
        data = bb.join(data_bits)

        if self.loader_name in ("QROM", "SelectSwap"):
            selection, data = bb.add(
                self.inverse_data_loading, selection=selection, target0_=data
            )
            bb.free(data)
        else:
            selection = bb.add(
                self.inverse_data_loading, selection=selection, target0_=data
            )
            for junk_register in junk_registers:
                if isinstance(junk_register, np.ndarray):
                    for junk_soq in junk_register.flat:
                        bb.free(junk_soq)
                else:
                    bb.free(junk_register)

        return {"selection": selection, "rotation_target": rotation_target}


def selection_bitsize(num_entries: int) -> int:
    return math.ceil(math.log2(num_entries))


DATA_BITSIZE = 16
DATA_SIZES = [
    {"num_entries": 2**exponent, "data_bitsize": DATA_BITSIZE}
    for exponent in range(8, 21, 2)
]
LOADER_NAMES = ("QROM", "SelectSwap", "QROAMClean")
DISPLAY_NAMES = {"QROM": "QROM", "SelectSwap": "SelectSWAP", "QROAMClean": "QROAM"}
OPTIMIZATION_MODES = ("qubit-optimized", "Toffoli-optimized")


def log_block_sizes_for(loader_name: str, selection_bits: int):
    # QROM does not expose a block-size tradeoff; plot it twice as the fixed
    # baseline for the qubit- and Toffoli-optimized views.
    if loader_name == "QROM":
        return [None]
    return list(range(selection_bits + 1))


bloqs_by_config = {}
for spec in DATA_SIZES:
    num_entries = spec["num_entries"]
    data_bitsize = spec["data_bitsize"]
    bits = selection_bitsize(num_entries)
    for loader_name in LOADER_NAMES:
        for log_block_size in log_block_sizes_for(loader_name, bits):
            key = (num_entries, data_bitsize, loader_name, log_block_size)
            bloqs_by_config[key] = DataLoadingControlledRotation(
                loader_name=loader_name,
                num_entries=num_entries,
                data_bitsize=data_bitsize,
                selection_bitsize=bits,
                log_block_size=log_block_size,
            )

# `bloqs_by_config` contains the raw scanned bloqs. Each bloq performs:
# data_loading -> data-register-controlled rotations -> inverse_data_loading.


def qubit_count_from_forward_loader(bloq: DataLoadingControlledRotation):
    """Estimate peak qubits without decomposing adjoint loader bloqs.

    Some Qualtran versions fail to decompose `SelectSwapQROM` adjoint during
    `QubitCount`, even though the forward loader is countable. The full
    load-rotate-unload wrapper only adds the persistent rotation target to
    the loader's peak width, so the ancilla count relative to the wrapper
    signature is `loader_peak - selection_bitsize`.
    """

    loader_peak_qubits = get_cost_value(bloq.data_loading, QubitCount())
    return int(loader_peak_qubits) + 1


def resource_counts(bloq: DataLoadingControlledRotation):
    gate_counts = get_cost_value(
        bloq, QECGatesCost(), generalizer=generalize_cswap_approx
    )
    toffoli_count = gate_counts.total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"]
    try:
        peak_qubits = get_cost_value(bloq, QubitCount())
    except RuntimeError as exc:
        if "SelectSwapQROM" not in str(exc):
            raise
        peak_qubits = qubit_count_from_forward_loader(bloq)
    external_qubits = bloq.signature.n_qubits()
    ancilla_count = max(0, int(peak_qubits) - int(external_qubits))
    return int(toffoli_count), ancilla_count


raw_records = []
for (num_entries, data_bitsize, loader_name, log_block_size), bloq in bloqs_by_config.items():
    toffoli_count, ancilla_count = resource_counts(bloq)
    raw_records.append(
        {
            "num_entries": num_entries,
            "data_bitsize": data_bitsize,
            "selection_bitsize": selection_bitsize(num_entries),
            "loader": loader_name,
            "log_block_size": log_block_size,
            "block_size": None if log_block_size is None else 2**log_block_size,
            "toffoli_count": toffoli_count,
            "ancilla_count": ancilla_count,
            "bloq": bloq,
        }
    )


def best_record(candidates, optimization_mode: str):
    if optimization_mode == "qubit-optimized":
        return min(candidates, key=lambda r: (r["ancilla_count"], r["toffoli_count"]))
    if optimization_mode == "Toffoli-optimized":
        return min(candidates, key=lambda r: (r["toffoli_count"], r["ancilla_count"]))
    raise ValueError(f"Unknown optimization_mode={optimization_mode!r}")


records = []
for spec in DATA_SIZES:
    num_entries = spec["num_entries"]
    for loader_name in LOADER_NAMES:
        candidates = [
            record
            for record in raw_records
            if record["num_entries"] == num_entries and record["loader"] == loader_name
        ]
        for optimization_mode in OPTIMIZATION_MODES:
            selected = dict(best_record(candidates, optimization_mode))
            selected["optimization_mode"] = optimization_mode
            records.append(selected)


def fit_polynomial_order(series, metric_name: str):
    x = np.array([record["num_entries"] for record in series], dtype=float)
    y = np.array([max(record[metric_name], 1) for record in series], dtype=float)
    order, log_coefficient = np.polyfit(np.log2(x), np.log2(y), 1)
    coefficient = 2**log_coefficient
    return float(order), float(coefficient)


def fitted_values(series, metric_name: str):
    order, coefficient = fit_polynomial_order(series, metric_name)
    x = np.array([record["num_entries"] for record in series], dtype=float)
    return coefficient * x**order


def plot_metric(metric_name: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "qubit-optimized": {"linestyle": "-", "marker": "o"},
        "Toffoli-optimized": {"linestyle": "--", "marker": "s"},
    }
    for loader_name in LOADER_NAMES:
        for optimization_mode in OPTIMIZATION_MODES:
            series = sorted(
                (
                    record
                    for record in records
                    if record["loader"] == loader_name
                    and record["optimization_mode"] == optimization_mode
                ),
                key=lambda record: record["num_entries"],
            )
            order, _ = fit_polynomial_order(series, metric_name)
            label = f"{DISPLAY_NAMES[loader_name]}, {optimization_mode}, order={order:.2f}"
            x = [record["num_entries"] for record in series]
            y = [record[metric_name] for record in series]
            ax.plot(x, y, label=label, **styles[optimization_mode])
            ax.plot(x, fitted_values(series, metric_name), color=ax.lines[-1].get_color(), alpha=0.35)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of data entries")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by optimized data-loading strategy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


toffoli_fig, toffoli_ax = plot_metric("toffoli_count", "Toffoli count")
ancilla_fig, ancilla_ax = plot_metric("ancilla_count", "Ancilla count")
# `records` contains the optimized curves; `raw_records` contains all scanned block sizes.

# %% cell 1


