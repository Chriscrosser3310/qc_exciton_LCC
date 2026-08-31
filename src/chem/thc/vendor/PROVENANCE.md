# Vendored modules

Copied verbatim, unmodified, from

    /resnick/groups/changroup/members/jsun3/xprize/THC_general/

on 2026-08-25. These are jsun3's code, not ours. They are vendored so that the
optimizer in the parent directory reproduces the *identical* fitting algorithm
rather than a reimplementation of it, and so this folder stays runnable if the
source tree moves.

| file | why it is needed |
|---|---|
| `utils.py` | the THC fit itself: `thc_ovvo_build_Lbar`, `thc_ovvo_solve_w_intermediate_from_mo`, `thc_solve_w_error2_from_intermediate`, `torch_lstsq_oinv_PSD`, `fourier_transform_3d`, `add_k`, `negative_k`, `is_k_ordered` |
| `optimize_X_common.py` | Adam driver and state handling: `run_adam`, `make_schedule`, `optimization_loss_from_X_log_reg`, `to_tensor`, `load_isdf`, `save_isdf`, `ao_from_state`, `X_mo_from_ao`, `state_from_ao`, `negative_k_indices`, `symmetrize_real_gauge` |
| `libsymm.py` | only used when symmetrization is requested (`--symm`) |

`system_common.py` is deliberately **not** vendored. Its only role upstream is to
resolve data directories by naming convention; `optimize.py` takes explicit paths
instead.

`optimize_X_common.get_device()` imports a `gpu_register` module that does not exist
in the source tree, so it is never called here -- `optimize.py` does its own device
selection with a CPU fallback.

Do not edit these files. If upstream changes, re-copy and note the date.
04fbd98c6f3ea4aad203f5611647849cbc703e5f20b91ddd2f135e2040861226  utils.py
bad6f8c562afe73c35b4988222c4ffcbcf171c98dcbbbb4a312a3a279c048806  optimize_X_common.py
c993ad6b85e73777734cfecd7772c42ae44eb7c4c3d451c72aaf61e44ec57272  libsymm.py
