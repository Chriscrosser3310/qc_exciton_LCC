"""THC collocation-matrix optimization with selectable norm penalties.

`penalties` holds the norms; `optimize.py` and `evaluate.py` are runnable scripts.
`vendor/` contains jsun3's fitting code, copied verbatim -- see `vendor/PROVENANCE.md`.
Norm *evaluation* lives in the sibling package `chem.lcu_norms`; this package is about
*optimizing* against those norms.
"""

from . import penalties
from .penalties import W_KINDS, X_KINDS, abs_norm_loss, norm_tag, w_norm, x_norm

__all__ = ["penalties", "abs_norm_loss", "norm_tag", "w_norm", "x_norm", "X_KINDS", "W_KINDS"]
