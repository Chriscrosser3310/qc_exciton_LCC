r"""Exact QROAM block-size selection.

The QROAM (select-swap) tradeoff for a table of :math:`M` entries with a :math:`w`-bit
output word, batched :math:`\Lambda` entries at a time, costs

.. math::
    T_{\mathrm{fwd}}(\Lambda) = \lceil M/\Lambda \rceil + (\Lambda - 1)\,w,
    \qquad
    T_{\mathrm{adj}}(\Lambda') = \lceil M/\Lambda' \rceil + \Lambda' ,

minimized near :math:`\Lambda^\star \simeq \sqrt{M/w}` and :math:`\sqrt{M}`.

Why a closed form is not enough
-------------------------------
Two selectors in this package used closed forms of the shape
``int(log2(dim))`` / ``round(log2(sqrt(...)))``.  Both are wrong for the sizes this
project actually uses:

* ``int(log2(d))`` **floors**, so a non-power-of-two dimension is costed as if the table
  were smaller than it is.  At :math:`N_{\mathrm{IP}} = 224` the table
  :math:`8 \times 224 = 1792` was treated as :math:`2^{10} = 1024`, which selected
  :math:`\Lambda = 4` where the optimum is :math:`8` -- 544 Toffolis instead of 448.
  At :math:`256` the floor happens to be exact, so the *larger* table came out
  **cheaper** than the smaller one.
* rounding :math:`\log_2\Lambda^\star` to an integer before splitting it across
  dimensions quantizes so coarsely that doubling the table can leave :math:`\Lambda`
  unchanged, and the per-layer cost then stops scaling with the table.

Since :math:`\Lambda` is a power of two per dimension and the dimensions are small, the
exact optimum is just a brute-force search over a grid of at most a few hundred points.
That is what this module does.  It is exact for non-power-of-two dimensions, which the
closed forms were not.
"""

from __future__ import annotations

from itertools import product
from math import ceil
from typing import Sequence, Tuple


def _log_caps(data_shape: Sequence[int]) -> Tuple[int, ...]:
    """Largest per-dimension log block size: ``2**bs <= dim``.

    This is ``floor(log2 d)``, NOT ``ceil``.  An earlier version used
    ``bit_length(d-1) = ceil(log2 d)``, which for every non-power-of-two dimension
    proposes a block **larger than the dimension** -- Qualtran rejects those outright
    (``assert 1 <= 2**bs <= ilen`` in ``with_log_block_sizes``).  The measured optimiser
    silently skipped them, so the live path was unaffected, but the closed-form optimiser
    would happily return an infeasible choice.
    """
    return tuple(max(0, int(d).bit_length() - 1) for d in data_shape)


def qroam_cost(data_shape: Sequence[int], log_block_sizes: Sequence[int],
               word_bits: int, *, adjoint: bool = False) -> int:
    r"""Toffoli cost of one QROAM lookup at the given per-dimension log block sizes.

    ``batches = prod(ceil(d_i / 2^{lk_i}))`` counts the unary-iteration steps, and
    ``block = prod(2^{lk_i})`` the swap-network width.  Uses ``ceil`` per dimension, so
    non-power-of-two dimensions are handled exactly.
    """
    batches = 1
    block = 1
    for d, lk in zip(data_shape, log_block_sizes):
        k = 1 << int(lk)
        batches *= ceil(int(d) / k)
        block *= k
    per_unit = 1 if adjoint else int(word_bits)
    return batches + (block - 1) * per_unit


def optimal_log_block_sizes(data_shape: Sequence[int], word_bits: int, *,
                            adjoint: bool = False) -> Tuple[int, ...]:
    r"""Brute-force the cost-minimizing per-dimension log block sizes.

    Exact for any ``data_shape`` (no power-of-two assumption).  Ties are broken toward
    the smaller total block size, i.e. toward fewer ancilla qubits.
    """
    caps = _log_caps(data_shape)
    best = None
    for combo in product(*(range(c + 1) for c in caps)):
        cost = qroam_cost(data_shape, combo, word_bits, adjoint=adjoint)
        key = (cost, sum(combo))
        if best is None or key < best[0]:
            best = (key, combo)
    assert best is not None
    return best[1]


def optimal_log_block_sizes_1d(n_entries: int, word_bits: int, *,
                               adjoint: bool = False) -> Tuple[int]:
    """Single-address convenience wrapper."""
    return optimal_log_block_sizes((n_entries,), word_bits, adjoint=adjoint)


# ---------------------------------------------------------------------------
# Measured variant -- minimize Qualtran's ACTUAL QROAM cost
# ---------------------------------------------------------------------------


def _measured_cost(data_shape, log_block_sizes, target_bitsizes, adjoint):
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
    try:
        from .toffoli_cost import toffoli_count
    except ImportError:
        from toffoli_cost import toffoli_count

    try:
        if adjoint:
            bloq = QROAMCleanAdjoint.build_from_bitsize(
                data_shape, target_bitsizes=target_bitsizes,
                log_block_sizes=log_block_sizes,
            )
        else:
            bloq = QROAMClean.build_from_bitsize(
                data_shape, target_bitsizes=target_bitsizes,
                log_block_sizes=log_block_sizes,
            )
        # Toffoli-equivalent: n_ccz alone drops the select-swap network entirely,
        # which would drive this search to the largest possible Lambda.
        return toffoli_count(bloq)
    except Exception:                                       # noqa: BLE001
        return None


def optimal_log_block_sizes_measured(data_shape, target_bitsizes, *, adjoint=False):
    r"""Brute-force the log block sizes that minimize Qualtran's *measured* QROAM cost.

    :func:`optimal_log_block_sizes` minimizes the textbook closed form
    ``ceil(M/L) + (L-1)w``.  Qualtran's ``QROAMClean`` does not cost exactly that -- it
    carries clean-ancilla bookkeeping and its own batching -- so minimizing the closed
    form can select a Lambda that is *worse* in the counter that actually reports the
    number.  This variant asks the counter directly, which is the only way to be sure
    the selected Lambda is the one the emitted circuit wants.

    Falls back to the closed-form optimum for any candidate Qualtran cannot build.
    """
    data_shape = tuple(int(d) for d in data_shape)
    caps = _log_caps(data_shape)
    best = None
    for combo in product(*(range(c + 1) for c in caps)):
        cost = _measured_cost(data_shape, combo, tuple(target_bitsizes), adjoint)
        if cost is None:
            continue
        key = (cost, sum(combo))
        if best is None or key < best[0]:
            best = (key, combo)
    if best is None:
        w = int(max(target_bitsizes))
        return optimal_log_block_sizes(data_shape, w, adjoint=adjoint)
    return best[1]
