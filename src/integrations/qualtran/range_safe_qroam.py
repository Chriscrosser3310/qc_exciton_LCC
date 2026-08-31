r"""Range-safety surcharge for QROAM address registers (P-13).

Plain unary iteration over a table of :math:`L` entries in a :math:`w`-qubit address
register is **unrestricted**: for :math:`x \ge L` it is free to write an arbitrary table
entry, because the controls that would distinguish those addresses are exactly what the
construction drops to get its :math:`L-1` cost.  What is needed here is the **restricted**
(range-safe) form,

.. math::
    U\lvert x\rangle\lvert 0\rangle = \lvert x\rangle\lvert d_x\rangle \;(x<L),
    \qquad
    U\lvert x\rangle\lvert 0\rangle = \lvert x\rangle\lvert 0\rangle \;(x\ge L),

so out-of-range addresses leave the target at :math:`\lvert 0\rangle` rather than picking
up garbage.

Why this project needs it
-------------------------
1. :math:`L` is not a power of two for any published mesh -- :math:`N_k = 27, 125, 216,
   343`, :math:`N_{\mathrm{THC}} = 208`, and their products;
2. **the walk operator's reflection populates the whole address register** (ledger
   `C-16`), so out-of-range basis states are genuinely reached -- neither of the two
   escape clauses that license unrestricted iteration (Rupprecht--Wölk 2026 §3) has been
   established for this walk;
3. ragged block-diagonal tables, where the valid within-block range depends on the block.

Qualtran does **not** do this.  Measured, ``QROM.build_from_bitsize((L,), ...)`` costs
:math:`L-2` for every :math:`L` -- the unrestricted count, with no dependence on how far
:math:`L` sits below :math:`2^w`:

======  ====  ==============  =================  ==================
``L``   w     Qualtran QROM   unrestricted L-1   range-safe L-1+d
======  ====  ==============  =================  ==================
27      5     25              26                 28
106     7     104             105                108
216     8     214             215                217
343     9     341             342                346
======  ====  ==============  =================  ==================

The constant
------------
From `verify/check_unary_iteration.py`, checked exhaustively for every :math:`L` and every
:math:`w \le 11` (4094 cases, no exceptions):

.. math::  \delta(L, w) \;=\; w - \operatorname{popcount}(L-1)

extra AND gates for the restricted iteration against the unrestricted one -- one extra
control per level of the tree where only the left half is live, and those levels are
exactly the zero bits of :math:`L-1`.  It is :math:`0` whenever :math:`L` is a power of
two, and :math:`O(\log L)` otherwise -- **additive and logarithmic, not a padding factor
on the table**.

Multi-dimensional addresses
---------------------------
For a chained (P-14 construction C) lookup each register is restricted over its *own*
range and the surplus of one never multiplies into the others, so the surcharges **add**:

.. math::  \delta_{\text{total}} = \sum_i \big( w_i - \operatorname{popcount}(L_i - 1) \big).

That is what :func:`range_safety_delta` returns.  Flattening and zero-padding the table
instead is a valid but worse implementation (+9 % to +29 % on the cases in the ledger);
this module charges the chained cost.
"""

from __future__ import annotations

from typing import Sequence


def _delta_1d(n_entries: int) -> int:
    """``w - popcount(L-1)``; zero when ``L`` is a power of two."""
    L = int(n_entries)
    if L <= 1:
        return 0
    w = (L - 1).bit_length()
    return w - bin(L - 1).count('1')


def range_safety_delta(data_shape: Sequence[int]) -> int:
    """Extra AND gates to make a QROAM over ``data_shape`` range-safe.

    Additive over address registers (each is restricted over its own range).
    """
    return sum(_delta_1d(d) for d in data_shape)


def emit_range_safety(ret, data_shape: Sequence[int], count: int = 1) -> None:
    """Add the range-safety surcharge for ``count`` lookups over ``data_shape`` to ``ret``.

    The AND gates are uncomputed by measurement (0 Toffoli), so only the compute side is
    charged -- the same convention the QROAM erasures use.
    """
    from qualtran.bloqs.mcmt import And

    d = range_safety_delta(data_shape) * int(count)
    if d:
        ret[And()] += d
