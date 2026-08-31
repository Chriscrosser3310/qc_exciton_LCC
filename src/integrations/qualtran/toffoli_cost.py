r"""The Toffoli-equivalent cost metric.

Qualtran's ``QECGatesCost`` reports ``n_ccz`` (Toffoli / CCZ) and ``n_t`` (bare T)
**separately**.  For QROAM in particular the two halves of the construction land in
different buckets:

* the unary-iteration address traversal, ``ceil(M/Lambda)``, is charged as ``n_ccz``;
* the **select-swap network**, ``(Lambda - 1) * w``, is charged as ``n_t``.

So ``n_ccz`` alone *omits the entire space-time tradeoff term*.  Reading it as "the
Toffoli count" makes a QROAM look cheaper the larger its block size, without bound --
which inverts the tradeoff, hides the cost of big tables, and can even go negative.

At the standard 4 T per Toffoli the correct aggregate is

.. math::  C_{\mathrm{Toffoli}} = n_{ccz} + n_t / 4 ,

and it reproduces the analytic model's QROAM term exactly (to the model's own
controlled-unary-iteration constant of 1):

======  =========  ==========  ================  ===========
Lambda  ``n_ccz``  ``n_t/4``   ``n_ccz+n_t/4``   model
======  =========  ==========  ================  ===========
1       1022       0           1022              1023
4       254        192         446               447
8       126        448         574               575
128     6          8128        8134              8135
======  =========  ==========  ================  ===========

Use :func:`toffoli_count` for every reported number and every block-size search.
"""

from __future__ import annotations

from qualtran.resource_counting import get_cost_value, QECGatesCost


def toffoli_count(bloq) -> int:
    r"""Toffoli-equivalent cost of ``bloq``: ``n_ccz + n_t/4``.

    Counts the QROAM select-swap network, which ``n_ccz`` alone silently drops.
    """
    d = get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()
    return int(d['n_ccz'] + d['n_t'] / 4)


def toffoli_parts(bloq) -> dict:
    """``{'n_ccz', 'n_t', 'toffoli'}`` -- useful when diagnosing a tradeoff choice."""
    d = get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()
    return {'n_ccz': int(d['n_ccz']), 'n_t': int(d['n_t']),
            'toffoli': int(d['n_ccz'] + d['n_t'] / 4)}
