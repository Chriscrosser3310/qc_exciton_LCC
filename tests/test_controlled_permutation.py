from __future__ import annotations

import numpy as np
import pytest


qualtran = pytest.importorskip("qualtran")
_ = qualtran

from integrations.qualtran.block_encoding.controlled_permutation import (
    MultiControlledRegisterPermutation,
    build_multi_controlled_register_permutation,
)


def test_multi_controlled_register_permutation_basic_mapping():
    # (0,2,1) swaps second and third register when ctrl=1.
    bloq = build_multi_controlled_register_permutation(
        register_bitsizes=(2, 3, 3),
        permutation=(0, 2, 1),
        control_values=(1,),
    )
    assert isinstance(bloq, MultiControlledRegisterPermutation)
    assert bloq.swap_sequence == ((1, 2),)

    ctrl_out, r0, r1, r2 = bloq.call_classically(ctrl=np.array([1]), r0=1, r1=3, r2=5)
    assert np.array_equal(ctrl_out, np.array([1]))
    assert (r0, r1, r2) == (1, 5, 3)

    ctrl_out, r0, r1, r2 = bloq.call_classically(ctrl=np.array([0]), r0=1, r1=3, r2=5)
    assert np.array_equal(ctrl_out, np.array([0]))
    assert (r0, r1, r2) == (1, 3, 5)


def test_multi_controlled_register_permutation_multi_control():
    bloq = build_multi_controlled_register_permutation(
        register_bitsizes=(2, 2, 2),
        permutation=(2, 1, 0),
        control_values=(1, 0),
    )

    c, r0, r1, r2 = bloq.call_classically(ctrl=np.array([1, 0]), r0=0, r1=1, r2=2)
    assert np.array_equal(c, np.array([1, 0]))
    assert (r0, r1, r2) == (2, 1, 0)

    c, r0, r1, r2 = bloq.call_classically(ctrl=np.array([1, 1]), r0=0, r1=1, r2=2)
    assert np.array_equal(c, np.array([1, 1]))
    assert (r0, r1, r2) == (0, 1, 2)


def test_multi_controlled_register_permutation_signature_and_cost():
    bloq = build_multi_controlled_register_permutation(
        register_bitsizes=(2, 2, 2),
        permutation=(0, 2, 1),
        control_values=(1, 1),
    )
    reg_names = [r.name for r in bloq.signature]
    assert reg_names == ["ctrl", "r0", "r1", "r2"]

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    qec = get_cost_value(bloq, QECGatesCost())
    assert qec is not None


def test_multi_controlled_register_permutation_rejects_incompatible_sizes():
    # r1 and r2 sizes differ, so (0,2,1) cannot be implemented with fixed signature.
    with pytest.raises(ValueError):
        build_multi_controlled_register_permutation(
            register_bitsizes=(2, 3, 4),
            permutation=(0, 2, 1),
            control_values=(1,),
        )

