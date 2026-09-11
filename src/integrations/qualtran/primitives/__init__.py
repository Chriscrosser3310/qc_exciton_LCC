"""Cost primitives implemented exactly as `context/primitives.md` specifies them."""

from .arithmetic import CtrlModularAddNoCarry, ModularAddSubtract, RippleCompare
from .bse import IrrepBlocks, SymmetryAdaptedBSE, diamond_blocks_6x6x6_c8
from .diagonal import DiagonalBlockEncoding
from .eigendecomposition import FusedEigendecompositionBlockEncoding
from .isometry import FusedColumnIsometry, power_of_two_at_least
from .qroam import RangeSafeQROAM, best_schedule, lookup_ancillas, lookup_toffolis
from .relabel import ReversibleRelabel
from .space_group import SpaceGroupData, SpaceGroupTransform, diamond_6x6x6_c8
from .staircase import BlockStaircase
from .state_prep import IndexedLoadAllStatePreparation
from .unitary import FusedBlockUnitarySynthesis, even_padded_sizes

__all__ = [
    "BlockStaircase",
    "CtrlModularAddNoCarry",
    "DiagonalBlockEncoding",
    "FusedBlockUnitarySynthesis",
    "FusedColumnIsometry",
    "FusedEigendecompositionBlockEncoding",
    "IndexedLoadAllStatePreparation",
    "IrrepBlocks",
    "ModularAddSubtract",
    "RangeSafeQROAM",
    "ReversibleRelabel",
    "RippleCompare",
    "SpaceGroupData",
    "SpaceGroupTransform",
    "SymmetryAdaptedBSE",
    "best_schedule",
    "diamond_6x6x6_c8",
    "diamond_blocks_6x6x6_c8",
    "even_padded_sizes",
    "lookup_ancillas",
    "lookup_toffolis",
    "power_of_two_at_least",
]
