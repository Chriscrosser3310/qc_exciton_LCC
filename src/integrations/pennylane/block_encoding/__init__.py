"""PennyLane block-encoding utilities."""

from .exciton_block_encoding import (
    exciton_block_encoding,
    one_particle_f_sum_block_encoding,
    two_particle_v_sum_block_encoding,
    two_particle_w_sum_block_encoding,
)
from .entry_oracle import controlled_entry_oracle_2d, entry_oracle_2d
from .index_oracle import one_particle_index_oracle, two_particle_index_oracle
from .qrom import qrom_table_2d
from .sparse_block_encoding import (
    one_particle_sparse_block_encoding,
    two_particle_sparse_block_encoding,
)

__all__ = [
    "qrom_table_2d",
    "entry_oracle_2d",
    "controlled_entry_oracle_2d",
    "one_particle_index_oracle",
    "two_particle_index_oracle",
    "one_particle_sparse_block_encoding",
    "two_particle_sparse_block_encoding",
    "one_particle_f_sum_block_encoding",
    "two_particle_w_sum_block_encoding",
    "two_particle_v_sum_block_encoding",
    "exciton_block_encoding",
]
