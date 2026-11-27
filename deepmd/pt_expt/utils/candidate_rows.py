# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cached candidate neighbor rows for the compact canonical graph.

一次重建把 rcut + skin 的候选邻居按中心原子排成行并存为自有张量
(行指针、行序源索引、预乘参考晶胞的位移), 之后每步只需以当前坐标
重算几何并压实 -- 不再重复排序与重排。热步由原生算子
``deepmd::dpa4c_candidate_rows_select`` 完成; 本模块只负责缓存的生命
周期与失效判定。
"""

from __future__ import (
    annotations,
)

import torch

from deepmd.pt_expt.utils.canonical_graph import (
    canonical_index_dtype,
)


def candidate_rows_available() -> bool:
    """Whether the native candidate-row operator is loaded."""
    return hasattr(torch.ops.deepmd, "dpa4c_candidate_rows_select")


class CandidateRowCache:
    """Candidate rows of one frame slot, rebuilt when their basis expires.

    The cache owns every tensor it stores; nothing aliases Vesin internals.
    Rebuild uses only the public Vesin interface at ``rcut + skin``. A hot
    step is valid while the cell is unchanged and no atom has moved farther
    than ``skin / 2`` from its reference position (Euclidean, per plan C).
    """

    def __init__(self, rcut: float, skin: float, node_count: int) -> None:
        self.rcut = float(rcut)
        self.skin = float(skin)
        self.node_count = int(node_count)
        self.row_ptr: torch.Tensor | None = None
        self.source: torch.Tensor | None = None
        self.shift_vec: torch.Tensor | None = None
        self.ref_pos: torch.Tensor | None = None
        self.ref_cell: torch.Tensor | None = None

    def matches(
        self, rcut: float, skin: float, node_count: int, positions: torch.Tensor
    ) -> bool:
        """Whether this cache was built for the same search basis."""
        return (
            self.row_ptr is not None
            and self.rcut == float(rcut)
            and self.skin == float(skin)
            and self.node_count == int(node_count)
            and positions.shape[0] == self.node_count
            and positions.device == self.row_ptr.device
            and positions.dtype == torch.float64
        )

    def rebuild(self, positions: torch.Tensor, cell: torch.Tensor) -> None:
        """Rebuild the candidate rows from the public Vesin interface."""
        from vesin_torch import (
            NeighborList,
        )

        candidates = NeighborList(
            cutoff=self.rcut + self.skin,
            full_list=True,
            sorted=False,
            algorithm="cell_list",
            skin=0,
        )
        ii, jj, shifts = candidates.compute(
            points=positions.detach(),
            box=cell.detach(),
            periodic=True,
            quantities="ijS",
            copy=True,
        )
        order = torch.argsort(ii, stable=True)
        ii, jj, shifts = ii[order], jj[order], shifts[order]
        counts = torch.bincount(ii, minlength=self.node_count).to(torch.int64)
        self.row_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=counts.device),
                counts.cumsum(0),
            )
        )
        self.source = jj.to(canonical_index_dtype(self.node_count)).contiguous()
        self.shift_vec = (
            shifts.to(torch.float64) @ cell.detach().to(torch.float64)
        ).contiguous()
        self.ref_pos = positions.detach().clone()
        self.ref_cell = cell.detach().clone()

    def valid_for(self, positions: torch.Tensor, cell: torch.Tensor) -> bool:
        """Cell unchanged and every atom within skin / 2 of its reference."""
        if not torch.equal(cell, self.ref_cell):
            return False
        moved = (positions - self.ref_pos).norm(dim=1).max()
        return bool(moved < self.skin / 2)

    def hot(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact this frame's edges from the candidate rows."""
        assert self.row_ptr is not None and self.source is not None
        return torch.ops.deepmd.dpa4c_candidate_rows_select(
            self.row_ptr,
            self.source,
            self.shift_vec,
            positions.contiguous(),
            self.rcut,
        )
