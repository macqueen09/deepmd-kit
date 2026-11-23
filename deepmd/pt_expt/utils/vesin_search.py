# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persistent geometric neighbor searches shared by inference input schemas."""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import (
        NDArray,
    )
    from vesin_torch import (
        NeighborList,
    )


class VesinNeighborSearch:
    """Reuse Vesin search storage and optional Verlet candidates.

    Each frame owns a calculator. Vesin refreshes displacements and filters
    candidates at the requested cutoff on every call, and rebuilds candidates
    when motion exceeds half the skin or the cell or point count changes.
    Search results are copied so later calls cannot overwrite indices retained
    by an earlier autograd graph.

    Parameters
    ----------
    skin : float, default: 0.0
        Extra candidate-search radius in angstrom. Zero disables Verlet caching.
    sorted : bool, default: False
        Whether pairs are returned in ascending center-index order.
    """

    def __init__(self, skin: float = 0.0, *, sorted: bool = False) -> None:
        if not math.isfinite(skin) or skin < 0:
            raise ValueError(
                f"neighbor_skin must be finite and non-negative, got {skin}"
            )
        self.skin = float(skin)
        self.sorted = sorted
        self._key: tuple[float, torch.device] | None = None
        # 每个 frame 槽存 (计算器, GPU 晶胞快照, 主机晶胞快照)。主机快照只在
        # 调用者提供了可信主机镜像时存在, 供失效判定做无同步的主机比较。
        self._frames: dict[
            int,
            tuple[NeighborList, torch.Tensor | None, np.ndarray | None],
        ] = {}
        # 候选行缓存(计划C): 普通单帧 canonical 路径的热步用它跳过每帧
        # 排序与重排; 其余路径不碰它, 失效即整体重建。
        self.candidate_row_cache: object | None = None

    def compute(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None,
        rcut: float,
        *,
        frame: int = 0,
        return_vectors: bool = False,
        host_cell: NDArray[np.floating] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return neighbor indices and image shifts or displacement vectors.

        Parameters
        ----------
        positions : torch.Tensor
            Coordinates in angstrom with shape (N, 3). The search detaches
            these coordinates and runs on their device.
        cell : torch.Tensor or None
            Cell vectors in angstrom with shape (3, 3), with the same dtype
            and device as ``positions``. None selects non-periodic boundaries.
        rcut : float
            Neighbor cutoff radius in angstrom.
        frame : int, optional
            Non-negative cache slot, by default 0. Each frame in a batch uses
            a distinct slot, reused for its successive configurations.
        return_vectors : bool, optional
            Return neighbor-minus-center vectors in angstrom instead of
            lattice shifts. These vectors are detached from the coordinates
            and are intended for inference with explicit force outputs.
        host_cell : numpy.ndarray, optional
            Host mirror of ``cell`` carrying the values that were uploaded to
            the device for THIS call, or None when the caller has no trusted
            host copy. When both this frame's snapshot and ``host_cell``
            exist, the cache-invalidation cell comparison runs on the host
            without a device sync; otherwise the original GPU ``torch.equal``
            path decides. The mirror must hold the same values as ``cell`` --
            passing a stale mirror can silently reuse a wrong Verlet cache.

        Returns
        -------
        ii : torch.Tensor
            Local center indices with shape (E,), int64.
        jj : torch.Tensor
            Local neighbor indices with shape (E,), int64.
        geometry : torch.Tensor
            Integer lattice shifts with shape (E, 3), int64, or displacement
            vectors with the dtype of ``positions`` when ``return_vectors``
            is True. All outputs reside on the device of ``positions`` and
            retain their values across subsequent searches.
        """
        from vesin_torch import (
            NeighborList,
        )

        device = positions.device
        key = (float(rcut), device)
        if self._key != key:
            self._frames.clear()
            self._key = key
        calculator, previous_cell, previous_host = self._frames.get(
            frame, (None, None, None)
        )
        # Cell deformation can bring uncached periodic images inside the
        # cutoff even when Vesin's single-cell displacement bound permits
        # reuse. A trusted host mirror decides without synchronizing the GPU;
        # without one, comparing CUDA cells synchronizes this decision.
        cells_match = (
            np.array_equal(host_cell, previous_host)
            if host_cell is not None and previous_host is not None
            else (
                cell is not None
                and previous_cell is not None
                and torch.equal(cell, previous_cell)
            )
        )
        if calculator is None or (
            self.skin > 0
            and ((cell is None) != (previous_cell is None) or not cells_match)
        ):
            calculator = NeighborList(
                cutoff=float(rcut),
                full_list=True,
                sorted=self.sorted,
                skin=self.skin,
            )
            # Independent snapshots detect in-place updates to the input; the
            # host snapshot corresponds to the value uploaded for this call.
            previous_cell = (
                cell.detach().clone() if self.skin > 0 and cell is not None else None
            )
            previous_host = (
                np.array(host_cell, copy=True)
                if self.skin > 0 and host_cell is not None
                else None
            )
            self._frames[frame] = calculator, previous_cell, previous_host
        box = (
            cell
            if cell is not None
            else torch.zeros((3, 3), dtype=positions.dtype, device=device)
        )
        with torch.device(device):
            try:
                ii, jj, geometry = calculator.compute(
                    points=positions.detach(),
                    box=box.detach(),
                    periodic=cell is not None,
                    quantities="ijD" if return_vectors else "ijS",
                )
            except RuntimeError as error:
                # An empty CUDA Verlet cache can expose a null pair buffer that
                # PyTorch cannot wrap. An uncached search supplies valid empty
                # tensors; nonempty results and unrelated errors still fail.
                if (
                    device.type != "cuda"
                    or self.skin == 0
                    or str(error)
                    != "The specified pointer resides on host memory and is not "
                    "registered with any CUDA device."
                ):
                    raise
                ii, jj, geometry = NeighborList(
                    cutoff=float(rcut), full_list=True
                ).compute(
                    points=positions.detach(),
                    box=box.detach(),
                    periodic=cell is not None,
                    quantities="ijD" if return_vectors else "ijS",
                )
                if ii.numel() != 0:
                    raise
        if not return_vectors:
            geometry = geometry.to(torch.int64)
        return ii.to(torch.int64), jj.to(torch.int64), geometry.reshape(-1, 3)
