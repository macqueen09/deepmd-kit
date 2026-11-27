# SPDX-License-Identifier: LGPL-3.0-or-later
"""Host-mirror cell invalidation (plan H1).

The trusted host mirror lets the Vesin cache and the candidate-row cache
decide invalidation on the host without reading GPU cells. These tests pin
the semantics: value (not identity) comparison, copy snapshots that catch
in-place edits, per-frame isolation, periodic switching, and the single-atom
multi-image shrink counterexample of upstream #6085.
"""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)

import numpy as np
import pytest
import torch

from deepmd.pt_expt.utils.vesin_search import (
    VesinNeighborSearch,
)

RCUT = 2.5


def _load_native_op_if_needed() -> bool:
    """注册候选行算子: 本树构建产物或安装目录里找, 找不到返回 False."""
    from deepmd.pt_expt.utils.candidate_rows import (
        candidate_rows_available,
    )

    if candidate_rows_available():
        return True
    tree = Path(__file__).resolve().parents[4]
    for built in (
        tree / "source" / "build" / "op" / "pt" / "libdeepmd_op_pt.so",
        tree / "deepmd" / "lib" / "libdeepmd_op_pt.so",
    ):
        if built.is_file():
            torch.ops.load_library(str(built))
            return candidate_rows_available()
    return False


def _system(nloc: int = 16, seed: int = 0, edge: float = 6.0):
    rng = np.random.default_rng(seed)
    cell = torch.eye(3, dtype=torch.float64) * edge
    pos = torch.from_numpy(rng.random((nloc, 3)) * (edge - 1.0)).to(torch.float64)
    return pos, cell


def _calculator(search: VesinNeighborSearch, frame: int = 0):
    return search._frames.get(frame, (None, None, None))[0]


def test_host_mirror_compares_values_not_identity() -> None:
    # 同值新数组(不同对象)必须复用; 变值必须重建 -- 不能退化为身份判断。
    pos, cell = _system()
    search = VesinNeighborSearch(skin=1.0)
    search.compute(pos, cell, RCUT, host_cell=cell.numpy())
    first = _calculator(search)
    same_values = (cell * 1.0).numpy()
    assert same_values is not cell.numpy()
    search.compute(pos, cell, RCUT, host_cell=same_values)
    assert _calculator(search) is first
    grown = (cell * 1.1).numpy()
    search.compute(pos, cell, RCUT, host_cell=grown)
    assert _calculator(search) is not first


def test_inplace_host_box_edit_is_detected() -> None:
    # 快照是拷贝: 调用者原位改盒后, 新值对旧快照必然不等 -> 重建。
    pos, cell = _system(seed=1)
    host = cell.clone().numpy()
    search = VesinNeighborSearch(skin=1.0)
    search.compute(pos, cell, RCUT, host_cell=host)
    first = _calculator(search)
    host[0, 0] *= 1.25  # 原位修改调用者的数组
    search.compute(pos, cell * 1.0, RCUT, host_cell=host)
    assert _calculator(search) is not first


def test_tilted_cell_change_is_detected() -> None:
    pos, _ = _system(seed=2)
    tilted = torch.tensor(
        [[6.0, 0.5, 0.0], [0.0, 6.0, 0.4], [0.1, 0.0, 6.0]], dtype=torch.float64
    )
    search = VesinNeighborSearch(skin=1.0)
    search.compute(pos, tilted, RCUT, host_cell=tilted.numpy())
    first = _calculator(search)
    # 同值新数组: 复用
    search.compute(pos, tilted.clone(), RCUT, host_cell=tilted.detach().clone().numpy())
    assert _calculator(search) is first
    other = tilted.clone()
    other[2, 1] += 0.7
    search.compute(pos, other, RCUT, host_cell=other.numpy())
    assert _calculator(search) is not first


def test_frame_slots_stay_isolated() -> None:
    # frame 槽各自独立: 改 frame 1 的盒不能动 frame 0 的计算器。
    pos, cell = _system(seed=3)
    search = VesinNeighborSearch(skin=1.0)
    search.compute(pos, cell, RCUT, frame=0, host_cell=cell.numpy())
    zero = _calculator(search, 0)
    other = cell * 1.2
    search.compute(pos, other, RCUT, frame=1, host_cell=other.numpy())
    assert _calculator(search, 1) is not None
    assert _calculator(search, 0) is zero
    grown = cell * 1.3
    search.compute(pos, grown, RCUT, frame=1, host_cell=grown.numpy())
    assert _calculator(search, 0) is zero


def test_periodic_none_switch_rebuilds() -> None:
    # 周期 <-> 非周期切换: 镜像 None 与值之间的转移都要重建。
    pos, cell = _system(seed=4)
    search = VesinNeighborSearch(skin=1.0)
    search.compute(pos, cell, RCUT, host_cell=cell.numpy())
    first = _calculator(search)
    search.compute(pos, None, RCUT)
    second = _calculator(search)
    assert second is not first
    search.compute(pos, cell, RCUT, host_cell=cell.numpy())
    assert _calculator(search) is not second


def test_mirror_then_no_mirror_stays_exact() -> None:
    # 有镜像调用后无镜像调用: 回退 GPU 比较仍精确, 不会把旧缓存误判为有效。
    pos, cell = _system(seed=5)
    search = VesinNeighborSearch(skin=1.0)
    search.compute(pos, cell, RCUT, host_cell=cell.numpy())
    first = _calculator(search)
    # 无镜像 + 同 GPU 盒 -> 复用
    search.compute(pos, cell, RCUT)
    assert _calculator(search) is first
    # 无镜像 + 变化的 GPU 盒 -> 重建(GPU torch.equal 路径)
    grown = cell * 1.05
    search.compute(pos, grown, RCUT)
    assert _calculator(search) is not first


def test_single_atom_multi_image_shrink_matches_fresh() -> None:
    # 上游 #6085 反例: 晶胞边长小于 rcut, 单原子靠多周期像成为邻居;
    # 收缩/膨胀晶胞必须重建且结果与全新搜索一致。
    rcut = 2.5
    pos = torch.zeros((1, 3), dtype=torch.float64)

    def neighbors(cell_edge: float, search: VesinNeighborSearch):
        cell = torch.eye(3, dtype=torch.float64) * cell_edge
        ii, jj, _ = search.compute(
            pos, cell, rcut, host_cell=cell.numpy(), return_vectors=False
        )
        return ii.numel()

    search = VesinNeighborSearch(skin=1.0)
    n_wide = neighbors(2.0, search)
    fresh_wide = VesinNeighborSearch(skin=1.0)
    assert n_wide == neighbors(2.0, fresh_wide)
    assert n_wide > 0  # 自身的周期像在截止内
    n_narrow = neighbors(1.6, search)
    fresh_narrow = VesinNeighborSearch(skin=1.0)
    assert n_narrow == neighbors(1.6, fresh_narrow)
    assert n_narrow > n_wide  # 更小的晶胞 -> 更多的周期像
    n_back = neighbors(2.0, search)
    assert n_back == n_wide


class TestCandidateRowHostMirror:
    """Candidate-row cache layer of the host mirror."""

    def test_valid_for_uses_host_values(self) -> None:
        from deepmd.pt_expt.utils.candidate_rows import (
            CandidateRowCache,
        )

        pos, cell = _system(seed=6)
        cache = CandidateRowCache(RCUT, 1.0, pos.shape[0])
        cache.rebuild(pos, cell, host_cell=cell.numpy())
        small = pos + 0.01
        # 同值新数组: 有效
        assert cache.valid_for(small, cell, host_cell=(cell * 1.0).numpy())
        # 主机值变化: 无效(不读 GPU 就能判)
        assert not cache.valid_for(small, cell, host_cell=(cell * 1.01).numpy())
        # 原位改主机数组: 无效
        host = cell.numpy().copy()
        assert cache.valid_for(small, cell, host_cell=host)
        host[1, 1] *= 1.2
        assert not cache.valid_for(small, cell, host_cell=host)

    def test_rebuilt_without_mirror_falls_back_to_gpu(self) -> None:
        from deepmd.pt_expt.utils.candidate_rows import (
            CandidateRowCache,
        )

        pos, cell = _system(seed=7)
        cache = CandidateRowCache(RCUT, 1.0, pos.shape[0])
        cache.rebuild(pos, cell)  # 无主机镜像
        assert cache.ref_cell_host is None
        small = pos + 0.01
        # 之后带镜像调用: 存量快照缺失 -> GPU 比较仍然精确
        assert cache.valid_for(small, cell, host_cell=cell.numpy())
        assert not cache.valid_for(small, cell * 1.02, host_cell=(cell * 1.02).numpy())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="builder-level mirror needs CUDA"
)
class TestBuilderLevelMirror:
    """The builder threading must not change the produced graphs."""

    def test_canonical_hot_path_unchanged_by_mirror(self) -> None:
        assert _load_native_op_if_needed(), "native candidate-row op unavailable"
        from deepmd.pt_expt.utils.vesin_graph_builder import (
            build_canonical_graph_vesin,
        )

        rng = np.random.default_rng(8)
        cell = torch.eye(3, dtype=torch.float64, device="cuda") * 6.0
        frames = [
            torch.from_numpy(rng.random((24, 3)) * 5.0).to("cuda", torch.float64)
            + torch.tensor([0.0, 0.0, i * 0.01], device="cuda")
            for i in range(3)
        ]
        with_mirror = VesinNeighborSearch(skin=1.0)
        without = VesinNeighborSearch(skin=1.0)
        host = cell.cpu().numpy()
        for frame in frames:
            g1 = build_canonical_graph_vesin(
                frame.unsqueeze(0),
                cell.unsqueeze(0),
                RCUT,
                with_mirror,
                with_source_csr=False,
                host_box=host,
            )
            g2 = build_canonical_graph_vesin(
                frame.unsqueeze(0),
                cell.unsqueeze(0),
                RCUT,
                without,
                with_source_csr=False,
            )
            assert torch.equal(g1.destination_row_ptr, g2.destination_row_ptr)
            assert torch.equal(g1.source, g2.source)
            assert torch.equal(g1.edge_vec, g2.edge_vec)
