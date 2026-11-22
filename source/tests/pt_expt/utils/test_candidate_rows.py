# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from unittest import (
    mock,
)

import numpy as np
import pytest
import torch

from deepmd.pt_expt.utils.canonical_graph import (
    validate_canonical_graph_shapes,
)

vesin_builder = pytest.importorskip("deepmd.pt_expt.utils.vesin_graph_builder")
from deepmd.pt_expt.utils.vesin_search import (
    VesinNeighborSearch,
)
from deepmd.pt_expt.utils.candidate_rows import (
    candidate_rows_available,
)

if not candidate_rows_available():
    # 算子尚未注册时, 尝试加载本树构建出的算子库; 没有构建产物则跳过.
    _built = Path(__file__).resolve().parents[3] / "build/op/pt/libdeepmd_op_pt.so"
    if _built.is_file():
        torch.ops.load_library(str(_built))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not candidate_rows_available(),
    reason="candidate-row hot path needs CUDA and the native operator",
)


def _system(nloc: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    cell = torch.eye(3, dtype=torch.float64, device="cuda") * 6.0
    pos = torch.from_numpy(rng.random((nloc, 3)) * 5.0).to("cuda", torch.float64)
    return pos, cell


def _frames():
    """固定种子的同一组扰动帧: 两态对照必须吃完全相同的输入."""
    pos, _ = _system()
    noise = [
        torch.from_numpy(
            np.random.default_rng(11 + i).standard_normal((32, 3)) * 0.02
        ).to("cuda", torch.float64)
        for i in range(4)
    ]
    return [pos + n for n in noise]


def _graphs(with_hot: bool):
    """Build the same frames through the hot path and the reference path."""
    _, cell = _system()
    search = VesinNeighborSearch(skin=1.0)
    frames = _frames()

    def run(frame):
        coord = frame.unsqueeze(0)
        box = cell.unsqueeze(0)
        if with_hot:
            return vesin_builder.build_canonical_graph_vesin(
                coord, box, 2.5, search, with_source_csr=False
            )
        with mock.patch(
            "deepmd.pt_expt.utils.candidate_rows.candidate_rows_available",
            return_value=False,
        ):
            return vesin_builder.build_canonical_graph_vesin(
                coord, box, 2.5, search, with_source_csr=False
            )

    return [run(f) for f in frames], search


def _payload(graph):
    """(center, source, 几何) 行, 按三键排序以便逐位多重集对照."""
    count = int(graph.destination_row_ptr[-1])
    center = torch.repeat_interleave(
        torch.arange(32, device="cuda"), graph.destination_row_ptr.diff()
    )
    keys = (
        torch.cat(
            (
                center[:, None].to(torch.float64),
                graph.source[:count].to(torch.int64)[:, None].to(torch.float64),
                graph.edge_vec[:count].to(torch.float64),
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    return keys[np.lexsort((keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))]


def test_hot_frames_match_reference_bitwise() -> None:
    hot, _ = _graphs(with_hot=True)
    ref, _ = _graphs(with_hot=False)
    assert len(hot) == len(ref) == 4
    for h, r in zip(hot, ref, strict=True):
        validate_canonical_graph_shapes(h, 32)
        # 行指针与每行边数逐位一致
        assert torch.equal(h.destination_row_ptr, r.destination_row_ptr)
        # 边多重集(含周期像几何)逐位一致
        np.testing.assert_array_equal(_payload(h), _payload(r))
        # 普通请求的源 CSR 两槽占位语义一致
        assert torch.equal(h.source_row_ptr, r.source_row_ptr)
        assert h.source_row_ptr.shape == (2,)
        assert h.source.shape[0] >= 2


def test_cache_rebuild_triggers() -> None:
    pos, cell = _system(seed=3)
    search = VesinNeighborSearch(skin=1.0)
    coord = (pos, cell.unsqueeze(0))

    def build(p):
        return vesin_builder.build_canonical_graph_vesin(
            p.unsqueeze(0), coord[1], 2.5, search, with_source_csr=False
        )

    build(pos)
    first = search.candidate_row_cache
    assert first is not None
    ref_before = first.ref_pos.clone()
    # 位移远小于 skin/2: 缓存不重建, 参考坐标不变
    build(pos + 0.05)
    assert search.candidate_row_cache is first
    assert torch.equal(search.candidate_row_cache.ref_pos, ref_before)
    # 位移超过 skin/2: 重建, 参考坐标更新
    far = pos + 0.6
    build(far)
    rebuilt = search.candidate_row_cache
    assert rebuilt is first
    assert not torch.equal(rebuilt.ref_pos, ref_before)
    assert torch.allclose(rebuilt.ref_pos, far)
    # 晶胞变化: 重建(参考晶胞更新)
    grown = cell * 1.2
    vesin_builder.build_canonical_graph_vesin(
        far.unsqueeze(0), grown.unsqueeze(0), 2.5, search, with_source_csr=False
    )
    assert torch.equal(search.candidate_row_cache.ref_cell, grown)


def test_atomic_request_keeps_reference_path() -> None:
    pos, cell = _system(seed=5)
    search = VesinNeighborSearch(skin=1.0)
    graph = vesin_builder.build_canonical_graph_vesin(
        pos.unsqueeze(0), cell.unsqueeze(0), 2.5, search
    )
    # with_source_csr=True(逐原子维里路线)不启用热路径: 源 CSR 是真实行指针
    assert search.candidate_row_cache is None
    assert int(graph.source_row_ptr[-1]) == int(graph.destination_row_ptr[-1])


def test_zero_edge_guard_slots() -> None:
    # 两个相距很远的原子: 零边, 热路径与参考路径都给出两槽零占位
    cell = torch.eye(3, dtype=torch.float64, device="cuda") * 20.0
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=torch.float64, device="cuda"
    )
    search = VesinNeighborSearch(skin=1.0)
    hot = vesin_builder.build_canonical_graph_vesin(
        pos.unsqueeze(0), cell.unsqueeze(0), 2.5, search, with_source_csr=False
    )
    assert int(hot.destination_row_ptr[-1]) == 0
    assert hot.source.shape[0] == 2
    assert torch.count_nonzero(hot.source) == 0
    assert torch.count_nonzero(hot.edge_vec) == 0
    validate_canonical_graph_shapes(hot, 2)


def test_virtual_atoms_fall_back() -> None:
    pos, cell = _system(seed=7)
    search = VesinNeighborSearch(skin=1.0)
    mask = torch.ones(1, pos.shape[0], dtype=torch.bool, device="cuda")
    mask[0, 3] = False  # 一个虚拟原子
    graph = vesin_builder.build_canonical_graph_vesin(
        pos.unsqueeze(0),
        cell.unsqueeze(0),
        2.5,
        search,
        node_mask=mask,
        with_source_csr=False,
    )
    assert search.candidate_row_cache is None
    assert int(graph.destination_row_ptr[3 + 1]) == int(graph.destination_row_ptr[3])
