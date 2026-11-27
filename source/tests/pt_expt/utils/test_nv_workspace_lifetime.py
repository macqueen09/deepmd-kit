# SPDX-License-Identifier: LGPL-3.0-or-later
"""Search workspace must not overlap geometry construction after decode."""

import weakref

import pytest
import torch

from deepmd.pt_expt.utils import (
    nv_graph_builder as builder,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 1, 3])
def test_decoded_edges_release_search_workspace(monkeypatch, device, edges):
    pytest.importorskip("nvalchemiops")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    references = []
    original_geometry = builder.neighbor_graph_from_ijs

    def search(coord, box, rcut, start_capacity, node_index):
        matrix = torch.ones((3, 2), dtype=torch.int32, device=device)
        counts = torch.tensor([int(i < edges) for i in range(3)], device=device)
        shifts = torch.zeros((3, 2, 3), dtype=torch.int32, device=device)
        references.extend(weakref.ref(x) for x in (matrix, counts, shifts, node_index))
        return coord, box, matrix, counts, shifts

    def geometry(*args, **kwargs):
        assert len(references) == 4
        assert all(ref() is None for ref in references)
        return original_geometry(*args, **kwargs)

    monkeypatch.setattr(builder, "nv_search_matrix", search)
    monkeypatch.setattr(builder, "neighbor_graph_from_ijs", geometry)
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device
    )
    graph = builder.build_neighbor_graph_nv(
        coord, torch.zeros((1, 3), dtype=torch.int64, device=device), None, 2.5
    )
    assert int(graph.edge_mask.sum()) == edges
    assert torch.isfinite(graph.edge_vec).all()
